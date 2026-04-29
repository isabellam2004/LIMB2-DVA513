// ==========================================
// PROJECT LIMB - Rehabilitation Vision System
// Complete two-stage BlazePose pipeline
//
// HOW THE SYSTEM WORKS:
//   The camera feeds two parallel AI pipelines simultaneously:
//
//   Stage 1 - Pose Detector (224x224):
//     Scans the full image to find WHERE the person is.
//     Outputs a bounding box (a rectangle around the body).
//
//   Stage 2 - Pose Landmark (256x256):
//     Receives a tightly cropped image of just the person.
//     Outputs 33 precise body keypoints (joints) with 3D coordinates.
//
// Compile using:
//   g++ -std=c++17 limb_feature_extraction.cpp -o limb \
//       $(pkg-config --cflags --libs opencv4) \
//       -ldepthai-core -ldepthai-opencv
// ==========================================


#include <iostream>   
#include <fstream>    
#include <sstream>   
#include <cmath>     
#include <vector>     
#include <string>    
#include <chrono>     
#include <algorithm>  
#include <filesystem> 

// Make sure the mathematical constant Pi is defined for our angle calculations.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#include "depthai/depthai.hpp" 
#include <opencv2/opencv.hpp>  

// Using namespaces so we don't have to type "std::" or "cv::" before every command.
using namespace std;
using namespace std::chrono;
using namespace cv;
namespace fs = std::filesystem; // Short name for the filesystem library


// CONSTANTS

constexpr float VISIBILITY_THRESHOLD  = 0.5f;   // Minimum confidence required to trust a detected joint (50%)
constexpr float TORSO_OFFSET_MM       = 300.0f; // Virtual spine length used as a reference for shoulder flexion
constexpr int   DETECTOR_INPUT_SIZE   = 224;    // AI person detector requires 224x224 input
constexpr int   LANDMARK_INPUT_SIZE   = 256;    // AI joint locator requires 256x256 input
constexpr int   NUM_ANCHORS           = 2254;   // Number of predefined search grid boxes for the BlazePose detector
constexpr float DETECTION_THRESHOLD   = 0.5f;   // Minimum confidence to confirm a person is in the frame
constexpr float CROP_PADDING          = 0.20f;  // Extra padding added when cropping around the person
constexpr double TIME_PER_RECORDING   = 8.0;    // Seconds per movement (8s gives more meaningful GA data)
constexpr double ARM_MATCH_TOLERANCE  = 50.0;   // Millimeter tolerance for biometric matching

// DATA STRUCTURES

// A structure to hold a point in 3D space.
struct Point3D { 
    float x; 
    float y; 
    float z; 
};

// A structure to store where the AI thinks the person is (Bounding Box).
struct BoundingBoxDetection {
    float centerX; 
    float centerY; 
    float width; 
    float height; 
    float confidenceScore;
    bool  isValidPerson;
};

// A structure to define the area we want to zoom in on (crop).
struct CropRectangle { 
    float x1; 
    float y1; 
    float x2; 
    float y2; 
};

// A structure used by the AI to scan specific regions.
struct AnchorPoint { 
    float centerX; 
    float centerY; 
};

// A structure that holds all the math data for one single frame of video.
struct BiometricFrame {
    double timestamp;
    int    userId;
    string sessionType;
    double upperArmLength;
    double forearmLength;
    double elbowFlexionAngle;
    double shoulderFlexionAngle;
    double wristVelocity;
    double wristAcceleration;
    double wristJerk;
};

// Saved patient profile read from JSON files
struct PatientProfile {
    int    patientId;
    double averageUpperArmLength;
    double averageForearmLength;
    double maxElbowAngle;
    double maxJerk;
};

// Handles terminal menu choices.
struct SessionConfig {
    string sessionType;
    string informationMessage;
    bool   isValid;
};

// ==========================================
// MENU LOGIC (Separated from main() for clarity)
// ==========================================

SessionConfig configureSession(int menuChoice, int patientId) {
    if (menuChoice == 1) {
        if (patientId <= 0) {
            return {"", "Invalid patient ID: " + to_string(patientId), false};
        }
        return {"Training",
                "Starting CALIBRATION for Patient " + to_string(patientId) +
                " (Will record 9 JSON files for backend)", true};
    }
    if (menuChoice == 2) {
        return {"Live_Tracking",
                "Starting LIVE TRACKING (Identifies patient automatically via biometrics)", true};
    }
    return {"", "Invalid menu choice: " + to_string(menuChoice), false};
}

// ==========================================
// KINEMATIC ANALYSIS
// ==========================================

class KinematicAnalyzer {
public:
    // This function calculates the straight-line distance between two points in 3D space.
    static double calculateDistanceBetweenPoints(Point3D point1, Point3D point2) {
        return sqrt(pow(point2.x - point1.x, 2) + pow(point2.y - point1.y, 2) + pow(point2.z - point1.z, 2));
    }

    // This function calculates the angle at the middle joint (point2).
    static double calculateAngleBetweenThreePoints(Point3D point1, Point3D point2, Point3D point3) {
        Point3D vectorA = {point1.x - point2.x, point1.y - point2.y, point1.z - point2.z};
        Point3D vectorB = {point3.x - point2.x, point3.y - point2.y, point3.z - point2.z};
        
        double dotProduct  = (vectorA.x * vectorB.x) + (vectorA.y * vectorB.y) + (vectorA.z * vectorB.z);
        double magnitudeA = sqrt(vectorA.x * vectorA.x + vectorA.y * vectorA.y + vectorA.z * vectorA.z);
        double magnitudeB = sqrt(vectorB.x * vectorB.x + vectorB.y * vectorB.y + vectorB.z * vectorB.z);
        
        if (magnitudeA == 0 || magnitudeB == 0) return 0.0;
        
        return acos(max(-1.0, min(1.0, dotProduct / (magnitudeA * magnitudeB)))) * (180.0 / M_PI);
    }
};

// ==========================================
// AI HELPER FUNCTIONS (BlazePose)
// ==========================================

// Generates 2254 anchor points according to BlazePose specifications
vector<AnchorPoint> generateSearchAnchors() {
    vector<AnchorPoint> anchorList;
    anchorList.reserve(NUM_ANCHORS);
    const int strideSizes[] = {8, 16, 32, 32, 32};
    
    for (int stride : strideSizes) {
        int gridHeight = (int)ceil((float)DETECTOR_INPUT_SIZE / stride);
        int gridWidth = (int)ceil((float)DETECTOR_INPUT_SIZE / stride);
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                for (int k = 0; k < 2; k++) {
                    anchorList.push_back({(x + 0.5f) / gridWidth, (y + 0.5f) / gridHeight});
                }
            }
        }
    }
    return anchorList;
}

// Decodes the detector's output and returns the best bounding box detection
BoundingBoxDetection decodeBestPersonDetection(const vector<float>& regressionData,
                                               const vector<float>& confidenceScores,
                                               const vector<AnchorPoint>& anchors)
{
    if ((int)regressionData.size() < NUM_ANCHORS * 12 || (int)confidenceScores.size() < NUM_ANCHORS) {
        return {0, 0, 0, 0, 0, false};
    }

    BoundingBoxDetection bestDetection = {0, 0, 0, 0, -1.0f, false};
    
    for (int i = 0; i < NUM_ANCHORS; i++) {
        float percentageScore = 1.0f / (1.0f + exp(-confidenceScores[i]));  // Sigmoid activation
        
        if (percentageScore < DETECTION_THRESHOLD) continue;
        
        float boxCenterX = anchors[i].centerX + regressionData[i * 12 + 0] / DETECTOR_INPUT_SIZE;
        float boxCenterY = anchors[i].centerY + regressionData[i * 12 + 1] / DETECTOR_INPUT_SIZE;
        float boxWidth   =                      regressionData[i * 12 + 2] / DETECTOR_INPUT_SIZE;
        float boxHeight  =                      regressionData[i * 12 + 3] / DETECTOR_INPUT_SIZE;
        
        if (percentageScore > bestDetection.confidenceScore) {
            bestDetection = {boxCenterX, boxCenterY, boxWidth, boxHeight, percentageScore, true};
        }
    }
    return bestDetection;
}

// Calculates a square crop rectangle with safety padding
CropRectangle computeZoomArea(const BoundingBoxDetection& detection) {
    float maximumSize = max(detection.width, detection.height) * (1.0f + 2.0f * CROP_PADDING);
    auto keepWithinBorders = [](float value){ return max(0.0f, min(1.0f, value)); };
    
    return { keepWithinBorders(detection.centerX - maximumSize * 0.5f),
             keepWithinBorders(detection.centerY - maximumSize * 0.5f),
             keepWithinBorders(detection.centerX + maximumSize * 0.5f),
             keepWithinBorders(detection.centerY + maximumSize * 0.5f) };
}

// Extracts a single keypoint and maps it back to the full 1080p coordinates
bool extractJointCoordinates(const vector<float>& neuralNetworkData, int jointIndex,
                              float imageWidth, float imageHeight,
                              const CropRectangle& cropArea,
                              Point3D& point3D, Point& pixelPoint)
{
    int baseIndex = jointIndex * 5;
    if (baseIndex + 4 >= (int)neuralNetworkData.size()) return false;
    if (neuralNetworkData[baseIndex + 3] < VISIBILITY_THRESHOLD) return false;

    float normalizedX = neuralNetworkData[baseIndex];
    float normalizedY = neuralNetworkData[baseIndex + 1];
    float normalizedZ = neuralNetworkData[baseIndex + 2];

    // Map from crop space back to the full image space
    float cropWidth = cropArea.x2 - cropArea.x1;
    float cropHeight = cropArea.y2 - cropArea.y1;
    float fullX = cropArea.x1 + normalizedX * cropWidth;
    float fullY = cropArea.y1 + normalizedY * cropHeight;

    pixelPoint  = Point((int)(fullX * imageWidth), (int)(fullY * imageHeight));
    point3D = {fullX * 1000.0f, fullY * 1000.0f, normalizedZ * 1000.0f};
    return true;
}

// ==========================================
// JSON EXPORT
// ==========================================

void saveRecordingToJSON(const vector<BiometricFrame>& recordedData, int patientId, const string& suffix) {
    string filename = "patient_" + to_string(patientId) + "_" + suffix + ".json";
    ofstream jsonFile(filename);
    
    if (!jsonFile.is_open()) {
        cerr << "[ERROR] Could not open '" << filename << "' for writing.\n";
        return;
    }
    
    jsonFile << "[\n";
    for (size_t i = 0; i < recordedData.size(); i++) {
        jsonFile << "  {\n"
                 << "    \"time\": "               << recordedData[i].timestamp            << ",\n"
                 << "    \"upper_arm_length\": "   << recordedData[i].upperArmLength       << ",\n"
                 << "    \"forearm_length\": "     << recordedData[i].forearmLength        << ",\n"
                 << "    \"elbow_flexion\": "      << recordedData[i].elbowFlexionAngle    << ",\n"
                 << "    \"shoulder_flexion\": "   << recordedData[i].shoulderFlexionAngle << ",\n"
                 << "    \"wrist_velocity\": "     << recordedData[i].wristVelocity        << ",\n"
                 << "    \"wrist_acceleration\": " << recordedData[i].wristAcceleration    << ",\n"
                 << "    \"wrist_jerk\": "         << recordedData[i].wristJerk            << "\n"
                 << "  }";
        
        if (i < recordedData.size() - 1) {
            jsonFile << ",\n"; 
        } else {
            jsonFile << "\n";
        }
    }
    jsonFile << "]\n";
    jsonFile.close();
    cout << "[EXPORT] Successfully saved Recording: " << filename << " (" << recordedData.size() << " frames)\n";
}

// ==========================================
// JSON PARSING (Reads local files for auto-ID)
// ==========================================

// Reads a specific numerical field from a simple JSON line.
// Example: parses "    \"upper_arm_length\": 312.4," → 312.4
static double parseJsonField(const string& jsonLine, const string& searchKey) {
    size_t position = jsonLine.find("\"" + searchKey + "\":");
    if (position == string::npos) return -1.0;
    
    size_t valueStart = jsonLine.find(":", position) + 1;
    
    // Skip empty spaces
    while (valueStart < jsonLine.size() && (jsonLine[valueStart] == ' ' || jsonLine[valueStart] == '\t')) {
        valueStart++;
    }
    return stod(jsonLine.substr(valueStart));
}

// Reads all patient_X_*.json files and returns average arm lengths per patient
vector<PatientProfile> loadAllPatientProfiles() {
    vector<PatientProfile> profileList;

    for (const auto& entry : fs::directory_iterator(".")) {
        string filename = entry.path().filename().string();

        // Filter by filename: must start with "patient_" and end with ".json"
        if (filename.rfind("patient_", 0) != 0) continue;
        if (filename.find(".json") == string::npos) continue;

        // Extract patient ID from the filename (e.g., patient_101_move1_slow.json → 101)
        size_t firstUnderscore  = filename.find('_');
        size_t secondUnderscore = filename.find('_', firstUnderscore + 1);
        if (firstUnderscore == string::npos || secondUnderscore == string::npos) continue;

        int filePatientId = -1;
        try {
            filePatientId = stoi(filename.substr(firstUnderscore + 1, secondUnderscore - firstUnderscore - 1));
        } catch (...) { 
            continue; 
        }

        // Read the file and collect data
        ifstream jsonFile(filename);
        if (!jsonFile.is_open()) continue;

        double totalUpperArmLength = 0, totalForearmLength = 0, totalElbowAngle = 0, totalJerk = 0;
        int frameCount = 0;
        string jsonLine;

        while (getline(jsonFile, jsonLine)) {
            double extractedValue;
            if ((extractedValue = parseJsonField(jsonLine, "upper_arm_length")) > 0) { 
                totalUpperArmLength += extractedValue; 
                frameCount++; 
            }
            if ((extractedValue = parseJsonField(jsonLine, "forearm_length")) > 0) {
                totalForearmLength += extractedValue;
            }
            if ((extractedValue = parseJsonField(jsonLine, "elbow_flexion")) > 0) {
                totalElbowAngle += extractedValue;
            }
            
            size_t jerkPosition = jsonLine.find("\"wrist_jerk\":");
            if (jerkPosition != string::npos) {
                try { 
                    totalJerk += abs(stod(jsonLine.substr(jsonLine.find(":", jerkPosition) + 1))); 
                } catch (...) {}
            }
        }
        jsonFile.close();

        if (frameCount == 0) continue;

        // If a profile for this ID already exists, merge them (calculate average)
        bool isProfileFound = false;
        for (auto& existingProfile : profileList) {
            if (existingProfile.patientId == filePatientId) {
                existingProfile.averageUpperArmLength = (existingProfile.averageUpperArmLength + totalUpperArmLength / frameCount) / 2.0;
                existingProfile.averageForearmLength  = (existingProfile.averageForearmLength  + totalForearmLength / frameCount) / 2.0;
                existingProfile.maxElbowAngle         = max(existingProfile.maxElbowAngle, totalElbowAngle / frameCount);
                existingProfile.maxJerk               = max(existingProfile.maxJerk, totalJerk / frameCount);
                isProfileFound = true;
                break;
            }
        }
        
        if (!isProfileFound) {
            profileList.push_back({
                filePatientId,
                totalUpperArmLength / frameCount,
                totalForearmLength / frameCount,
                totalElbowAngle / frameCount,
                totalJerk / frameCount
            });
        }
    }
    return profileList;
}

// AUTO-IDENTIFICATION LOGIC

int identifyPatientFromBiometrics(double liveUpperArmLength, double liveForearmLength,
                                   const vector<PatientProfile>& savedProfiles)
{
    int    bestMatchId = -1;
    double bestDifference = ARM_MATCH_TOLERANCE; // Must be under the tolerance to count as a match

    for (const auto& profile : savedProfiles) {
        double currentDifference = abs(liveUpperArmLength - profile.averageUpperArmLength) +
                                   abs(liveForearmLength  - profile.averageForearmLength);
                                   
        if (currentDifference < bestDifference) {
            bestDifference = currentDifference;
            bestMatchId  = profile.patientId;
        }
    }
    return bestMatchId; // Returns -1 if no match was found
}



int main() {
    //  Menu 
    int menuChoice = 0, patientId = -1;

    cout << " PROJECT LIMB: VISION SYSTEM STARTUP   \n"
         << "  1. Training Phase (Record 9 JSON movements)\n"
         << "  2. Live Tracking  (Auto-identify patient)\n"
         << "Select mode: ";
    cin >> menuChoice;

    if (menuChoice == 1) {
        cout << "Enter new Patient ID: ";
        cin  >> patientId;
    }

    SessionConfig currentSession = configureSession(menuChoice, patientId);
    if (!currentSession.isValid) {
        cerr << "[ERROR] " << currentSession.informationMessage << "\n";
        return 1;
    }
    cout << "[INFO] " << currentSession.informationMessage << "\n";

    //  Check if AI blob files exist 
    const string detectionModelPath = "pose_detection_sh4.blob";
    const string landmarkModelPath = "pose_landmark_full_sh4.blob";
    
    for (const auto& filePath : {detectionModelPath, landmarkModelPath}) {
        if (!fs::exists(filePath)) {
            cerr << "[ERROR] Could not find '" << filePath
                 << "'. Ensure it is placed in the same folder as the executable.\n";
            return 1;
        }
        cout << "[OK]   " << filePath << " found.\n";
    }

    //  Load patient profiles from JSON files (for Live Tracking) 
    vector<PatientProfile> loadedPatientProfiles;
    if (currentSession.sessionType == "Live_Tracking") {
        loadedPatientProfiles = loadAllPatientProfiles();
        cout << "[OK]   " << loadedPatientProfiles.size() << " patient profiles loaded.\n";
        
        if (loadedPatientProfiles.empty()) {
            cout << "[WARNING] No saved profiles were found. Run Training mode first.\n";
        }
    }

    //  Anchors and kinematic states 
    auto searchAnchors = generateSearchAnchors();

    vector<BiometricFrame> listOfRecordedFrames;
    int    currentRecordingStep  = 0;

    bool   recordingTimerStarted = false;
    double recordingStepStartTime = 0.0;

    const string recordingNames[9] = {
        "move1_slow", "move1_med", "move1_fast",
        "move2_slow", "move2_med", "move2_fast",
        "move3_slow", "move3_med", "move3_fast"
    };

    Point3D previousWristPosition    = {0.0, 0.0, 0.0};
    double  previousFrameTime        = 0.0;
    double  previousWristVelocity    = 0.0;
    double  previousWristAcceleration = 0.0;
    bool    isFirstFrameEver         = true;
    bool    isPersonDetected         = false;
    CropRectangle currentZoomArea    = {0.f, 0.f, 1.f, 1.f};

    //  DepthAI Camera Pipeline 
    dai::Pipeline cameraPipeline;

    auto colorCameraNode = cameraPipeline.create<dai::node::ColorCamera>();
    colorCameraNode->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    colorCameraNode->setInterleaved(false);
    colorCameraNode->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);
    colorCameraNode->setPreviewSize(DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE);

    //  Branch 1: Person Detector 
    auto neuralNetworkDetector = cameraPipeline.create<dai::node::NeuralNetwork>();
    neuralNetworkDetector->setBlobPath(detectionModelPath);
    neuralNetworkDetector->setNumInferenceThreads(2);   
    neuralNetworkDetector->input.setBlocking(false);    
    colorCameraNode->preview.link(neuralNetworkDetector->input);

    auto videoOutputDetector = cameraPipeline.create<dai::node::XLinkOut>();
    videoOutputDetector->setStreamName("detector_output");
    neuralNetworkDetector->out.link(videoOutputDetector->input);

    //  Branch 2: Landmarker with dynamic cropping 
    auto imageManipulator = cameraPipeline.create<dai::node::ImageManip>();
    imageManipulator->initialConfig.setResize(LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE);
    imageManipulator->initialConfig.setFrameType(dai::ImgFrame::Type::RGB888p);
    imageManipulator->setWaitForConfigInput(false);  // Critical: Prevents the branch from freezing
    imageManipulator->setMaxOutputFrameSize(LANDMARK_INPUT_SIZE * LANDMARK_INPUT_SIZE * 3); 
    colorCameraNode->video.link(imageManipulator->inputImage);

    auto inputManipulatorConfig = cameraPipeline.create<dai::node::XLinkIn>();
    inputManipulatorConfig->setStreamName("manipulator_config");
    inputManipulatorConfig->out.link(imageManipulator->inputConfig);

    auto neuralNetworkLandmarker = cameraPipeline.create<dai::node::NeuralNetwork>();
    neuralNetworkLandmarker->setBlobPath(landmarkModelPath);
    neuralNetworkLandmarker->setNumInferenceThreads(2);
    neuralNetworkLandmarker->input.setBlocking(false);
    imageManipulator->out.link(neuralNetworkLandmarker->input);

    auto videoOutputLandmarker = cameraPipeline.create<dai::node::XLinkOut>();
    videoOutputLandmarker->setStreamName("landmarker_output");
    neuralNetworkLandmarker->out.link(videoOutputLandmarker->input);

    // Raw RGB Video Output for Display
    auto videoOutputRgb = cameraPipeline.create<dai::node::XLinkOut>();
    videoOutputRgb->setStreamName("rgb_video");
    colorCameraNode->video.link(videoOutputRgb->input);

    //  Main Infinite Loop 
    try {
        dai::Device cameraDevice(cameraPipeline);
        cout << "[OK] Camera connected. Press 'q' to exit.\n";

        auto queueDetector          = cameraDevice.getOutputQueue("detector_output",  4, false);
        auto queueLandmarker        = cameraDevice.getOutputQueue("landmarker_output",4, false);
        auto queueRgbVideo          = cameraDevice.getOutputQueue("rgb_video",        4, false);
        auto queueManipulatorConfig = cameraDevice.getInputQueue ("manipulator_config");

        auto programStartTime = steady_clock::now();

        while (true) {

            //  Step 1: Detector – Find the person 
            if (auto incomingDetectionData = queueDetector->tryGet<dai::NNData>()) {
                auto regressionData = incomingDetectionData->getLayerFp16("Identity");
                auto confidenceScores = incomingDetectionData->getLayerFp16("Identity_1");

                auto detectionResult = decodeBestPersonDetection(regressionData, confidenceScores, searchAnchors);
                isPersonDetected = detectionResult.isValidPerson;

                if (detectionResult.isValidPerson) {
                    currentZoomArea = computeZoomArea(detectionResult);
                    dai::ImageManipConfig newCameraConfig;
                    newCameraConfig.setCropRect(currentZoomArea.x1, currentZoomArea.y1,
                                                currentZoomArea.x2, currentZoomArea.y2);
                    newCameraConfig.setResize(LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE);
                    newCameraConfig.setKeepAspectRatio(false);
                    queueManipulatorConfig->send(newCameraConfig);
                }
            }

            //  Step 2: Landmarker + Kinematics 
            auto incomingLandmarkData = queueLandmarker->tryGet<dai::NNData>();
            auto incomingRgbFrame = queueRgbVideo->tryGet<dai::ImgFrame>();

            if (incomingLandmarkData && incomingRgbFrame) {
                Mat displayFrame = incomingRgbFrame->getCvFrame();
                double currentTimeInSeconds = duration_cast<milliseconds>(steady_clock::now() - programStartTime).count() / 1000.0;

                // Start the recording timer using the boolean flag
                if (!recordingTimerStarted) {
                    recordingTimerStarted = true;
                    recordingStepStartTime = currentTimeInSeconds;
                }

                auto neuralNetworkArray = incomingLandmarkData->getLayerFp16("Identity");

                if (neuralNetworkArray.size() < 33 * 5) {
                    cerr << "[WARNING] Landmark size " << neuralNetworkArray.size() << " < 165, skipping.\n";
                    imshow("Project LIMB", displayFrame);
                    if (waitKey(1) == 'q') break;
                    continue;
                }

                float imageWidth = (float)displayFrame.cols;
                float imageHeight = (float)displayFrame.rows;

                // Tracking LEFT arm: shoulder=11, elbow=13, wrist=15
                Point3D pointShoulder, pointElbow, pointWrist;
                Point   pixelShoulder, pixelElbow, pixelWrist;
                
                bool isShoulderVisible = extractJointCoordinates(neuralNetworkArray, 11, imageWidth, imageHeight, currentZoomArea, pointShoulder, pixelShoulder);
                bool isElbowVisible    = extractJointCoordinates(neuralNetworkArray, 13, imageWidth, imageHeight, currentZoomArea, pointElbow,    pixelElbow);
                bool isWristVisible    = extractJointCoordinates(neuralNetworkArray, 15, imageWidth, imageHeight, currentZoomArea, pointWrist,    pixelWrist);

                // Draw physical skeleton overlay
                if (isShoulderVisible && isElbowVisible) line(displayFrame, pixelShoulder, pixelElbow, Scalar(255, 255, 0), 4);
                if (isElbowVisible && isWristVisible)    line(displayFrame, pixelElbow,    pixelWrist, Scalar(255, 255, 0), 4);
                if (isShoulderVisible) circle(displayFrame, pixelShoulder, 8, Scalar(0, 0, 255),  -1);
                if (isElbowVisible)    circle(displayFrame, pixelElbow,    8, Scalar(0, 255, 0),  -1);
                if (isWristVisible)    circle(displayFrame, pixelWrist,    8, Scalar(255, 0, 0),  -1);

                // Kinematic variables
                double currentUpperArmLength = 0, currentForearmLength = 0;
                double elbowFlexionAngle = 0, shoulderFlexionAngle = 0;
                double currentWristVelocity = 0, currentWristAcceleration = 0, currentWristJerk = 0;

                if (isShoulderVisible && isElbowVisible && isWristVisible) {
                    currentUpperArmLength = KinematicAnalyzer::calculateDistanceBetweenPoints(pointShoulder, pointElbow);
                    currentForearmLength  = KinematicAnalyzer::calculateDistanceBetweenPoints(pointElbow,    pointWrist);
                    elbowFlexionAngle     = KinematicAnalyzer::calculateAngleBetweenThreePoints(pointShoulder, pointElbow, pointWrist);

                    Point3D virtualTorsoPoint = {pointShoulder.x, pointShoulder.y + TORSO_OFFSET_MM, pointShoulder.z};
                    shoulderFlexionAngle = KinematicAnalyzer::calculateAngleBetweenThreePoints(virtualTorsoPoint, pointShoulder, pointElbow);

                    if (!isFirstFrameEver) {
                        double timeDifference = currentTimeInSeconds - previousFrameTime;
                        if (timeDifference > 0) {
                            double distanceMoved = KinematicAnalyzer::calculateDistanceBetweenPoints(previousWristPosition, pointWrist);
                            currentWristVelocity = distanceMoved / timeDifference;
                            currentWristAcceleration = (currentWristVelocity - previousWristVelocity) / timeDifference;
                            currentWristJerk = (currentWristAcceleration - previousWristAcceleration) / timeDifference;
                        }
                    }

                    //  TRAINING MODE (JSON Export) 
                    if (currentSession.sessionType == "Training" && currentRecordingStep < 9) {
                        listOfRecordedFrames.push_back({
                            currentTimeInSeconds, patientId, currentSession.sessionType,
                            currentUpperArmLength, currentForearmLength,
                            elbowFlexionAngle, shoulderFlexionAngle,
                            currentWristVelocity, currentWristAcceleration, currentWristJerk
                        });

                        if (currentTimeInSeconds - recordingStepStartTime > TIME_PER_RECORDING) {
                            saveRecordingToJSON(listOfRecordedFrames, patientId,
                                                recordingNames[currentRecordingStep]);
                            listOfRecordedFrames.clear();
                            currentRecordingStep++;
                            recordingStepStartTime = currentTimeInSeconds;
                            
                            if (currentRecordingStep < 9) {
                                cout << "[INFO] Next movement task: "
                                     << recordingNames[currentRecordingStep] << "\n";
                            }
                        }
                    }

                    //  LIVE TRACKING MODE (Auto-ID only) 
                    if (currentSession.sessionType == "Live_Tracking") {
                        // Identify based on the actual loaded profiles
                        if (patientId == -1) {
                            patientId = identifyPatientFromBiometrics(
                                currentUpperArmLength, currentForearmLength, loadedPatientProfiles);
                        }

                        if (patientId != -1) {
                            // In the future, backend logic (GA integration and robotic control)
                            // will be triggered here for the identified patient.
                        }
                    }

                    previousWristPosition    = pointWrist;
                    previousFrameTime        = currentTimeInSeconds;
                    previousWristVelocity    = currentWristVelocity;
                    previousWristAcceleration = currentWristAcceleration;
                    isFirstFrameEver         = false;
                }

                //  HUD DISPLAY 
                putText(displayFrame, "SYSTEM MODE: " + currentSession.sessionType,
                        Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

                if (currentSession.sessionType == "Training") {
                    putText(displayFrame, "PATIENT ID: " + to_string(patientId),
                            Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
                            
                    if (currentRecordingStep < 9) {
                        double timeElapsed = currentTimeInSeconds - recordingStepStartTime;
                        string taskString = "TASK " + to_string(currentRecordingStep + 1) +
                                         "/9: " + recordingNames[currentRecordingStep] +
                                         "  [" + to_string((int)timeElapsed) + "/" +
                                         to_string((int)TIME_PER_RECORDING) + "s]";
                        putText(displayFrame, taskString, Point(20, 150),
                                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 165, 255), 2);
                    } else {
                        putText(displayFrame, "ALL 9 RECORDINGS COMPLETE - Press 'Q'",
                                Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                    }
                } else {
                    if (patientId == -1)
                        putText(displayFrame, "STATUS: SCANNING BIOMETRICS...",
                                Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                    else
                        putText(displayFrame, "STATUS: MATCHED - Patient " + to_string(patientId),
                                Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                }

                putText(displayFrame, "Left Elbow Angle: " + to_string((int)elbowFlexionAngle) + " degrees",
                        Point(20, 110), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

                if (!isPersonDetected)
                    putText(displayFrame, "! NO PERSON DETECTED",
                            Point(20, 190), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                else if (!isShoulderVisible || !isElbowVisible || !isWristVisible)
                    putText(displayFrame, "! LOW VISIBILITY",
                            Point(20, 190), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 165, 255), 2);

                imshow("Project LIMB - Rehabilitation Vision System", displayFrame);
            }

            if (waitKey(1) == 'q') break;
        }

    } catch (const exception& e) {
        cerr << "[CRITICAL ERROR] The camera crashed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}