#if 0
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "depthai/depthai.hpp"
#include <opencv2/opencv.hpp>

using namespace std;                                      // Use standard namespace for strings, vectors, and IO.
using namespace std::chrono;                              // Use chrono for high-resolution timing.
using namespace cv;                                      // Use OpenCV for image processing and visualization.
namespace fs = std::filesystem;                           // Alias for filesystem operations like directory scanning.

constexpr float TORSO_OFFSET_MM       = 300.0f;           // Vertical distance to create a torso point for shoulder angle math.
constexpr double TIME_PER_RECORDING   = 8.0;              // Fixed time limit for each movement capture in Training mode.
constexpr double ARM_MATCH_TOLERANCE  = 50.0;             // Threshold for biometric matching of arm segment lengths.
constexpr float SMOOTHING_ALPHA       = 0.35f;            // EMA filter coefficient; smaller values increase smoothness/latency.
constexpr float MAX_POINT_JUMP_MM     = 180.0f;           // Max allowed distance between frames to reject tracking artifacts.
constexpr float MIN_ARM_SEGMENT_MM    = 80.0f;            // Minimum length for a human arm segment to be considered valid.
constexpr float MAX_ARM_SEGMENT_MM    = 700.0f;           // Maximum length for a human arm segment to be considered valid.
constexpr float MAX_CONNECTED_Z_DIFF_MM = 600.0f;         // Max depth jump allowed between two connected skeletal joints.
constexpr float MAX_HAND_WRIST_Z_DIFF_MM = 450.0f;        // Max depth difference allowed between wrist and finger keypoints.
constexpr float MAX_POSE_ARM_Z_DIFF_MM = 850.0f;          // Max depth allowed for full pose landmarks relative to the arm.
constexpr float DEPTH_FOREGROUND_PERCENTILE = 0.25f;      // Percentile used to select front-most pixels from a depth ROI.

struct Point3D { float x; float y; float z; };            // Structure representing a coordinate in 3D space.

struct NamedPoint3D {                                     // Structure linking a descriptive name to a 3D point.
    string name;                                          // The label for the landmark (e.g., "wrist").
    Point3D point;                                        // The spatial data for the landmark.
};

struct BiometricFrame {                                   // A snapshot of all biometric data for a single frame.
    double timestamp;                                     // Time index relative to session start.
    int    userId;                                        // Unique patient identifier.
    string sessionType;                                   // Type of session (Training or Live).
    double upperArmLength;                                // Distance from shoulder to elbow.
    double forearmLength;                                 // Distance from elbow to wrist.
    double elbowFlexionAngle;                             // Bend angle of the elbow joint.
    double shoulderFlexionAngle;                          // Orientation angle of the humerus.
    double wristVelocity;                                 // Instantaneous speed of the wrist.
    double wristAcceleration;                             // Rate of change in wrist velocity.
    double wristJerk;                                     // Rate of change in wrist acceleration (smoothness).
    std::vector<NamedPoint3D> armLandmarks3D;             // Labels and coordinates for primary arm joints.
    std::vector<Point3D> poseLandmarks3D;                 // Coordinates for the full 17-point body pose.
    std::vector<Point3D> handLandmarks3D;                 // Coordinates for the 21-point hand landmarks.
};

struct PatientProfile {                                   // Stored biometric averages for a specific user.
    int    patientId;                                     // Unique ID for the patient.
    double averageUpperArmLength;                         // Mean humerus length across previous recordings.
    double averageForearmLength;                          // Mean radius/ulna length across previous recordings.
    double maxElbowAngle;                                 // Peak range of motion captured for the elbow.
    double maxJerk;                                       // Peak jerk recorded (lower indicates more fluid motion).
};

struct SessionConfig {                                    // Configuration state for the active program loop.
    string sessionType;                                   // Name of the mode (Training/Live).
    string informationMessage;                            // Status message for the UI.
    bool   isValid;                                       // Success flag for session initialization.
};

const std::vector<std::pair<int, int>> HAND_CONNECTIONS = { // Mapping of which hand points to connect for drawing.
    {0,1}, {1,2}, {2,3}, {3,4},
    {0,5}, {5,6}, {6,7}, {7,8},
    {5,9}, {9,10}, {10,11}, {11,12},
    {9,13}, {13,14}, {14,15}, {15,16},
    {13,17}, {0,17}, {17,18}, {18,19}, {19,20}
};

class KinematicAnalyzer {                                 // Utility class for biometric calculations.
public:
    static double calculateDistanceBetweenPoints(Point3D p1, Point3D p2) { // 3D Euclidean distance between two points.
        return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
    }

    static double calculateAngleBetweenThreePoints(Point3D p1, Point3D p2, Point3D p3) { // Angle at joint p2.
        Point3D vA = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z}; // Vector from joint to first point.
        Point3D vB = {p3.x - p2.x, p3.y - p2.y, p3.z - p2.z}; // Vector from joint to third point.

        double dot  = (vA.x * vB.x) + (vA.y * vB.y) + (vA.z * vB.z); // Dot product of vectors.
        double magA = sqrt(vA.x * vA.x + vA.y * vA.y + vA.z * vA.z); // Magnitude of Vector A.
        double magB = sqrt(vB.x * vB.x + vB.y * vB.y + vB.z * vB.z); // Magnitude of Vector B.

        if (magA == 0 || magB == 0) return 0.0;           // Guard against division by zero.
        return acos(max(-1.0, min(1.0, dot / (magA * magB)))) * (180.0 / M_PI); // Return degrees.
    }
};

static bool isValidNormalizedPoint(const dai::Point2f& point) { // Verify AI keypoint is within 0-1 frame bounds.
    return point.x >= 0.0f && point.x <= 1.0f && point.y >= 0.0f && point.y <= 1.0f;
}

static Point2f normalizedToPixel(const dai::Point2f& point, const Size& size) { // Map 0-1 range to image pixels.
    return Point2f(point.x * static_cast<float>(size.width - 1),
                   point.y * static_cast<float>(size.height - 1));
}

static Point2f scalePixelBetweenFrames(const Point2f& point, const Size& fromSize, const Size& toSize) { // Resize pixel coordinate.
    if (fromSize.width <= 1 || fromSize.height <= 1 || toSize.width <= 1 || toSize.height <= 1) {
        return Point2f(0.0f, 0.0f);
    }

    float xNorm = point.x / static_cast<float>(fromSize.width - 1); // Normalize X.
    float yNorm = point.y / static_cast<float>(fromSize.height - 1); // Normalize Y.
    return Point2f(xNorm * static_cast<float>(toSize.width - 1),
                   yNorm * static_cast<float>(toSize.height - 1));
}

static Point2f clampPixelPoint(const Point2f& point, const Size& size) { // Restrict pixel to image borders.
    return Point2f(
        std::clamp(point.x, 0.0f, static_cast<float>(size.width - 1)),
        std::clamp(point.y, 0.0f, static_cast<float>(size.height - 1))
    );
}

static bool isValid3DPoint(const Point3D& point) {        // Check if depth sensor successfully captured point.
    return point.z > 0.0f;
}

static Point3D invalidPoint3D() {                         // Return an empty point for failed tracking.
    return {0.0f, 0.0f, 0.0f};
}

static void writePoint3DJSON(ofstream& jsonFile, const Point3D& point) { // Format a 3D coordinate for JSON.
    jsonFile << "{\"x\": " << point.x
             << ", \"y\": " << point.y
             << ", \"z\": " << point.z
             << ", \"valid\": " << (isValid3DPoint(point) ? "true" : "false") << "}";
}

static Point3D smoothPoint3D(const Point3D& current, const Point3D& previous, bool hasPrevious) { // EMA Filter.
    if (!isValid3DPoint(current)) {                       // If current frame failed, use last valid frame.
        return hasPrevious ? previous : Point3D{0.0f, 0.0f, 0.0f};
    }

    if (!hasPrevious || !isValid3DPoint(previous)) {      // If first valid frame, initialize filter.
        return current;
    }

    double jump = KinematicAnalyzer::calculateDistanceBetweenPoints(previous, current);
    if (jump > MAX_POINT_JUMP_MM) {                       // Filter out extreme tracking glitches.
        return previous;
    }

    return {                                              // Blend new and old data.
        previous.x + (current.x - previous.x) * SMOOTHING_ALPHA,
        previous.y + (current.y - previous.y) * SMOOTHING_ALPHA,
        previous.z + (current.z - previous.z) * SMOOTHING_ALPHA
    };
}

static std::vector<Point3D> smoothPointVector3D(const std::vector<Point3D>& current, const std::vector<Point3D>& previous) {
    std::vector<Point3D> smoothed;                        // Container for filtered landmarks.
    smoothed.reserve(current.size());                     // Optimize memory allocation.

    for (size_t i = 0; i < current.size(); i++) {         // Iterate and smooth each skeletal joint.
        bool hasPrevious = i < previous.size() && isValid3DPoint(previous[i]);
        Point3D previousPoint = hasPrevious ? previous[i] : Point3D{0.0f, 0.0f, 0.0f};
        smoothed.push_back(smoothPoint3D(current[i], previousPoint, hasPrevious));
    }

    return smoothed;
}

static bool isPlausibleArmGeometry(const Point3D& shoulder, const Point3D& elbow, const Point3D& wrist) { // Biology check.
    if (!isValid3DPoint(shoulder) || !isValid3DPoint(elbow) || !isValid3DPoint(wrist)) {
        return false;                                     // Fail if any arm joint is missing.
    }

    double upperArm = KinematicAnalyzer::calculateDistanceBetweenPoints(shoulder, elbow);
    double forearm = KinematicAnalyzer::calculateDistanceBetweenPoints(elbow, wrist);
    double shoulderElbowZDiff = std::abs(shoulder.z - elbow.z);
    double elbowWristZDiff = std::abs(elbow.z - wrist.z);

    return upperArm >= MIN_ARM_SEGMENT_MM &&              // Check for valid bone lengths.
           upperArm <= MAX_ARM_SEGMENT_MM &&
           forearm >= MIN_ARM_SEGMENT_MM &&
           forearm <= MAX_ARM_SEGMENT_MM &&
           shoulderElbowZDiff <= MAX_CONNECTED_Z_DIFF_MM && // Check for valid depth continuity.
           elbowWristZDiff <= MAX_CONNECTED_Z_DIFF_MM;
}

static void rejectHandDepthOutliers(std::vector<Point3D>& handLandmarks, const Point3D& wrist) { // Depth sanity check.
    if (!isValid3DPoint(wrist)) return;                   // Exit if wrist reference is missing.

    for (auto& point : handLandmarks) {                   // Invalidate fingers too far behind/ahead of wrist.
        if (isValid3DPoint(point) && std::abs(point.z - wrist.z) > MAX_HAND_WRIST_Z_DIFF_MM) {
            point = invalidPoint3D();
        }
    }
}

static void rejectPoseDepthOutliers(std::vector<Point3D>& poseLandmarks, const Point3D& shoulder, const Point3D& elbow, const Point3D& wrist) {
    if (!isValid3DPoint(shoulder) || !isValid3DPoint(elbow) || !isValid3DPoint(wrist)) return;

    float armReferenceZ = (shoulder.z + elbow.z + wrist.z) / 3.0f; // Center of arm depth.
    for (auto& point : poseLandmarks) {                   // Filter full body points based on arm distance.
        if (isValid3DPoint(point) && std::abs(point.z - armReferenceZ) > MAX_POSE_ARM_Z_DIFF_MM) {
            point = invalidPoint3D();
        }
    }
}

std::vector<uint8_t> toPlanar(const cv::Mat& mat) {       // Convert OpenCV image to NN planar format.
    cv::Mat resized;
    cv::resize(mat, resized, cv::Size(224, 224));          // Scaling for Hand Landmark model.
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);    // Color space conversion.

    std::vector<uint8_t> data(224 * 224 * 3);             // Planar buffer (R-R-R, G-G-G, B-B-B).
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);                         // Separate color planes.
    memcpy(data.data(), channels[0].data, 224 * 224);     // Copy Red.
    memcpy(data.data() + 224 * 224, channels[1].data, 224 * 224); // Copy Green.
    memcpy(data.data() + 2 * 224 * 224, channels[2].data, 224 * 224); // Copy Blue.
    return data;
}

Point3D get3DPointFromNormalizedPoint(Point2f normalizedPoint, Mat& depthMat, const std::vector<std::vector<float>>& intrinsics) {
    int dw = depthMat.cols;                               // Get depth image resolution.
    int dh = depthMat.rows;
    if (dw <= 0 || dh <= 0) {                             // Guard against empty sensor data.
        return {0.0f, 0.0f, 0.0f};
    }

    int dx = std::clamp(static_cast<int>(normalizedPoint.x * dw), 0, dw - 1); // Map to pixel X.
    int dy = std::clamp(static_cast<int>(normalizedPoint.y * dh), 0, dh - 1); // Map to pixel Y.

    std::vector<uint16_t> validDepths;                    // Buffer for neighbor pixels.
    for(int i = -4; i <= 4; i++) {                        // Scan 9x9 area around the AI landmark.
        for(int j = -4; j <= 4; j++) {
            int nx = std::clamp(dx + j, 0, dw - 1);
            int ny = std::clamp(dy + i, 0, dh - 1);
            uint16_t z_val = depthMat.at<uint16_t>(ny, nx); // Get depth in mm.

            if(z_val > 0 && z_val < 5000) {               // Filter valid range (0-5 meters).
                validDepths.push_back(z_val);
            }
        }
    }

    float z_mm = 0.0f;
    if (!validDepths.empty()) {                           // Find stable depth using foreground percentile.
        std::sort(validDepths.begin(), validDepths.end());
        size_t index = static_cast<size_t>((validDepths.size() - 1) * DEPTH_FOREGROUND_PERCENTILE);
        z_mm = static_cast<float>(validDepths[index]);
    }

    if(z_mm > 0) {                                        // Pinhole camera back-projection.
        float x_mm = (dx - intrinsics[0][2]) * z_mm / intrinsics[0][0]; // Calculate world X.
        float y_mm = (dy - intrinsics[1][2]) * z_mm / intrinsics[1][1]; // Calculate world Y.
        return {x_mm, y_mm, z_mm};
    }
    return {0.0f, 0.0f, 0.0f};
}

Point3D get3DPointFromFramePixel(Point2f framePt, const Size& frameSize, Mat& depthMat, const std::vector<std::vector<float>>& intrinsics) {
    if (frameSize.width <= 1 || frameSize.height <= 1) {  // Basic sanity check.
        return {0.0f, 0.0f, 0.0f};
    }

    Point2f normalizedPoint(                              // Re-normalize pixel back to 0-1 range.
        framePt.x / static_cast<float>(frameSize.width - 1),
        framePt.y / static_cast<float>(frameSize.height - 1)
    );
    normalizedPoint.x = std::clamp(normalizedPoint.x, 0.0f, 1.0f);
    normalizedPoint.y = std::clamp(normalizedPoint.y, 0.0f, 1.0f);
    return get3DPointFromNormalizedPoint(normalizedPoint, depthMat, intrinsics);
}

SessionConfig configureSession(int menuChoice, int patientId) { // Startup menu logic.
    if (menuChoice == 1) {                                // Training/Calibration mode setup.
        if (patientId <= 0) return {"", "Invalid patient ID.", false};
        return {"Training", "Starting CALIBRATION for Patient " + to_string(patientId) + " (Records 9 JSONs)", true};
    }
    if (menuChoice == 2) {                                // Live system mode setup.
        return {"Live_Tracking", "Starting LIVE TRACKING (Auto-identifies patient)", true};
    }
    return {"", "Invalid choice.", false};
}

void saveRecordingToJSON(const vector<BiometricFrame>& data, int patientId, const string& suffix) { // Data persistence.
    string filename = "patient_" + to_string(patientId) + "_" + suffix + ".json";
    ofstream jsonFile(filename);                          // Open file for writing.
    if (!jsonFile.is_open()) return;

    jsonFile << "[\n";                                    // Start JSON array.
    for (size_t i = 0; i < data.size(); i++) {
        jsonFile << "  {\n"                               // Store all kinematic metrics per frame.
                 << "    \"time\": "               << data[i].timestamp            << ",\n"
                 << "    \"upper_arm_length\": "   << data[i].upperArmLength       << ",\n"
                 << "    \"forearm_length\": "     << data[i].forearmLength        << ",\n"
                 << "    \"elbow_flexion\": "      << data[i].elbowFlexionAngle    << ",\n"
                 << "    \"shoulder_flexion\": "   << data[i].shoulderFlexionAngle << ",\n"
                 << "    \"wrist_velocity\": "     << data[i].wristVelocity        << ",\n"
                 << "    \"wrist_acceleration\": " << data[i].wristAcceleration    << ",\n"
                 << "    \"wrist_jerk\": "         << data[i].wristJerk            << ",\n";

        jsonFile << "    \"arm_landmarks_3d\": [";        // Write primary limb labels.
        for (size_t j = 0; j < data[i].armLandmarks3D.size(); j++) {
            jsonFile << "{\"name\": \"" << data[i].armLandmarks3D[j].name << "\", \"x\": "
                     << data[i].armLandmarks3D[j].point.x << ", \"y\": "
                     << data[i].armLandmarks3D[j].point.y << ", \"z\": "
                     << data[i].armLandmarks3D[j].point.z << ", \"valid\": "
                     << (isValid3DPoint(data[i].armLandmarks3D[j].point) ? "true" : "false") << "}";
            if (j < data[i].armLandmarks3D.size() - 1) jsonFile << ", ";
        }
        jsonFile << "],\n";

        jsonFile << "    \"pose_landmarks_3d\": [";       // Write the full body skeleton indices.
        for (size_t j = 0; j < data[i].poseLandmarks3D.size(); j++) {
            jsonFile << "{\"id\": " << j << ", \"x\": " << data[i].poseLandmarks3D[j].x
                     << ", \"y\": " << data[i].poseLandmarks3D[j].y
                     << ", \"z\": " << data[i].poseLandmarks3D[j].z
                     << ", \"valid\": " << (isValid3DPoint(data[i].poseLandmarks3D[j]) ? "true" : "false") << "}";
            if (j < data[i].poseLandmarks3D.size() - 1) jsonFile << ", ";
        }
        jsonFile << "],\n";

        jsonFile << "    \"hand_landmarks_3d\": [";       // Write finger keypoints.
        for (size_t j = 0; j < data[i].handLandmarks3D.size(); j++) {
            writePoint3DJSON(jsonFile, data[i].handLandmarks3D[j]);
            if (j < data[i].handLandmarks3D.size() - 1) jsonFile << ", ";
        }
        jsonFile << "]\n  }";
        jsonFile << (i < data.size() - 1 ? ",\n" : "\n"); // Array delimiter.
    }
    jsonFile << "]\n";                                    // End JSON array.
    jsonFile.close();
    cout << "[EXPORT] Saved: " << filename << "\n";
}

double parseJsonField(const string& line, const string& key) { // Mini-parser for biometric files.
    size_t pos = line.find("\"" + key + "\":");           // Locate the field key.
    if (pos == string::npos) return -1.0;
    size_t valStart = line.find(":", pos) + 1;            // Jump to the value.
    while (valStart < line.size() && (line[valStart] == ' ' || line[valStart] == '\t')) valStart++;
    return stod(line.substr(valStart));                   // Convert string value to double.
}

vector<PatientProfile> loadAllPatientProfiles() {         // Profile database initialization.
    vector<PatientProfile> profiles;
    for (const auto& entry : fs::directory_iterator(".")) { // Scan local directory for JSON files.
        string fname = entry.path().filename().string();
        if (fname.rfind("patient_", 0) != 0 || fname.find(".json") == string::npos) continue;

        size_t u1 = fname.find('_');                      // Parse ID between underscores.
        size_t u2 = fname.find('_', u1 + 1);
        if (u1 == string::npos || u2 == string::npos) continue;

        int pId = -1;
        try { pId = stoi(fname.substr(u1 + 1, u2 - u1 - 1)); } catch (...) { continue; }

        ifstream jsonFile(fname);                         // Open found patient data.
        if (!jsonFile.is_open()) continue;

        double tUal = 0, tFl = 0, tElbow = 0, tJerk = 0;
        int frames = 0;
        string line;
        while (getline(jsonFile, line)) {                 // Accumulate averages for recognition.
            double val;
            if ((val = parseJsonField(line, "upper_arm_length")) > 0) { tUal += val; frames++; }
            if ((val = parseJsonField(line, "forearm_length")) > 0) tFl += val;
            if ((val = parseJsonField(line, "elbow_flexion")) > 0) tElbow += val;

            size_t jPos = line.find("\"wrist_jerk\":");
            if (jPos != string::npos) {                   // Parse jerk separately due to JSON nesting.
                try { tJerk += abs(stod(line.substr(line.find(":", jPos) + 1))); } catch (...) {}
            }
        }
        if (frames == 0) continue;

        bool found = false;
        for (auto& p : profiles) {                        // Average multiple files for the same user.
            if (p.patientId == pId) {
                p.averageUpperArmLength = (p.averageUpperArmLength + tUal / frames) / 2.0;
                p.averageForearmLength  = (p.averageForearmLength + tFl / frames) / 2.0;
                p.maxElbowAngle         = max(p.maxElbowAngle, tElbow / frames);
                p.maxJerk               = max(p.maxJerk, tJerk / frames);
                found = true; break;
            }
        }
        if (!found) profiles.push_back({pId, tUal/frames, tFl/frames, tElbow/frames, tJerk/frames});
    }
    return profiles;
}

int identifyPatientFromBiometrics(double uArm, double fArm, const vector<PatientProfile>& profiles) {
    int bestId = -1;                                      // Target ID.
    double bestDiff = ARM_MATCH_TOLERANCE;                // Starting matching threshold.
    for (const auto& p : profiles) {                      // Find patient with the most similar arm proportions.
        double diff = abs(uArm - p.averageUpperArmLength) + abs(fArm - p.averageForearmLength);
        if (diff < bestDiff) { bestDiff = diff; bestId = p.patientId; }
    }
    return bestId;
}

int main() {                                              // Execution start.
    int menuChoice = 0, patientId = -1;
    cout << " PROJECT LIMB: VISION SYSTEM STARTUP   \n"   // Operator UI Header.
         << "  1. Training Phase (Record 9 JSON movements)\n"
         << "  2. Live Tracking  (Auto-identify patient)\n"
         << "Select mode: ";
    cin >> menuChoice;

    if (menuChoice == 1) {                                // Mode 1 requires an ID to label files.
        cout << "Enter new Patient ID: ";
        cin >> patientId;
    }

    SessionConfig session = configureSession(menuChoice, patientId); // Prepare software state.
    if (!session.isValid) return 1;                       // Halt if configuration failed.
    cout << "[INFO] " << session.informationMessage << "\n";

    if (!fs::exists("hand_landmark_full_sh4.blob")) {      // Vital AI dependency check.
        cerr << "[ERROR] Could not find hand_landmark_full_sh4.blob\n";
        return 1;
    }

    vector<PatientProfile> loadedProfiles;
    if (session.sessionType == "Live_Tracking") {         // Load biometrics for auto-matching.
        loadedProfiles = loadAllPatientProfiles();
        cout << "[OK] " << loadedProfiles.size() << " patient profiles loaded.\n";
    }

    vector<BiometricFrame> recordedFrames;                // RAM buffer for recordings.
    int currentRecordingStep = 0;                         // Counter for the 9 calibration tasks.
    bool recordingTimerStarted = false;                   // State tracker for recording clock.
    double recordingStepStartTime = 0.0;                  // Start timestamp for current task.
    const string recordingNames[9] = {"move1_slow", "move1_med", "move1_fast", "move2_slow", "move2_med", "move2_fast", "move3_slow", "move3_med", "move3_fast"};

    Point3D prevWrist = {0,0,0};                          // Last wrist position for derivatives.
    double prevTime = 0, prevVel = 0, prevAccel = 0;      // Last frame kinematics.
    bool isFirstFrame = true;                             // Flag to initialize temporal calculations.
    bool hasFilteredArm = false;                          // Filter state initialization.
    Point3D filteredShoulder = {0,0,0}, filteredElbow = {0,0,0}, filteredWrist = {0,0,0};
    std::vector<Point3D> filteredPoseLandmarks3D, filteredHandLandmarks3D;

    auto device = make_shared<dai::Device>();             // Connect to the OAK-D hardware.
    dai::Pipeline pipeline{device};                       // Create hardware process graph.

    auto cameraNode = pipeline.create<dai::node::Camera>(); // Configure RGB sensor.
    cameraNode->build(dai::CameraBoardSocket::CAM_A);

    auto monoLeft = pipeline.create<dai::node::Camera>();  // Configure left stereo sensor.
    monoLeft->build(dai::CameraBoardSocket::CAM_B);
    auto monoRight = pipeline.create<dai::node::Camera>(); // Configure right stereo sensor.
    monoRight->build(dai::CameraBoardSocket::CAM_C);

    auto stereo = pipeline.create<dai::node::StereoDepth>(); // Setup depth processing node.
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::FAST_ACCURACY);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);  // Align depth frame to RGB image.
    stereo->setOutputSize(640, 400);

    monoLeft->requestOutput({640, 400})->link(stereo->left); // Link stereo pairs.
    monoRight->requestOutput({640, 400})->link(stereo->right);

    auto detectionNetwork = pipeline.create<dai::node::DetectionNetwork>(); // Setup YOLOv8 Node.
    dai::NNModelDescription modelDesc;
    modelDesc.model = (device->getPlatform() == dai::Platform::RVC2) ?
        "luxonis/yolov8-nano-pose-estimation:coco-512x288" :
        "luxonis/yolov8-large-pose-estimation:coco-640x352";
    detectionNetwork->build(cameraNode, modelDesc);       // Build network for pose estimation.

    auto lmNn = pipeline.create<dai::node::NeuralNetwork>(); // Setup Hand Landmark Node.
    lmNn->setBlobPath("hand_landmark_full_sh4.blob");

    auto qRgb = detectionNetwork->passthrough.createOutputQueue(); // Queue for RGB frames.
    auto qDet = detectionNetwork->out.createOutputQueue(); // Queue for AI detections.
    auto qDepth = stereo->depth.createOutputQueue(4, false); // Queue for depth maps.

    auto qLmOut = lmNn->out.createOutputQueue(4, false);  // Queue for hand landmarks results.
    auto qLmIn = lmNn->input.createInputQueue();          // Queue to send crops to hand AI.

    pipeline.start();                                     // Activate the hardware pipeline.
    auto calibData = device->readCalibration();           // Load lens intrinsics for 3D math.
    auto startTime = steady_clock::now();                 // System start time.

    Mat frame, depthMat;                                  // Buffers for visual data.
    cv::Rect pendingHandRoi(0, 0, 0, 0);                  // ROI sent to AI.
    cv::Rect currentHandRoi(0, 0, 0, 0);                  // ROI associated with current AI results.

    bool handAiBusy = false;                              // Lock for hand landmark node.
    std::vector<Point2f> currentHandPoints;               // 2D finger landmarks.
    float currentHandScore = 0.0f;                        // AI confidence in hand detection.

    while(pipeline.isRunning()) {                         // Main live processing loop.
        auto inRgb = qRgb->tryGet<dai::ImgFrame>();       // Try to grab latest RGB frame.
        auto inDet = qDet->tryGet<dai::ImgDetections>();  // Try to grab latest AI pose data.
        auto inDepth = qDepth->tryGet<dai::ImgFrame>();   // Try to grab latest depth map.
        auto inLm = qLmOut->tryGet<dai::NNData>();        // Try to grab latest hand AI data.

        if(inRgb) frame = inRgb->getCvFrame();            // Convert to OpenCV format.
        if(inDepth) depthMat = inDepth->getFrame();       // Convert to OpenCV format.

        double curTimeSecs = duration_cast<milliseconds>(steady_clock::now() - startTime).count() / 1000.0;
        if (!recordingTimerStarted) {                     // Start clock upon first sensor activity.
            recordingTimerStarted = true;
            recordingStepStartTime = curTimeSecs;
        }

        if(inLm != nullptr) {                             // Process asynchronous Hand AI results.
            handAiBusy = false;                           // Unlock hand node.

            dai::TensorInfo infoScore;                    // Extract confidence score layer.
            if(inLm->getLayer("Identity_1", infoScore)) {
                cv::Mat fp16Mat(1, 1, CV_16FC1, (void*)(inLm->getData().data() + infoScore.offset));
                cv::Mat fp32Mat;
                fp16Mat.convertTo(fp32Mat, CV_32FC1);     // Convert FP16 to FP32.
                currentHandScore = fp32Mat.at<float>(0, 0);
            }

            dai::TensorInfo infoLm;                       // Extract landmark coordinate layer.
            if (currentHandScore > 0.3f && pendingHandRoi.area() > 0 && inLm->getLayer("Identity_dense/BiasAdd/Add", infoLm)) {
                cv::Mat fp16Mat(1, 63, CV_16FC1, (void*)(inLm->getData().data() + infoLm.offset));
                cv::Mat fp32Mat;
                fp16Mat.convertTo(fp32Mat, CV_32FC1);     // Convert FP16 to FP32.
                float* landmarks = (float*)fp32Mat.data;

                currentHandRoi = pendingHandRoi;          // Sync ROI with results.
                currentHandPoints.clear();
                for (int i = 0; i < 21; i++) {            // Normalize landmarks into full-frame space.
                    float lmX = landmarks[i * 3] / 224.0f;
                    float lmY = landmarks[i * 3 + 1] / 224.0f;
                    Point2f mappedPoint(
                        currentHandRoi.x + lmX * static_cast<float>(currentHandRoi.width - 1),
                        currentHandRoi.y + lmY * static_cast<float>(currentHandRoi.height - 1)
                    );

                    if (!frame.empty()) {                 // Clamp to frame edges.
                        mappedPoint = clampPixelPoint(mappedPoint, frame.size());
                    }
                    currentHandPoints.push_back(mappedPoint);
                }
            } else {                                      // Reset tracking on low confidence.
                currentHandPoints.clear();
                currentHandRoi = cv::Rect(0, 0, 0, 0);
            }
        }

        bool personVisible = false, lowVisibility = false; // Detection state.
        double elbowFlexion = 0, shoulderFlexion = 0;     // Measurement state.

        if(inDet != nullptr && !frame.empty() && !depthMat.empty()) { // Processing block.
            Mat cleanFrame = frame.clone();               // Keep copy for cropping.

            for(const auto& detection : inDet->detections) { // Iterate through YOLOv8 detections.
                auto keypoints = detection.keypoints->getPoints2f(); // Get 2D skeletal keypoints.

                if (keypoints.size() > 9) {               // Validate person has enough joint data.
                    int fw = frame.cols, fh = frame.rows;

                    bool armKeypointsValid =              // Require shoulder, elbow, and wrist.
                        isValidNormalizedPoint(keypoints[5]) &&
                        isValidNormalizedPoint(keypoints[7]) &&
                        isValidNormalizedPoint(keypoints[9]);

                    if (armKeypointsValid) {              // Execute kinematic extraction.
                        Point2f sPx = normalizedToPixel(keypoints[5], frame.size()); // 2D Shoulder.
                        Point2f ePx = normalizedToPixel(keypoints[7], frame.size()); // 2D Elbow.
                        Point2f wPx = normalizedToPixel(keypoints[9], frame.size()); // 2D Wrist.
                        personVisible = true;

                        line(frame, sPx, ePx, Scalar(0, 255, 255), 4); // Draw arm bones.
                        line(frame, ePx, wPx, Scalar(0, 255, 255), 4);
                        circle(frame, sPx, 8, Scalar(0, 0, 255), -1); // Draw joint markers.
                        circle(frame, ePx, 8, Scalar(0, 255, 0), -1);
                        circle(frame, wPx, 8, Scalar(255, 0, 0), -1);

                        auto intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, depthMat.cols, depthMat.rows);

                        std::vector<Point3D> poseLandmarks3D; // Extract all pose points to 3D space.
                        poseLandmarks3D.reserve(keypoints.size());
                        for (const auto& keypoint : keypoints) {
                            if (isValidNormalizedPoint(keypoint)) {
                                poseLandmarks3D.push_back(get3DPointFromNormalizedPoint(Point2f(keypoint.x, keypoint.y), depthMat, intrinsics));
                            } else {
                                poseLandmarks3D.push_back({0.0f, 0.0f, 0.0f});
                            }
                        }

                        Point3D pShoulder = get3DPointFromNormalizedPoint(Point2f(keypoints[5].x, keypoints[5].y), depthMat, intrinsics);
                        Point3D pElbow    = get3DPointFromNormalizedPoint(Point2f(keypoints[7].x, keypoints[7].y), depthMat, intrinsics);
                        Point3D pWrist    = get3DPointFromNormalizedPoint(Point2f(keypoints[9].x, keypoints[9].y), depthMat, intrinsics);

                        if (isPlausibleArmGeometry(pShoulder, pElbow, pWrist)) { // Geometric validation.
                            pShoulder = smoothPoint3D(pShoulder, filteredShoulder, hasFilteredArm); // Filter jitter.
                            pElbow = smoothPoint3D(pElbow, filteredElbow, hasFilteredArm);
                            pWrist = smoothPoint3D(pWrist, filteredWrist, hasFilteredArm);

                            if (!isPlausibleArmGeometry(pShoulder, pElbow, pWrist)) { // Re-check biology after smoothing.
                                lowVisibility = true;
                                continue;
                            }

                            filteredShoulder = pShoulder; // Update filter states.
                            filteredElbow = pElbow;
                            filteredWrist = pWrist;
                            hasFilteredArm = true;

                            rejectPoseDepthOutliers(poseLandmarks3D, pShoulder, pElbow, pWrist); // Sanitize skeleton depth.
                            poseLandmarks3D = smoothPointVector3D(poseLandmarks3D, filteredPoseLandmarks3D);
                            rejectPoseDepthOutliers(poseLandmarks3D, pShoulder, pElbow, pWrist);
                            if (poseLandmarks3D.size() > 9) { // Ensure key arm joints are exact.
                                poseLandmarks3D[5] = pShoulder;
                                poseLandmarks3D[7] = pElbow;
                                poseLandmarks3D[9] = pWrist;
                            }
                            filteredPoseLandmarks3D = poseLandmarks3D;

                            std::vector<NamedPoint3D> armLandmarks3D = { // Final kinematic joint set.
                                {"left_shoulder", pShoulder},
                                {"left_elbow", pElbow},
                                {"left_wrist", pWrist}
                            };

                            std::vector<Point3D> handLandmarks3D; // Extract finger points to 3D space.
                            handLandmarks3D.reserve(currentHandPoints.size());
                            for (const auto& handPoint : currentHandPoints) {
                                handLandmarks3D.push_back(get3DPointFromFramePixel(handPoint, frame.size(), depthMat, intrinsics));
                            }
                            rejectHandDepthOutliers(handLandmarks3D, pWrist); // Sanitize hand depth.
                            handLandmarks3D = smoothPointVector3D(handLandmarks3D, filteredHandLandmarks3D);
                            rejectHandDepthOutliers(handLandmarks3D, pWrist);
                            filteredHandLandmarks3D = handLandmarks3D;

                            // For Inverse Kinematics
                            // if (session.sessionType == "Live_Tracking")
                            // {
                            //     std::vector<float> floatJoints = {
                            //         pShoulder.x, pShoulder.y, pShoulder.z,
                            //         pElbow.x, pElbow.y, pElbow.z,
                            //         pWrist.x, pWrist.y, pWrist.z
                            //     };

                            //     fabrik(floatJoints)
                            // }

                            double curUarm = KinematicAnalyzer::calculateDistanceBetweenPoints(pShoulder, pElbow); // Segments.
                            double curFarm = KinematicAnalyzer::calculateDistanceBetweenPoints(pElbow, pWrist);

                            elbowFlexion = KinematicAnalyzer::calculateAngleBetweenThreePoints(pShoulder, pElbow, pWrist);
                            Point3D vTorso = {pShoulder.x, pShoulder.y + TORSO_OFFSET_MM, pShoulder.z}; // Virtual spine point.
                            shoulderFlexion = KinematicAnalyzer::calculateAngleBetweenThreePoints(vTorso, pShoulder, pElbow);

                            double curVel = 0, curAccel = 0, curJerk = 0; // Temporal derivatives.
                            if (!isFirstFrame) {
                                double dt = curTimeSecs - prevTime; // Calculate frame time delta.
                                if (dt > 0) {
                                    double distMoved = KinematicAnalyzer::calculateDistanceBetweenPoints(prevWrist, pWrist);
                                    curVel = distMoved / dt; // 1st derivative (velocity).
                                    curAccel = (curVel - prevVel) / dt; // 2nd derivative (acceleration).
                                    curJerk = (curAccel - prevAccel) / dt; // 3rd derivative (jerk).
                                }
                            }

                            if (session.sessionType == "Training" && currentRecordingStep < 9) { // Capture logic.
                                recordedFrames.push_back({ // Push frame to buffer.
                                    curTimeSecs, patientId, session.sessionType,
                                    curUarm, curFarm, elbowFlexion, shoulderFlexion,
                                    curVel, curAccel, curJerk,
                                    armLandmarks3D,
                                    poseLandmarks3D,
                                    handLandmarks3D
                                });

                                if (curTimeSecs - recordingStepStartTime > TIME_PER_RECORDING) { // Interval check.
                                    saveRecordingToJSON(recordedFrames, patientId, recordingNames[currentRecordingStep]);
                                    recordedFrames.clear(); // Clear RAM for next task.
                                    currentRecordingStep++; // Advance task.
                                    recordingStepStartTime = curTimeSecs; // Reset clock.
                                }
                            }

                            if (session.sessionType == "Live_Tracking" && patientId == -1) { // Recognition logic.
                                patientId = identifyPatientFromBiometrics(curUarm, curFarm, loadedProfiles);
                            }

                            prevWrist = pWrist; // Shift temporal states.
                            prevTime = curTimeSecs;
                            prevVel = curVel;
                            prevAccel = curAccel;
                            isFirstFrame = false;
                        } else {
                            lowVisibility = true; // Geometry failed plausibility check.
                        }

                        if (!handAiBusy) {                 // Dynamic Hand ROI Targeting logic.
                            Point2f armDir = wPx - ePx;    // Direction of forearm.
                            double armLen = norm(armDir);
                            if(armLen == 0) armLen = 1;
                            armDir = armDir / armLen;      // Unit vector.

                            Point2f handCenter = wPx + armDir * static_cast<float>(armLen * 0.3); // Predict hand center.
                            int boxSize = std::max(static_cast<int>(armLen * 1.5), 120); // Dynamic box scaling.

                            int x = std::clamp(static_cast<int>(std::lround(handCenter.x)) - boxSize / 2, 0, fw - 1); // Border clamp ROI.
                            int y = std::clamp(static_cast<int>(std::lround(handCenter.y)) - boxSize / 2, 0, fh - 1);
                            int wBox = std::min(boxSize, fw - x);
                            int hBox = std::min(boxSize, fh - y);
                            int side = std::min(wBox, hBox); // Force square ROI.

                            if (side > 50) {              // Submit crop to Hand AI.
                                pendingHandRoi = cv::Rect(x, y, side, side);
                                cv::Mat croppedHand = cleanFrame(pendingHandRoi).clone();

                                auto img = std::make_shared<dai::ImgFrame>();
                                img->setType(dai::ImgFrame::Type::RGB888p);
                                img->setWidth(224);
                                img->setHeight(224);
                                img->setData(toPlanar(croppedHand));
                                qLmIn->send(img);         // Send to Neural Network input queue.

                                handAiBusy = true;        // Set lock.
                            }
                        }
                    } else {
                        lowVisibility = true;             // AI failed to detect arm joints.
                    }
                }
            }
        }

        if (!frame.empty()) {                             // UI Rendering.
            if (currentHandScore > 0.3f && currentHandPoints.size() == 21) {
                for (int i = 0; i < 21; i++) {            // Draw finger joint points.
                    circle(frame, currentHandPoints[i], 4, Scalar(255, 0, 255), -1);
                }
                for (auto conn : HAND_CONNECTIONS) {      // Draw finger connections.
                    line(frame, currentHandPoints[conn.first], currentHandPoints[conn.second], Scalar(255, 100, 255), 2);
                }
                if (currentHandRoi.area() > 0) {          // Draw hand tracking box.
                    rectangle(frame, currentHandRoi, Scalar(100, 100, 100), 2);
                }
            }

            putText(frame, "SYSTEM MODE: " + session.sessionType, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

            if (session.sessionType == "Training") {      // Training Mode Overlays.
                putText(frame, "PATIENT ID: " + to_string(patientId), Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

                if (currentRecordingStep < 9) {
                    double timeElapsed = curTimeSecs - recordingStepStartTime;
                    string taskStr = "TASK " + to_string(currentRecordingStep + 1) + "/9: " + recordingNames[currentRecordingStep] +
                                     "  [" + to_string((int)timeElapsed) + "/" + to_string((int)TIME_PER_RECORDING) + "s]";
                    putText(frame, taskStr, Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 165, 255), 2);
                } else {
                    putText(frame, "ALL 9 RECORDINGS COMPLETE - Press 'Q'", Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                }
            } else {                                      // Live Tracking Mode Overlays.
                if (patientId == -1)
                    putText(frame, "STATUS: SCANNING BIOMETRICS...", Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                else
                    putText(frame, "STATUS: MATCHED - Patient " + to_string(patientId), Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            }

            if (!personVisible && inDet != nullptr)       // Warning flags.
                putText(frame, "! NO PERSON DETECTED", Point(20, 190), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            else if (lowVisibility)
                putText(frame, "! LOW VISIBILITY", Point(20, 190), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 165, 255), 2);

            imshow("Project LIMB - Rehabilitation Vision System", frame); // Render window.
        }

        if(waitKey(1) == 'q') break;                      // Escape mechanism.
    }

    return 0;                                             // Successful termination.
}
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef LIBNOP_INCLUDE_NOP_UTILITY_COMPILER_H_
#define LIBNOP_INCLUDE_NOP_UTILITY_COMPILER_H_
#define NOP_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif
#define NOP_GNU_USED [[gnu::used]]
#if __has_cpp_attribute(clang::fallthrough)
#define NOP_FALLTHROUGH [[clang::fallthrough]]
#elif NOP_GCC_VERSION >= 70000
#define NOP_FALLTHROUGH [[fallthrough]]
#else
#define NOP_FALLTHROUGH
#endif
#define NOP_TEMPLATE template
#endif

#include "nop/types/variant.h"
#include "depthai/depthai.hpp"
#include <opencv2/opencv.hpp>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"

using namespace std;
using namespace std::chrono;
using namespace cv;
namespace fs = std::filesystem;

namespace mp = mediapipe;
namespace mp_core = mediapipe::tasks::core;
namespace mp_vision_core = mediapipe::tasks::vision::core;
namespace mp_hand = mediapipe::tasks::vision::hand_landmarker;
namespace mp_pose = mediapipe::tasks::vision::pose_landmarker;
namespace mp_containers = mediapipe::tasks::components::containers;

constexpr float TORSO_OFFSET_MM = 300.0f;
constexpr double TIME_PER_RECORDING = 8.0;
constexpr double RECORDING_PAUSE_SECONDS = 3.0;
constexpr int MIN_RECORDING_FRAMES_TO_SAVE = 5;
constexpr double MAX_RECORDING_EXTRA_SECONDS = 5.0;
constexpr double ARM_MATCH_TOLERANCE = 50.0;

constexpr float MAX_POINT_JUMP_MM = 350.0f;
constexpr float MIN_ARM_SEGMENT_MM = 80.0f;
constexpr float MAX_ARM_SEGMENT_MM = 700.0f;
constexpr float MAX_CONNECTED_Z_DIFF_MM = 600.0f;
constexpr float DEPTH_FOREGROUND_PERCENTILE = 0.25f;
constexpr int DEPTH_SAMPLE_RADIUS = 2;
constexpr double PERSON_MATCH_SCORE_THRESHOLD = 170.0;
constexpr double PERSON_MATCH_AMBIGUITY_MARGIN = 18.0;
constexpr double PERSON_MATCH_REQUIRED_MARGIN = 12.0;
constexpr int LIVE_IDENTIFICATION_MIN_FRAMES = 12;
constexpr int TRAINING_REPEATS_PER_SPEED = 2;
constexpr double GA_MOVEMENT_MATCH_THRESHOLD = 45.0;
constexpr double MOVEMENT_REFERENCE_MATCH_THRESHOLD = 95.0;
constexpr double LIVE_PATIENT_MOVEMENT_MATCH_THRESHOLD = 220.0;
constexpr double LIVE_PATIENT_SWITCH_MARGIN = 95.0;
constexpr int LIVE_PATIENT_LOCK_FRAMES = 10;
constexpr int LIVE_PATIENT_SWITCH_CONFIRM_FRAMES = 999999;
constexpr double LIVE_MOVEMENT_WINDOW_SECONDS = 2.5;
constexpr int LIVE_MOVEMENT_MIN_SAMPLES = 8;
constexpr double TRACK_HOLD_SECONDS = 1.40;

constexpr bool TRACK_LEFT_ARM = true;
constexpr int POSE_LEFT_SHOULDER = 11;
constexpr int POSE_LEFT_ELBOW = 13;
constexpr int POSE_LEFT_WRIST = 15;
constexpr int POSE_RIGHT_SHOULDER = 12;
constexpr int POSE_RIGHT_ELBOW = 14;
constexpr int POSE_RIGHT_WRIST = 16;

constexpr float POSE_MIN_VISIBILITY = 0.22f;
constexpr float KALMAN_PROCESS_NOISE = 4e-2f;
constexpr float KALMAN_MEASUREMENT_NOISE = 2.5e-2f;
constexpr float KALMAN_MAX_SNAP_PIXELS = 75.0f;
constexpr bool ENABLE_HAND_TRACKING = true;
constexpr float MAX_HAND_WRIST_PIXEL_DISTANCE = 180.0f;
constexpr float MAX_HAND_SELECTION_PIXEL_DISTANCE = 240.0f;
constexpr float MAX_HAND_POSE_WRIST_3D_DIFF_MM = 260.0f;
constexpr float HAND_WRIST_2D_BLEND = 0.90f;
constexpr float HAND_WRIST_BLEND = 0.75f;
constexpr bool ENHANCE_AI_INPUT_FOR_DARK_CLOTHES = true;

struct Point3D {
    float x;
    float y;
    float z;
};

struct NamedPoint3D {
    string name;
    Point3D point;
};

struct BiometricFrame {
    double timestamp;
    int userId;
    string sessionType;
    double upperArmLength;
    double forearmLength;
    double elbowFlexionAngle;
    double shoulderFlexionAngle;
    double wristVelocity;
    double wristAcceleration;
    double wristJerk;
    vector<NamedPoint3D> armLandmarks3D;
    vector<Point3D> poseLandmarks3D;
    vector<Point3D> handLandmarks3D;
};

struct PatientProfile {
    int patientId;
    double averageUpperArmLength;
    double averageForearmLength;
    double averageArmRatio;
    double averageTotalArmLength;
    double averageShoulderWidth;
    double averageElbowAngle;
    double averageAbsJerk;
    double averageWristVelocity;
    int sampleCount;
    int shoulderWidthSampleCount;
};

struct SessionConfig {
    string sessionType;
    string informationMessage;
    bool isValid;
};

struct GAJointRange {
    double startAngle = 0.0;
    double maxAngle = 0.0;
    double smoothness = 0.0;
    bool valid = false;
};

struct GAMovementProfile {
    int patientId = -1;
    string movementName;
    string sourcePath;
    double fitness = 0.0;
    GAJointRange shoulderFlexion;
    GAJointRange elbowFlexion;
};

struct MovementReferenceProfile {
    int patientId = -1;
    string movementName;
    double averageUpperArmLength = 0.0;
    double averageForearmLength = 0.0;
    double averageShoulderFlexion = 0.0;
    double minShoulderFlexion = 0.0;
    double maxShoulderFlexion = 0.0;
    double averageElbowFlexion = 0.0;
    double minElbowFlexion = 0.0;
    double maxElbowFlexion = 0.0;
    double averageWristVelocity = 0.0;
    double averageAbsJerk = 0.0;
    int sampleCount = 0;
};

struct LiveMovementSample {
    double timestamp = 0.0;
    double shoulderFlexion = 0.0;
    double elbowFlexion = 0.0;
    double wristVelocity = 0.0;
    double absJerk = 0.0;
};

struct LiveMovementFeatures {
    double latestShoulderFlexion = 0.0;
    double latestElbowFlexion = 0.0;
    double averageShoulderFlexion = 0.0;
    double minShoulderFlexion = 0.0;
    double maxShoulderFlexion = 0.0;
    double averageElbowFlexion = 0.0;
    double minElbowFlexion = 0.0;
    double maxElbowFlexion = 0.0;
    double averageWristVelocity = 0.0;
    double averageAbsJerk = 0.0;
    int sampleCount = 0;
};

const vector<pair<int, int>> HAND_CONNECTIONS = {
    {0,1}, {1,2}, {2,3}, {3,4},
    {0,5}, {5,6}, {6,7}, {7,8},
    {5,9}, {9,10}, {10,11}, {11,12},
    {9,13}, {13,14}, {14,15}, {15,16},
    {13,17}, {0,17}, {17,18}, {18,19}, {19,20}
};

const vector<pair<int, int>> ARM_CONNECTIONS = {
    {TRACK_LEFT_ARM ? POSE_LEFT_SHOULDER : POSE_RIGHT_SHOULDER,
    TRACK_LEFT_ARM ? POSE_LEFT_ELBOW : POSE_RIGHT_ELBOW},
    {TRACK_LEFT_ARM ? POSE_LEFT_ELBOW : POSE_RIGHT_ELBOW,
    TRACK_LEFT_ARM ? POSE_LEFT_WRIST : POSE_RIGHT_WRIST}
};

static fs::path jsonDataDir() {
    return fs::path("Camera_test") / "json_files";
}

static fs::path gaProfileDir() {
    return jsonDataDir() / "ga_profiles";
}

static void ensureJsonDataDirs() {
    std::error_code ec;
    fs::create_directories(jsonDataDir(), ec);
    fs::create_directories(gaProfileDir(), ec);
    if (ec) {
        cerr << "[WARN] Could not create JSON folders: " << ec.message() << "\n";
    }
}

class KinematicAnalyzer {
public:
    static double calculateDistanceBetweenPoints(Point3D p1, Point3D p2) {
        return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
    }

    static double calculateAngleBetweenThreePoints(Point3D p1, Point3D p2, Point3D p3) {
        Point3D vA = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
        Point3D vB = {p3.x - p2.x, p3.y - p2.y, p3.z - p2.z};

        double dot = (vA.x * vB.x) + (vA.y * vB.y) + (vA.z * vB.z);
        double magA = sqrt(vA.x * vA.x + vA.y * vA.y + vA.z * vA.z);
        double magB = sqrt(vB.x * vB.x + vB.y * vB.y + vB.z * vB.z);

        if (magA == 0 || magB == 0) return 0.0;
        return acos(max(-1.0, min(1.0, dot / (magA * magB)))) * (180.0 / M_PI);
    }
};

class KalmanPoint2D {
public:
    Point2f update(const Point2f& measurement, double dt) {
        if (!initialized) {
            init(measurement);
            return measurement;
        }

        dt = std::clamp(dt, 1.0 / 120.0, 0.20);
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, static_cast<float>(dt), 0,
            0, 1, 0, static_cast<float>(dt),
            0, 0, 1, 0,
            0, 0, 0, 1);

        Mat prediction = kf.predict();
        Point2f predicted(prediction.at<float>(0), prediction.at<float>(1));

        if (norm(predicted - measurement) > KALMAN_MAX_SNAP_PIXELS) {
            init(measurement);
            return measurement;
        }

        Mat meas(2, 1, CV_32F);
        meas.at<float>(0) = measurement.x;
        meas.at<float>(1) = measurement.y;
        Mat corrected = kf.correct(meas);
        lastOutput = Point2f(corrected.at<float>(0), corrected.at<float>(1));
        return lastOutput;
    }

    void reset() {
        initialized = false;
    }

    bool hasEstimate() const {
        return initialized;
    }

    Point2f getLastEstimate() const {
        return lastOutput;
    }

private:
    void init(const Point2f& p) {
        kf.init(4, 2, 0, CV_32F);
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        kf.measurementMatrix = Mat::zeros(2, 4, CV_32F);
        kf.measurementMatrix.at<float>(0, 0) = 1.0f;
        kf.measurementMatrix.at<float>(1, 1) = 1.0f;
        setIdentity(kf.processNoiseCov, Scalar::all(KALMAN_PROCESS_NOISE));
        setIdentity(kf.measurementNoiseCov, Scalar::all(KALMAN_MEASUREMENT_NOISE));
        setIdentity(kf.errorCovPost, Scalar::all(1.0f));
        kf.statePost.at<float>(0) = p.x;
        kf.statePost.at<float>(1) = p.y;
        kf.statePost.at<float>(2) = 0.0f;
        kf.statePost.at<float>(3) = 0.0f;
        lastOutput = p;
        initialized = true;
    }

    cv::KalmanFilter kf;
    Point2f lastOutput = Point2f(-1.0f, -1.0f);
    bool initialized = false;
};

static bool isValid3DPoint(const Point3D& point) {
    return point.z > 0.0f;
}

static bool isValidNormalizedPoint(float x, float y) {
    return x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f;
}

static Point2f normalizedToPixel(float x, float y, const cv::Size& size) {
    return Point2f(x * static_cast<float>(size.width - 1),
        y * static_cast<float>(size.height - 1));
}

static Point2f clampPixelPoint(const Point2f& point, const cv::Size& size) {
    return Point2f(
        std::clamp(point.x, 0.0f, static_cast<float>(size.width - 1)),
        std::clamp(point.y, 0.0f, static_cast<float>(size.height - 1))
    );
}

static bool landmarkVisible(const mp_containers::NormalizedLandmark& landmark) {
    if (!isValidNormalizedPoint(landmark.x, landmark.y)) return false;
    if (landmark.visibility.has_value() && landmark.visibility.value() < POSE_MIN_VISIBILITY) return false;
    if (landmark.presence.has_value() && landmark.presence.value() < POSE_MIN_VISIBILITY) return false;
    return true;
}

static bool isPlausibleArmGeometry(const Point3D& shoulder, const Point3D& elbow, const Point3D& wrist) {
    if (!isValid3DPoint(shoulder) || !isValid3DPoint(elbow) || !isValid3DPoint(wrist)) {
        return false;
    }

    double upperArm = KinematicAnalyzer::calculateDistanceBetweenPoints(shoulder, elbow);
    double forearm = KinematicAnalyzer::calculateDistanceBetweenPoints(elbow, wrist);
    double shoulderElbowZDiff = std::abs(shoulder.z - elbow.z);
    double elbowWristZDiff = std::abs(elbow.z - wrist.z);

    return upperArm >= MIN_ARM_SEGMENT_MM &&
        upperArm <= MAX_ARM_SEGMENT_MM &&
        forearm >= MIN_ARM_SEGMENT_MM &&
        forearm <= MAX_ARM_SEGMENT_MM &&
        shoulderElbowZDiff <= MAX_CONNECTED_Z_DIFF_MM &&
        elbowWristZDiff <= MAX_CONNECTED_Z_DIFF_MM;
}

static Point3D chooseStableWrist3D(const Point3D& handWrist,
    const Point3D& poseWrist,
    const Point3D& elbow) {
    if (!isValid3DPoint(handWrist)) return poseWrist;
    if (!isValid3DPoint(elbow)) return isValid3DPoint(poseWrist) ? poseWrist : handWrist;
    if (isValid3DPoint(poseWrist)) {
        double handPoseDistance = KinematicAnalyzer::calculateDistanceBetweenPoints(poseWrist, handWrist);
        if (handPoseDistance > MAX_HAND_POSE_WRIST_3D_DIFF_MM) {
            return poseWrist;
        }
    }

    double forearm = KinematicAnalyzer::calculateDistanceBetweenPoints(elbow, handWrist);
    double zDiff = std::abs(elbow.z - handWrist.z);
    bool handWristPlausible =
        forearm >= MIN_ARM_SEGMENT_MM &&
        forearm <= MAX_ARM_SEGMENT_MM &&
        zDiff <= MAX_CONNECTED_Z_DIFF_MM;

    if (handWristPlausible && isValid3DPoint(poseWrist)) {
        return {
            poseWrist.x + (handWrist.x - poseWrist.x) * HAND_WRIST_BLEND,
            poseWrist.y + (handWrist.y - poseWrist.y) * HAND_WRIST_BLEND,
            poseWrist.z + (handWrist.z - poseWrist.z) * HAND_WRIST_BLEND
        };
    }
    if (handWristPlausible) return handWrist;
    return isValid3DPoint(poseWrist) ? poseWrist : handWrist;
}

static void fuseWristToHand(vector<Point2f>& posePixels,
    vector<Point2f>& handPixels,
    int wristId,
    const cv::Size& frameSize,
    KalmanPoint2D& fusedWristKalman,
    double dt) {
    if (posePixels.size() <= static_cast<size_t>(wristId) || handPixels.size() != 21) {
        fusedWristKalman.reset();
        return;
    }
    if (posePixels[wristId].x < 0.0f || handPixels[0].x < 0.0f) {
        fusedWristKalman.reset();
        return;
    }

    double handPosePixelDistance = norm(handPixels[0] - posePixels[wristId]);
    if (handPosePixelDistance > MAX_HAND_WRIST_PIXEL_DISTANCE) {
        fusedWristKalman.reset();
        return;
    }

    Point2f targetWrist = posePixels[wristId] + (handPixels[0] - posePixels[wristId]) * HAND_WRIST_2D_BLEND;
    Point2f fusedWrist = clampPixelPoint(fusedWristKalman.update(targetWrist, dt), frameSize);
    Point2f handShift = fusedWrist - handPixels[0];

    for (auto& point : handPixels) {
        point = clampPixelPoint(point + handShift, frameSize);
    }
    posePixels[wristId] = fusedWrist;
    handPixels[0] = fusedWrist;
}

static Mat enhanceFrameForPoseDetection(const Mat& bgr) {
    if (!ENHANCE_AI_INPUT_FOR_DARK_CLOTHES || bgr.empty()) return bgr;

    Mat lab;
    cvtColor(bgr, lab, COLOR_BGR2Lab);

    vector<Mat> channels;
    split(lab, channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    merge(channels, lab);

    Mat enhanced;
    cvtColor(lab, enhanced, COLOR_Lab2BGR);
    enhanced.convertTo(enhanced, -1, 1.10, 12.0);
    return enhanced;
}

static mp::Image cvMatBgrToMediaPipeImage(const Mat& bgr) {
    Mat rgb;
    cvtColor(bgr, rgb, COLOR_BGR2RGB);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }

    auto imageFrame = make_shared<mp::ImageFrame>(
        mp::ImageFormat::SRGB,
        rgb.cols,
        rgb.rows,
        mp::ImageFrame::kDefaultAlignmentBoundary
    );
    imageFrame->CopyPixelData(
        mp::ImageFormat::SRGB,
        rgb.cols,
        rgb.rows,
        static_cast<int>(rgb.step),
        rgb.data,
        mp::ImageFrame::kDefaultAlignmentBoundary
    );
    return mp::Image(imageFrame);
}

Point3D get3DPointFromFramePixel(Point2f framePt,
    const cv::Size& frameSize,
    Mat& depthMat,
    const vector<vector<float>>& intrinsics) {
    if (frameSize.width <= 1 || frameSize.height <= 1 || depthMat.empty()) {
        return {0.0f, 0.0f, 0.0f};
    }
    if (intrinsics.size() < 2 || intrinsics[0].size() < 3 || intrinsics[1].size() < 3 ||
        intrinsics[0][0] == 0.0f || intrinsics[1][1] == 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }
    if (framePt.x < 0.0f || framePt.y < 0.0f ||
        framePt.x >= static_cast<float>(frameSize.width) ||
        framePt.y >= static_cast<float>(frameSize.height)) {
        return {0.0f, 0.0f, 0.0f};
    }

    int dw = depthMat.cols;
    int dh = depthMat.rows;
    int dx = std::clamp(static_cast<int>((framePt.x / static_cast<float>(frameSize.width - 1)) * dw), 0, dw - 1);
    int dy = std::clamp(static_cast<int>((framePt.y / static_cast<float>(frameSize.height - 1)) * dh), 0, dh - 1);

    vector<uint16_t> validDepths;
    constexpr int depthWindowSide = DEPTH_SAMPLE_RADIUS * 2 + 1;
    validDepths.reserve(depthWindowSide * depthWindowSide);

    for (int y = -DEPTH_SAMPLE_RADIUS; y <= DEPTH_SAMPLE_RADIUS; y++) {
        for (int x = -DEPTH_SAMPLE_RADIUS; x <= DEPTH_SAMPLE_RADIUS; x++) {
            int nx = std::clamp(dx + x, 0, dw - 1);
            int ny = std::clamp(dy + y, 0, dh - 1);
            uint16_t zVal = depthMat.at<uint16_t>(ny, nx);
            if (zVal > 0 && zVal < 5000) {
                validDepths.push_back(zVal);
            }
        }
    }

    if (validDepths.empty()) {
        return {0.0f, 0.0f, 0.0f};
    }

    size_t index = static_cast<size_t>((validDepths.size() - 1) * DEPTH_FOREGROUND_PERCENTILE);
    nth_element(validDepths.begin(), validDepths.begin() + index, validDepths.end());
    float zMm = static_cast<float>(validDepths[index]);
    float xMm = (dx - intrinsics[0][2]) * zMm / intrinsics[0][0];
    float yMm = (dy - intrinsics[1][2]) * zMm / intrinsics[1][1];
    return {xMm, yMm, zMm};
}

static vector<Point2f> extractPosePixels(const mp_pose::PoseLandmarkerResult& result,
    const cv::Size& frameSize,
    vector<KalmanPoint2D>& filters,
    double dt) {
    vector<Point2f> points;
    if (result.pose_landmarks.empty()) return points;

    const auto& landmarks = result.pose_landmarks[0].landmarks;
    points.resize(landmarks.size(), Point2f(-1.0f, -1.0f));
    if (filters.size() < landmarks.size()) filters.resize(landmarks.size());

    for (size_t i = 0; i < landmarks.size(); i++) {
        if (!landmarkVisible(landmarks[i])) {
            if (filters[i].hasEstimate()) {
                points[i] = clampPixelPoint(filters[i].getLastEstimate(), frameSize);
            }
            continue;
        }

        Point2f raw = normalizedToPixel(landmarks[i].x, landmarks[i].y, frameSize);
        points[i] = clampPixelPoint(filters[i].update(raw, dt), frameSize);
    }
    return points;
}

static int selectHandClosestToWrist(const mp_hand::HandLandmarkerResult& result,
    const Point2f& wristPx,
    const cv::Size& frameSize) {
    if (result.hand_landmarks.empty()) return -1;

    int best = -1;
    double bestDistance = numeric_limits<double>::max();
    for (size_t i = 0; i < result.hand_landmarks.size(); i++) {
        const auto& landmarks = result.hand_landmarks[i].landmarks;
        if (landmarks.empty() || !isValidNormalizedPoint(landmarks[0].x, landmarks[0].y)) continue;
        Point2f handWrist = normalizedToPixel(landmarks[0].x, landmarks[0].y, frameSize);
        double d = norm(handWrist - wristPx);
        if (d < bestDistance) {
            bestDistance = d;
            best = static_cast<int>(i);
        }
    }
    if (bestDistance > MAX_HAND_SELECTION_PIXEL_DISTANCE) return -1;
    return best;
}

static vector<Point2f> extractHandPixels(const mp_hand::HandLandmarkerResult& result,
    int handIndex,
    const cv::Size& frameSize,
    vector<KalmanPoint2D>& filters,
    double dt) {
    vector<Point2f> points;
    if (handIndex < 0 || handIndex >= static_cast<int>(result.hand_landmarks.size())) return points;

    const auto& landmarks = result.hand_landmarks[handIndex].landmarks;
    points.resize(landmarks.size(), Point2f(-1.0f, -1.0f));
    if (filters.size() < landmarks.size()) filters.resize(landmarks.size());

    for (size_t i = 0; i < landmarks.size(); i++) {
        if (!isValidNormalizedPoint(landmarks[i].x, landmarks[i].y)) {
            filters[i].reset();
            continue;
        }

        Point2f raw = normalizedToPixel(landmarks[i].x, landmarks[i].y, frameSize);
        points[i] = clampPixelPoint(filters[i].update(raw, dt), frameSize);
    }
    return points;
}

static vector<Point3D> pixelsTo3D(const vector<Point2f>& pixels,
    const cv::Size& frameSize,
    Mat& depthMat,
    const vector<vector<float>>& intrinsics) {
    vector<Point3D> points3D;
    points3D.reserve(pixels.size());
    for (const auto& p : pixels) {
        if (p.x < 0.0f || p.y < 0.0f) {
            points3D.push_back({0.0f, 0.0f, 0.0f});
        } else {
            points3D.push_back(get3DPointFromFramePixel(p, frameSize, depthMat, intrinsics));
        }
    }
    return points3D;
}

static void drawArm(Mat& frame, const vector<Point2f>& posePixels) {
    int shoulderId = TRACK_LEFT_ARM ? POSE_LEFT_SHOULDER : POSE_RIGHT_SHOULDER;
    int elbowId = TRACK_LEFT_ARM ? POSE_LEFT_ELBOW : POSE_RIGHT_ELBOW;
    int wristId = TRACK_LEFT_ARM ? POSE_LEFT_WRIST : POSE_RIGHT_WRIST;
    if (posePixels.size() <= static_cast<size_t>(wristId)) return;

    Point2f s = posePixels[shoulderId];
    Point2f e = posePixels[elbowId];
    Point2f w = posePixels[wristId];
    if (s.x < 0 || e.x < 0 || w.x < 0) return;

    line(frame, s, e, Scalar(0, 255, 255), 4, LINE_AA);
    line(frame, e, w, Scalar(0, 255, 255), 4, LINE_AA);
    circle(frame, s, 8, Scalar(0, 0, 255), -1, LINE_AA);
    circle(frame, e, 8, Scalar(0, 255, 0), -1, LINE_AA);
    circle(frame, w, 8, Scalar(255, 0, 0), -1, LINE_AA);
}

static void drawHand(Mat& frame, const vector<Point2f>& handPixels) {
    if (handPixels.size() != 21) return;
    for (auto conn : HAND_CONNECTIONS) {
        const Point2f& a = handPixels[conn.first];
        const Point2f& b = handPixels[conn.second];
        if (a.x >= 0 && b.x >= 0) {
            line(frame, a, b, Scalar(255, 100, 255), 2, LINE_AA);
        }
    }
    for (const auto& p : handPixels) {
        if (p.x >= 0 && p.y >= 0) {
            circle(frame, p, 4, Scalar(255, 0, 255), -1, LINE_AA);
        }
    }
}

static void writePoint3DJSON(ofstream& jsonFile, const Point3D& point) {
    jsonFile << "{\"x\": " << point.x
        << ", \"y\": " << point.y
        << ", \"z\": " << point.z
        << ", \"valid\": " << (isValid3DPoint(point) ? "true" : "false") << "}";
}

void saveRecordingToJSON(const vector<BiometricFrame>& data, int patientId, const string& suffix) {
    ensureJsonDataDirs();
    fs::path filename = jsonDataDir() / ("patient_" + to_string(patientId) + "_" + suffix + ".json");
    ofstream jsonFile(filename);
    if (!jsonFile.is_open()) {
        cerr << "[ERROR] Could not write JSON: " << filename << "\n";
        return;
    }

    jsonFile << "[\n";
    for (size_t i = 0; i < data.size(); i++) {
        jsonFile << " {\n"
            << " \"time\": " << data[i].timestamp << ",\n"
            << " \"upper_arm_length\": " << data[i].upperArmLength << ",\n"
            << " \"forearm_length\": " << data[i].forearmLength << ",\n"
            << " \"elbow_flexion\": " << data[i].elbowFlexionAngle << ",\n"
            << " \"shoulder_flexion\": " << data[i].shoulderFlexionAngle << ",\n"
            << " \"wrist_velocity\": " << data[i].wristVelocity << ",\n"
            << " \"wrist_acceleration\": " << data[i].wristAcceleration << ",\n"
            << " \"wrist_jerk\": " << data[i].wristJerk << ",\n";

        jsonFile << " \"arm_landmarks_3d\": [";
        for (size_t j = 0; j < data[i].armLandmarks3D.size(); j++) {
            jsonFile << "{\"name\": \"" << data[i].armLandmarks3D[j].name << "\", \"x\": "
                << data[i].armLandmarks3D[j].point.x << ", \"y\": "
                << data[i].armLandmarks3D[j].point.y << ", \"z\": "
                << data[i].armLandmarks3D[j].point.z << ", \"valid\": "
                << (isValid3DPoint(data[i].armLandmarks3D[j].point) ? "true" : "false") << "}";
            if (j < data[i].armLandmarks3D.size() - 1) jsonFile << ", ";
        }
        jsonFile << "],\n";

        jsonFile << " \"pose_landmarks_3d\": [";
        for (size_t j = 0; j < data[i].poseLandmarks3D.size(); j++) {
            jsonFile << "{\"id\": " << j << ", \"x\": " << data[i].poseLandmarks3D[j].x
                << ", \"y\": " << data[i].poseLandmarks3D[j].y
                << ", \"z\": " << data[i].poseLandmarks3D[j].z
                << ", \"valid\": " << (isValid3DPoint(data[i].poseLandmarks3D[j]) ? "true" : "false") << "}";
            if (j < data[i].poseLandmarks3D.size() - 1) jsonFile << ", ";
        }
        jsonFile << "],\n";

        jsonFile << " \"hand_landmarks_3d\": [";
        for (size_t j = 0; j < data[i].handLandmarks3D.size(); j++) {
            writePoint3DJSON(jsonFile, data[i].handLandmarks3D[j]);
            if (j < data[i].handLandmarks3D.size() - 1) jsonFile << ", ";
        }
        jsonFile << "]\n }";
        jsonFile << (i < data.size() - 1 ? ",\n" : "\n");
    }
    jsonFile << "]\n";
    jsonFile.close();
    if (data.empty()) {
        cout << "[EXPORT] Saved EMPTY recording for troubleshooting: " << fs::absolute(filename) << "\n";
    } else {
        cout << "[EXPORT] Saved: " << fs::absolute(filename) << " (" << data.size() << " frames)\n";
    }
}

double parseJsonField(const string& line, const string& key) {
    size_t pos = line.find("\"" + key + "\":");
    if (pos == string::npos) return -1.0;
    size_t valStart = line.find(":", pos) + 1;
    while (valStart < line.size() && (line[valStart] == ' ' || line[valStart] == '\t')) valStart++;
    try {
        return stod(line.substr(valStart));
    } catch (...) {
        return -1.0;
    }
}

static string readWholeFile(const fs::path& path) {
    ifstream file(path);
    if (!file.is_open()) return "";
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static double parseJsonFieldFromText(const string& text, const string& key, double fallback = 0.0) {
    size_t pos = text.find("\"" + key + "\":");
    if (pos == string::npos) return fallback;

    size_t valStart = text.find(":", pos);
    if (valStart == string::npos) return fallback;
    valStart++;
    while (valStart < text.size() && std::isspace(static_cast<unsigned char>(text[valStart]))) valStart++;

    try {
        return stod(text.substr(valStart));
    } catch (...) {
        return fallback;
    }
}

static bool parsePoseLandmarkPointFromLine(const string& line, int landmarkId, Point3D& point) {
    string spacedToken = "\"id\": " + to_string(landmarkId);
    string compactToken = "\"id\":" + to_string(landmarkId);
    size_t idPos = line.find(spacedToken);
    if (idPos == string::npos) idPos = line.find(compactToken);
    if (idPos == string::npos) return false;

    size_t objectStart = line.rfind("{", idPos);
    size_t objectEnd = line.find("}", idPos);
    if (objectStart == string::npos || objectEnd == string::npos || objectEnd <= objectStart) return false;

    string objectText = line.substr(objectStart, objectEnd - objectStart + 1);
    if (objectText.find("\"valid\": false") != string::npos) return false;

    point.x = static_cast<float>(parseJsonFieldFromText(objectText, "x", 0.0));
    point.y = static_cast<float>(parseJsonFieldFromText(objectText, "y", 0.0));
    point.z = static_cast<float>(parseJsonFieldFromText(objectText, "z", 0.0));
    return isValid3DPoint(point);
}

static string extractJsonObjectForKey(const string& text, const string& key) {
    size_t keyPos = text.find("\"" + key + "\"");
    if (keyPos == string::npos) return "";

    size_t open = text.find("{", keyPos);
    if (open == string::npos) return "";

    int depth = 0;
    for (size_t i = open; i < text.size(); i++) {
        if (text[i] == '{') depth++;
        if (text[i] == '}') {
            depth--;
            if (depth == 0) {
                return text.substr(open, i - open + 1);
            }
        }
    }
    return "";
}

static vector<pair<string, string>> extractTopLevelJsonObjects(const string& objectText) {
    vector<pair<string, string>> objects;
    size_t i = objectText.find('{');
    if (i == string::npos) return objects;
    i++;

    while (i < objectText.size()) {
        while (i < objectText.size() && (std::isspace(static_cast<unsigned char>(objectText[i])) || objectText[i] == ',')) i++;
        if (i >= objectText.size() || objectText[i] == '}') break;
        if (objectText[i] != '"') {
            i++;
            continue;
        }

        size_t keyStart = ++i;
        while (i < objectText.size() && objectText[i] != '"') i++;
        if (i >= objectText.size()) break;
        string key = objectText.substr(keyStart, i - keyStart);
        i++;

        while (i < objectText.size() && (std::isspace(static_cast<unsigned char>(objectText[i])) || objectText[i] == ':')) i++;
        if (i >= objectText.size() || objectText[i] != '{') continue;

        size_t valueStart = i;
        int depth = 0;
        bool inString = false;
        bool escaped = false;
        for (; i < objectText.size(); i++) {
            char c = objectText[i];
            if (inString) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    inString = false;
                }
                continue;
            }

            if (c == '"') {
                inString = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    objects.push_back({key, objectText.substr(valueStart, i - valueStart + 1)});
                    i++;
                    break;
                }
            }
        }
    }

    return objects;
}

static GAJointRange parseGAJointRange(const string& profileText, const string& jointName) {
    string jointBlock = extractJsonObjectForKey(profileText, jointName);
    if (jointBlock.empty()) return {};

    GAJointRange range;
    range.startAngle = parseJsonFieldFromText(jointBlock, "start_angle");
    range.maxAngle = parseJsonFieldFromText(jointBlock, "max_angle");
    range.smoothness = parseJsonFieldFromText(jointBlock, "smoothness");
    range.valid = true;
    return range;
}

static string movementNameFromGAFilename(const fs::path& path, int patientId) {
    string stem = path.stem().string();
    string prefix = "patient_" + to_string(patientId) + "_";
    if (stem.rfind(prefix, 0) == 0) {
        stem = stem.substr(prefix.size());
    }

    const vector<string> removable = {"_profile", "profile_", "ga_", "_ga"};
    for (const auto& token : removable) {
        size_t pos = stem.find(token);
        if (pos != string::npos) stem.erase(pos, token.size());
    }

    while (!stem.empty() && (stem.front() == '_' || stem.front() == '-')) stem.erase(stem.begin());
    while (!stem.empty() && (stem.back() == '_' || stem.back() == '-')) stem.pop_back();
    if (stem == "profile") return "general_profile";
    return stem.empty() ? "general_profile" : stem;
}

static vector<fs::path> findGAProfileJsonFiles() {
    vector<fs::path> files;
    const vector<fs::path> searchDirs = {
        jsonDataDir(),
        gaProfileDir(),
        "/Users/sarazrno/Downloads/GA_1"
    };

    for (const auto& dir : searchDirs) {
        std::error_code ec;
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) continue;

        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec || !entry.is_regular_file()) continue;
            string fname = entry.path().filename().string();
            if (fname.rfind("patient_", 0) == 0 &&
                fname.find("profile") != string::npos &&
                entry.path().extension() == ".json") {
                files.push_back(entry.path());
            }
        }
    }
    return files;
}

static vector<GAMovementProfile> loadGAMovementProfiles() {
    vector<GAMovementProfile> profiles;

    for (const auto& path : findGAProfileJsonFiles()) {
        string fname = path.filename().string();
        size_t u1 = fname.find('_');
        size_t u2 = fname.find('_', u1 + 1);
        if (u1 == string::npos || u2 == string::npos) continue;

        int pId = -1;
        try { pId = stoi(fname.substr(u1 + 1, u2 - u1 - 1)); } catch (...) { continue; }

        string text = readWholeFile(path);
        if (text.empty()) continue;

        pId = static_cast<int>(parseJsonFieldFromText(text, "patient_id", pId));

        auto addProfile = [&](const string& movementName, const string& movementBlock) {
            string profileBlock = extractJsonObjectForKey(movementBlock, "profile");
            if (profileBlock.empty()) profileBlock = movementBlock;

            GAMovementProfile profile;
            profile.patientId = pId;
            profile.movementName = movementName;
            profile.sourcePath = fs::absolute(path).string() + "#" + movementName;
            profile.fitness = parseJsonFieldFromText(movementBlock, "ga_fitness");
            profile.shoulderFlexion = parseGAJointRange(profileBlock, "sh_flex");
            profile.elbowFlexion = parseGAJointRange(profileBlock, "el_flex");

            if (profile.shoulderFlexion.valid || profile.elbowFlexion.valid) {
                profiles.push_back(profile);
            }
        };

        string movementsBlock = extractJsonObjectForKey(text, "movements");
        if (!movementsBlock.empty()) {
            for (const auto& movement : extractTopLevelJsonObjects(movementsBlock)) {
                addProfile(movement.first, movement.second);
            }
        } else {
            addProfile(movementNameFromGAFilename(path, pId), text);
        }
    }

    cout << "[OK] Loaded " << profiles.size() << " GA movement profiles.\n";
    return profiles;
}

static double scoreAngleAgainstGARange(double liveAngle, const GAJointRange& range) {
    if (!range.valid) return 0.0;
    double lo = std::min(range.startAngle, range.maxAngle);
    double hi = std::max(range.startAngle, range.maxAngle);
    double span = std::max(1.0, hi - lo);
    double mid = (lo + hi) * 0.5;
    double centerScore = std::abs(liveAngle - mid) / span * 20.0;

    if (liveAngle < lo) return centerScore + (lo - liveAngle);
    if (liveAngle > hi) return centerScore + (liveAngle - hi);
    return centerScore;
}

static const GAMovementProfile* selectActiveGAProfile(const vector<GAMovementProfile>& profiles,
    int patientId,
    double shoulderFlexion,
    double elbowFlexion,
    double* bestScoreOut = nullptr) {
    const GAMovementProfile* best = nullptr;
    double bestScore = std::numeric_limits<double>::max();

    for (const auto& profile : profiles) {
        if (profile.patientId != patientId) continue;

        double score =
            scoreAngleAgainstGARange(shoulderFlexion, profile.shoulderFlexion) +
            scoreAngleAgainstGARange(elbowFlexion, profile.elbowFlexion);

        if (score < bestScore) {
            bestScore = score;
            best = &profile;
        }
    }

    if (bestScoreOut) *bestScoreOut = bestScore;
    if (best && bestScore <= GA_MOVEMENT_MATCH_THRESHOLD) return best;
    return nullptr;
}

static vector<fs::path> findPatientJsonFiles() {
    vector<fs::path> files;
    const vector<fs::path> searchDirs = {jsonDataDir()};

    for (const auto& dir : searchDirs) {
        std::error_code ec;
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) continue;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!entry.is_regular_file()) continue;
            string fname = entry.path().filename().string();
            if (fname.find("_partial") != string::npos) continue;
            if (fname.rfind("patient_", 0) == 0 && entry.path().extension() == ".json") {
                files.push_back(entry.path());
            }
        }
    }
    return files;
}

static string baseMovementNameFromFilename(const string& filename) {
    size_t movePos = filename.find("_move");
    if (movePos == string::npos) return "";
    size_t start = movePos + 1;
    size_t end = filename.find('_', start);
    if (end == string::npos) end = filename.find('.', start);
    if (end == string::npos || end <= start) return "";
    return filename.substr(start, end - start);
}

static vector<MovementReferenceProfile> loadMovementReferenceProfiles() {
    map<pair<int, string>, MovementReferenceProfile> mergedProfiles;

    for (const auto& path : findPatientJsonFiles()) {
        string fname = path.filename().string();
        string movementName = baseMovementNameFromFilename(fname);
        if (movementName.empty()) continue;

        size_t u1 = fname.find('_');
        size_t u2 = fname.find('_', u1 + 1);
        if (u1 == string::npos || u2 == string::npos) continue;

        int pId = -1;
        try { pId = stoi(fname.substr(u1 + 1, u2 - u1 - 1)); } catch (...) { continue; }

        ifstream jsonFile(path);
        if (!jsonFile.is_open()) continue;

        MovementReferenceProfile fileProfile;
        fileProfile.patientId = pId;
        fileProfile.movementName = movementName;
        fileProfile.minShoulderFlexion = std::numeric_limits<double>::max();
        fileProfile.minElbowFlexion = std::numeric_limits<double>::max();
        fileProfile.maxShoulderFlexion = -std::numeric_limits<double>::max();
        fileProfile.maxElbowFlexion = -std::numeric_limits<double>::max();

        double tUal = 0.0, tFl = 0.0, tShoulder = 0.0, tElbow = 0.0, tVel = 0.0, tAbsJerk = 0.0;
        int frames = 0, velocitySamples = 0, jerkSamples = 0;

        string line;
        while (getline(jsonFile, line)) {
            double val;
            bool hasFrameMetric = false;
            if ((val = parseJsonField(line, "upper_arm_length")) > 0) {
                tUal += val;
                hasFrameMetric = true;
            }
            if ((val = parseJsonField(line, "forearm_length")) > 0) tFl += val;
            if ((val = parseJsonField(line, "shoulder_flexion")) > 0) {
                tShoulder += val;
                fileProfile.minShoulderFlexion = std::min(fileProfile.minShoulderFlexion, val);
                fileProfile.maxShoulderFlexion = std::max(fileProfile.maxShoulderFlexion, val);
            }
            if ((val = parseJsonField(line, "elbow_flexion")) > 0) {
                tElbow += val;
                fileProfile.minElbowFlexion = std::min(fileProfile.minElbowFlexion, val);
                fileProfile.maxElbowFlexion = std::max(fileProfile.maxElbowFlexion, val);
            }
            if ((val = parseJsonField(line, "wrist_velocity")) >= 0) {
                tVel += val;
                velocitySamples++;
            }
            size_t jPos = line.find("\"wrist_jerk\":");
            if (jPos != string::npos) {
                try {
                    tAbsJerk += abs(stod(line.substr(line.find(":", jPos) + 1)));
                    jerkSamples++;
                } catch (...) {}
            }
            if (hasFrameMetric) frames++;
        }

        if (frames == 0) continue;

        fileProfile.sampleCount = frames;
        fileProfile.averageUpperArmLength = tUal / frames;
        fileProfile.averageForearmLength = tFl / frames;
        fileProfile.averageShoulderFlexion = tShoulder / frames;
        fileProfile.averageElbowFlexion = tElbow / frames;
        fileProfile.averageWristVelocity = velocitySamples > 0 ? tVel / velocitySamples : 0.0;
        fileProfile.averageAbsJerk = jerkSamples > 0 ? tAbsJerk / jerkSamples : 0.0;
        if (fileProfile.minShoulderFlexion == std::numeric_limits<double>::max()) fileProfile.minShoulderFlexion = fileProfile.averageShoulderFlexion;
        if (fileProfile.minElbowFlexion == std::numeric_limits<double>::max()) fileProfile.minElbowFlexion = fileProfile.averageElbowFlexion;
        if (fileProfile.maxShoulderFlexion == -std::numeric_limits<double>::max()) fileProfile.maxShoulderFlexion = fileProfile.averageShoulderFlexion;
        if (fileProfile.maxElbowFlexion == -std::numeric_limits<double>::max()) fileProfile.maxElbowFlexion = fileProfile.averageElbowFlexion;

        auto key = make_pair(pId, movementName);
        auto it = mergedProfiles.find(key);
        if (it == mergedProfiles.end()) {
            mergedProfiles[key] = fileProfile;
            continue;
        }

        MovementReferenceProfile& merged = it->second;
        int totalSamples = merged.sampleCount + fileProfile.sampleCount;
        merged.averageUpperArmLength =
            (merged.averageUpperArmLength * merged.sampleCount + fileProfile.averageUpperArmLength * fileProfile.sampleCount) / totalSamples;
        merged.averageForearmLength =
            (merged.averageForearmLength * merged.sampleCount + fileProfile.averageForearmLength * fileProfile.sampleCount) / totalSamples;
        merged.averageShoulderFlexion =
            (merged.averageShoulderFlexion * merged.sampleCount + fileProfile.averageShoulderFlexion * fileProfile.sampleCount) / totalSamples;
        merged.averageElbowFlexion =
            (merged.averageElbowFlexion * merged.sampleCount + fileProfile.averageElbowFlexion * fileProfile.sampleCount) / totalSamples;
        merged.averageWristVelocity =
            (merged.averageWristVelocity * merged.sampleCount + fileProfile.averageWristVelocity * fileProfile.sampleCount) / totalSamples;
        merged.averageAbsJerk =
            (merged.averageAbsJerk * merged.sampleCount + fileProfile.averageAbsJerk * fileProfile.sampleCount) / totalSamples;
        merged.minShoulderFlexion = std::min(merged.minShoulderFlexion, fileProfile.minShoulderFlexion);
        merged.maxShoulderFlexion = std::max(merged.maxShoulderFlexion, fileProfile.maxShoulderFlexion);
        merged.minElbowFlexion = std::min(merged.minElbowFlexion, fileProfile.minElbowFlexion);
        merged.maxElbowFlexion = std::max(merged.maxElbowFlexion, fileProfile.maxElbowFlexion);
        merged.sampleCount = totalSamples;
    }

    vector<MovementReferenceProfile> profiles;
    profiles.reserve(mergedProfiles.size());
    for (const auto& item : mergedProfiles) {
        profiles.push_back(item.second);
    }
    cout << "[OK] Loaded " << profiles.size() << " movement reference profiles from training JSON.\n";
    return profiles;
}

vector<PatientProfile> loadAllPatientProfiles() {
    vector<PatientProfile> profiles;
    for (const auto& path : findPatientJsonFiles()) {
        string fname = path.filename().string();
        size_t u1 = fname.find('_');
        size_t u2 = fname.find('_', u1 + 1);
        if (u1 == string::npos || u2 == string::npos) continue;

        int pId = -1;
        try { pId = stoi(fname.substr(u1 + 1, u2 - u1 - 1)); } catch (...) { continue; }

        ifstream jsonFile(path);
        if (!jsonFile.is_open()) continue;

        double tUal = 0, tFl = 0, tElbow = 0, tAbsJerk = 0, tVel = 0, tShoulderWidth = 0;
        int frames = 0;
        int jerkSamples = 0;
        int velocitySamples = 0;
        int shoulderWidthSamples = 0;
        string line;
        while (getline(jsonFile, line)) {
            double val;
            if ((val = parseJsonField(line, "upper_arm_length")) > 0) { tUal += val; frames++; }
            if ((val = parseJsonField(line, "forearm_length")) > 0) tFl += val;
            if ((val = parseJsonField(line, "elbow_flexion")) > 0) tElbow += val;
            if ((val = parseJsonField(line, "wrist_velocity")) >= 0) {
                tVel += val;
                velocitySamples++;
            }
            size_t jPos = line.find("\"wrist_jerk\":");
            if (jPos != string::npos) {
                try {
                    tAbsJerk += abs(stod(line.substr(line.find(":", jPos) + 1)));
                    jerkSamples++;
                } catch (...) {}
            }
            if (line.find("\"pose_landmarks_3d\"") != string::npos) {
                Point3D leftShoulder, rightShoulder;
                if (parsePoseLandmarkPointFromLine(line, POSE_LEFT_SHOULDER, leftShoulder) &&
                    parsePoseLandmarkPointFromLine(line, POSE_RIGHT_SHOULDER, rightShoulder)) {
                    double shoulderWidth = KinematicAnalyzer::calculateDistanceBetweenPoints(leftShoulder, rightShoulder);
                    if (shoulderWidth >= 120.0 && shoulderWidth <= 700.0) {
                        tShoulderWidth += shoulderWidth;
                        shoulderWidthSamples++;
                    }
                }
            }
        }
        if (frames == 0) continue;

        double avgUpperArm = tUal / frames;
        double avgForearm = tFl / frames;
        PatientProfile fileProfile{
            pId,
            avgUpperArm,
            avgForearm,
            avgForearm > 0.0 ? avgUpperArm / avgForearm : 0.0,
            avgUpperArm + avgForearm,
            shoulderWidthSamples > 0 ? tShoulderWidth / shoulderWidthSamples : 0.0,
            tElbow / frames,
            jerkSamples > 0 ? tAbsJerk / jerkSamples : 0.0,
            velocitySamples > 0 ? tVel / velocitySamples : 0.0,
            frames,
            shoulderWidthSamples
        };

        bool found = false;
        for (auto& p : profiles) {
            if (p.patientId == pId) {
                int totalSamples = p.sampleCount + fileProfile.sampleCount;
                p.averageUpperArmLength =
                    (p.averageUpperArmLength * p.sampleCount + fileProfile.averageUpperArmLength * fileProfile.sampleCount) / totalSamples;
                p.averageForearmLength =
                    (p.averageForearmLength * p.sampleCount + fileProfile.averageForearmLength * fileProfile.sampleCount) / totalSamples;
                p.averageArmRatio =
                    (p.averageArmRatio * p.sampleCount + fileProfile.averageArmRatio * fileProfile.sampleCount) / totalSamples;
                p.averageTotalArmLength =
                    (p.averageTotalArmLength * p.sampleCount + fileProfile.averageTotalArmLength * fileProfile.sampleCount) / totalSamples;
                int totalShoulderWidthSamples = p.shoulderWidthSampleCount + fileProfile.shoulderWidthSampleCount;
                if (totalShoulderWidthSamples > 0) {
                    p.averageShoulderWidth =
                        (p.averageShoulderWidth * p.shoulderWidthSampleCount + fileProfile.averageShoulderWidth * fileProfile.shoulderWidthSampleCount) / totalShoulderWidthSamples;
                    p.shoulderWidthSampleCount = totalShoulderWidthSamples;
                }
                p.averageElbowAngle =
                    (p.averageElbowAngle * p.sampleCount + fileProfile.averageElbowAngle * fileProfile.sampleCount) / totalSamples;
                p.averageAbsJerk =
                    (p.averageAbsJerk * p.sampleCount + fileProfile.averageAbsJerk * fileProfile.sampleCount) / totalSamples;
                p.averageWristVelocity =
                    (p.averageWristVelocity * p.sampleCount + fileProfile.averageWristVelocity * fileProfile.sampleCount) / totalSamples;
                p.sampleCount = totalSamples;
                found = true;
                break;
            }
        }
        if (!found) profiles.push_back(fileProfile);
    }
    cout << "[OK] Loaded " << profiles.size() << " patient profiles from JSON.\n";
    return profiles;
}

int identifyPatientFromBiometrics(double uArm,
    double fArm,
    double shoulderWidth,
    double elbowAngle,
    double wristVelocity,
    double absJerk,
    const vector<PatientProfile>& profiles,
    double* bestScoreOut = nullptr) {
    struct CandidateScore {
        int patientId = -1;
        double lengthScore = 0.0;
        double tieBreakerScore = 0.0;
    };

    vector<CandidateScore> candidates;
    double bestIdentityScore = std::numeric_limits<double>::max();

    for (const auto& p : profiles) {
        double armLengthScore =
            abs(uArm - p.averageUpperArmLength) +
            abs(fArm - p.averageForearmLength);
        double liveRatio = fArm > 0.0 ? uArm / fArm : 0.0;
        double ratioScore = 80.0 * abs(liveRatio - p.averageArmRatio);
        double totalArmScore = 0.35 * abs((uArm + fArm) - p.averageTotalArmLength);
        double shoulderScore = 0.0;
        if (shoulderWidth > 0.0 && p.averageShoulderWidth > 0.0 && p.shoulderWidthSampleCount >= 10) {
            shoulderScore = 0.75 * abs(shoulderWidth - p.averageShoulderWidth);
        }

        double identityScore = armLengthScore + ratioScore + totalArmScore + shoulderScore;
        double elbowTieScore = 0.20 * abs(elbowAngle - p.averageElbowAngle);
        double velocityTieScore = p.averageWristVelocity > 0.0
            ? 0.02 * abs(wristVelocity - p.averageWristVelocity)
            : 0.0;
        double jerkTieScore = p.averageAbsJerk > 0.0
            ? 0.001 * abs(absJerk - p.averageAbsJerk)
            : 0.0;
        double tieBreakerScore = identityScore + elbowTieScore + velocityTieScore + jerkTieScore;

        candidates.push_back({p.patientId, identityScore, tieBreakerScore});
        bestIdentityScore = std::min(bestIdentityScore, identityScore);
    }

    if (candidates.empty() || bestIdentityScore > PERSON_MATCH_SCORE_THRESHOLD) {
        if (bestScoreOut) *bestScoreOut = candidates.empty() ? -1.0 : bestIdentityScore;
        return -1;
    }

    CandidateScore best;
    bool hasBest = false;
    bool ambiguous = false;
    for (const auto& candidate : candidates) {
        bool closeToBest = candidate.lengthScore <= bestIdentityScore + PERSON_MATCH_AMBIGUITY_MARGIN;
        if (!closeToBest) continue;

        if (!hasBest) {
            best = candidate;
            hasBest = true;
            continue;
        }

        ambiguous = true;
        if (candidate.tieBreakerScore < best.tieBreakerScore) {
            best = candidate;
        }
    }

    if (!hasBest) {
        if (bestScoreOut) *bestScoreOut = bestIdentityScore;
        return -1;
    }

    double secondBestScore = std::numeric_limits<double>::max();
    for (const auto& candidate : candidates) {
        if (candidate.patientId == best.patientId) continue;
        secondBestScore = std::min(secondBestScore, candidate.lengthScore);
    }

    if (std::isfinite(secondBestScore) &&
        secondBestScore - best.lengthScore < PERSON_MATCH_REQUIRED_MARGIN) {
        double secondBestTieScore = std::numeric_limits<double>::max();
        for (const auto& candidate : candidates) {
            if (candidate.patientId == best.patientId) continue;
            secondBestTieScore = std::min(secondBestTieScore, candidate.tieBreakerScore);
        }

        if (!std::isfinite(secondBestTieScore) ||
            secondBestTieScore - best.tieBreakerScore < PERSON_MATCH_REQUIRED_MARGIN) {
            if (bestScoreOut) *bestScoreOut = best.tieBreakerScore;
            return -1;
        }
    }

    if (bestScoreOut) {
        *bestScoreOut = ambiguous ? best.tieBreakerScore : best.lengthScore;
    }
    return best.patientId;
}

static int identifyPatientFromMovementReferences(double uArm,
    double fArm,
    double elbowAngle,
    double wristVelocity,
    double absJerk,
    const vector<MovementReferenceProfile>& movementProfiles,
    double* bestScoreOut = nullptr) {
    int bestId = -1;
    double bestScore = PERSON_MATCH_SCORE_THRESHOLD;

    for (const auto& profile : movementProfiles) {
        double lengthScore =
            abs(uArm - profile.averageUpperArmLength) +
            abs(fArm - profile.averageForearmLength);

        double styleScore =
            lengthScore +
            0.12 * abs(elbowAngle - profile.averageElbowFlexion) +
            0.015 * abs(wristVelocity - profile.averageWristVelocity) +
            4.0 * abs(log1p(absJerk) - log1p(profile.averageAbsJerk));

        if (styleScore < bestScore) {
            bestScore = styleScore;
            bestId = profile.patientId;
        }
    }

    if (bestScoreOut) *bestScoreOut = movementProfiles.empty() ? -1.0 : bestScore;
    return bestId;
}

static void updateLiveMovementWindow(deque<LiveMovementSample>& window,
    double timestamp,
    double shoulderFlexion,
    double elbowFlexion,
    double wristVelocity,
    double absJerk) {
    window.push_back({timestamp, shoulderFlexion, elbowFlexion, wristVelocity, absJerk});
    while (!window.empty() && timestamp - window.front().timestamp > LIVE_MOVEMENT_WINDOW_SECONDS) {
        window.pop_front();
    }
}

static LiveMovementFeatures summarizeLiveMovementWindow(const deque<LiveMovementSample>& window) {
    LiveMovementFeatures features;
    if (window.empty()) return features;

    features.sampleCount = static_cast<int>(window.size());
    features.latestShoulderFlexion = window.back().shoulderFlexion;
    features.latestElbowFlexion = window.back().elbowFlexion;
    features.minShoulderFlexion = std::numeric_limits<double>::max();
    features.minElbowFlexion = std::numeric_limits<double>::max();
    features.maxShoulderFlexion = -std::numeric_limits<double>::max();
    features.maxElbowFlexion = -std::numeric_limits<double>::max();

    for (const auto& sample : window) {
        features.averageShoulderFlexion += sample.shoulderFlexion;
        features.averageElbowFlexion += sample.elbowFlexion;
        features.averageWristVelocity += sample.wristVelocity;
        features.averageAbsJerk += sample.absJerk;
        features.minShoulderFlexion = std::min(features.minShoulderFlexion, sample.shoulderFlexion);
        features.maxShoulderFlexion = std::max(features.maxShoulderFlexion, sample.shoulderFlexion);
        features.minElbowFlexion = std::min(features.minElbowFlexion, sample.elbowFlexion);
        features.maxElbowFlexion = std::max(features.maxElbowFlexion, sample.elbowFlexion);
    }

    features.averageShoulderFlexion /= features.sampleCount;
    features.averageElbowFlexion /= features.sampleCount;
    features.averageWristVelocity /= features.sampleCount;
    features.averageAbsJerk /= features.sampleCount;
    return features;
}

static double scoreLiveMovementAgainstReference(const LiveMovementFeatures& live,
    const MovementReferenceProfile& reference) {
    if (live.sampleCount < LIVE_MOVEMENT_MIN_SAMPLES) return std::numeric_limits<double>::max();

    double liveShoulderRange = live.maxShoulderFlexion - live.minShoulderFlexion;
    double refShoulderRange = reference.maxShoulderFlexion - reference.minShoulderFlexion;
    double liveElbowRange = live.maxElbowFlexion - live.minElbowFlexion;
    double refElbowRange = reference.maxElbowFlexion - reference.minElbowFlexion;
    double shoulderRangePenalty = 0.0;
    if (live.latestShoulderFlexion < reference.minShoulderFlexion) {
        shoulderRangePenalty = reference.minShoulderFlexion - live.latestShoulderFlexion;
    } else if (live.latestShoulderFlexion > reference.maxShoulderFlexion) {
        shoulderRangePenalty = live.latestShoulderFlexion - reference.maxShoulderFlexion;
    }

    return
        2.4 * abs(live.averageShoulderFlexion - reference.averageShoulderFlexion) +
        1.0 * abs(live.maxShoulderFlexion - reference.maxShoulderFlexion) +
        0.55 * abs(liveShoulderRange - refShoulderRange) +
        1.0 * shoulderRangePenalty +
        0.45 * abs(live.averageElbowFlexion - reference.averageElbowFlexion) +
        0.20 * abs(liveElbowRange - refElbowRange) +
        0.006 * abs(live.averageWristVelocity - reference.averageWristVelocity) +
        1.0 * abs(log1p(live.averageAbsJerk) - log1p(reference.averageAbsJerk));
}

static const MovementReferenceProfile* selectActiveMovementReference(
    const vector<MovementReferenceProfile>& movementProfiles,
    int patientId,
    const LiveMovementFeatures& live,
    double* bestScoreOut = nullptr) {
    const MovementReferenceProfile* best = nullptr;
    double bestScore = std::numeric_limits<double>::max();

    for (const auto& profile : movementProfiles) {
        if (profile.patientId != patientId) continue;

        double score = scoreLiveMovementAgainstReference(live, profile);
        if (score < bestScore) {
            bestScore = score;
            best = &profile;
        }
    }

    if (bestScoreOut) *bestScoreOut = bestScore;
    if (best && bestScore <= MOVEMENT_REFERENCE_MATCH_THRESHOLD) return best;
    return nullptr;
}

static double scoreLivePatientForMovementReferences(double uArm,
    double fArm,
    const LiveMovementFeatures& live,
    const MovementReferenceProfile& reference) {
    if (live.sampleCount < LIVE_MOVEMENT_MIN_SAMPLES) return std::numeric_limits<double>::max();

    double lengthScore =
        abs(uArm - reference.averageUpperArmLength) +
        abs(fArm - reference.averageForearmLength);
    double movementScore = scoreLiveMovementAgainstReference(live, reference);
    return lengthScore + 0.55 * movementScore;
}

static double bestLivePatientScoreForId(double uArm,
    double fArm,
    const LiveMovementFeatures& live,
    const vector<MovementReferenceProfile>& movementProfiles,
    int patientId) {
    double bestScore = std::numeric_limits<double>::max();
    for (const auto& profile : movementProfiles) {
        if (profile.patientId != patientId) continue;
        bestScore = std::min(bestScore, scoreLivePatientForMovementReferences(uArm, fArm, live, profile));
    }
    return bestScore;
}

static int identifyPatientFromLiveMovementReferences(double uArm,
    double fArm,
    const LiveMovementFeatures& live,
    const vector<MovementReferenceProfile>& movementProfiles,
    double* bestScoreOut = nullptr) {
    int bestId = -1;
    double bestScore = LIVE_PATIENT_MOVEMENT_MATCH_THRESHOLD;

    for (const auto& profile : movementProfiles) {
        double score = scoreLivePatientForMovementReferences(uArm, fArm, live, profile);
        if (score < bestScore) {
            bestScore = score;
            bestId = profile.patientId;
        }
    }

    if (bestScoreOut) *bestScoreOut = movementProfiles.empty() ? -1.0 : bestScore;
    return bestId;
}

static const GAMovementProfile* findGAProfileForMovement(const vector<GAMovementProfile>& profiles,
    int patientId,
    const string& movementName) {
    for (const auto& profile : profiles) {
        if (profile.patientId == patientId && profile.movementName == movementName) {
            return &profile;
        }
    }
    return nullptr;
}

SessionConfig configureSession(int menuChoice, int patientId) {
    if (menuChoice == 1) {
        if (patientId <= 0) return {"", "Invalid patient ID.", false};
        return {"Training", "Starting CALIBRATION for Patient " + to_string(patientId) + " (Records 18 JSONs)", true};
    }
    if (menuChoice == 2) {
        return {"Live_Tracking", "Starting LIVE TRACKING (Auto-identifies patient)", true};
    }
    return {"", "Invalid choice.", false};
}

static vector<string> makeTrainingRecordingNames() {
    vector<string> names;
    const vector<string> movements = {"move1", "move2", "move3"};
    const vector<string> speeds = {"slow", "med", "fast"};

    for (const auto& movement : movements) {
        for (const auto& speed : speeds) {
            for (int setId = 1; setId <= TRAINING_REPEATS_PER_SPEED; setId++) {
                names.push_back(movement + "_" + speed + "_set" + to_string(setId));
            }
        }
    }

    return names;
}

static double recordingDurationForName(const string& recordingName) {
    if (recordingName.find("_fast") != string::npos) return 4.0;
    if (recordingName.find("_med") != string::npos) return 6.0;
    return TIME_PER_RECORDING;
}

static string firstExistingPath(const vector<string>& paths) {
    for (const auto& path : paths) {
        if (fs::exists(path)) return path;
    }
    return paths.empty() ? string{} : paths.front();
}

int main(int argc, char** argv) {
    string poseModelPath = argc > 1 ? argv[1] : firstExistingPath({
        "pose_landmarker_full.task",
        "Camera_test/pose_landmarker_full.task",
        "pose_landmarker_lite.task",
        "Camera_test/pose_landmarker_lite.task"
    });
    string handModelPath = argc > 2 ? argv[2] : firstExistingPath({
        "hand_landmarker.task",
        "Camera_test/hand_landmarker.task"
    });

    if (!fs::exists(poseModelPath)) {
        cerr << "[ERROR] Missing pose model: " << poseModelPath << "\n";
        return 1;
    }
    if (ENABLE_HAND_TRACKING && !fs::exists(handModelPath)) {
        cerr << "[ERROR] Missing hand model: " << handModelPath << "\n";
        return 1;
    }

    int menuChoice = 0, patientId = -1;
    cout << " PROJECT LIMB: VISION SYSTEM STARTUP\n"
        << "  1. Training Phase (Record 18 JSON movements)\n"
        << "  2. Live Tracking  (Auto-identify patient)\n"
        << "Select mode: ";
    cin >> menuChoice;

    if (menuChoice == 1) {
        cout << "Enter new Patient ID: ";
        cin >> patientId;
    }

    SessionConfig session = configureSession(menuChoice, patientId);
    if (!session.isValid) return 1;
    ensureJsonDataDirs();
    cout << "[INFO] " << session.informationMessage << "\n";
    cout << "[INFO] JSON folder: " << fs::absolute(jsonDataDir()) << "\n";

    auto poseOptions = make_unique<mp_pose::PoseLandmarkerOptions>();
    poseOptions->base_options.model_asset_path = poseModelPath;
    poseOptions->running_mode = mp_vision_core::RunningMode::VIDEO;
    poseOptions->num_poses = 1;
    poseOptions->min_pose_detection_confidence = 0.42f;
    poseOptions->min_pose_presence_confidence = 0.42f;
    poseOptions->min_tracking_confidence = 0.42f;

    auto poseOr = mp_pose::PoseLandmarker::Create(std::move(poseOptions));
    if (!poseOr.ok()) {
        cerr << "[ERROR] PoseLandmarker create failed: " << poseOr.status() << "\n";
        return 1;
    }
    unique_ptr<mp_pose::PoseLandmarker> poseLandmarker = std::move(poseOr.value());

    unique_ptr<mp_hand::HandLandmarker> handLandmarker;
    if (ENABLE_HAND_TRACKING) {
        auto handOptions = make_unique<mp_hand::HandLandmarkerOptions>();
        handOptions->base_options.model_asset_path = handModelPath;
        handOptions->running_mode = mp_vision_core::RunningMode::VIDEO;
        handOptions->num_hands = 2;
        handOptions->min_hand_detection_confidence = 0.50f;
        handOptions->min_hand_presence_confidence = 0.50f;
        handOptions->min_tracking_confidence = 0.50f;

        auto handOr = mp_hand::HandLandmarker::Create(std::move(handOptions));
        if (!handOr.ok()) {
            cerr << "[ERROR] HandLandmarker create failed: " << handOr.status() << "\n";
            return 1;
        }
        handLandmarker = std::move(handOr.value());
    }

    vector<PatientProfile> loadedProfiles;
    vector<GAMovementProfile> gaMovementProfiles;
    vector<MovementReferenceProfile> movementReferenceProfiles;
    if (session.sessionType == "Live_Tracking") {
        loadedProfiles = loadAllPatientProfiles();
        gaMovementProfiles = loadGAMovementProfiles();
        movementReferenceProfiles = loadMovementReferenceProfiles();
    }

    vector<BiometricFrame> recordedFrames;
    int currentRecordingStep = 0;
    bool recordingTimerStarted = false;
    bool recordingPauseActive = true;
    double recordingStepStartTime = 0.0;
    const vector<string> recordingNames = makeTrainingRecordingNames();
    const int totalRecordingSteps = static_cast<int>(recordingNames.size());

    Point3D prevWrist = {0, 0, 0};
    double prevTime = 0, prevVel = 0, prevAccel = 0;
    bool isFirstFrame = true;
    int liveIdentificationFrames = 0;
    double liveUpperArmSum = 0.0;
    double liveForearmSum = 0.0;
    double liveShoulderWidthSum = 0.0;
    int liveShoulderWidthSamples = 0;
    double liveElbowSum = 0.0;
    double liveVelocitySum = 0.0;
    double liveAbsJerkSum = 0.0;
    double liveBestScore = -1.0;
    int livePatientCandidateId = -1;
    int livePatientCandidateFrames = 0;
    int livePatientSwitchCandidateId = -1;
    int livePatientSwitchCandidateFrames = 0;
    string activeGAMovement = "waiting";
    string activeGAFile = "";
    double activeGAScore = 0.0;
    deque<LiveMovementSample> liveMovementWindow;
    auto resetLiveIdentificationWindow = [&]() {
        liveIdentificationFrames = 0;
        liveUpperArmSum = 0.0;
        liveForearmSum = 0.0;
        liveShoulderWidthSum = 0.0;
        liveShoulderWidthSamples = 0;
        liveElbowSum = 0.0;
        liveVelocitySum = 0.0;
        liveAbsJerkSum = 0.0;
    };

    vector<KalmanPoint2D> poseKalman(33);
    vector<KalmanPoint2D> handKalman(21);
    KalmanPoint2D fusedWristKalman;
    vector<Point2f> lastGoodPosePixels;
    vector<Point2f> lastGoodHandPixels;
    double lastGoodTrackingTime = -1000.0;

    auto device = make_shared<dai::Device>();
    dai::Pipeline pipeline{device};

    auto cameraNode = pipeline.create<dai::node::Camera>();
    // FIXAD: Gamla syntaxen för socket (baserat på dina fel)
    cameraNode->build(dai::CameraBoardSocket::CAM_A);

    auto monoLeft = pipeline.create<dai::node::Camera>();
    monoLeft->build(dai::CameraBoardSocket::CAM_B);
    auto monoRight = pipeline.create<dai::node::Camera>();
    monoRight->build(dai::CameraBoardSocket::CAM_C);

    auto stereo = pipeline.create<dai::node::StereoDepth>();
    // FIXAD: FAST_ACCURACY istället för HIGH_ACCURACY
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::FAST_ACCURACY);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);
    stereo->setOutputSize(640, 400);

    monoLeft->requestOutput({640, 400})->link(stereo->left);
    monoRight->requestOutput({640, 400})->link(stereo->right);

    auto rgbOut = cameraNode->requestOutput({640, 400}, dai::ImgFrame::Type::BGR888i);
    auto qRgb = rgbOut->createOutputQueue(1, false);
    auto qDepth = stereo->depth.createOutputQueue(1, false);

    pipeline.start();

    auto calibData = device->readCalibration();
    auto startTime = steady_clock::now();
    double lastFrameTimeSecs = 0.0;
    int64_t lastTimestampMs = -1;

    Mat frame, depthMat;

    while (true) {
        bool personVisible = false;
        bool armLockActive = false;
        auto inRgb = qRgb->tryGet<dai::ImgFrame>();
        auto inDepth = qDepth->tryGet<dai::ImgFrame>();

        if (inRgb) frame = inRgb->getCvFrame();
        if (inDepth) depthMat = inDepth->getFrame(true);
        
        if (frame.empty() || depthMat.empty()) {
            if (waitKey(1) == 'q') break;
            continue;
        }

        double curTimeSecs = duration_cast<milliseconds>(steady_clock::now() - startTime).count() / 1000.0;
        double dt = lastFrameTimeSecs > 0 ? curTimeSecs - lastFrameTimeSecs : 1.0 / 30.0;
        lastFrameTimeSecs = curTimeSecs;
        int64_t timestampMs = static_cast<int64_t>(curTimeSecs * 1000.0);
        if (timestampMs <= lastTimestampMs) timestampMs = lastTimestampMs + 1;
        lastTimestampMs = timestampMs;

        if (!recordingTimerStarted) {
            recordingTimerStarted = true;
            recordingStepStartTime = curTimeSecs;
            recordingPauseActive = session.sessionType == "Training";
        }

        if (session.sessionType == "Training" &&
            currentRecordingStep < totalRecordingSteps &&
            recordingPauseActive &&
            curTimeSecs - recordingStepStartTime >= RECORDING_PAUSE_SECONDS) {
            recordedFrames.clear();
            recordingPauseActive = false;
            recordingStepStartTime = curTimeSecs;
            prevTime = 0.0;
            prevVel = 0.0;
            prevAccel = 0.0;
            isFirstFrame = true;
        }

        Mat aiFrame = enhanceFrameForPoseDetection(frame);
        mp::Image mpImage = cvMatBgrToMediaPipeImage(aiFrame);
        auto poseResultOr = poseLandmarker->DetectForVideo(mpImage, timestampMs);

        if (poseResultOr.ok()) {
            const auto& poseResult = poseResultOr.value();
            vector<Point2f> posePixels = extractPosePixels(poseResult, frame.size(), poseKalman, dt);
            int shoulderId = TRACK_LEFT_ARM ? POSE_LEFT_SHOULDER : POSE_RIGHT_SHOULDER;
            int elbowId = TRACK_LEFT_ARM ? POSE_LEFT_ELBOW : POSE_RIGHT_ELBOW;
            int wristId = TRACK_LEFT_ARM ? POSE_LEFT_WRIST : POSE_RIGHT_WRIST;

            bool armPoseValid = posePixels.size() > static_cast<size_t>(wristId) &&
                posePixels[shoulderId].x >= 0 &&
                posePixels[elbowId].x >= 0 &&
                posePixels[wristId].x >= 0;

            if (!armPoseValid &&
                lastGoodPosePixels.size() > static_cast<size_t>(wristId) &&
                curTimeSecs - lastGoodTrackingTime <= TRACK_HOLD_SECONDS) {
                posePixels = lastGoodPosePixels;
                armPoseValid = true;
                armLockActive = true;
            }

            if (armPoseValid) {
                vector<Point2f> handPixels;
                Point2f poseWristBeforeHand = posePixels[wristId];
                if (ENABLE_HAND_TRACKING && handLandmarker) {
                    auto handResultOr = handLandmarker->DetectForVideo(mpImage, timestampMs);
                    if (handResultOr.ok()) {
                        const auto& handResult = handResultOr.value();
                        int handIndex = selectHandClosestToWrist(handResult, posePixels[wristId], frame.size());
                        handPixels = extractHandPixels(handResult, handIndex, frame.size(), handKalman, dt);
                    }
                }
                fuseWristToHand(posePixels, handPixels, wristId, frame.size(), fusedWristKalman, dt);

                personVisible = true;
                drawArm(frame, posePixels);
                drawHand(frame, handPixels);
                lastGoodPosePixels = posePixels;
                lastGoodHandPixels = handPixels;
                lastGoodTrackingTime = curTimeSecs;

                auto intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, depthMat.cols, depthMat.rows);
                vector<Point3D> poseLandmarks3D = pixelsTo3D(posePixels, frame.size(), depthMat, intrinsics);
                vector<Point3D> handLandmarks3D = pixelsTo3D(handPixels, frame.size(), depthMat, intrinsics);
                double curShoulderWidth = 0.0;
                if (poseLandmarks3D.size() > static_cast<size_t>(POSE_RIGHT_SHOULDER) &&
                    isValid3DPoint(poseLandmarks3D[POSE_LEFT_SHOULDER]) &&
                    isValid3DPoint(poseLandmarks3D[POSE_RIGHT_SHOULDER])) {
                    curShoulderWidth = KinematicAnalyzer::calculateDistanceBetweenPoints(
                        poseLandmarks3D[POSE_LEFT_SHOULDER],
                        poseLandmarks3D[POSE_RIGHT_SHOULDER]
                    );
                    if (curShoulderWidth < 120.0 || curShoulderWidth > 700.0) {
                        curShoulderWidth = 0.0;
                    }
                }

                Point3D pShoulder = poseLandmarks3D[shoulderId];
                Point3D pElbow = poseLandmarks3D[elbowId];
                Point3D rawPoseWrist3D = get3DPointFromFramePixel(posePixels[wristId], frame.size(), depthMat, intrinsics);
                Point3D handWrist3D = {0.0f, 0.0f, 0.0f};
                if (handLandmarks3D.size() == 21 && handPixels.size() == 21 && handPixels[0].x >= 0.0f) {
                    double handPosePixelDistance = norm(handPixels[0] - poseWristBeforeHand);
                    if (handPosePixelDistance <= MAX_HAND_WRIST_PIXEL_DISTANCE) {
                        handWrist3D = handLandmarks3D[0];
                    }
                }
                Point3D pWrist = chooseStableWrist3D(handWrist3D, rawPoseWrist3D, pElbow);

                if (poseLandmarks3D.size() > static_cast<size_t>(wristId)) {
                    poseLandmarks3D[wristId] = pWrist;
                }
                if (handLandmarks3D.size() == 21) {
                    handLandmarks3D[0] = pWrist;
                }

                if (isPlausibleArmGeometry(pShoulder, pElbow, pWrist)) {
                    vector<NamedPoint3D> armLandmarks3D = {
                        {TRACK_LEFT_ARM ? "left_shoulder" : "right_shoulder", pShoulder},
                        {TRACK_LEFT_ARM ? "left_elbow" : "right_elbow", pElbow},
                        {TRACK_LEFT_ARM ? "left_wrist" : "right_wrist", pWrist}
                    };

                    double curUarm = KinematicAnalyzer::calculateDistanceBetweenPoints(pShoulder, pElbow);
                    double curFarm = KinematicAnalyzer::calculateDistanceBetweenPoints(pElbow, pWrist);
                    double elbowFlexion = KinematicAnalyzer::calculateAngleBetweenThreePoints(pShoulder, pElbow, pWrist);
                    Point3D vTorso = {pShoulder.x, pShoulder.y + TORSO_OFFSET_MM, pShoulder.z};
                    double shoulderFlexion = KinematicAnalyzer::calculateAngleBetweenThreePoints(vTorso, pShoulder, pElbow);

                    double curVel = 0, curAccel = 0, curJerk = 0;
                    if (!isFirstFrame) {
                        double kDt = curTimeSecs - prevTime;
                        if (kDt > 0) {
                            double distMoved = KinematicAnalyzer::calculateDistanceBetweenPoints(prevWrist, pWrist);
                            curVel = distMoved / kDt;
                            curAccel = (curVel - prevVel) / kDt;
                            curJerk = (curAccel - prevAccel) / kDt;
                        }
                    }

                    if (session.sessionType == "Training" && currentRecordingStep < totalRecordingSteps && !recordingPauseActive) {
                        recordedFrames.push_back({curTimeSecs, patientId, session.sessionType, curUarm, curFarm, elbowFlexion, shoulderFlexion, curVel, curAccel, curJerk, armLandmarks3D, poseLandmarks3D, handLandmarks3D});
                    }

                    if (session.sessionType == "Live_Tracking") {
                        updateLiveMovementWindow(liveMovementWindow, curTimeSecs, shoulderFlexion, elbowFlexion, curVel, abs(curJerk));
                    }

                    if (session.sessionType == "Live_Tracking" && patientId == -1) {
                        liveIdentificationFrames++;
                        liveUpperArmSum += curUarm;
                        liveForearmSum += curFarm;
                        if (curShoulderWidth > 0.0) {
                            liveShoulderWidthSum += curShoulderWidth;
                            liveShoulderWidthSamples++;
                        }
                        liveElbowSum += elbowFlexion;
                        liveVelocitySum += curVel;
                        liveAbsJerkSum += abs(curJerk);

                        if (liveIdentificationFrames >= LIVE_IDENTIFICATION_MIN_FRAMES) {
                            int lengthMatchedPatientId = identifyPatientFromBiometrics(
                                liveUpperArmSum / liveIdentificationFrames,
                                liveForearmSum / liveIdentificationFrames,
                                liveShoulderWidthSamples > 0 ? liveShoulderWidthSum / liveShoulderWidthSamples : 0.0,
                                liveElbowSum / liveIdentificationFrames,
                                liveVelocitySum / liveIdentificationFrames,
                                liveAbsJerkSum / liveIdentificationFrames,
                                loadedProfiles,
                                &liveBestScore
                            );
                            if (lengthMatchedPatientId != -1) {
                                if (lengthMatchedPatientId == livePatientCandidateId) {
                                    livePatientCandidateFrames++;
                                } else {
                                    livePatientCandidateId = lengthMatchedPatientId;
                                    livePatientCandidateFrames = 1;
                                }

                                if (livePatientCandidateFrames >= LIVE_PATIENT_LOCK_FRAMES) {
                                    patientId = lengthMatchedPatientId;
                                }
                            } else {
                                resetLiveIdentificationWindow();
                                livePatientCandidateId = -1;
                                livePatientCandidateFrames = 0;
                            }
                        }
                    }

                    if (session.sessionType == "Live_Tracking" && patientId != -1) {
                        LiveMovementFeatures liveMovement = summarizeLiveMovementWindow(liveMovementWindow);
                        double movementReferenceScore = std::numeric_limits<double>::max();
                        const MovementReferenceProfile* movementMatch = selectActiveMovementReference(
                            movementReferenceProfiles,
                            patientId,
                            liveMovement,
                            &movementReferenceScore
                        );

                        if (movementMatch) {
                            activeGAMovement = movementMatch->movementName;
                            activeGAScore = movementReferenceScore;
                            const GAMovementProfile* gaMatch = findGAProfileForMovement(
                                gaMovementProfiles,
                                patientId,
                                activeGAMovement
                            );
                            activeGAFile = gaMatch ? gaMatch->sourcePath : "";
                        } else if (movementReferenceProfiles.empty()) {
                            const GAMovementProfile* gaMatch = selectActiveGAProfile(
                                gaMovementProfiles,
                                patientId,
                                shoulderFlexion,
                                elbowFlexion,
                                &activeGAScore
                            );

                            if (gaMatch) {
                                activeGAMovement = gaMatch->movementName;
                                activeGAFile = gaMatch->sourcePath;
                            } else {
                                activeGAMovement = liveMovement.sampleCount < LIVE_MOVEMENT_MIN_SAMPLES ? "collecting movement..." : "no move match";
                                activeGAFile = "";
                            }
                        } else {
                            activeGAMovement = liveMovement.sampleCount < LIVE_MOVEMENT_MIN_SAMPLES ? "collecting movement..." : "no move match";
                            activeGAFile = "";
                            activeGAScore = movementReferenceScore;
                        }
                    }

                    if (session.sessionType == "Training" && currentRecordingStep < totalRecordingSteps && recordingPauseActive) {
                        prevWrist = pWrist;
                        prevTime = curTimeSecs;
                        prevVel = 0.0;
                        prevAccel = 0.0;
                        isFirstFrame = true;
                    } else {
                        prevWrist = pWrist;
                        prevTime = curTimeSecs;
                        prevVel = curVel;
                        prevAccel = curAccel;
                        isFirstFrame = false;
                    }
                }
            }
        }

        if (session.sessionType == "Training" &&
            currentRecordingStep < totalRecordingSteps &&
            !recordingPauseActive) {
            double activeRecordingDuration = recordingDurationForName(recordingNames[currentRecordingStep]);
            if (curTimeSecs - recordingStepStartTime > activeRecordingDuration) {
                bool hasEnoughFrames = recordedFrames.size() >= static_cast<size_t>(MIN_RECORDING_FRAMES_TO_SAVE);
                bool canWaitForMoreFrames =
                    !hasEnoughFrames &&
                    curTimeSecs - recordingStepStartTime <= activeRecordingDuration + MAX_RECORDING_EXTRA_SECONDS;

                if (canWaitForMoreFrames) {
                    putText(frame, "WAITING FOR VALID ARM FRAMES...",
                        cv::Point(20, 250), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 220, 80), 2);
                } else {
                    saveRecordingToJSON(recordedFrames, patientId, recordingNames[currentRecordingStep]);
                    recordedFrames.clear();
                    currentRecordingStep++;
                    recordingStepStartTime = curTimeSecs;
                    recordingPauseActive = currentRecordingStep < totalRecordingSteps;
                    prevTime = 0.0;
                    prevVel = 0.0;
                    prevAccel = 0.0;
                    isFirstFrame = true;
                }
            }
        }

        bool holdingLastTrack = false;
        if (!personVisible && curTimeSecs - lastGoodTrackingTime <= TRACK_HOLD_SECONDS) {
            drawArm(frame, lastGoodPosePixels);
            drawHand(frame, lastGoodHandPixels);
            personVisible = true;
            holdingLastTrack = true;
        }

        putText(frame, "SYSTEM MODE: " + session.sessionType, cv::Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

        if (session.sessionType == "Training") {
            putText(frame, "PATIENT ID: " + to_string(patientId), cv::Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

            if (currentRecordingStep < totalRecordingSteps) {
                double timeElapsed = curTimeSecs - recordingStepStartTime;
                double activeRecordingDuration = recordingDurationForName(recordingNames[currentRecordingStep]);
                if (recordingPauseActive) {
                    int secondsLeft = std::max(0, static_cast<int>(std::ceil(RECORDING_PAUSE_SECONDS - timeElapsed)));
                    string taskStr = "GET READY: TASK " + to_string(currentRecordingStep + 1) + "/" + to_string(totalRecordingSteps) + ": " + recordingNames[currentRecordingStep] +
                        " starts in " + to_string(secondsLeft) + "s";
                    putText(frame, taskStr, cv::Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 220, 80), 2);
                    putText(frame, "Recording paused - move into start position",
                        cv::Point(20, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 220, 80), 2);
                } else {
                    string taskStr = "RECORDING TASK " + to_string(currentRecordingStep + 1) + "/" + to_string(totalRecordingSteps) + ": " + recordingNames[currentRecordingStep] +
                        "  [" + to_string(static_cast<int>(timeElapsed)) + "/" + to_string(static_cast<int>(activeRecordingDuration)) + "s]";
                    putText(frame, taskStr, cv::Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 165, 255), 2);
                    putText(frame, "FRAMES SAVED IN BUFFER: " + to_string(recordedFrames.size()),
                        cv::Point(20, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(180, 255, 180), 2);
                }
            } else {
                putText(frame, "ALL " + to_string(totalRecordingSteps) + " RECORDINGS COMPLETE - Press 'Q'", cv::Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            }
        } else {
            string scoreText = liveBestScore >= 0.0 ? to_string(static_cast<int>(liveBestScore)) : "waiting";
            if (patientId == -1) {
                putText(frame, "STATUS: SCANNING BIOMETRICS...", cv::Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                putText(frame, "ID samples: " + to_string(liveIdentificationFrames) + "/" + to_string(LIVE_IDENTIFICATION_MIN_FRAMES),
                    cv::Point(20, 110), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(180, 255, 180), 2);
                putText(frame, "Profiles loaded: " + to_string(loadedProfiles.size()) + "  Score: " + scoreText,
                    cv::Point(20, 145), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(180, 255, 180), 2);
            } else {
                putText(frame, "STATUS: MATCHED - Patient " + to_string(patientId), cv::Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                putText(frame, "Match score: " + scoreText,
                    cv::Point(20, 110), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(180, 255, 180), 2);
                putText(frame, "GA profile: " + activeGAMovement,
                    cv::Point(20, 145), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(120, 220, 255), 2);
                string gaScoreText = std::isfinite(activeGAScore) ? to_string(static_cast<int>(activeGAScore)) : "waiting";
                putText(frame, "GA score: " + gaScoreText,
                    cv::Point(20, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(120, 220, 255), 2);
                if (!activeGAFile.empty()) {
                    putText(frame, "Use: " + fs::path(activeGAFile).filename().string(),
                        cv::Point(20, 210), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(120, 220, 255), 2);
                }
            }
        }

        if (armLockActive) {
            putText(frame, "ARM LOCK: following last stable arm",
                cv::Point(20, 220), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 220, 80), 2);
        } else if (holdingLastTrack) {
            putText(frame, "TRACK HOLD: arm temporarily occluded",
                cv::Point(20, 220), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 220, 80), 2);
        } else if (!personVisible && poseResultOr.ok()) {
            putText(frame, "! NO PERSON DETECTED", cv::Point(20, 220), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        imshow("Project LIMB - Rehabilitation Vision System", frame);
        if (waitKey(1) == 'q') break;
    }

    if (session.sessionType == "Training" && !recordingPauseActive && !recordedFrames.empty() && currentRecordingStep < totalRecordingSteps) {
        saveRecordingToJSON(recordedFrames, patientId, recordingNames[currentRecordingStep] + "_partial");
    }

    poseLandmarker->Close();
    if (handLandmarker) handLandmarker->Close();
    return 0;
}
