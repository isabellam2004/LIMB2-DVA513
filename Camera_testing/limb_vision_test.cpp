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
constexpr double ARM_MATCH_TOLERANCE = 50.0;

constexpr float MAX_POINT_JUMP_MM = 350.0f;
constexpr float MIN_ARM_SEGMENT_MM = 80.0f;
constexpr float MAX_ARM_SEGMENT_MM = 700.0f;
constexpr float MAX_CONNECTED_Z_DIFF_MM = 600.0f;
constexpr float DEPTH_FOREGROUND_PERCENTILE = 0.25f;
constexpr int DEPTH_SAMPLE_RADIUS = 2;
constexpr double PERSON_MATCH_SCORE_THRESHOLD = 170.0;
constexpr double PERSON_MATCH_AMBIGUITY_MARGIN = 35.0;
constexpr int LIVE_IDENTIFICATION_MIN_FRAMES = 12;
constexpr double GA_MOVEMENT_MATCH_THRESHOLD = 45.0;

constexpr bool TRACK_LEFT_ARM = true;
constexpr int POSE_LEFT_SHOULDER = 11;
constexpr int POSE_LEFT_ELBOW = 13;
constexpr int POSE_LEFT_WRIST = 15;
constexpr int POSE_RIGHT_SHOULDER = 12;
constexpr int POSE_RIGHT_ELBOW = 14;
constexpr int POSE_RIGHT_WRIST = 16;

constexpr float POSE_MIN_VISIBILITY = 0.35f;
constexpr float KALMAN_PROCESS_NOISE = 4e-2f;
constexpr float KALMAN_MEASUREMENT_NOISE = 2.5e-2f;
constexpr float KALMAN_MAX_SNAP_PIXELS = 75.0f;
constexpr bool ENABLE_HAND_TRACKING = true;
constexpr float MAX_HAND_WRIST_PIXEL_DISTANCE = 140.0f;
constexpr float MAX_HAND_SELECTION_PIXEL_DISTANCE = 180.0f;
constexpr float MAX_HAND_POSE_WRIST_3D_DIFF_MM = 260.0f;
constexpr float HAND_WRIST_2D_BLEND = 0.90f;
constexpr float HAND_WRIST_BLEND = 0.75f;

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
    double averageElbowAngle;
    double averageAbsJerk;
    double averageWristVelocity;
    int sampleCount;
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
    return fs::path("Camera_testing") / "json_files";
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
        return Point2f(corrected.at<float>(0), corrected.at<float>(1));
    }

    void reset() {
        initialized = false;
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
        initialized = true;
    }

    cv::KalmanFilter kf;
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
            filters[i].reset();
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
    if (data.empty()) {
        cout << "[EXPORT] Skipped empty recording: " << suffix << "\n";
        return;
    }

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
    cout << "[EXPORT] Saved: " << fs::absolute(filename) << " (" << data.size() << " frames)\n";
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

        string profileBlock = extractJsonObjectForKey(text, "profile");
        if (profileBlock.empty()) profileBlock = text;

        GAMovementProfile profile;
        profile.patientId = pId;
        profile.movementName = movementNameFromGAFilename(path, pId);
        profile.sourcePath = fs::absolute(path).string();
        profile.fitness = parseJsonFieldFromText(text, "ga_fitness");
        profile.shoulderFlexion = parseGAJointRange(profileBlock, "sh_flex");
        profile.elbowFlexion = parseGAJointRange(profileBlock, "el_flex");

        if (profile.shoulderFlexion.valid || profile.elbowFlexion.valid) {
            profiles.push_back(profile);
        }
    }

    cout << "[OK] Loaded " << profiles.size() << " GA movement profiles.\n";
    return profiles;
}

static double scoreAngleAgainstGARange(double liveAngle, const GAJointRange& range) {
    if (!range.valid) return 0.0;
    double lo = std::min(range.startAngle, range.maxAngle);
    double hi = std::max(range.startAngle, range.maxAngle);
    if (liveAngle < lo) return lo - liveAngle;
    if (liveAngle > hi) return liveAngle - hi;

    double span = std::max(1.0, hi - lo);
    double progress = std::abs(liveAngle - range.startAngle) / span;
    return progress * 5.0;
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

        double tUal = 0, tFl = 0, tElbow = 0, tAbsJerk = 0, tVel = 0;
        int frames = 0;
        int jerkSamples = 0;
        int velocitySamples = 0;
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
        }
        if (frames == 0) continue;

        PatientProfile fileProfile{
            pId,
            tUal / frames,
            tFl / frames,
            tElbow / frames,
            jerkSamples > 0 ? tAbsJerk / jerkSamples : 0.0,
            velocitySamples > 0 ? tVel / velocitySamples : 0.0,
            frames
        };

        bool found = false;
        for (auto& p : profiles) {
            if (p.patientId == pId) {
                int totalSamples = p.sampleCount + fileProfile.sampleCount;
                p.averageUpperArmLength =
                    (p.averageUpperArmLength * p.sampleCount + fileProfile.averageUpperArmLength * fileProfile.sampleCount) / totalSamples;
                p.averageForearmLength =
                    (p.averageForearmLength * p.sampleCount + fileProfile.averageForearmLength * fileProfile.sampleCount) / totalSamples;
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
    double bestLengthScore = std::numeric_limits<double>::max();

    for (const auto& p : profiles) {
        double lengthScore =
            abs(uArm - p.averageUpperArmLength) +
            abs(fArm - p.averageForearmLength);

        double tieBreakerScore =
            lengthScore +
            0.20 * abs(elbowAngle - p.averageElbowAngle) +
            0.02 * abs(wristVelocity - p.averageWristVelocity) +
            6.0 * abs(log1p(absJerk) - log1p(p.averageAbsJerk));

        candidates.push_back({p.patientId, lengthScore, tieBreakerScore});
        bestLengthScore = std::min(bestLengthScore, lengthScore);
    }

    if (candidates.empty() || bestLengthScore > PERSON_MATCH_SCORE_THRESHOLD) {
        if (bestScoreOut) *bestScoreOut = candidates.empty() ? -1.0 : bestLengthScore;
        return -1;
    }

    CandidateScore best;
    bool hasBest = false;
    bool ambiguous = false;
    for (const auto& candidate : candidates) {
        bool closeToBest = candidate.lengthScore <= bestLengthScore + PERSON_MATCH_AMBIGUITY_MARGIN;
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
        if (bestScoreOut) *bestScoreOut = bestLengthScore;
        return -1;
    }

    if (bestScoreOut) {
        *bestScoreOut = ambiguous ? best.tieBreakerScore : best.lengthScore;
    }
    return best.patientId;
}

SessionConfig configureSession(int menuChoice, int patientId) {
    if (menuChoice == 1) {
        if (patientId <= 0) return {"", "Invalid patient ID.", false};
        return {"Training", "Starting CALIBRATION for Patient " + to_string(patientId) + " (Records 9 JSONs)", true};
    }
    if (menuChoice == 2) {
        return {"Live_Tracking", "Starting LIVE TRACKING (Auto-identifies patient)", true};
    }
    return {"", "Invalid choice.", false};
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
        "pose_landmarker_lite.task",
        "Camera_testing/pose_landmarker_lite.task",
        "pose_landmarker_full.task",
        "Camera_testing/pose_landmarker_full.task"
    });
    string handModelPath = argc > 2 ? argv[2] : firstExistingPath({
        "hand_landmarker.task",
        "Camera_testing/hand_landmarker.task"
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
        << "  1. Training Phase (Record 9 JSON movements)\n"
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
    poseOptions->min_pose_detection_confidence = 0.6f;
    poseOptions->min_pose_presence_confidence = 0.6f;
    poseOptions->min_tracking_confidence = 0.6f;

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
        handOptions->min_hand_detection_confidence = 0.6f;
        handOptions->min_hand_presence_confidence = 0.6f;
        handOptions->min_tracking_confidence = 0.6f;

        auto handOr = mp_hand::HandLandmarker::Create(std::move(handOptions));
        if (!handOr.ok()) {
            cerr << "[ERROR] HandLandmarker create failed: " << handOr.status() << "\n";
            return 1;
        }
        handLandmarker = std::move(handOr.value());
    }

    vector<PatientProfile> loadedProfiles;
    vector<GAMovementProfile> gaMovementProfiles;
    if (session.sessionType == "Live_Tracking") {
        loadedProfiles = loadAllPatientProfiles();
        gaMovementProfiles = loadGAMovementProfiles();
    }

    vector<BiometricFrame> recordedFrames;
    int currentRecordingStep = 0;
    bool recordingTimerStarted = false;
    bool recordingPauseActive = true;
    double recordingStepStartTime = 0.0;
    const string recordingNames[9] = {
        "move1_slow", "move1_med", "move1_fast",
        "move2_slow", "move2_med", "move2_fast",
        "move3_slow", "move3_med", "move3_fast"
    };

    Point3D prevWrist = {0, 0, 0};
    double prevTime = 0, prevVel = 0, prevAccel = 0;
    bool isFirstFrame = true;
    int liveIdentificationFrames = 0;
    double liveUpperArmSum = 0.0;
    double liveForearmSum = 0.0;
    double liveElbowSum = 0.0;
    double liveVelocitySum = 0.0;
    double liveAbsJerkSum = 0.0;
    double liveBestScore = -1.0;
    string activeGAMovement = "waiting";
    string activeGAFile = "";
    double activeGAScore = 0.0;

    vector<KalmanPoint2D> poseKalman(33);
    vector<KalmanPoint2D> handKalman(21);
    KalmanPoint2D fusedWristKalman;

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

        mp::Image mpImage = cvMatBgrToMediaPipeImage(frame);
        auto poseResultOr = poseLandmarker->DetectForVideo(mpImage, timestampMs);

        if (poseResultOr.ok()) {
            const auto& poseResult = poseResultOr.value();
            vector<Point2f> posePixels = extractPosePixels(poseResult, frame.size(), poseKalman, dt);
            int shoulderId = TRACK_LEFT_ARM ? POSE_LEFT_SHOULDER : POSE_RIGHT_SHOULDER;
            int elbowId = TRACK_LEFT_ARM ? POSE_LEFT_ELBOW : POSE_RIGHT_ELBOW;
            int wristId = TRACK_LEFT_ARM ? POSE_LEFT_WRIST : POSE_RIGHT_WRIST;

            if (posePixels.size() > static_cast<size_t>(wristId) &&
                posePixels[shoulderId].x >= 0 &&
                posePixels[elbowId].x >= 0 &&
                posePixels[wristId].x >= 0) {
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

                auto intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, depthMat.cols, depthMat.rows);
                vector<Point3D> poseLandmarks3D = pixelsTo3D(posePixels, frame.size(), depthMat, intrinsics);
                vector<Point3D> handLandmarks3D = pixelsTo3D(handPixels, frame.size(), depthMat, intrinsics);

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

                    bool justStartedRecording = false;
                    if (session.sessionType == "Training" && currentRecordingStep < 9 && recordingPauseActive) {
                        if (curTimeSecs - recordingStepStartTime >= RECORDING_PAUSE_SECONDS) {
                            recordedFrames.clear();
                            recordingPauseActive = false;
                            recordingStepStartTime = curTimeSecs;
                            prevWrist = pWrist;
                            prevTime = curTimeSecs;
                            prevVel = 0.0;
                            prevAccel = 0.0;
                            isFirstFrame = true;
                            justStartedRecording = true;
                        }
                    }

                    if (session.sessionType == "Training" && currentRecordingStep < 9 && !recordingPauseActive && !justStartedRecording) {
                        recordedFrames.push_back({curTimeSecs, patientId, session.sessionType, curUarm, curFarm, elbowFlexion, shoulderFlexion, curVel, curAccel, curJerk, armLandmarks3D, poseLandmarks3D, handLandmarks3D});
                        double activeRecordingDuration = recordingDurationForName(recordingNames[currentRecordingStep]);
                        if (curTimeSecs - recordingStepStartTime > activeRecordingDuration) {
                            saveRecordingToJSON(recordedFrames, patientId, recordingNames[currentRecordingStep]);
                            recordedFrames.clear();
                            currentRecordingStep++;
                            recordingStepStartTime = curTimeSecs;
                            recordingPauseActive = currentRecordingStep < 9;
                        }
                    }

                    if (session.sessionType == "Live_Tracking" && patientId == -1) {
                        liveIdentificationFrames++;
                        liveUpperArmSum += curUarm;
                        liveForearmSum += curFarm;
                        liveElbowSum += elbowFlexion;
                        liveVelocitySum += curVel;
                        liveAbsJerkSum += abs(curJerk);

                        if (liveIdentificationFrames >= LIVE_IDENTIFICATION_MIN_FRAMES) {
                            patientId = identifyPatientFromBiometrics(
                                liveUpperArmSum / liveIdentificationFrames,
                                liveForearmSum / liveIdentificationFrames,
                                liveElbowSum / liveIdentificationFrames,
                                liveVelocitySum / liveIdentificationFrames,
                                liveAbsJerkSum / liveIdentificationFrames,
                                loadedProfiles,
                                &liveBestScore
                            );
                        }
                    }

                    if (session.sessionType == "Live_Tracking" && patientId != -1) {
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
                            activeGAMovement = "no GA match";
                            activeGAFile = "";
                        }
                    }

                    if (session.sessionType == "Training" && currentRecordingStep < 9 && (recordingPauseActive || justStartedRecording)) {
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

        putText(frame, "SYSTEM MODE: " + session.sessionType, cv::Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

        if (session.sessionType == "Training") {
            putText(frame, "PATIENT ID: " + to_string(patientId), cv::Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

            if (currentRecordingStep < 9) {
                double timeElapsed = curTimeSecs - recordingStepStartTime;
                double activeRecordingDuration = recordingDurationForName(recordingNames[currentRecordingStep]);
                if (recordingPauseActive) {
                    int secondsLeft = std::max(0, static_cast<int>(std::ceil(RECORDING_PAUSE_SECONDS - timeElapsed)));
                    string taskStr = "GET READY: TASK " + to_string(currentRecordingStep + 1) + "/9: " + recordingNames[currentRecordingStep] +
                        " starts in " + to_string(secondsLeft) + "s";
                    putText(frame, taskStr, cv::Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 220, 80), 2);
                    putText(frame, "Recording paused - move into start position",
                        cv::Point(20, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 220, 80), 2);
                } else {
                    string taskStr = "RECORDING TASK " + to_string(currentRecordingStep + 1) + "/9: " + recordingNames[currentRecordingStep] +
                        "  [" + to_string(static_cast<int>(timeElapsed)) + "/" + to_string(static_cast<int>(activeRecordingDuration)) + "s]";
                    putText(frame, taskStr, cv::Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 165, 255), 2);
                    putText(frame, "FRAMES SAVED IN BUFFER: " + to_string(recordedFrames.size()),
                        cv::Point(20, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(180, 255, 180), 2);
                }
            } else {
                putText(frame, "ALL 9 RECORDINGS COMPLETE - Press 'Q'", cv::Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
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
                if (!activeGAFile.empty()) {
                    putText(frame, "Use: " + fs::path(activeGAFile).filename().string(),
                        cv::Point(20, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(120, 220, 255), 2);
                }
            }
        }

        if (!personVisible && poseResultOr.ok()) {
            putText(frame, "! NO PERSON DETECTED", cv::Point(20, 220), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        imshow("Project LIMB - Rehabilitation Vision System", frame);
        if (waitKey(1) == 'q') break;
    }

    if (session.sessionType == "Training" && !recordingPauseActive && !recordedFrames.empty() && currentRecordingStep < 9) {
        saveRecordingToJSON(recordedFrames, patientId, recordingNames[currentRecordingStep] + "_partial");
    }

    poseLandmarker->Close();
    if (handLandmarker) handLandmarker->Close();
    return 0;
}
