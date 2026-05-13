#include "fabrik.h"
#include <iostream>
#include <vector>
#include <utility>

using namespace Eigen;

void create_points_list(std::vector<Vector3f>& pointsList, std::vector<float>& flatCoords)
{
    for(size_t i = 0; i < flatCoords.size(); i += 3)
    {
        pointsList.push_back(Vector3f(flatCoords[i], flatCoords[i + 1], flatCoords[i + 2]));
    }

    return;
}

void backward_forward_pass(std::vector<Vector3f>& joints, const std::vector<float>& lengths, Vector3f target, Vector3f rootPos)
{
    // -------- BACKWARD PASS -------- //

    joints.back() = target; // Put the last joint onto target

    // Go back through the chain:
    for(int i = joints.size() - 2; i >= 0; i--)
    {
        Vector3f dir = (joints[i] - joints[i + 1]).normalized();    // The direction from current joint to the next joint
        joints[i] = joints[i + 1] + dir * lengths[i];   // Place the joint right according to the segment length
    }

    // -------- FORWARD PASS -------- //

    joints[0] = rootPos;  // Lock the root to rootPos

    // Go forward through the chain:
    for(int i = 1; i < joints.size(); i++)
    {
        Vector3f dir = (joints[i] - joints[i - 1]).normalized();    // The direction from the previous joint to the current joint
        joints[i] = joints[i - 1] + dir * lengths[i - 1];   // Place the joint right according to the segment length
    }

    return;
}

std::pair<std::vector<Vector3f>, std::vector<Vector3f>> fabrik(std::vector<float>& flatArmCoords, std::vector<float>& flatHandCoords, std::vector<float>& flatTargetCoords)
{
    // Constant arm and finger lengths:
    const std::vector<float> armLens = {
        50, 30
    };
    const std::vector<float> handLens = {
        30, 10, 5, 5,
        60, 8, 8, 8,
        60, 10, 10, 10,
        60, 9, 9, 9,
        60, 5, 5, 5
    };

    // Check 3D-coordinates:
    if (flatArmCoords.size() % 3 != 0 || flatHandCoords.size() % 3 != 0 || flatTargetCoords.size() % 3 != 0)
    {
        std::cout << "Wrong coordinate size!";
        return {};
    }

    // Create 3D-coordinate lists:
    std::vector<Vector3f> armJoints;
    create_points_list(armJoints, flatArmCoords);

    std::vector<Vector3f> handJoints;
    create_points_list(handJoints, flatHandCoords);

    std::vector<Vector3f> targets;
    create_points_list(targets, flatTargetCoords);

    // Do the algorithm for the arm:
    backward_forward_pass(armJoints, armLens, targets[0], Vector3f(0, 0, 0));

    // Get the wrists position:
    Vector3f wristPos = armJoints.back();
    
    int targetCounter = 1;
    const int jointsPerFinger = 5;

    // Do the algorithm for each finger:
    for(int i = 0; i < 5; i++)
    {
        std::vector<float> tmpHandLens;
        for(int j = 0; j < 4; j++)  // Take out the lengths for the currentt finger
        {
            tmpHandLens.push_back(handLens[i * 4 + j]);
        }

        std::vector<Vector3f> tmpFingerJoints;
        for(int j = 0; j < jointsPerFinger; j++)    // Take out the joints for the current finger
        {
            tmpFingerJoints.push_back(handJoints[i * jointsPerFinger + j]);
        }

        backward_forward_pass(tmpFingerJoints, tmpHandLens, targets[targetCounter], wristPos);

        // Copy the new positions into handJoints:
        for(int j = 0; j < jointsPerFinger; j++)
        {
            handJoints[i * jointsPerFinger + j] = tmpFingerJoints[j];
        }

        targetCounter++;
    }

    return {armJoints, handJoints};
}