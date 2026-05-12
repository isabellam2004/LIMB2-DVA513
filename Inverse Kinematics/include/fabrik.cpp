#include "fabrik.h"
#include <iostream>

using namespace Eigen;

void create_joints_list(std::vector<Vector3f>& jointsList, std::vector<float>& flatCoords)
{
    for(size_t i = 0; i < flatCoords.size(); i += 3)
    {
        jointsList.push_back(Vector3f(flatCoords[i], flatCoords[i + 1], flatCoords[i + 2]));
    }
}

void fabrik(std::vector<float>& flatCoords, std::vector<float>& lengths, Vector3f target)
{
    // Check 3D coordinates
    if (flatCoords.size() % 3 != 0)
    {
        std::cout << "Wrong coordinate size!";
        return;
    }

    std::vector<Vector3f> joints;
    create_joints_list(joints, flatCoords);

    // -------- BACKWARD PASS -------- //

    joints.back() = target; // Put the last joint (the hand) onto target

    // Go back through the chain: hand -> elbow -> shoulder
    for(int i = joints.size() - 2; i >= 0; i--)
    {
        Vector3f dir = (joints[i] - joints[i + 1]).normalized();    // Calculates the direction from current joint to the next
        joints[i] = joints[i + 1] + dir * lengths[i];   // Put the joint at the right distance (segment length)
    }

    // -------- FORWARD PASS -------- //

    joints[0] = Vector3f(0, 0, 0);  // Lock the root (shoulder) to its original position

    // Go forward through the chain: shoulder -> elbow -> hand
    for(int i = 1; i < joints.size(); i++)
    {
        Vector3f dir = (joints[i] - joints[i - 1]).normalized();    // The direction from the previous joint to the current joint
        joints[i] = joints[i - 1] + dir * lengths[i - 1];   // Place the joint right according to the segment length
    }


    // -------- TEST --------
    std::cout << "Joint positions:\n";

    for(auto& joint : joints)
    {
        std::cout
            << "x: " << joint.x()
            << ", y: " << joint.y()
            << ", z: " << joint.z()
            << std::endl;
    }
}