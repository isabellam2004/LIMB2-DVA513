#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;


void fabrik(std::vector<Vector3f>& joints, std::vector<float>& lengths, Vector3f target)
{
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
}

int main()
{
    std::vector<Vector3f> joints;   // The arms joints

    // Create the joints:
    Vector3f shoulder(0, 0, 0);
    Vector3f elbow(0, 0, 30);
    Vector3f hand(0, 0, 60);

    // Add the joints to joints:
    joints.push_back(shoulder);
    joints.push_back(elbow);
    joints.push_back(hand);

    std::vector<float> lengths; // Segment lengths

    // Create the segment lengths:
    float shoulderToElbow(50);
    float elbowToHand(30);

    // Add the lengths to lengths:
    lengths.push_back(shoulderToElbow);
    lengths.push_back(elbowToHand);

    Vector3f target(45, 19, 30);    // The target

    fabrik(joints, lengths, target);

    // Test
    std::cout << "Joint positions:\n";

    for(auto& joint : joints)
    {
        std::cout
            << "x: " << joint.x()
            << ", y: " << joint.y()
            << ", z: " << joint.z()
            << std::endl;
    }


    // std::cout << "Start OK\n";

    // float error = (hand - target).norm();
    // std::cout << "Error: " << error << std::endl;

    return 0;
}