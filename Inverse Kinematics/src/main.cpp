#include "fabrik.h"
#include <iostream>

using namespace Eigen;

int main()
{   
    // Joints coordinates:
    std::vector<float> flatJointsCoords = {
        0, 0, 0,
        0, 0, 30,
        0, 0, 60
    };

    std::vector<float> lengths; // Segment lengths

    // Create the segment lengths:
    float shoulderToElbow(50);
    float elbowToHand(30);

    // Add the lengths to lengths:
    lengths.push_back(shoulderToElbow);
    lengths.push_back(elbowToHand);

    Vector3f target(45, 19, 30);    // The target

    fabrik(flatJointsCoords, lengths, target);

    return 0;
}