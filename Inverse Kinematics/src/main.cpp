#include "fabrik.h"
#include <iostream>

using namespace Eigen;

int main()
{   
    // Joints coordinates:
    std::vector<float> flatArmCoords = {
        0.0f,  0.0f, 0.0f,  // Soulder
        30.0f, 0.0f, 0.0f,  // Elbow
        55.0f, 0.0f, 0.0f   // Wrist
    };
    std::vector<float> flatHandCoords = {
        // Finger 1 (thumb):
        55.0f, 0.0f, 0.0f,  60.0f, -2.0f, 5.0f,  65.0f, -3.0f, 8.0f,  68.0f, -4.0f, 10.0f, 70.0f, -5.0f, 12.0f,
        // Finger 2:
        55.0f, 0.0f, 0.0f,  62.0f, 1.0f, 3.0f,   68.0f, 1.5f, 3.5f,   73.0f, 1.5f, 4.0f,  76.0f, 1.0f, 4.5f,
        // Finger 3:
        55.0f, 0.0f, 0.0f,  63.0f, 2.0f, 0.0f,   70.0f, 2.5f, 0.0f,   76.0f, 2.5f, 0.0f,  80.0f, 2.0f, 0.0f,
        // Finger 4:
        55.0f, 0.0f, 0.0f,  62.0f, 1.0f, -3.0f,  69.0f, 1.5f, -3.5f,  74.0f, 1.5f, -4.0f, 77.0f, 1.0f, -4.5f,
        // Finger 5:
        55.0f, 0.0f, 0.0f,  60.0f, 0.0f, -6.0f,  65.0f, -1.0f, -7.0f, 69.0f, -1.5f, -8.0f, 72.0f, -2.0f, -9.0f
    };
    std::vector<float> flatTargetCoords = {
        50.0f, 15.0f, 5.0f,   // The wrists target
        // The finger tops targets:
        65.0f, 10.0f, 15.0f,
        75.0f, 20.0f, 8.0f,
        80.0f, 22.0f, 0.0f,
        75.0f, 20.0f, -8.0f,
        70.0f, 15.0f, -15.0f
    };

    auto result = fabrik(flatArmCoords, flatHandCoords, flatTargetCoords);  // auto gives the type automatically

    // -------- TEST --------
    std::cout << "Joint positions:\n";

    std::cout << "Arm:\n";
    for(auto& joint : result.first)
    {
        std::cout
            << joint.x()
            << ", " << joint.y()
            << ", " << joint.z()
            << std::endl;
    }

    std::cout << "Hand:\n";
    for(auto& joint : result.second)
    {
        std::cout
            << joint.x()
            << ", " << joint.y()
            << ", " << joint.z()
            << std::endl;
    }

    return 0;
}