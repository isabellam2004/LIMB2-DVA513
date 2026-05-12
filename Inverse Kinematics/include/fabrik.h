// If MATHUTILS_H is not defined, define it (start of header guard)
#ifndef FABRIK_H

// Define MATHUTILS_H to prevent multiple inclusions
#define FABRIK_H

#include <vector>
#include <Eigen/Dense>

// Function declarations (only the function names and parameters, not the logic)
void create_joints_list(std::vector<Eigen::Vector3f>& jointsList, std::vector<float>& flatCoords);
void fabrik(std::vector<float>& flatCoords, std::vector<float>& lengths, Eigen::Vector3f target);

#endif