// If MATHUTILS_H is not defined, define it (start of header guard)
#ifndef FABRIK_H

// Define MATHUTILS_H to prevent multiple inclusions
#define FABRIK_H

#include <vector>
#include <Eigen/Dense>

// Function declarations (only the function names and parameters, not the logic)
void create_points_list(std::vector<Eigen::Vector3f>& pointsList, std::vector<float>& flatCoords);
void backward_forward_pass(std::vector<Eigen::Vector3f>& joints, const std::vector<float>& lengths, Eigen::Vector3f target, Eigen::Vector3f rootPos);
std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>> fabrik(std::vector<float>& flatArmCoords, std::vector<float>& flatHandCoords, std::vector<float>& flatTargetCoords);

#endif