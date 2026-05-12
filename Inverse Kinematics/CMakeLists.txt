cmake_minimum_required(VERSION 3.16)

project(LIMB_IK)

set(CMAKE_CXX_STANDARD 17)

add_executable(limb_IK
    src/FABRIK.cpp
)

target_include_directories(limb_IK PRIVATE
    libs/eigen
)