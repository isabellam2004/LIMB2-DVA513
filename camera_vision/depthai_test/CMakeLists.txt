# cmake_minimum_required(VERSION 3.16)
# project(LIMB2)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Point on depthai-core in external/
# set(DEPTHAI_DIR "${CMAKE_SOURCE_DIR}/external/depthai-core")

# # Point on the dir containing OpenCVConfig.cmake in opencv
# set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/external/opencv/build")

# # Help CMake find the config-file
# list(APPEND CMAKE_PREFIX_PATH "${DEPTHAI_DIR}")

# # Find depthai
# find_package(depthai CONFIG REQUIRED)

# # Find opencv
# find_package(OpenCV REQUIRED)

# # Create executable from src/
# add_executable(LIMB_app src/main.cpp)

# # Link library
# target_link_libraries(LIMB_app PRIVATE depthai::core)


# ----------------------------------------------------------------------


cmake_minimum_required(VERSION 3.14)
# 3.14 is the minimum version that reliably supports C++17, target-based linking,
# and the modern find_package() behaviour used by the DepthAI SDK.

project(LimbVision)
# Project name used by CMake for IDE integration and build artefact naming.

# ─────────────────────────────────────────────────────────────────────────────
# C++ STANDARD
# ─────────────────────────────────────────────────────────────────────────────
# C++17 is required for:
#   <filesystem>  - scanning the directory for patient JSON files
#   constexpr     - compile-time constants used throughout the pipeline
#   structured bindings and other modern syntax
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Hard error if the compiler does not support C++17
set(CMAKE_CXX_EXTENSIONS OFF)        # Use -std=c++17, NOT -std=gnu++17 (keeps code portable)

set(DEPTHAI_DIR "${CMAKE_SOURCE_DIR}/external/depthai-core/lib/cmake/depthai")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TYPE
# ─────────────────────────────────────────────────────────────────────────────
# Default to Release if no build type was specified on the command line.
# Release enables compiler optimisations (-O2 / -O3) which significantly
# improve AI inference and video processing performance.
# Use 'cmake .. -DCMAKE_BUILD_TYPE=Debug' when debugging with gdb.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# FIND DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

# DepthAI SDK (Luxonis OAK-D camera driver and pipeline API)
# Must be installed first - see SETUP INSTRUCTIONS at the bottom of this file.
# CONFIG mode tells CMake to use the depthai-config.cmake file installed by the SDK,
# rather than trying to find the library itself with a FindXxx.cmake module.
find_package(depthai CONFIG REQUIRED)

# OpenCV (used for video display, drawing skeleton lines and HUD text on screen)
# Install on Ubuntu: sudo apt install libopencv-dev
find_package(OpenCV REQUIRED)

# Print the found versions so the developer can verify the correct libraries were picked up
message(STATUS "DepthAI found: ${depthai_VERSION}")
message(STATUS "OpenCV found:  ${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTABLE
# ─────────────────────────────────────────────────────────────────────────────
# FIX: Updated source file name to match the actual file (limb_final.cpp).
# The original CMakeLists.txt referenced 'limb_feature_extraction.cpp' which
# no longer exists - this would cause a build error.
add_executable(limb_app /camera_test/limb_final.cpp)

# ─────────────────────────────────────────────────────────────────────────────
# INCLUDE DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
# Tells the compiler where to find header files for our dependencies.
# PRIVATE means these paths are only used when compiling limb_app itself,
# not exposed to any other targets that might link against it.
target_include_directories(limb_app PRIVATE
    ${OpenCV_INCLUDE_DIRS}   # e.g. /usr/include/opencv4
)
# Note: DepthAI include paths are handled automatically via the imported target
# depthai::core, so they do not need to be listed here manually.

# ─────────────────────────────────────────────────────────────────────────────
# LINK LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
# Links the compiled executable against the required libraries.
#
# depthai::core   - Core DepthAI SDK: pipeline, nodes, XLink communication with the camera
# depthai::opencv - DepthAI OpenCV integration: getCvFrame() and related helpers
# ${OpenCV_LIBS}  - All OpenCV modules found above (core, highgui, imgproc, videoio, etc.)
target_link_libraries(limb_app PRIVATE
    depthai::core
    depthai::opencv
    ${OpenCV_LIBS}
)

# ─────────────────────────────────────────────────────────────────────────────
# COMPILER WARNINGS
# ─────────────────────────────────────────────────────────────────────────────
# Enable common warnings to catch potential bugs at compile time.
# These apply only to our own code, not to the DepthAI/OpenCV library headers.
target_compile_options(limb_app PRIVATE
    -Wall       # Enable all standard warnings
    -Wextra     # Enable extra warnings beyond -Wall
    -Wpedantic  # Enforce strict ISO C++ compliance
)

# ─────────────────────────────────────────────────────────────────────────────
# COPY BLOB FILES TO BUILD DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
# The two AI model files must be in the same directory as the executable at runtime.
# This command automatically copies them from the source tree to the build directory
# every time the project is built, so you never have to copy them manually.
configure_file(
    ${CMAKE_SOURCE_DIR}/camera_test/pose_detection_sh4.blob
    ${CMAKE_BINARY_DIR}/camera_test/pose_detection_sh4.blob
    COPYONLY
)
configure_file(
    ${CMAKE_SOURCE_DIR}/camera_test/pose_landmark_full_sh4.blob
    ${CMAKE_BINARY_DIR}/camera_test/pose_landmark_full_sh4.blob
    COPYONLY
)

# ─────────────────────────────────────────────────────────────────────────────
# SETUP INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Run these commands once to install the DepthAI SDK from source:
#
#   git clone https://github.com/luxonis/depthai-core.git --recursive
#   cd depthai-core
#   mkdir build && cd build
#   cmake .. -DBUILD_SHARED_LIBS=ON
#   make -j$(nproc)
#   sudo make install
#
# Then build this project:
#
#   cd /path/to/your/project        (folder containing limb_final.cpp and this CMakeLists.txt)
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#   ./limb_app