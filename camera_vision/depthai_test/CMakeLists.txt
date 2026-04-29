cmake_minimum_required(VERSION 3.16)
project(LIMB2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Peka på depthai-core i external/
set(DEPTHAI_DIR "${CMAKE_SOURCE_DIR}/external/depthai-core-v3.5.0-win64")

# Hjälp CMake hitta config-filen
list(APPEND CMAKE_PREFIX_PATH "${DEPTHAI_DIR}")

# Hitta depthai
find_package(depthai CONFIG REQUIRED)

# Skapa executable från src/
add_executable(my_app src/main.cpp)

# Länka biblioteket
target_link_libraries(my_app PRIVATE depthai::core)