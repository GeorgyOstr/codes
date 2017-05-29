project(vision)
cmake_minimum_required(VERSION 2.4.8)
cmake_policy(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(connectedComponents)
add_subdirectory(surveillance)

add_subdirectory(dummy_example)
