cmake_minimum_required(VERSION 2.4.8)

project(connectedComponents)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

add_library(connectedComponents components.cpp components.hpp)

add_executable(main main.cpp )

target_link_libraries(main ${OpenCV_LIBS} connectedComponents)
