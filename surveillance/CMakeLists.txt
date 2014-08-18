cmake_minimum_required(VERSION 2.4.8)

project(surveillance)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

add_executable(surveillance capture.cpp )
target_link_libraries(surveillance ${OpenCV_LIBS})
