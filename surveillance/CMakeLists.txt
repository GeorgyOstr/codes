cmake_minimum_required(VERSION 2.4.8)

find_package(OpenCV REQUIRED)

add_executable(surveillance capture.cpp )
target_link_libraries(surveillance ${OpenCV_LIBS})
