project(surveillance)

include_directories("${CMAKE_SOURCE_DIR}/connectedComponents/")
add_executable(surveillance capture.cpp )
target_link_libraries(surveillance ${OpenCV_LIBS} connectedComponents)
