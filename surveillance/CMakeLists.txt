project(surveillance)

add_executable(surveillance capture.cpp )
target_link_libraries(surveillance ${OpenCV_LIBS})
