project(dummy)

add_executable(dummy main.cpp)
target_link_libraries(dummy ${OpenCV_LIBS})
