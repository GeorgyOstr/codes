project(connectedComponents)

add_library(connectedComponents components.cpp components.hpp)
add_executable(test_app main.cpp )
target_link_libraries(test_app ${OpenCV_LIBS} connectedComponents)
