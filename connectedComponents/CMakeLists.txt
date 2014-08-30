project(connectedComponents)

add_library(connectedComponents components.cpp components.hpp)
add_executable(main main.cpp )
target_link_libraries(main ${OpenCV_LIBS} connectedComponents)
