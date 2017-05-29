set(project_name "dummy")

project(${project_name})

add_executable(${project_name} main.cpp)
target_link_libraries(${project_name} ${OpenCV_LIBS})
