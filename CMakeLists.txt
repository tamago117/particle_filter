cmake_minimum_required(VERSION 3.9)
project(particle_filter)

set(CMAKE_CXX_STANDARD 11)

find_package(PythonLibs 3.8)

include_directories(
        include
        lib
        ${PYTHON_INCLUDE_DIRS}
)

add_executable(particle_filter
        include/particle_filter.cpp
        main.cpp
)
#target_include_directories(dynamic_window_approach PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(particle_filter
        ${PYTHON_LIBRARIES}
)