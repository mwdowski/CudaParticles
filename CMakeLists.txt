cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CudaParticles
        VERSION 1.0
        LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")


add_library(application application/application.cpp application/application.hpp)
target_include_directories(application PRIVATE application)
target_include_directories(application PRIVATE ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
target_link_libraries(application ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(main_gpu gpu/main.cu)
add_executable(main_cpu cpu/main.cpp)

set_target_properties(main_gpu PROPERTIES CUDA_ARCHITECTURES "72")

target_link_libraries(main_gpu ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
target_link_libraries(main_cpu application)
