cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(CudaParticles
        VERSION 1.0
        LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Thrust REQUIRED)
find_package(glfw3 REQUIRED)

thrust_create_target(Thrust)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(DEFAULT_CUDA_ARCHITECTURE "72")

add_library(particles_set particles/particles_set/particles_set.hpp)
set_target_properties(particles_set PROPERTIES LINKER_LANGUAGE CXX)

add_library(kernels kernels/kernels.cuh)
set_target_properties(kernels PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(kernels PUBLIC kernels)
target_link_libraries(kernels ${CUDA_RUNTIME_LIBRARY})
set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})

add_library(macros macros/macros.cuh)
set_target_properties(macros PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(macros ${CUDA_RUNTIME_LIBRARY})
set_target_properties(macros PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})

add_library(engine particles/engine/engine.cuh)
target_include_directories(engine PUBLIC particles/particles_set PUBLIC ${CUDA_INCLUDE_DIRS} PUBLIC macros kernels)
target_link_libraries(engine ${CUDA_RUNTIME_LIBRARY})
target_link_libraries(engine Thrust)
set_target_properties(engine PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(engine PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})

add_library(application application/application.cu application/application.cuh)
target_include_directories(application PUBLIC application PUBLIC particles/particles_set PUBLIC particles/engine)
target_include_directories(application PUBLIC ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(application ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CUDA_RUNTIME_LIBRARY} ${OPENGL_gl_LIBRARY} glfw)
set_target_properties(application PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})

add_executable(main_gpu gpu/main.cu)
set_target_properties(main_gpu PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})
target_link_libraries(main_gpu ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CUDA_RUNTIME_LIBRARY} application macros)
#target_include_directories(main_gpu PUBLIC application)
#add_executable(main_cpu cpu/main.cpp)


#target_link_libraries(main_cpu application)
