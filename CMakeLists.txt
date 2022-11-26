cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CudaParticles
        VERSION 1.0
        LANGUAGES CXX CUDA)

add_executable(main_gpu gpu/main.cu)
add_executable(main_cpu cpu/main.cpp)

set_target_properties(main_gpu PROPERTIES CUDA_ARCHITECTURES "72")