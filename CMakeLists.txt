cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_test
        VERSION 1.0
        LANGUAGES CXX CUDA)
        
add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "72")