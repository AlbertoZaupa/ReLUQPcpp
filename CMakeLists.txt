cmake_minimum_required(VERSION 3.16)
project(MyCUDAProject LANGUAGES CXX CUDA)

# Set CUDA architecture (modify as needed)
set(CMAKE_CUDA_ARCHITECTURES 60)

# Enable C++11 or later
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the shared library and source files
add_library(solver SHARED solver.cu linalg.cu)

# Link against cuBLAS
find_package(CUDA REQUIRED)
target_link_libraries(solver PRIVATE cublas)

# Optional: Set the output directory for the shared library
set_target_properties(solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
