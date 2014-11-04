project(firepony)
cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

option(CPU
       "Build for the CPU"
       OFF)

# default to debug builds
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# grab gcc flags
include("gcc.cmake")

if (CPU)
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DRUN_ON_CPU=1)
endif()

# locate required packages
# CUDA 7.0 is required to build
find_package(CUDA 7.0 REQUIRED)
# enable C++11 support
set(CUDA_NVCC_FLAGS "-std=c++11")
# build for Kepler
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=sm_35 -use_fast_math)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# set cuda debug flags
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -g")
endif()

include_directories(${CMAKE_SOURCE_DIR})

# build dependencies
include("contrib/boost.cmake")
include("contrib/gamgee.cmake")
include("contrib/zlib.cmake")
include("contrib/cub.cmake")

add_subdirectory(firepony)
