project(firepony)
cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

option(USE_SYSTEM_BOOST
       "Use the system-wide installed version of Boost"
       ON)

# default to debug builds
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# grab gcc flags
include("gcc.cmake")

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

find_library(RT_LIBRARY rt)
if (RT_LIBRARY)
    set(SYSTEM_LINK_LIBRARIES ${SYSTEM_LINK_LIBRARIES} ${RT_LIBRARY})
endif()

# set cuda debug flags
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -g")
endif()

include_directories(${CMAKE_SOURCE_DIR})

# build dependencies
if (NOT USE_SYSTEM_BOOST)
    include("contrib/boost.cmake")
endif()

include("contrib/gamgee.cmake")
include("contrib/zlib.cmake")
include("contrib/cub.cmake")

add_subdirectory(device)

set(firepony_common_sources
    alignment_data.h
    command_line.cu
    command_line.h
    gamgee_loader.cu
    gamgee_loader.h
    io_thread.cu
    io_thread.h
    mmap.cu
    mmap.h
    serialization.h
    serialization_inl.h
    string_database.cu
    string_database.h
    types.h
    runtime_options.h)

cuda_add_library(firepony-common ${firepony_common_sources})
add_dependencies(firepony-common gamgee zlib)

cuda_add_executable(firepony firepony.cu)
target_link_libraries(firepony firepony-common firepony-device ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})

cuda_add_executable(firepony-loader firepony-loader.cu)
target_link_libraries(firepony-loader firepony-common firepony-device ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})
