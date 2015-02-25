project(firepony)
cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

option(USE_SYSTEM_BOOST
       "Use the system-wide installed version of Boost"
       ON)

option(USE_EXTERNAL_THRUST
       "Build using Thrust from Github (do not use the CUDA toolkit version)"
       ON)

option(ENABLE_CUDA_BACKEND
       "Enable the GPU-accelerated CUDA backend"
       ON)

option(ENABLE_CPU_BACKEND
       "Enable the CPU backend based on Intel Threading Building Blocks"
       ON)

if (ENABLE_CUDA_BACKEND)
    add_definitions(-DENABLE_CUDA_BACKEND=1)
endif()

if (ENABLE_CPU_BACKEND)
    add_definitions(-DENABLE_TBB_BACKEND=1)
endif()

# default to debug builds
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# grab gcc flags
include("gcc.cmake")

# build dependencies
if (NOT USE_SYSTEM_BOOST)
    include("contrib/boost.cmake")
    include_directories(${Boost_INCLUDE_DIR})
endif()

if (USE_EXTERNAL_THRUST)
    include("contrib/thrust.cmake")
endif()

include("contrib/gamgee.cmake")
include("contrib/zlib.cmake")
include("contrib/cub.cmake")

if (ENABLE_CPU_BACKEND)
    include("contrib/tbb.cmake")
    set(SYSTEM_LINK_LIBRARIES ${SYSTEM_LINK_LIBRARIES} ${tbb_LIB})
endif()

# locate required packages
# CUDA 7.0 is required to build
find_package(CUDA 7.0 REQUIRED)
# enable C++11 support
set(CUDA_NVCC_FLAGS "-std=c++11")
# build for Kepler
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=sm_35 -use_fast_math)

if (USE_EXTERNAL_THRUST)
    # inject our Thrust include path to override the CUDA toolkit include path
    # (this is a bit of a hack)
    set(CUDA_INCLUDE_DIRS ${thrust_INCLUDE} -I${CUDA_INCLUDE_DIRS})
endif()

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

add_subdirectory(device)

set(firepony_common_sources
    alignment_data.h
    command_line.cu
    command_line.h
    io_thread.cu
    io_thread.h
    runtime_options.h
    segmented_database.h
    string_database.cu
    string_database.h
    types.h
    version.h

    loader/alignments.cu
    loader/alignments.h
    loader/reference.cu
    loader/reference.h
    loader/variants.cu
    loader/variants.h
    )

cuda_add_library(firepony-common ${firepony_common_sources})
add_dependencies(firepony-common gamgee zlib)
if (ENABLE_CPU_BACKEND)
    add_dependencies(firepony-common tbb)
endif()
if (USE_EXTERNAL_THRUST)
    add_dependencies(firepony-common thrust)
endif()

cuda_add_executable(firepony firepony.cu)
target_link_libraries(firepony firepony-common firepony-device ${tbb_LIB} ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})

cuda_build_clean_target()
