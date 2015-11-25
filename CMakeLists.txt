project(firepony)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake-local")

include(ExternalProject)

# if the environment variable FIREPONY_GPU_ARCHITECTURE exists, take the list of architectures from it
if (DEFINED ENV{FIREPONY_GPU_ARCHITECTURE})
    set(GPU_ARCHITECTURE $ENV{FIREPONY_GPU_ARCHITECTURE} CACHE STRING "GPU architectures to generate code for (defaults to sm_30, sm_35, sm_52)")
else ()
    # allow setting the list of GPU architectures on the command line
    set(GPU_ARCHITECTURE "sm_30,sm_35,sm_52" CACHE STRING "GPU architectures to generate code for (defaults to sm_30, sm_35, sm_52)")
endif ()

# build lift and grab propagated build parameters
# this has to be done early to avoid conflicts with CUDA command line flags
add_subdirectory(modules/lift)
include_directories(${LIFT_INCLUDE_DIRS})

# default to debug builds
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# grab gcc flags
include("gcc.cmake")

# build our external projects
include("contrib/zlib.cmake")
include("contrib/htslib.cmake")

# locate required packages
# CUDA 7.0 is required to build
# note: CUDA 7.5 causes hangs in Firepony!
find_package(CUDA 7.0 REQUIRED)
# enable C++11 support
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
# enable device lambda support
# xxxnsubtil: not usable yet
#list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

# set optimization flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)

# add nvcc flags to generate code for all enabled architectures
string(REPLACE "," ";" GPU_ARCHITECTURE ${GPU_ARCHITECTURE})
foreach(ARCH IN LISTS GPU_ARCHITECTURE)
    string(REPLACE "sm_" "compute_" COMPUTE ${ARCH})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        --generate-code arch=${COMPUTE},code=${ARCH})
endforeach(ARCH)

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

include("cmake-local/build_info.cmake")

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(device)

set(firepony_common_sources
    alignment_data.h
    command_line.cu
    command_line.h
    io_thread.cu
    io_thread.h
    mmap.cu
    mmap.h
    output.cu
    output.h
    runtime_options.h
    segmented_database.h
    sequence_database.h
    serialization.h
    serialization_inl.h
    string_database.cu
    string_database.h
    table_formatter.cu
    table_formatter.h
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
add_dependencies(firepony-common htslib zlib lift)

cuda_add_executable(firepony firepony.cu)
target_link_libraries(firepony firepony-device firepony-common ${htslib_LIB} ${zlib_LIB} ${LIFT_LINK_LIBRARIES})

cuda_add_executable(firepony-loader firepony-loader.cu)
target_link_libraries(firepony-loader firepony-common ${htslib_LIB} ${zlib_LIB} ${LIFT_LINK_LIBRARIES})
add_dependencies(firepony-loader zlib htslib)

cuda_build_clean_target()

install(TARGETS firepony firepony-loader
        RUNTIME DESTINATION bin)
