/*
 * Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/omp/vector.h>
#include <thrust/system/cpp/vector.h>

#include <thrust/execution_policy.h>

namespace firepony
{

enum target_system
{
    // the target system decides where computation will run
    // note: host has a special meaning; it represents the host process memory space and doesn't run any computation
    host,
    cuda,
    omp,
    cpp,
};

template <target_system system, typename T>
struct backend_vector_type
{ };

template <typename T>
struct backend_vector_type<host, T>
{
    typedef thrust::host_vector<T> vector_type;
};

template <typename T>
struct backend_vector_type<cuda, T>
{
    typedef thrust::system::cuda::vector<T> vector_type;
};

template <typename T>
struct backend_vector_type<omp, T>
{
    typedef thrust::system::omp::vector<T> vector_type;
};

template <typename T>
struct backend_vector_type<cpp, T>
{
    typedef thrust::system::cpp::vector<T> vector_type;
};

template <target_system system>
struct backend_policy
{ };

template <>
struct backend_policy<cuda>
{
    static inline decltype(thrust::cuda::par)& execution_policy(void)
    {
        return thrust::cuda::par;
    }
};

template <>
struct backend_policy<omp>
{
    static inline decltype(thrust::omp::par)& execution_policy(void)
    {
        return thrust::omp::par;
    }
};


template <>
struct backend_policy<cpp>
{
    static inline decltype(thrust::cpp::par)& execution_policy(void)
    {
        return thrust::cpp::par;
    }
};

} // namespace firepony

// really ugly trick to force arbitrary device function instantiation
// note: we intentionally never instantiate device functions for the host backend
#define INSTANTIATE(fun) \
        namespace __ ## fun ## __instantiation {    \
            auto *ptr_cuda = fun<firepony::cuda>;   \
            auto *ptr_omp = fun<firepony::omp>;     \
            auto *ptr_cpp = fun<firepony::cpp>;     \
    }

#define METHOD_INSTANTIATE(base, method) \
        namespace __ ## base ## __ ## method ## __instantiation { \
            auto ptr_cuda = &base<firepony::cuda>::method;       \
            auto ptr_omp = &base<firepony::omp>::method;         \
            auto ptr_cpp = &base<firepony::cpp>::method;         \
    }
