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

#ifndef ENABLE_CUDA_BACKEND
#define ENABLE_CUDA_BACKEND 0
#endif

#ifndef ENABLE_CPP_BACKEND
#define ENABLE_CPP_BACKEND 0
#endif

#ifndef ENABLE_OMP_BACKEND
#define ENABLE_OMP_BACKEND 0
#endif

namespace firepony
{

enum target_system
{
    // the target system decides where computation will run
    // note: host has a special meaning; it represents the host process memory space and doesn't run any computation
    host,
#if ENABLE_CUDA_BACKEND
    cuda,
#endif
#if ENABLE_CPP_BACKEND
    cpp,
#endif
#if ENABLE_OMP_BACKEND
    omp,
#endif
};

template <target_system system, typename T>
struct backend_vector_type
{ };


template <target_system system>
struct backend_policy
{ };

// host backend definitions
template <typename T>
struct backend_vector_type<host, T>
{
    typedef thrust::host_vector<T> vector_type;
};
// no launch policy for host

// cuda backend definitions
#if ENABLE_CUDA_BACKEND
template <typename T>
struct backend_vector_type<cuda, T>
{
    typedef thrust::system::cuda::vector<T> vector_type;
};

template <>
struct backend_policy<cuda>
{
    static inline decltype(thrust::cuda::par)& execution_policy(void)
    {
        return thrust::cuda::par;
    }
};
#endif

// openmp backend
#if ENABLE_OMP_BACKEND
template <typename T>
struct backend_vector_type<omp, T>
{
    typedef thrust::system::omp::vector<T> vector_type;
};

template <>
struct backend_policy<omp>
{
    static inline decltype(thrust::omp::par)& execution_policy(void)
    {
        return thrust::omp::par;
    }
};
#endif

// cpp threads backend
#if ENABLE_CPP_BACKEND
template <typename T>
struct backend_vector_type<cpp, T>
{
    typedef thrust::system::cpp::vector<T> vector_type;
};

template <>
struct backend_policy<cpp>
{
    static inline decltype(thrust::cpp::par)& execution_policy(void)
    {
        return thrust::cpp::par;
    }
};
#endif

} // namespace firepony

// ugly macro hackery to force arbitrary device function / method instantiation
// note: we intentionally never instantiate device functions for the host system
#if ENABLE_CUDA_BACKEND
#define __FUNC_CUDA(fun) auto *ptr_cuda = fun<firepony::cuda>;
#define __METHOD_CUDA(base, method) auto ptr_cuda = &base<firepony::cuda>::method;
#else
#define __FUNC_CUDA(fun) ;
#define __METHOD_CUDA(base, method) ;
#endif

#if ENABLE_CPP_BACKEND
#define __FUNC_CPP(fun) auto *ptr_cpp = fun<firepony::cpp>;
#define __METHOD_CPP(base, method) auto ptr_cpp = &base<firepony::cpp>::method;
#else
#define __FUNC_CPP(fun) ;
#define __METHOD_CPP(base, method) ;
#endif

#if ENABLE_OMP_BACKEND
#define __FUNC_OMP(fun) auto *ptr_omp= fun<firepony::omp>;
#define __METHOD_OMP(base, method) auto ptr_omp = &base<firepony::omp>::method;
#else
#define __FUNC_OMP(fun) ;
#define __METHOD_OMP(base, method) ;
#endif

// free function instantiation
#define INSTANTIATE(fun) \
        namespace __ ## fun ## __instantiation {    \
            __FUNC_CUDA(fun);                       \
            __FUNC_CPP(fun);                        \
            __FUNC_OMP(fun);                        \
    }

// method instantiation
#define METHOD_INSTANTIATE(base, method) \
        namespace __ ## base ## __ ## method ## __instantiation {   \
            __METHOD_CUDA(base, method);                            \
            __METHOD_CPP(base, method);                             \
            __METHOD_OMP(base, method);                             \
    }
