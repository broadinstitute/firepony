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

#ifndef ENABLE_CUDA_BACKEND
#define ENABLE_CUDA_BACKEND 0
#endif

#ifndef ENABLE_TBB_BACKEND
#define ENABLE_TBB_BACKEND 0
#endif

#include <thrust/device_vector.h>

#if ENABLE_CUDA_BACKEND
#include <thrust/system/cuda/vector.h>
#endif

#if ENABLE_TBB_BACKEND
#include <thrust/system/tbb/vector.h>
#endif

#include <thrust/execution_policy.h>

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
#if ENABLE_TBB_BACKEND
    intel_tbb,
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

// threading building blocks backend
#if ENABLE_TBB_BACKEND
template <typename T>
struct backend_vector_type<intel_tbb, T>
{
    typedef thrust::system::tbb::vector<T> vector_type;
};

template <>
struct backend_policy<intel_tbb>
{
    static inline decltype(thrust::tbb::par)& execution_policy(void)
    {
        return thrust::tbb::par;
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

#if ENABLE_TBB_BACKEND
#define __FUNC_TBB(fun) auto *ptr_TBB= fun<firepony::intel_tbb>;
#define __METHOD_TBB(base, method) auto ptr_TBB = &base<firepony::intel_tbb>::method;
#else
#define __FUNC_TBB(fun) ;
#define __METHOD_TBB(base, method) ;
#endif

// free function instantiation
#define INSTANTIATE(fun) \
        namespace __ ## fun ## __instantiation {    \
            __FUNC_CUDA(fun);                       \
            __FUNC_TBB(fun);                        \
    }

// method instantiation
#define METHOD_INSTANTIATE(base, method) \
        namespace __ ## base ## __ ## method ## __instantiation {   \
            __METHOD_CUDA(base, method);                            \
            __METHOD_TBB(base, method);                             \
    }
