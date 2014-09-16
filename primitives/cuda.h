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

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
    #define CUDA_HOST   __host__
    #define CUDA_DEVICE __device__
#else
    #define CUDA_HOST_DEVICE
    #define CUDA_HOST
    #define CUDA_DEVICE
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#define CUDA_DEVICE_COMPILATION
#endif
