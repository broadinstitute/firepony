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

namespace bqsr
{

/// a meta-function to convert a type to const
///
template <typename T> struct to_const           { typedef T type; };
template <typename T> struct to_const<T&>       { typedef const T& type; };
template <typename T> struct to_const<T*>       { typedef const T* type; };
template <typename T> struct to_const<const T&> { typedef const T& type; };
template <typename T> struct to_const<const T*> { typedef const T* type; };

/// a meta-function to convert potentially unsigned integrals to their signed counter-part
///
template <typename T> struct signed_type {};
template <> struct signed_type<uint32> { typedef int32 type; };
template <> struct signed_type<uint64> { typedef int64 type; };
template <> struct signed_type<int32>  { typedef int32 type; };
template <> struct signed_type<int64>  { typedef int64 type; };

/// a meta-function to convert potentially signed integrals to their unsigned counter-part
///
template <typename T> struct unsigned_type {};
template <> struct unsigned_type<uint32> { typedef uint32 type; };
template <> struct unsigned_type<uint64> { typedef uint64 type; };
template <> struct unsigned_type<int32>  { typedef uint32 type; };
template <> struct unsigned_type<int64>  { typedef uint64 type; };

CUDA_HOST_DEVICE constexpr bool is_pow2(uint32 C)
{
    return (C & (C-1)) == 0u;
}

/// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
template <typename L, typename R>
inline CUDA_HOST_DEVICE L divide_ri(const L x, const R y)
{
    return L((x + (y - 1)) / y);
}

template <typename T>
inline CUDA_HOST_DEVICE T min(const T a, const T b)
{
    return (a < b ? a : b);
}

template <typename T>
inline CUDA_HOST_DEVICE T max(const T a, const T b)
{
    return (a > b ? a : b);
}

} // namespace bqsr
