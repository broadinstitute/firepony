/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

namespace firepony {

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

} // namespace firepony
