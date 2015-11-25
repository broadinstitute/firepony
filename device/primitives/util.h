/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <lift/sys/compute_device.h>
#include <lift/sys/cuda/compute_device_cuda.h>

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

template <target_system sys_dest, target_system sys_source, typename T>
inline void cross_device_copy(const lift::compute_device& dest_device, allocation<sys_dest, T>& dest, size_t dest_offset,
                              const lift::compute_device& source_device, allocation<sys_source, T>& source, size_t source_offset,
                              size_t len)
{
#if ENABLE_CUDA_BACKEND
    if (sys_dest == cuda && sys_source == cuda)
    {
        const auto& dst = (const lift::compute_device_cuda&)dest_device;
        const auto& src = (const lift::compute_device_cuda&)source_device;

        T *ptr_src;
        T *ptr_dest;

        ptr_src = source.data() + source_offset;
        ptr_dest = dest.data() + dest_offset;

        cudaMemcpyPeer(ptr_dest, dst.config.device, ptr_src, src.config.device, len * sizeof(T));
    } else
#endif
        thrust::copy_n(source.t_begin() + source_offset, len, dest.t_begin() + dest_offset);
}

} // namespace firepony
