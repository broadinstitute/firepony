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

#define DISABLE_OUTPUT_ROUNDING 0

// help the eclipse error parser deal with CUDA keywords
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __host__
#define __constant__
#define __const__
#define __restrict__
#define ENABLE_CUDA_BACKEND 1
#define ENABLE_TBB_BACKEND 1
#endif

#include <stdint.h>

namespace firepony {

typedef uint64_t uint64;
typedef int64_t int64;
typedef uint32_t uint32;
typedef int32_t int32;
typedef uint16_t uint16;
typedef int16_t int16;
typedef uint8_t uint8;
typedef int8_t int8;

} // namespace firepony

#include "device/primitives/backends.h"
#include "device/primitives/cuda.h"
#include "device/primitives/vector.h"
#include "device/primitives/packed_stream.h"
#include "device/primitives/packed_vector.h"

namespace firepony {

struct cigar_op
{
    uint32 len:24, op:4;

    enum
    {
        OP_M     = 0,
        OP_I     = 1,
        OP_D     = 2,
        OP_N     = 3,
        OP_S     = 4,
        OP_H     = 5,
        OP_P     = 6,
        OP_MATCH = 7,
        OP_X     = 8,
    };

    CUDA_HOST_DEVICE char ascii_op(void) const
    {
        return op == 0 ? 'M' :
               op == 1 ? 'I' :
               op == 2 ? 'D' :
               op == 3 ? 'N' :
               op == 4 ? 'S' :
               op == 5 ? 'H' :
               op == 6 ? 'P' :
               op == 7 ? '=' :
                         'X';
    }
};

template <target_system system> struct firepony_context;

// shorthand to denote vectors that are meant to be used on host vs device code
// (this does not represent a functional change, merely indicates what memory space a given vector is meant to live in)
template <target_system system, typename T> using d_vector = vector<system, T>;
template <typename T> using h_vector = vector<host, T>;

template <target_system system> using d_vector_u8 = d_vector<system, uint8>;
                                using h_vector_u8 = vector<host, uint8>;
template <target_system system> using d_vector_u16 = vector<system, uint16>;
                                using h_vector_u16 = vector<host, uint16>;
template <target_system system> using d_vector_u32 = vector<system, uint32>;
                                using h_vector_u32 = vector<host, uint32>;
template <target_system system> using d_vector_u64 = vector<system, uint64>;
                                using h_vector_u64 = vector<host, uint64>;

template <target_system system> using d_vector_i16 = vector<system, int16>;
                                using h_vector_i16 = vector<host, int16>;
template <target_system system> using d_vector_i32 = vector<system, int32>;
                                using h_vector_i32 = vector<host, int32>;
template <target_system system> using d_vector_i64 = vector<system, int64>;
                                using h_vector_i64 = vector<host, int64>;

template <target_system system> using d_vector_f32 = vector<system, float>;
                                using h_vector_f32 = vector<host, float>;
template <target_system system> using d_vector_f64 = vector<system, double>;
                                using h_vector_f64 = vector<host, double>;

template <target_system system> using d_packed_vector_1b = packed_vector<system, 1>;
                                using h_packed_vector_1b = packed_vector<host, 1>;
template <target_system system> using d_packed_vector_2b = packed_vector<system, 2>;
                                using h_packed_vector_2b = packed_vector<host, 2>;
template <target_system system> using d_packed_vector_4b = packed_vector<system, 4>;
                                using h_packed_vector_4b = packed_vector<host, 4>;

template <target_system system> using d_vector_dna16 = d_packed_vector_4b<system>;
template <target_system system> using d_stream_dna16 = typename d_vector_dna16<system>::const_stream_type;
                                using h_vector_dna16 = h_packed_vector_4b;
                                using h_stream_dna16 = typename h_vector_dna16::const_stream_type;

template <target_system system> using d_vector_u32_2 = vector<system, uint2>;
                                using h_vector_u32_2 = vector<host, uint2>;
template <target_system system> using d_vector_u16_2 = vector<system, ushort2>;
                                using h_vector_u16_2 = vector<host, ushort2>;

// we only use 1 bit per entry on the active location list
// however, because we write to this using multiple threads that are scanning reads concurrently,
// we need to make sure we don't access the same dword from different threads
// sizing this to 4 bits per symbol ensures that, because input reads are padded to a dword boundary
template <target_system system> using d_vector_active_location_list = d_vector_dna16<system>;
template <target_system system> using d_stream_active_location_list = typename d_vector_dna16<system>::view;
                                using h_vector_active_location_list = h_vector_dna16;
                                using h_stream_active_location_list = typename h_vector_dna16::view;

template <target_system system> using d_vector_cigar_op = vector<system, cigar_op>;
                                using h_vector_cigar_op = vector<host, cigar_op>;

} // namespace firepony

