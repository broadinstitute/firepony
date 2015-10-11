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

#include <lift/types.h>
#include <lift/decorators.h>
#include <lift/memory.h>

namespace firepony {
using namespace lift;
} // namespace firepony

//#include "device/primitives/backends.h"
//#include "device/primitives/cuda.h"
//#include "device/primitives/vector.h"
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

// shorthand for DNA packed vector/stream types
template <target_system system> using vector_dna16 = packed_vector<system, 4>;
template <target_system system> using stream_dna16 = typename vector_dna16<system>::const_stream_type;

} // namespace firepony

