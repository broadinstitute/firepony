/*
 * Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
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

// help the eclipse error parser deal with CUDA keywords
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __host__
#define __constant__
#define __const__
#define __restrict__
#define ENABLE_CPU 1
#define ENABLE_GPU 1
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

#include "device/primitives/cuda.h"
#include "device/primitives/vector.h"
#include "device/primitives/packed_stream.h"
#include "device/primitives/packed_vector.h"

namespace firepony {

struct firepony_context;

template <typename T> using H_Vector = firepony::vector<host_tag, T>;
template <uint32 bits> using H_PackedVector = firepony::packed_vector<host_tag, bits>;

typedef H_Vector<uint8> H_VectorU8;
typedef H_Vector<uint16> H_VectorU16;
typedef H_Vector<uint32> H_VectorU32;
typedef H_Vector<uint64> H_VectorU64;

typedef H_Vector<int32> H_VectorI32;

typedef H_Vector<double> H_VectorF64;
typedef H_Vector<float> H_VectorF32;

typedef H_PackedVector<4> H_VectorDNA16;
typedef H_VectorDNA16::const_stream_type H_StreamDNA16;

typedef H_PackedVector<1> H_PackedVector_1b;
typedef H_PackedVector<2> H_PackedVector_2b;

typedef H_Vector<uint2> H_VectorU32_2;
typedef H_Vector<ushort2> H_VectorU16_2;

// we only use 1 bit per entry on the active location list
// however, because we write to this using multiple threads that are scanning reads concurrently,
// we need to make sure we don't access the same dword from different threads
// sizing this to 4 bits per symbol ensures that, because input reads are padded to a dword boundary
typedef H_VectorDNA16 H_ActiveLocationList;
typedef H_ActiveLocationList::view H_ActiveLocationStream;

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

typedef H_Vector<cigar_op> H_VectorCigarOp;

} // namespace firepony
