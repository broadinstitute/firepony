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

#include "../types.h"

namespace firepony {

//#define RUN_ON_CPU
#ifdef RUN_ON_CPU
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
#error must build with -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
#endif
typedef host_tag target_system_tag;
#else
typedef firepony::device_tag target_system_tag;
#endif

template <typename T> using D_Vector = firepony::vector<target_system_tag, T>;
template <uint32 bits> using D_PackedVector = firepony::packed_vector<target_system_tag, bits>;

typedef D_Vector<uint8> D_VectorU8;
typedef D_Vector<uint16> D_VectorU16;
typedef D_Vector<uint32> D_VectorU32;
typedef D_Vector<uint64> D_VectorU64;

typedef D_Vector<int32> D_VectorI32;

typedef D_Vector<double> D_VectorF64;
typedef D_Vector<float> D_VectorF32;

typedef D_PackedVector<4> D_VectorDNA16;
typedef D_VectorDNA16::const_stream_type D_StreamDNA16;

typedef D_PackedVector<1> D_PackedVector_1b;

typedef D_PackedVector<2> D_PackedVector_2b;

typedef D_Vector<uint2> D_VectorU32_2;
typedef D_Vector<ushort2> D_VectorU16_2;

// we only use 1 bit per entry on the active location list
// however, because we write to this using multiple threads that are scanning reads concurrently,
// we need to make sure we don't access the same dword from different threads
// sizing this to 4 bits per symbol ensures that, because input reads are padded to a dword boundary
typedef D_VectorDNA16 D_ActiveLocationList;
typedef D_ActiveLocationList::view D_ActiveLocationStream;

typedef D_Vector<cigar_op> D_VectorCigarOp;

} // namespace firepony
