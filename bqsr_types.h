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

#include <nvbio/basic/types.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/packed_vector.h>

using namespace nvbio;

struct bqsr_context;

typedef nvbio::vector<device_tag, uint8> D_VectorU8;
typedef nvbio::vector<host_tag, uint8> H_VectorU8;
typedef nvbio::vector<device_tag, uint16> D_VectorU16;
typedef nvbio::vector<host_tag, uint16> H_VectorU16;
typedef nvbio::vector<device_tag, uint32> D_VectorU32;
typedef nvbio::vector<host_tag, uint32> H_VectorU32;
typedef nvbio::vector<device_tag, uint64> D_VectorU64;
typedef nvbio::vector<host_tag, uint64> H_VectorU64;

typedef nvbio::vector<device_tag, int32> D_VectorI32;
typedef nvbio::vector<host_tag, int32> H_VectorI32;

//template <typename system_tag> using Vector_DNA16 = nvbio::PackedVector<system_tag, 4>;
typedef nvbio::PackedVector<host_tag, 4> H_VectorDNA16;
typedef nvbio::PackedVector<device_tag, 4> D_VectorDNA16;
typedef H_VectorDNA16::stream_type H_StreamDNA16;
typedef D_VectorDNA16::stream_type D_StreamDNA16;

typedef nvbio::vector<device_tag, uint2> D_VectorU32_2;
typedef nvbio::vector<host_tag, uint2> H_VectorU32_2;
typedef nvbio::vector<device_tag, ushort2> D_VectorU16_2;
typedef nvbio::vector<host_tag, ushort2> H_VectorU16_2;

// we only use 1 bit per entry on the active location list
// however, because we write to this using multiple threads that are scanning reads concurrently,
// we need to make sure we don't access the same dword from different threads
// sizing this to 4 bits per symbol ensures that, because input reads are padded to a dword boundary
typedef D_VectorDNA16 D_ActiveLocationList;
typedef H_VectorDNA16 H_ActiveLocationList;
typedef D_ActiveLocationList::plain_view_type D_ActiveLocationStream;
typedef H_ActiveLocationList::plain_view_type H_ActiveLocationStream;

typedef D_VectorU16 D_ReadOffsetList;
typedef H_VectorU16 H_ReadOffsetList;
