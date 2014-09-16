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

//#include <nvbio/basic/types.h>
//#include <nvbio/basic/vector.h>

#include "bqsr_types.h"

using namespace nvbio;

uint32 bqsr_string_hash(const char* s);

// prepare temp_storage to store num_elements to be packed into a bit vector
void pack_prepare_storage_2bit(D_VectorU8& storage, uint32 num_elements);
void pack_prepare_storage_1bit(D_VectorU8& storage, uint32 num_elements);

// packs a vector of uint8 into a bit vector
void pack_to_2bit(D_PackedVector_2b& dest, D_VectorU8& src);
void pack_to_1bit(D_PackedVector_1b& dest, D_VectorU8& src);
