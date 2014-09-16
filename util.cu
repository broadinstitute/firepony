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

#include "bqsr_types.h"
#include "util.h"

uint32 bqsr_string_hash(const char* s)
{
    uint32 h = 0;
    while (*s)
        h = h * 101 + (uint32) *s++;
    return h;
}

// the following two structures are the result of a couple of hours trying to do this with templates...
struct pack_uint8_to_2bit_vector
{
    D_PackedVector_2b::plain_view_type dest;
    D_VectorU8::plain_view_type src;

    pack_uint8_to_2bit_vector(D_PackedVector_2b::plain_view_type dest,
                              D_VectorU8::plain_view_type src)
        : dest(dest), src(src)
    { }

    NVBIO_HOST_DEVICE void operator() (const uint32 word_index)
    {
        const uint8 *input = &src[word_index * D_PackedVector_2b::SYMBOLS_PER_WORD];
        for(uint32 i = 0; i < D_PackedVector_2b::SYMBOLS_PER_WORD; i++)
        {
            dest[word_index * D_PackedVector_2b::SYMBOLS_PER_WORD + i] = input[i];
        }
    }
};

struct pack_uint8_to_1bit_vector
{
    D_PackedVector_1b::plain_view_type dest;
    D_VectorU8::plain_view_type src;

    pack_uint8_to_1bit_vector(D_PackedVector_1b::plain_view_type dest,
                              D_VectorU8::plain_view_type src)
        : dest(dest), src(src)
    { }

    NVBIO_HOST_DEVICE void operator() (const uint32 word_index)
    {
        const uint8 *input = &src[word_index * D_PackedVector_1b::SYMBOLS_PER_WORD];
        for(uint32 i = 0; i < D_PackedVector_1b::SYMBOLS_PER_WORD; i++)
        {
            dest[word_index * D_PackedVector_1b::SYMBOLS_PER_WORD + i] = input[i];
        }
    }
};

// prepare temp storage to store num_elements that will be packed into a bit vector
template<typename D_PackedVector_Dest>
static void pack_prepare_storage(D_VectorU8& src, uint32 num_elements)
{
    src.resize(nvbio::util::divide_ri(num_elements, D_PackedVector_Dest::SYMBOLS_PER_WORD) * D_PackedVector_Dest::SYMBOLS_PER_WORD);
}

// prepare temp_storage to store num_elements to be packed into a 1-bit vector
void pack_prepare_storage_2bit(D_VectorU8& storage, uint32 num_elements)
{
    pack_prepare_storage<D_PackedVector_2b>(storage, num_elements);
}

// prepare temp_storage to store num_elements to be packed into a 2-bit vector
void pack_prepare_storage_1bit(D_VectorU8& storage, uint32 num_elements)
{
    pack_prepare_storage<D_PackedVector_1b>(storage, num_elements);
}

void pack_to_2bit(D_PackedVector_2b& dest, D_VectorU8& src)
{
    dest.resize(src.size());
    thrust::for_each(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(0) + nvbio::util::divide_ri(src.size(), D_PackedVector_2b::SYMBOLS_PER_WORD),
                     pack_uint8_to_2bit_vector(nvbio::plain_view(dest), nvbio::plain_view(src)));
}

void pack_to_1bit(D_PackedVector_1b& dest, D_VectorU8& src)
{
    dest.resize(src.size());
    thrust::for_each(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(0) + nvbio::util::divide_ri(src.size(), D_PackedVector_1b::SYMBOLS_PER_WORD),
                     pack_uint8_to_1bit_vector(nvbio::plain_view(dest), nvbio::plain_view(src)));
}
