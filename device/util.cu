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

#include "../types.h"

#include "primitives/backends.h"
#include "primitives/cuda.h"
#include "primitives/parallel.h"
#include "primitives/util.h"

#include "util.h"

namespace firepony {

// the following two structures are the result of a couple of hours trying to do this with templates...
template <target_system system>
struct pack_uint8_to_2bit_vector
{
    typename d_packed_vector_2b<system>::view dest;
    typename d_vector_u8<system>::view src;

    pack_uint8_to_2bit_vector(typename d_packed_vector_2b<system>::view dest,
                              typename d_vector_u8<system>::view src)
        : dest(dest), src(src)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 word_index)
    {
        const uint8 *input = &src[word_index * d_packed_vector_2b<system>::SYMBOLS_PER_WORD];
        for(uint32 i = 0; i < d_packed_vector_2b<system>::SYMBOLS_PER_WORD; i++)
        {
            dest[word_index * d_packed_vector_2b<system>::SYMBOLS_PER_WORD + i] = input[i];
        }
    }
};

template <target_system system>
struct pack_uint8_to_1bit_vector
{
    typename d_packed_vector_1b<system>::view dest;
    typename d_vector_u8<system>::view src;

    pack_uint8_to_1bit_vector(typename d_packed_vector_1b<system>::view dest,
                              typename d_vector_u8<system>::view src)
        : dest(dest), src(src)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 word_index)
    {
        const uint8 *input = &src[word_index * d_packed_vector_1b<system>::SYMBOLS_PER_WORD];
        for(uint32 i = 0; i < d_packed_vector_1b<system>::SYMBOLS_PER_WORD; i++)
        {
            dest[word_index * d_packed_vector_1b<system>::SYMBOLS_PER_WORD + i] = input[i];
        }
    }
};

// prepare temp storage to store num_elements that will be packed into a bit vector
template <target_system system, typename packed_vector_dest>
static void pack_prepare_storage(d_vector_u8<system>& src, uint32 num_elements)
{
    src.resize(divide_ri(num_elements, packed_vector_dest::SYMBOLS_PER_WORD) * packed_vector_dest::SYMBOLS_PER_WORD);
}

// prepare temp_storage to store num_elements to be packed into a 1-bit vector
template <target_system system>
void pack_prepare_storage_2bit(d_vector_u8<system>& storage, uint32 num_elements)
{
    pack_prepare_storage<system, d_packed_vector_2b<system> >(storage, num_elements);
}
INSTANTIATE(pack_prepare_storage_2bit);

// prepare temp_storage to store num_elements to be packed into a 2-bit vector
template <target_system system>
void pack_prepare_storage_1bit(d_vector_u8<system>& storage, uint32 num_elements)
{
    pack_prepare_storage<system, d_packed_vector_1b<system> >(storage, num_elements);
}
INSTANTIATE(pack_prepare_storage_1bit);

template <target_system system>
void pack_to_2bit(d_packed_vector_2b<system>& dest, d_vector_u8<system>& src)
{
    dest.resize(src.size());
    parallel<system>::for_each(thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(0) + divide_ri(src.size(), d_packed_vector_2b<system>::SYMBOLS_PER_WORD),
                               pack_uint8_to_2bit_vector<system>(dest, src));
}
INSTANTIATE(pack_to_2bit);

template <target_system system>
void pack_to_1bit(d_packed_vector_1b<system>& dest, d_vector_u8<system>& src)
{
    dest.resize(src.size());
    parallel<system>::for_each(thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(0) + divide_ri(src.size(), d_packed_vector_1b<system>::SYMBOLS_PER_WORD),
                               pack_uint8_to_1bit_vector<system>(dest, src));
}
INSTANTIATE(pack_to_1bit);

// round a double to the Nth decimal place
// this is meant to workaround broken printf() rounding in glibc
double round_n(double val, int n)
{
    // xxxnsubtil: i suspect this might cause loss of precision if the initial exponent is large
    val = val * pow(10.0, n);
    val = round(val);
    val = val / pow(10.0, n);
    return val;
}

} // namespace firepony
