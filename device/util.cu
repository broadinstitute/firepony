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

#include <lift/backends.h>
#include <lift/parallel.h>

#include "../types.h"

#include "primitives/util.h"

#include "util.h"

namespace firepony {

// packs a uint8 into an N-bit-per-symbol packed vector
template <target_system system, uint32 N>
struct pack_uint8
{
    typename packed_vector<system, N>::view dest;
    pointer<system, uint8> src;

    pack_uint8(typename packed_vector<system, N>::view dest,
               pointer<system, uint8> src)
        : dest(dest), src(src)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 word_index)
    {
        const uint8 *input = &src[word_index * packed_vector<system, N>::SYMBOLS_PER_WORD];
        for(uint32 i = 0; i < packed_vector<system, N>::SYMBOLS_PER_WORD; i++)
        {
            dest[word_index * packed_vector<system, N>::SYMBOLS_PER_WORD + i] = input[i];
        }
    }
};

// prepare temp storage to store num_elements that will be packed into a bit vector
template <target_system system, typename packed_vector_dest>
static void pack_prepare_storage(allocation<system, uint8>& src, uint32 num_elements)
{
    src.resize(divide_ri(num_elements, packed_vector_dest::SYMBOLS_PER_WORD) * packed_vector_dest::SYMBOLS_PER_WORD);
}

// prepare temp_storage to store num_elements to be packed into a 1-bit vector
template <target_system system>
void pack_prepare_storage_2bit(allocation<system, uint8>& storage, uint32 num_elements)
{
    pack_prepare_storage<system, packed_vector<system, 2> >(storage, num_elements);
}
INSTANTIATE(pack_prepare_storage_2bit);

// prepare temp_storage to store num_elements to be packed into a 2-bit vector
template <target_system system>
void pack_prepare_storage_1bit(allocation<system, uint8>& storage, uint32 num_elements)
{
    pack_prepare_storage<system, packed_vector<system, 1> >(storage, num_elements);
}
INSTANTIATE(pack_prepare_storage_1bit);

template <target_system system>
void pack_to_2bit(packed_vector<system, 2>& dest, allocation<system, uint8>& src)
{
    dest.resize(src.size());
    parallel<system>::for_each(thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(0) + divide_ri(src.size(), packed_vector<system, 2>::SYMBOLS_PER_WORD),
                               pack_uint8<system, 2>(dest, src));
}
INSTANTIATE(pack_to_2bit);

template <target_system system>
void pack_to_1bit(packed_vector<system, 1>& dest, allocation<system, uint8>& src)
{
    dest.resize(src.size());
    parallel<system>::for_each(thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(0) + divide_ri(src.size(), packed_vector<system, 1>::SYMBOLS_PER_WORD),
                               pack_uint8<system, 1>(dest, src));
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
