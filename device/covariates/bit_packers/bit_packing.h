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

// defines the base types for implementing bit-packing chains that track covariate values

#include "../../../types.h"
#include "../../from_nvbio/dna.h"
#include "../../from_nvbio/alphabet.h"

namespace firepony {

struct covariate_key_set
{
    covariate_key M;
    covariate_key I;
    covariate_key D;
};

// generates a bit mask with the lowest N_bits set
#define BITMASK(N_bits) ((1 << (N_bits)) - 1)

// constexpr versions of min and max for uint32
constexpr uint32 constexpr_max(uint32 a, uint32 b)
{
    return (a > b ? a : b);
}

constexpr uint32 constexpr_min(uint32 a, uint32 b)
{
    return (a < b ? a : b);
}

// base covariate class
// COVARIATE_IS_SPARSE determines whether this covariate can skip generating keys for certain events
template<target_system system, typename PrevCovariate, uint32 BITS, bool COVARIATE_IS_SPARSE = false>
struct covariate
{
    static constexpr bool covariate_is_sparse = COVARIATE_IS_SPARSE;

    // the previous covariate in the chain
    // (first covariate will have this set to struct covariate_null)
    typedef PrevCovariate PreviousCovariate;

    enum {
        // how many bits we reserved for this covariate
        num_bits = BITS,
        // our index in the chain
        index = PreviousCovariate::index + 1,
        // our own offset
        offset = PreviousCovariate::next_offset,
        // our bit mask
        mask = BITMASK(num_bits) << PreviousCovariate::next_offset,
        // the total number of bits used up until (and including) this node in the chain
        bits_used = PreviousCovariate::next_offset + num_bits,
        // the bit offset for the next covariate in the chain
        next_offset = bits_used,
        // our maximum integral value (also used as an invalid token)
        max_value = (1 << num_bits) - 1,
        // sentinel value used when a given covariate key is skipped
        invalid_key_pattern = max_value,
    };

    static_assert(offset + num_bits <= sizeof(covariate_key) * 8, "covariate set too large");

protected:
    static CUDA_HOST_DEVICE covariate_key_set build_key(covariate_key_set input_key, covariate_key_set data,
                                                        typename firepony_context<system>::view ctx,
                                                        const typename alignment_batch_device<system>::const_view batch,
                                                        uint32 read_index, uint16 bp_offset, uint32 cigar_event_index)
    {
        // add in our bits
        input_key.M = input_key.M | (data.M << offset);
        input_key.I = input_key.I | (data.I << offset);
        input_key.D = input_key.D | (data.D << offset);

        // pass along to next in chain
        return PreviousCovariate::encode(ctx, batch, read_index, bp_offset, cigar_event_index, input_key);
    }

public:
    static CUDA_HOST_DEVICE uint32 decode(covariate_key input, uint32 target_index)
    {
        if (target_index == index)
        {
            covariate_key out = (input >> offset) & BITMASK(num_bits);
            return out;
        } else {
            return PreviousCovariate::decode(input, target_index);
        }
    }

    static constexpr CUDA_HOST_DEVICE bool is_sparse(const uint32 target_index)
    {
        return (target_index == index ? covariate_is_sparse : PreviousCovariate::is_sparse(target_index));
    }

    static constexpr CUDA_HOST_DEVICE uint32 invalid_key(const uint32 target_index)
    {
        return (target_index == index ? invalid_key_pattern : PreviousCovariate::invalid_key(target_index));
    }

    static constexpr CUDA_HOST_DEVICE uint32 key_mask(const uint32 target_index)
    {
        return (target_index == index ? mask : PreviousCovariate::key_mask(target_index));
    }
};

// chain terminator
template <target_system system>
struct covariate_null
{
    typedef covariate_null PreviousCovariate;

    enum {
        index = 0,
        mask = 0,
        next_offset = 0
    };

    static CUDA_HOST_DEVICE covariate_key_set encode(typename firepony_context<system>::view ctx,
                                                     const typename alignment_batch_device<system>::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        return input_key;
    }

    static CUDA_HOST_DEVICE uint32 decode(covariate_key input, uint32 target_index)
    {
        return 0;
    }

    static constexpr CUDA_HOST_DEVICE bool is_sparse(const uint32 target_index)
    {
        return false;
    }

    static constexpr CUDA_HOST_DEVICE uint32 invalid_key(const uint32 target_index)
    {
        return 0;
    }

    static constexpr CUDA_HOST_DEVICE uint32 key_mask(const uint32 target_index)
    {
        return 0;
    }
};

} // namespace firepony
