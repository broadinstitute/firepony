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

// defines the base types for implementing bit-packing chains that track covariate values

#include "../../types.h"
#include "../../primitives/cuda.h"
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
template<typename PrevCovariate, uint32 BITS, bool COVARIATE_IS_SPARSE = false>
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
                                                        context::view ctx,
                                                        const alignment_batch_device::const_view batch,
                                                        uint32 read_index, uint16 bp_offset, uint32 cigar_event_index)
    {
        // add in our bits
        // xxxnsubtil: the bitwise AND is in theory not necessary here, investigate removing it
        input_key.M = input_key.M | (((data.M) & BITMASK(num_bits)) << offset);
        input_key.I = input_key.I | (((data.I) & BITMASK(num_bits)) << offset);
        input_key.D = input_key.D | (((data.D) & BITMASK(num_bits)) << offset);

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
struct covariate_null
{
    typedef covariate_null PreviousCovariate;

    enum {
        index = 0,
        mask = 0,
        next_offset = 0
    };

    static CUDA_HOST_DEVICE covariate_key_set encode(context::view ctx,
                                                     const alignment_batch_device::const_view batch,
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
