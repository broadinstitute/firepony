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

#include "bqsr_types.h"
#include "bqsr_context.h"
#include "covariates.h"
#include "covariates_table.h"
#include "cigar.h"

#include "primitives/cuda.h"

#define BITMASK(bits) ((1 << bits) - 1)

struct covariate_key_set
{
    covariate_key M;
    covariate_key I;
    covariate_key D;
};

typedef enum {
    EVENT_M = 0,
    EVENT_I = 1,
    EVENT_D = 2
} CovariateEventIndex;

// base covariate class
// COVARIATE_IS_SPARSE determines whether this covariate can skip generating keys for certain events
template<typename PrevCovariate, uint32 bits, bool COVARIATE_IS_SPARSE = false>
struct covariate
{
    static constexpr bool covariate_is_sparse = COVARIATE_IS_SPARSE;

    // the previous covariate in the chain
    // (first covariate will have this set to struct covariate_null)
    typedef PrevCovariate PreviousCovariate;

    enum {
        // our index in the chain
        index = PreviousCovariate::index + 1,
        // our own offset
        offset = PreviousCovariate::next_offset,
        // our bit mask
        mask = BITMASK(bits) << PreviousCovariate::next_offset,
        // the offset for the next covariate in the chain
        next_offset = PreviousCovariate::next_offset + bits,
        // our maximum integral value (also used as an invalid token)
        max_value = (1 << bits) - 1,
        // sentinel value used when a given covariate key is skipped
        invalid_key_pattern = max_value,
    };

    static_assert(offset + bits <= sizeof(covariate_key) * 8, "covariate set too large");

protected:
    static CUDA_HOST_DEVICE covariate_key_set build_key(covariate_key_set input_key, covariate_key_set data,
                                                        bqsr_context::view ctx,
                                                        const alignment_batch_device::const_view batch,
                                                        uint32 read_index, uint16 bp_offset, uint32 cigar_event_index)
    {
        // add in our bits
        input_key.M = input_key.M | (((data.M) & BITMASK(bits)) << offset);
        input_key.I = input_key.I | (((data.I) & BITMASK(bits)) << offset);
        input_key.D = input_key.D | (((data.D) & BITMASK(bits)) << offset);

        // pass along to next in chain
        return PreviousCovariate::encode(ctx, batch, read_index, bp_offset, cigar_event_index, input_key);
    }

public:
    static CUDA_HOST_DEVICE uint32 decode(covariate_key input, uint32 target_index)
    {
        if (target_index == index)
        {
            covariate_key out = (input >> offset) & BITMASK(bits);
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

    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
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
};

template<typename PreviousCovariate = covariate_null>
struct covariate_ReadGroup : public covariate<PreviousCovariate, 8>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        uint32 read_group = batch.read_group[read_index];
        return covariate<PreviousCovariate, 8>::build_key(input_key,
                                                          { read_group, read_group, read_group },
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};

// this is a little weird, since it doesn't actually change
template <typename PreviousCovariate = covariate_null>
struct covariate_EventTracker : public covariate<PreviousCovariate, 2>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        return covariate<PreviousCovariate, 2>::build_key(input_key,
                                                          { cigar_event::M, cigar_event::I, cigar_event::D },
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};

template <typename PreviousCovariate = covariate_null>
struct covariate_QualityScore : public covariate<PreviousCovariate, 8>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        const CRQ_index idx = batch.crq_index(read_index);
        // xxxnsubtil: 45 is the "default" base quality for when insertion/deletion qualities are not available in the read
        // we should eventually grab these from the alignment data itself if they're present
        covariate_key_set key = { batch.qualities[idx.qual_start + bp_offset],
                                  45,
                                  45 };

        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};

template <typename PreviousCovariate = covariate_null>
struct covariate_Cycle_Illumina : public covariate<PreviousCovariate, 10, true>
{
    typedef covariate<PreviousCovariate, 10, true> base;

    enum {
        CUSHION_FOR_INDELS = 4
    };

    static CUDA_HOST_DEVICE covariate_key keyFromCycle(const int cycle)
    {
        // no negative values because values must fit into the first few bits of the long
        covariate_key result = abs(cycle);
//        assert(result <= (base::max_value >> 1));

        // xxxnsubtil: investigate if we can do sign propagation here instead
        result = result << 1; // shift so we can add the "sign" bit
        if (cycle < 0)
            result++;

        return result;
    }

    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const bool paired = batch.flags[read_index] & AlignmentFlags::PAIRED;
        const bool second_of_pair = batch.flags[read_index] & AlignmentFlags::READ2;
        const bool negative_strand = batch.flags[read_index] & AlignmentFlags::REVERSE;

        const auto& window = ctx.cigar.read_window_clipped[read_index];
        const int readLength = window.y - window.x + 1;
        const int readOrderFactor = (paired && second_of_pair) ? -1 : 1;
        const int i = bp_offset - window.x;

        int cycle;
        int increment;

        if (negative_strand)
        {
            cycle = readLength * readOrderFactor;
            increment = -1 * readOrderFactor;
        } else {
            cycle = readOrderFactor;
            increment = readOrderFactor;
        }

        cycle += i * increment;

        const int MAX_CYCLE_FOR_INDELS = readLength - CUSHION_FOR_INDELS - 1;

        const covariate_key substitutionKey = keyFromCycle(cycle);
        const covariate_key indelKey = (i < CUSHION_FOR_INDELS || i > MAX_CYCLE_FOR_INDELS) ? base::invalid_key_pattern : substitutionKey;

//        if (indelKey == base::invalid_key_pattern)
//        {
//            printf("indelKey == -1 -> i == %d MAX_CYCLE_FOR_INDELS = %d CUSHION_FOR_INDELS = %d readLength = %d\n", i, MAX_CYCLE_FOR_INDELS, CUSHION_FOR_INDELS, readLength);
//        }

//        printf("bp %d: window={%d,%d} paired=%d second_of_pair=%d negative_strand=%d readLength=%d readOrderFactor=%d increment=%d i=%d cycle=%d M=%d indel=%d\n",
//                bp_offset, window.x, window.y, paired, second_of_pair, negative_strand, readLength, readOrderFactor, increment, i, cycle, substitutionKey, indelKey);

//        printf("bp %d qual %d keys { %d %d %d }\n", bp_offset - window.x, batch.qualities[idx.qual_start + bp_offset],
//                (substitutionKey == base::invalid_key_pattern ? -1 : substitutionKey),
//                (indelKey == base::invalid_key_pattern ? -1 : indelKey),
//                (indelKey == base::invalid_key_pattern ? -1 : indelKey));

        return base::build_key(input_key, { substitutionKey, indelKey, indelKey },
                               ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};
