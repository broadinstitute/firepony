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
#include "from_nvbio/dna.h"
#include "from_nvbio/alphabet.h"

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
        // the offset for the next covariate in the chain
        next_offset = PreviousCovariate::next_offset + num_bits,
        // our maximum integral value (also used as an invalid token)
        max_value = (1 << num_bits) - 1,
        // sentinel value used when a given covariate key is skipped
        invalid_key_pattern = max_value,
    };

    static_assert(offset + num_bits <= sizeof(covariate_key) * 8, "covariate set too large");

protected:
    static CUDA_HOST_DEVICE covariate_key_set build_key(covariate_key_set input_key, covariate_key_set data,
                                                        bqsr_context::view ctx,
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

        return base::build_key(input_key, { substitutionKey, indelKey, indelKey },
                               ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};

constexpr uint32 constexpr_max(uint32 a, uint32 b)
{
    return (a > b ? a : b);
}

constexpr uint32 constexpr_min(uint32 a, uint32 b)
{
    return (a < b ? a : b);
}

// context covariate
// xxxnsubtil: need to implement low qual tail clipping
template <uint32 NUM_BASES_MISMATCH, uint32 NUM_BASES_INDEL, typename PreviousCovariate = covariate_null>
struct covariate_Context : public covariate<PreviousCovariate, 4 + constexpr_max(NUM_BASES_MISMATCH, NUM_BASES_INDEL) * 2 + 1, true>
{
    // bit calculation for the context: 4 length bits, 2 bits per base pair, 1 sentinel bit for invalid values
    typedef covariate<PreviousCovariate, 4 + constexpr_max(NUM_BASES_MISMATCH, NUM_BASES_INDEL) * 2 + 1, true> base;

    typedef enum {
        Mismatch,
        Indel
    } ContextType;

    // compile time variables
    enum {
        // input parameters: length of mismatch and indel contexts
        num_bases_mismatch = NUM_BASES_MISMATCH,
        num_bases_indel = NUM_BASES_INDEL,

        // the range of bases we need to track
        max_context_bases = constexpr_max(num_bases_mismatch, num_bases_indel),
        min_context_bases = constexpr_min(num_bases_mismatch, num_bases_indel),

        // context sizes in bits
        base_bits_context = max_context_bases * 2,
        base_bits_mismatch = num_bases_mismatch * 2,
        base_bits_indel = num_bases_indel * 2,

        length_bits = 4,
    };

    static_assert(max_context_bases <= BITMASK(length_bits), "not enough length bits to represent context size");

    static CUDA_DEVICE covariate_key reverse_dna4(covariate_key input)
    {
        constexpr uint32 shift_bits = (sizeof(covariate_key) * 8) - base_bits_context;
        constexpr covariate_key pattern_hi = covariate_key((0xAAAAAAAAAAAAAAAAULL) & BITMASK(base_bits_context));
        constexpr covariate_key pattern_lo = covariate_key((0x5555555555555555ULL) & BITMASK(base_bits_context));

        const auto rev = (sizeof(input) == 4 ? __brev(input) : __brevll(input)) >> shift_bits;
        return ((rev & pattern_hi) >> 1) | ((rev & pattern_lo) << 1);
    }

    static CUDA_DEVICE bool is_non_regular_base(uint8 b)
    {
        return b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::A &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::C &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::T &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::G;
    }

    // context encoding (MSB to LSB): BB[bp_offset] BB[bp_offset-1] BB[bp_offset-2] ... SSSS
    // B = base pair bit, S = size bit
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        const auto idx = batch.crq_index(read_index);
        const auto& window = ctx.cigar.read_window_clipped[read_index];

        covariate_key context = 0;
        covariate_key context_mismatch = base::invalid_key_pattern;
        covariate_key context_indel = base::invalid_key_pattern;

        // do we have enough bases for the smallest context?
        if (bp_offset >= window.x + (min_context_bases - 1))
        {
            const bool negative_strand = batch.flags[read_index] & AlignmentFlags::REVERSE;

            // how many context bases do we have before the current base?
            const int context_bases = min(bp_offset - window.x, max_context_bases - 1);
            // gather base pairs over the context region
            for(int i = bp_offset - context_bases; i <= bp_offset; i++)
            {
                const uint8 bp = batch.reads[idx.read_start + i];

                // note: we push these in reverse order for a forward-encoded read
                // we'll reverse the context below
                context <<= 2;
                context |= from_nvbio::iupac16_to_dna(bp);
            }


            if (!negative_strand)
            {
                // this is counter intuitive, but we pushed in reverse order above
                // therefore, reverse the context if this is *not* the negative strand
                context = reverse_dna4(context);
            } else {
                // since we pushed in reverse order, there's no need to reverse the read data, just compute the complement
                context = ~context & BITMASK(base_bits_context);
            }

            if (context_bases >= num_bases_mismatch - 1)
            {
                // remove any extra bases that are not required
                context_mismatch = context >> (max_context_bases - num_bases_mismatch) * 2;
                // add in the size
                context_mismatch = (context_mismatch << length_bits) | num_bases_mismatch;
            }

            if (context_bases >= num_bases_indel - 1)
            {
                // remove any extra bases that are not required
                context_indel = context >> (max_context_bases - num_bases_indel) * 2;
                // add in the size
                context_indel = (context_indel << length_bits) | num_bases_indel;
            }
        }

        return base::build_key(input_key, { context_mismatch, context_indel, context_indel },
                               ctx, batch, read_index, bp_offset, cigar_event_index);

    }
};
