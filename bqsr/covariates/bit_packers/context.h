/* Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
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

#include "bit_packing.h"

// context covariate
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

    static CUDA_DEVICE bool is_non_regular_base(uint8 b)
    {
        return b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::A &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::C &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::T &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::G;
    }

    // reverse a 2-bit encoded DNA sequence of (at most) base_bits_context length
    static CUDA_DEVICE covariate_key reverse_dna4(covariate_key input)
    {
        constexpr uint32 shift_bits = (sizeof(covariate_key) * 8) - base_bits_context;
        constexpr covariate_key pattern_hi = covariate_key((0xAAAAAAAAAAAAAAAAULL) & BITMASK(base_bits_context));
        constexpr covariate_key pattern_lo = covariate_key((0x5555555555555555ULL) & BITMASK(base_bits_context));

        const auto rev = (sizeof(input) == 4 ? __brev(input) : __brevll(input)) >> shift_bits;
        return ((rev & pattern_hi) >> 1) | ((rev & pattern_lo) << 1);
    }

    // context encoding (MSB to LSB): BB[bp_offset] BB[bp_offset-1] BB[bp_offset-2] ... SSSS
    // B = base pair bit, S = size bit
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        const auto idx = batch.crq_index(read_index);
        const auto& window = ctx.covariates.high_quality_window[read_index];
        const bool negative_strand = batch.flags[read_index] & AlignmentFlags::REVERSE;

        covariate_key context = 0;
        covariate_key context_mismatch = base::invalid_key_pattern;
        covariate_key context_indel = base::invalid_key_pattern;

        // which direction do we need to move in the read?
        const int direction = negative_strand ? 1 : -1;
        // how many context bases do we have available?
        const int context_bases = negative_strand ? min(window.y - bp_offset + 1, max_context_bases) : min(bp_offset - window.x + 1, max_context_bases);
        // where is our starting point?
        const int start_offset = bp_offset;
        // where is our stopping point?
        const int stop_offset = bp_offset + context_bases * direction;

        // do we have enough bases for the smallest context?
        if (context_bases >= min_context_bases)
        {
            int num_bases = 0;

            // gather base pairs over the context region, starting from the context base
            for(int i = start_offset; i != stop_offset; i += direction)
            {
                const uint8 bp = batch.reads[idx.read_start + i];

                if (is_non_regular_base(bp))
                {
                    break;
                }

                context <<= 2;
                context |= from_nvbio::iupac16_to_dna(bp);
                num_bases++;
            }

            if (negative_strand)
            {
                // we're on the negative strand, complement the context bits
                context = ~context & BITMASK(num_bases * 2);
            }

            if (num_bases >= num_bases_mismatch)
            {
                // remove any extra bases that are not required
                context_mismatch = context >> (num_bases - num_bases_mismatch) * 2;
                // add in the size
                context_mismatch = (context_mismatch << length_bits) | num_bases_mismatch;
            }

            if (num_bases >= num_bases_indel)
            {
                // remove any extra bases that are not required
                context_indel = context >> (num_bases - num_bases_indel) * 2;
                // add in the size
                context_indel = (context_indel << length_bits) | num_bases_indel;
            }
        }

        return base::build_key(input_key, { context_mismatch, context_indel, context_indel },
                               ctx, batch, read_index, bp_offset, cigar_event_index);

    }
};
