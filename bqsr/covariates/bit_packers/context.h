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
        const auto& window = ctx.covariates.high_quality_window[read_index];

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
