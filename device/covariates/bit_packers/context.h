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

#pragma once

#include "bit_packing.h"

namespace firepony {

// context covariate
template <target_system system, uint32 NUM_BASES_MISMATCH, uint32 NUM_BASES_INDEL, typename PreviousCovariate = covariate_null<system> >
struct covariate_Context : public covariate<system, PreviousCovariate, 4 + constexpr_max(NUM_BASES_MISMATCH, NUM_BASES_INDEL) * 2 + 1, true>
{
    // bit calculation for the context: 4 length bits, 2 bits per base pair, 1 sentinel bit for invalid values
    typedef covariate<system, PreviousCovariate, 4 + constexpr_max(NUM_BASES_MISMATCH, NUM_BASES_INDEL) * 2 + 1, true> base;

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

    static CUDA_HOST_DEVICE bool is_non_regular_base(uint8 b)
    {
        return b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::A &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::C &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::T &&
               b != from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::G;
    }

#if !CUDA_DEVICE_COMPILATION
    static uint32 __brev(uint32 x)
    {
        x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
        x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
        return ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1);
    }

    static uint64 __brevll(uint64 x)
    {
        x = ((x & 0xF0F0F0F0F0F0F0F0LLU) >> 4) | ((x & 0x0F0F0F0F0F0F0F0FLLU) << 4);
        x = ((x & 0xCCCCCCCCCCCCCCCCLLU) >> 2) | ((x & 0x3333333333333333LLU) << 2);
        return ((x & 0xAAAAAAAAAAAAAAAALLU) >> 1) | ((x & 0x5555555555555555LLU) << 1);
    }
#endif

    // reverse a 2-bit encoded DNA sequence of (at most) base_bits_context length
    static CUDA_HOST_DEVICE covariate_key reverse_dna4(covariate_key input)
    {
        constexpr uint32 shift_bits = (sizeof(covariate_key) * 8) - base_bits_context;
        constexpr covariate_key pattern_hi = covariate_key((0xAAAAAAAAAAAAAAAAULL) & BITMASK(base_bits_context));
        constexpr covariate_key pattern_lo = covariate_key((0x5555555555555555ULL) & BITMASK(base_bits_context));

        const auto rev = (sizeof(input) == 4 ? __brev(input) : __brevll(input)) >> shift_bits;
        return ((rev & pattern_hi) >> 1) | ((rev & pattern_lo) << 1);
    }

    // context encoding (MSB to LSB): BB[bp_offset] BB[bp_offset-1] BB[bp_offset-2] ... SSSS
    // B = base pair bit, S = size bit
    static CUDA_HOST_DEVICE covariate_key_set encode(firepony_context<system>& ctx,
                                                     const alignment_batch_device<system>& batch,
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
        const int context_bases = negative_strand ? min<int>(window.y - bp_offset + 1, max_context_bases) : min<int>(bp_offset - window.x + 1, max_context_bases);
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

} // namespace firepony
