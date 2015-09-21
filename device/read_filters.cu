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

#include <lift/parallel.h>

#include "from_nvbio/alphabet.h"

#include "firepony_context.h"
#include "alignment_data_device.h"
#include "read_filters.h"
#include "util.h"

namespace firepony {

// filter if any of the flags are set
template <target_system system, uint32 flags>
struct filter_if_any_set : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE bool operator() (const uint32 read_index)
    {
        if ((batch.flags[read_index] & flags) != 0)
        {
            return false;
        } else {
            return true;
        }
    }
};

// implements the GATK filters MappingQualityUnavailable and MappingQualityZero
template <target_system system>
struct filter_mapq : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE bool operator() (const uint32 read_index)
    {
        if (batch.mapq[read_index] == 0 ||
            batch.mapq[read_index] == 255)
        {
            return false;
        } else {
            return true;
        }
    }
};

// partially implements the GATK MalformedReadFilter
template <target_system system>
struct filter_if_read_malformed : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE bool operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        // read is not flagged as unmapped...
        if (!(batch.flags[read_index] & AlignmentFlags::UNMAP))
        {
            // ... but reference sequence ID is invalid (GATK: checkInvalidAlignmentStart)
            if (batch.chromosome[read_index] == uint32(-1))
            {
                return false;
            }

            // ... but alignment start is -1 (GATK: checkInvalidAlignmentStart)
            if (batch.alignment_start[read_index] == uint32(-1))
            {
                return false;
            }

            // ... but alignment aligns to negative number of bases in the reference (GATK: checkInvalidAlignmentEnd)
            if (ctx.alignment_windows[read_index].y <= ctx.alignment_windows[read_index].x)
            {
                return false;
            }

            // ... but read is aligned to a point after the end of the contig (GATK: checkAlignmentDisagreesWithHeader)
            auto& sequence = ctx.reference_db.get_chromosome(batch.chromosome[read_index]);
            if (ctx.alignment_windows[read_index].x >= sequence.bases.size())
            {
                return false;
            }

            // ... and it has a valid alignment start (tested before), but the CIGAR string is empty (GATK: checkCigarDisagreesWithAlignment)
            // xxxnsubtil: need to verify that this is exactly what GATK does
            if (idx.cigar_len == 0)
            {
                return false;
            }
        }

        // read is aligned to nonexistent contig but alignment start is valid
        // (GATK: checkAlignmentDisagreesWithHeader)
        if (batch.chromosome[read_index] == uint32(-1) && batch.alignment_start[read_index] != uint32(-1))
        {
            return false;
        }

        // read has no read group
        // (GATK: checkHasReadGroup)
        if (batch.read_group[read_index] == uint32(-1))
        {
            return false;
        }

        // read has different number of bases and base qualities
        // (GATK: checkMismatchBasesAndQuals)
        // xxxnsubtil: note that this is meaningless for BAM, but it's here anyway in case we end up parsing SAM files
        if (idx.qual_len != idx.read_len)
        {
            return false;
        }

        // read has no base sequence stored in the file
        // (GATK: checkSeqStored)
        if (idx.read_len == 0)
        {
            return false;
        }

        return true;
    }
};

// implements another part of the GATK MalformedReadFilter
template <target_system system>
struct filter_if_cigar_malformed : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE bool operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        // state for CIGAR sanity checks
        enum {
            LEADING_HARD_CLIPS,
            LEADING_SOFT_CLIPS,
            NO_CLIPS,
            TRAILING_SOFT_CLIPS,
            TRAILING_HARD_CLIPS,
        } clip_state = LEADING_HARD_CLIPS;

        for(uint32 i = idx.cigar_start; i < idx.cigar_start + idx.cigar_len; i++)
        {
            const struct cigar_op& ce = batch.cigars[i];

            // CIGAR contains N operators
            // (GATK: checkCigarIsSupported)
            if (ce.op == cigar_op::OP_N)
            {
                return false;
            }

            // CIGAR contains improperly placed soft or hard clipping operators
            // (this is not part of GATK but we rely on it for cigar expansion)
            if (ce.op == cigar_op::OP_H)
            {
                switch(clip_state)
                {
                case LEADING_HARD_CLIPS:
                    // no state change
                    break;

                case LEADING_SOFT_CLIPS:
                    // H in leading soft clip region is invalid
                    return false;

                case NO_CLIPS:
                case TRAILING_SOFT_CLIPS:
                    // first H after a non-clip operator enters the hard clipping region
                    clip_state = TRAILING_HARD_CLIPS;
                    break;

                case TRAILING_HARD_CLIPS:
                    // no state change
                    break;
                }
            } else if (ce.op == cigar_op::OP_S) {
                switch(clip_state)
                {
                case LEADING_HARD_CLIPS:
                    // first S enters the leading soft clipping region
                    clip_state = LEADING_SOFT_CLIPS;
                    break;

                case LEADING_SOFT_CLIPS:
                    // no change
                    break;

                case NO_CLIPS:
                    // first S after a non-clip operator enters the trailing soft clipping region
                    clip_state = TRAILING_SOFT_CLIPS;
                    break;

                case TRAILING_SOFT_CLIPS:
                    // no change
                    break;

                case TRAILING_HARD_CLIPS:
                    // we've entered the hard clip region, S not expected
                    return false;
                }
            } else {
                switch(clip_state)
                {
                case LEADING_HARD_CLIPS:
                case LEADING_SOFT_CLIPS:
                    // first non-clip operator enters the no-clipping region
                    clip_state = NO_CLIPS;
                    break;

                case NO_CLIPS:
                    // no change
                    break;

                case TRAILING_SOFT_CLIPS:
                case TRAILING_HARD_CLIPS:
                    // non-clip operator in a trailing clipping region is invalid
                    return false;
                }
            }
        }

        return true;
    }
};

// filter invalid reads
// (this runs prior to any other pieces of the context being populated)
template <target_system system>
void filter_invalid_reads(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    auto& active_read_list = context.active_read_list;
    auto& temp_u32 = context.temp_u32;
    uint32 num_active;
    uint32 start_count;

    // corresponds to the GATK filters MappingQualityUnavailable and MappingQualityZero
    filter_mapq<system> mapq_filter(context, batch.device);

    // this filter corresponds to the following GATK filters:
    // - DuplicateReadFilter
    // - FailsVendorQualityCheckFilter
    // - NotPrimaryAlignmentFilter
    // - UnmappedReadFilter
    filter_if_any_set<system,
                      AlignmentFlags::DUPLICATE |
                      AlignmentFlags::QC_FAIL |
                      AlignmentFlags::UNMAP |
                      AlignmentFlags::SECONDARY> flags_filter(context, batch.device);

    // implements part of the GATK filter MalformedReadFilter
    filter_if_cigar_malformed<system> malformed_cigar_filter(context, batch.device);

    start_count = active_read_list.size();
    num_active = active_read_list.size();

    // make sure the temp buffer is big enough
    context.temp_u32.resize(active_read_list.size());

    // set up a ping-pong queue between active_read_list and temp_u32
    auto pingpong = make_pingpong_queue(active_read_list, temp_u32);

    // apply the mapq filter
    num_active = parallel<system>::copy_if(pingpong.source().begin(),
                                           num_active,
                                           pingpong.dest().begin(),
                                           mapq_filter,
                                           context.temp_storage);
    pingpong.swap();

    if (num_active)
    {
        // apply the flags filters
        num_active = parallel<system>::copy_if(pingpong.source().begin(),
                                               num_active,
                                               pingpong.dest().begin(),
                                               flags_filter,
                                               context.temp_storage);
        pingpong.swap();
    }

    if (num_active)
    {
        // apply the malformed cigar filters
        num_active = parallel<system>::copy_if(pingpong.source().begin(),
                                               num_active,
                                               pingpong.dest().begin(),
                                               malformed_cigar_filter,
                                               context.temp_storage);
        pingpong.swap();
    }

    pingpong.source().resize(num_active);

    // copy back into active_read_list if needed
    if (pingpong.is_swapped())
    {
        active_read_list = pingpong.source();
    }

    // track how many reads we filtered
    context.stats.filtered_reads += start_count - num_active;
}
INSTANTIATE(filter_invalid_reads);

// filter invalid reads
// (this runs prior to any other pieces of the context being populated)
template <target_system system>
void filter_malformed_reads(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    auto& active_read_list = context.active_read_list;
    auto& temp_u32 = context.temp_u32;
    uint32 num_active;
    uint32 start_count;

    // implements part of the GATK filter MalformedReadFilter
    filter_if_read_malformed<system> malformed_read_filter(context, batch.device);

    start_count = active_read_list.size();
    num_active = active_read_list.size();

    // make sure the temp buffer is big enough
    context.temp_u32.resize(active_read_list.size());

    // apply the malformed read filter
    // this copies from active_read_list into temp_u32
    num_active = parallel<system>::copy_if(active_read_list.begin(),
                                           num_active,
                                           temp_u32.begin(),
                                           malformed_read_filter,
                                           context.temp_storage);

    // resize and copy back to active_read_list
    temp_u32.resize(num_active);
    active_read_list = temp_u32;

    // track how many reads we filtered
    context.stats.filtered_reads += start_count - num_active;
}
INSTANTIATE(filter_malformed_reads);

// filter non-regular bases (anything other than A, C, G, T)
template <target_system system>
struct filter_non_regular_bases : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
        {
            uint8 bp = batch.reads[i];
            if (bp != from_nvbio::AlphabetTraits<from_nvbio::Alphabet::DNA_IUPAC>::A &&
                bp != from_nvbio::AlphabetTraits<from_nvbio::Alphabet::DNA_IUPAC>::C &&
                bp != from_nvbio::AlphabetTraits<from_nvbio::Alphabet::DNA_IUPAC>::G &&
                bp != from_nvbio::AlphabetTraits<from_nvbio::Alphabet::DNA_IUPAC>::T)
            {
                ctx.active_location_list[i] = 0;
            }
        }
    }
};

/**
 * The lowest quality score for a base that is considered reasonable for statistical analysis.  This is
 * because Q 6 => you stand a 25% of being right, which means all bases are equally likely
 */
#define MIN_USABLE_Q_SCORE 6

// filter bases with quality < MIN_USABLE_Q_SCORE
template <target_system system>
struct filter_low_quality_bases : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        for(uint32 i = 0; i < idx.read_len; i++)
        {
            uint8 qual = batch.qualities[idx.qual_start + i];
            if (qual < MIN_USABLE_Q_SCORE)
            {
                ctx.active_location_list[idx.read_start + i] = 0;
            }
        }
    }
};

// apply per-BP filters to the batch
// (known SNPs are filtered elsewhere)
template <target_system system>
void filter_bases(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     filter_non_regular_bases<system>(context, batch.device));

    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     filter_low_quality_bases<system>(context, batch.device));
}
INSTANTIATE(filter_bases);

} // namespace firepony

