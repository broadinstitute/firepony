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

#include <nvbio/strings/alphabet.h>

#include "primitives/parallel.h"

#include "bqsr_types.h"
#include "bqsr_context.h"
#include "alignment_data.h"
#include "filters.h"

// filter if any of the flags are set
template<uint32 flags>
struct filter_if_any_set : public bqsr_lambda
{
    filter_if_any_set(bqsr_context::view ctx,
                      const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

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
struct filter_mapq : public bqsr_lambda
{
    filter_mapq(bqsr_context::view ctx,
                const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

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
struct filter_malformed_reads : public bqsr_lambda
{
    filter_malformed_reads(bqsr_context::view ctx,
                           const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

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
            if (ctx.sequence_alignment_windows[read_index].y >= ctx.bam_header.chromosome_lengths[batch.chromosome[read_index]])
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
struct filter_malformed_cigars : public bqsr_lambda
{
    filter_malformed_cigars(bqsr_context::view ctx,
                            const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

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

// apply read filters to the batch
void filter_reads(bqsr_context *context, const alignment_batch& batch)
{
    D_VectorU32& active_read_list = context->active_read_list;
    D_VectorU32& temp_u32 = context->temp_u32;
    uint32 num_active;
    uint32 start_count;

    // this filter corresponds to the following GATK filters:
    // - DuplicateReadFilter
    // - FailsVendorQualityCheckFilter
    // - NotPrimaryAlignmentFilter
    // - UnmappedReadFilter
    filter_if_any_set<AlignmentFlags::DUPLICATE |
                      AlignmentFlags::QC_FAIL |
                      AlignmentFlags::UNMAP |
                      AlignmentFlags::SECONDARY> flags_filter(*context, batch.device);

    // corresponds to the GATK filters MappingQualityUnavailable and MappingQualityZero
    filter_mapq mapq_filter(*context, batch.device);
    // corresponds to the GATK filter MalformedReadFilter
    filter_malformed_reads malformed_read_filter(*context, batch.device);
    filter_malformed_cigars malformed_cigar_filter(*context, batch.device);

    start_count = active_read_list.size();

    // make sure the temp buffer is big enough
    context->temp_u32.resize(active_read_list.size());

    // apply the flags filter, copying from active_read_list into temp_u32
    num_active = bqsr::copy_if(active_read_list.begin(),
                               active_read_list.size(),
                               temp_u32.begin(),
                               flags_filter);

    // apply the mapq filters, copying from temp_u32 into active_read_list
    num_active = bqsr::copy_if(temp_u32.begin(),
                               temp_u32.size(),
                               active_read_list.begin(),
                               mapq_filter);

    // apply the malformed read filters, copying from active_read_list into temp_u32
    num_active = bqsr::copy_if(active_read_list.begin(),
                               num_active,
                               temp_u32.begin(),
                               malformed_read_filter);

    // apply the malformed cigar filters, copying from temp_u32 into active_read_list
    num_active = bqsr::copy_if(temp_u32.begin(),
                               num_active,
                               active_read_list.begin(),
                               malformed_cigar_filter);

    // resize active_read_list
    active_read_list.resize(num_active);

    context->stats.filtered_reads += start_count - num_active;
}

// filter non-regular bases (anything other than A, C, G, T)
struct filter_non_regular_bases : public bqsr_lambda
{
    filter_non_regular_bases(bqsr_context::view ctx,
                             const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
        {
            uint8 bp = batch.reads[i];
            if (bp != nvbio::AlphabetTraits<nvbio::Alphabet::DNA_IUPAC>::A &&
                bp != nvbio::AlphabetTraits<nvbio::Alphabet::DNA_IUPAC>::C &&
                bp != nvbio::AlphabetTraits<nvbio::Alphabet::DNA_IUPAC>::G &&
                bp != nvbio::AlphabetTraits<nvbio::Alphabet::DNA_IUPAC>::T)
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
struct filter_low_quality_bases : public bqsr_lambda
{
    filter_low_quality_bases(bqsr_context::view ctx,
                             const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
        {
            uint8 qual = batch.qualities[i];
            if (qual < MIN_USABLE_Q_SCORE)
            {
                ctx.active_location_list[i] = 0;
            }
        }
    }
};

// apply per-BP filters to the batch
// (known SNPs are filtered elsewhere)
void filter_bases(bqsr_context *context, const alignment_batch& batch)
{
    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     filter_non_regular_bases(*context, batch.device));

    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     filter_low_quality_bases(*context, batch.device));
}
