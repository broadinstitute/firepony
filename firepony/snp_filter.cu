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

#include "bqsr_types.h"
#include "bqsr_context.h"
#include "snp_filter.h"

#include "primitives/algorithms.h"
#include "primitives/cuda.h"
#include "primitives/parallel.h"

// functor used to compute the read offset list
// for each read, fills in a list of uint16 values with the offset of each BP in the reference relative to the start of the alignment
struct compute_read_offset_list : public bqsr_lambda
{
    compute_read_offset_list(bqsr_context::view ctx,
                             const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const cigar_op *cigar = &batch.cigars[idx.cigar_start];
        uint16 *offset_list = &ctx.read_offset_list[idx.read_start];

        // create a list of offsets from the base alignment position for each BP in the read
        uint16 offset = 0;
        for(uint32 c = 0; c < idx.cigar_len; c++)
        {
            switch(cigar[c].op)
            {
            case cigar_op::OP_M:
            case cigar_op::OP_MATCH:
            case cigar_op::OP_X:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *offset_list = offset;
                    offset_list++;
                    offset++;
                }

                break;

            case cigar_op::OP_I:
            case cigar_op::OP_N:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *offset_list = offset;
                    offset_list++;
                }

                break;

            case cigar_op::OP_D:
            case cigar_op::OP_H:
            case cigar_op::OP_P:
                offset += cigar[c].len;
                break;

            case cigar_op::OP_S:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *offset_list = uint16(-1);
                    offset_list++;
                }

                break;
            }
        }
    }
};

// for each read, compute the offset of each read BP relative to the base alignment position of the read
void build_read_offset_list(bqsr_context *context,
                            const alignment_batch& batch)
{
    context->read_offset_list.resize(batch.device.reads.size());
    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     compute_read_offset_list(*context, batch.device));
}

// functor used to compute the alignment window list
// for each read, compute the end of the alignment window in the reference and the sequence
// xxxnsubtil: this is very similar to cigar_coordinates_expand, should merge
struct compute_alignment_window : public bqsr_lambda
{
    compute_alignment_window(bqsr_context::view ctx,
                             const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const uint16 *offset_list = &ctx.read_offset_list[idx.read_start];
        uint2& output = ctx.alignment_windows[read_index];
        ushort2& output_sequence = ctx.sequence_alignment_windows[read_index];

        // scan the offset list looking for the largest offset
        int c;
        for(c = (int) idx.read_len - 1; c >= 0; c--)
        {
            if (offset_list[c] != uint16(-1))
                    break;
        }

        if (c < 0)
        {
            // no alignment, emit an invalid window
            output = make_uint2(uint32(-1), uint32(-1));
        } else {
            // transform the start position
            output.x = ctx.reference.sequence_bp_start[batch.chromosome[read_index]] + batch.alignment_start[read_index];
            output.y = output.x + offset_list[c];

            output_sequence.x = batch.alignment_start[read_index];
            output_sequence.y = output_sequence.x + offset_list[c];
        }
    }
};

// for each read, compute the end of the alignment window in the reference
void build_alignment_windows(bqsr_context *ctx, const alignment_batch& batch)
{
    // set up the alignment window buffer
    ctx->alignment_windows.resize(batch.device.num_reads);
    ctx->sequence_alignment_windows.resize(batch.device.num_reads);
    // compute alignment windows
    thrust::for_each(ctx->active_read_list.begin(),
                     ctx->active_read_list.end(),
                     compute_alignment_window(*ctx, batch.device));
}

struct compute_vcf_ranges : public bqsr_lambda
{
    compute_vcf_ranges(bqsr_context::view ctx,
                       const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const auto& db = ctx.variant_db;

        // figure out the genome alignment window for this read
        const ushort2& reference_window_clipped = ctx.cigar.reference_window_clipped[read_index];

        const uint32 ref_sequence_id = batch.chromosome[read_index];
        const uint32 ref_sequence_base = ctx.reference.sequence_bp_start[ref_sequence_id];
        const uint32 ref_sequence_offset = ref_sequence_base + batch.alignment_start[read_index];

        const uint2 alignment_window = make_uint2(ref_sequence_offset + uint32(reference_window_clipped.x),
                                                  ref_sequence_offset + uint32(reference_window_clipped.y));

        // do a binary search for a feature that starts inside our read
        const uint32 *vcf_start = bqsr::lower_bound(alignment_window.x,
                                                    db.reference_window_start.begin(),
                                                    db.reference_window_start.size());

        // compute the initial vcf range
        uint2 vcf_range = make_uint2(vcf_start - db.reference_window_start.begin(),
                                     vcf_start - db.reference_window_start.begin());

        vcf_range.x = vcf_start - db.reference_window_start.begin();
        vcf_range.y = vcf_range.x;

        // do a linear search to find the end of the VCF range
        // (there are generally very few VCF entries for an average read length --- and often none --- so this is expected to be faster than a binary search)
        while(vcf_range.y < db.reference_window_start.size() - 1 &&
                db.reference_window_start[vcf_range.y + 1] <= alignment_window.y)
        {
            vcf_range.y++;
        }

        // expand the start of the range backwards to find the first feature that overlaps with our alignment window
        while(vcf_range.x != 0 &&
              db.reference_window_start[vcf_range.x - 1] + db.alignment_window_len[vcf_range.x - 1] >= alignment_window.x)
        {
            vcf_range.x--;
        }

        // check for overlap at the edges of the range
        if (db.reference_window_start[vcf_range.x] < alignment_window.x &&
                db.reference_window_start[vcf_range.y] + db.alignment_window_len[vcf_range.y] > alignment_window.y)
        {
            // emit an empty VCF range
            ctx.snp_filter.active_vcf_ranges[read_index] = make_uint2(uint32(-1), uint32(-1));
        } else {
            ctx.snp_filter.active_vcf_ranges[read_index] = vcf_range;
        }
    }
};

struct vcf_active_predicate
{
    D_VectorU32_2::view vcf_active;

    vcf_active_predicate(D_VectorU32_2::view vcf_active)
        : vcf_active(vcf_active)
    { }

    CUDA_HOST_DEVICE bool operator() (const uint32 vcf_id)
    {
        return vcf_active[vcf_id].x != uint32(-1);
    }
};

struct filter_bps : public bqsr_lambda
{
    filter_bps(bqsr_context::view ctx,
               const alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

public:
    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const auto& db = ctx.variant_db;

        const CRQ_index& idx = batch.crq_index(read_index);
        const ushort2& read_window_clipped = ctx.cigar.read_window_clipped[read_index];

        // figure out the genome alignment window for this read
        const uint32 ref_sequence_id = batch.chromosome[read_index];
        const uint32 ref_sequence_base = ctx.reference.sequence_bp_start[ref_sequence_id];
        const uint32 ref_sequence_offset = ref_sequence_base + batch.alignment_start[read_index];

        uint2 vcf_db_range = ctx.snp_filter.active_vcf_ranges[read_index];

        // traverse the VCF range and mark corresponding read BPs as inactive
        for(uint32 feature = vcf_db_range.x; feature <= vcf_db_range.y; feature++)
        {
            const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
            const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

            // compute the feature range as offset from alignment start
            const int feature_start = db.reference_window_start[feature] - ref_sequence_offset;
            const int feature_end = db.reference_window_start[feature] + db.alignment_window_len[feature] - ref_sequence_offset;

            // convert start and end to read coordinates
            uint32 read_start = 0xffffffff, read_end = 0xffffffff;
            uint32 ev;

            for(ev = cigar_start; ev < cigar_end; ev++)
            {
                const uint16 ref_coord = ctx.cigar.cigar_event_reference_coordinates[ev];

                if (ref_coord != uint16(-1) && int(ref_coord) >= feature_start)
                {
                    // if the feature starts inside a deletion, move ev forward until we find a read base
                    // (note that we preserve ev since the next loop relies on it being accurate --- moving ev back could land us in an insertion!)
                    uint32 ev_read = ev;
                    while(ev_read < cigar_end && ctx.cigar.cigar_event_read_coordinates[ev_read] == uint16(-1))
                    {
                        ev_read++;
                    }

                    read_start = ctx.cigar.cigar_event_read_coordinates[ev_read];
                    read_end = read_start;

                    break;
                }
            }

            for( ; ev < cigar_end; ev++)
            {
                const uint16 ref_coord = ctx.cigar.cigar_event_reference_coordinates[ev];

                if (int(ref_coord) <= feature_end)
                {
                    if (ctx.cigar.cigar_event_read_coordinates[ev] != uint16(-1))
                    {
                        read_end = ctx.cigar.cigar_event_read_coordinates[ev];
                    }
                } else {
                    break;
                }
            }

            if ((read_start < read_window_clipped.x && read_end < read_window_clipped.x) ||
                (read_start > read_window_clipped.y && read_end > read_window_clipped.y))
            {
                continue;
            }

            // truncate any portions of the feature range that fall outside the clipped read region
            read_start = max(read_start, read_window_clipped.x);
            read_start = min(read_start, read_window_clipped.y);
            read_end = max(read_end, read_window_clipped.x);
            read_end = min(read_end, read_window_clipped.y);

            for(uint32 dead_bp = read_start; dead_bp <= read_end; dead_bp++)
                ctx.active_location_list[idx.read_start + dead_bp] = 0;
        }
    }
};

// filter out known SNPs from the active BP list
// for each BP in batch, set the corresponding bit in active_loc_list to zero if it matches a known SNP
void filter_known_snps(bqsr_context *context, const alignment_batch& batch)
{
    snp_filter_context& snp = context->snp_filter;

    // compute the VCF ranges for each read
    snp.active_vcf_ranges.resize(batch.device.num_reads);
    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     compute_vcf_ranges(*context, batch.device));

    // build a list of reads with active VCF ranges
    snp.active_read_ids.resize(context->active_read_list.size());
    context->temp_u32 = context->active_read_list;

    uint32 num_active;
    num_active = bqsr::copy_if(context->temp_u32.begin(),
                               context->temp_u32.size(),
                               snp.active_read_ids.begin(),
                               vcf_active_predicate(snp.active_vcf_ranges),
                               context->temp_storage);

    snp.active_read_ids.resize(num_active);

    // finally apply the VCF filter
    // this will create zeros in the active location list for each BP that matches a known variant
    thrust::for_each(snp.active_read_ids.begin(),
                     snp.active_read_ids.end(),
                     filter_bps(*context, batch.device));
}
