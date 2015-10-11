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

#include <lift/parallel.h>

#include "device_types.h"
#include "firepony_context.h"
#include "snp_filter.h"
#include "cigar.h"

#include "primitives/algorithms.h"

namespace firepony {

// functor used to compute the read offset list
// for each read, fills in a list of uint16 values with the offset of each BP in the reference relative to the start of the alignment
template <target_system system>
struct compute_read_offset_list : public lambda<system>
{
    LAMBDA_INHERIT;

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
template <target_system system>
void build_read_offset_list(firepony_context<system>& context,
                            const alignment_batch<system>& batch)
{
    context.read_offset_list.resize(batch.device.reads.size());
    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_read_offset_list<system>(context, batch.device));
}
INSTANTIATE(build_read_offset_list);

// functor used to compute the alignment window list
// for each read, compute the end of the alignment window in the reference and the sequence
// xxxnsubtil: this is very similar to cigar_coordinates_expand, should merge
template <target_system system>
struct compute_alignment_window : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const uint16 *offset_list = &ctx.read_offset_list[idx.read_start];
        uint2& output = ctx.alignment_windows[read_index];

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
            output.x = batch.alignment_start[read_index];
            output.y = output.x + offset_list[c];
        }
    }
};

// for each read, compute the end of the alignment window in the reference
template <target_system system>
void build_alignment_windows(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    // set up the alignment window buffer
    context.alignment_windows.resize(batch.device.num_reads);
    // compute alignment windows
    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_alignment_window<system>(context, batch.device));
}
INSTANTIATE(build_alignment_windows);

template <target_system system>
struct compute_vcf_ranges : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        // check if we know the chromosome for this read
        auto ch = batch.chromosome[read_index];
        if (ch >= ctx.variant_db.data.size())
        {
            // dbsnp does not reference this chromosome
            // mark read as inactive for VCF search and exit
            ctx.snp_filter.active_vcf_ranges[read_index] = make_uint2(uint32(-1), uint32(-1));
            return;
        }

        const auto& db = ctx.variant_db.get_chromosome(ch);

        // figure out the genome alignment window for this read
        const ushort2& reference_window_clipped = ctx.cigar.reference_window_clipped[read_index];

        const uint32 ref_sequence_offset = batch.alignment_start[read_index];
        const uint2 alignment_window = make_uint2(ref_sequence_offset + uint32(reference_window_clipped.x),
                                                  ref_sequence_offset + uint32(reference_window_clipped.y));

        uint2 vcf_range;

        // do a binary search along the max end point array to find the first overlap
        vcf_range.x = lower_bound(alignment_window.x,
                                  db.max_end_point_left.begin(),
                                  db.max_end_point_left.size()) - db.max_end_point_left.begin();

        if (vcf_range.x >= db.feature_start.size())
        {
            // overlap not found, mark as inactive
            ctx.snp_filter.active_vcf_ranges[read_index] = make_uint2(uint32(-1), uint32(-1));
            return;
        }

        // now search along the min end point array to find the last overlap
        vcf_range.y = upper_bound(alignment_window.y,
                                  db.feature_start.begin(),
                                  db.feature_start.size()) - db.feature_start.begin();

        if (vcf_range.y <= vcf_range.x)
        {
            // overlap not found, mark range as inactive
            ctx.snp_filter.active_vcf_ranges[read_index] = make_uint2(uint32(-1), uint32(-1));
            return;
        }

        if (vcf_range.y >= db.feature_start.size())
        {
            // end point not found, clip to the end of the database
            vcf_range.y = db.feature_start.size() - 1;
        }

        ctx.snp_filter.active_vcf_ranges[read_index] = vcf_range;
    }
};

template <target_system system>
struct vcf_active_predicate
{
    pointer<system, uint2> vcf_active;

    vcf_active_predicate(pointer<system, uint2> vcf_active)
        : vcf_active(vcf_active)
    { }

    CUDA_HOST_DEVICE bool operator() (const uint32 vcf_id)
    {
        return vcf_active[vcf_id].x != uint32(-1);
    }
};

template <target_system system>
struct filter_bps : public lambda<system>
{
    LAMBDA_INHERIT;

public:
    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const auto& db = ctx.variant_db.get_chromosome(batch.chromosome[read_index]);

        const CRQ_index& idx = batch.crq_index(read_index);
        const ushort2& read_window_clipped = ctx.cigar.read_window_clipped[read_index];
        const ushort2& reference_window_clipped = ctx.cigar.reference_window_clipped[read_index];

        const auto alignment_start = batch.alignment_start[read_index];

        uint2 vcf_db_range = ctx.snp_filter.active_vcf_ranges[read_index];

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        // traverse the VCF range and mark corresponding read BPs as inactive
        for(uint32 feature = vcf_db_range.x; feature <= vcf_db_range.y; feature++)
        {
            // compute the feature range as offset from alignment start
            const int feature_start = db.feature_start[feature] - alignment_start;
            const int feature_end = db.feature_stop[feature] - alignment_start;

            // locate a starting point for matching the feature along the read: search for the first read bp with a known reference coordinate inside our feature
            uint32 ev;
            uint16 ref_coord = uint16(-1);

            for(ev = cigar_start; ev < cigar_end; ev++)
            {
                ref_coord = ctx.cigar.cigar_event_reference_coordinates[ev];

                if (ref_coord != uint16(-1) && ref_coord >= feature_start)
                {
                    break;
                }
            }

            if (ev == cigar_end)
            {
                // no match found
                continue;
            }

            // the rest of this function is best left alone

            // how many base pairs exist in the feature to the left of our starting point?
            int feature_bp_left = ref_coord - feature_start;
            uint32 ev_feature_start = ev;

            if (feature_bp_left > 0)
            {
                // we skipped some base pairs in the feature, which means there is an indel region to the left
                // move backwards in the indel region to compensate
                while(feature_bp_left && ev_feature_start > cigar_start)
                {
                    auto event = ctx.cigar.cigar_events[ev_feature_start];

                    if (event == cigar_event::S)
                    {
                        // we've reached the left clip region, stop
                        break;
                    }

                    // (mis)matches and deletions consume base pairs from the feature
                    // (note: matches should *never* show up here, since our starting point is essentially the first match/mismatch point along the read)
                    if (event == cigar_event::M || event == cigar_event::D)
                    {
                        feature_bp_left--;
                    }

                    ev_feature_start--;
                }

                // if we landed in an insertion region, move backwards to the beginning of the insertion region
                while (ev_feature_start > cigar_start && ctx.cigar.cigar_events[ev_feature_start - 1] == cigar_event::I)
                {
                    ev_feature_start--;
                }

                // if the matching event has no read coordinate, move forward again until we find the first read coordinate inside the feature range
                while (ev_feature_start < cigar_end && ctx.cigar.cigar_event_read_coordinates[ev_feature_start] == uint16(-1))
                {
                    ev_feature_start++;
                }
            } else {
                // if we didn't move backwards at all and we're in a deletion, then move backwards if the (reference) starting point for the feature is inside our clipping window
                if (ctx.cigar.cigar_event_reference_coordinates[ev_feature_start] <= reference_window_clipped.y)
                {
                    while (ev_feature_start > cigar_start && ctx.cigar.cigar_events[ev_feature_start] == cigar_event::D)
                    {
                        ev_feature_start--;
                    }
                }
            }

            // how many base pairs exist in the feature to the right of the starting point?
            int feature_bp_right = feature_end - ref_coord;

            // now walk forward from our starting point until we have enough base pairs to cover the feature
            // (note that we pre-increment here; this is because the starting point has already been consumed)
            while(feature_bp_right > 0 && ev < cigar_end - 1)
            {
                ev++;

                auto event = ctx.cigar.cigar_events[ev];

                if (event == cigar_event::S)
                {
                    // we've reached the right clip region, stop
                    break;
                }

                // matches and deletions consume base pairs from the feature
                if (event == cigar_event::M || event == cigar_event::D)
                {
                    feature_bp_right--;
                }
            }

            // if there's no read coordinate for the matching event, move backward again until we find the last read coordinate inside the feature range
            while(ev > cigar_start && ctx.cigar.cigar_event_read_coordinates[ev] == uint16(-1))
            {
                ev--;
            }

            // we adjusted both start and end points of the feature in event coordinates
            // if they crossed over, the feature didn't match any base pairs in the read
            if (ev < ev_feature_start)
            {
                continue;
            }

            uint32 read_start = ctx.cigar.cigar_event_read_coordinates[ev_feature_start];
            uint32 read_end   = ctx.cigar.cigar_event_read_coordinates[ev];

            if ((read_start < read_window_clipped.x && read_end < read_window_clipped.x) ||
                (read_start > read_window_clipped.y && read_end > read_window_clipped.y))
            {
                continue;
            }

            // truncate any portions of the feature range that fall outside the clipped read region
            read_start = max<uint32>(read_start, read_window_clipped.x);
            read_start = min<uint32>(read_start, read_window_clipped.y);
            read_end = max<uint32>(read_end, read_window_clipped.x);
            read_end = min<uint32>(read_end, read_window_clipped.y);

            for(uint32 dead_bp = read_start; dead_bp <= read_end; dead_bp++)
                ctx.active_location_list[idx.read_start + dead_bp] = 0;
        }
    }
};

// filter out known SNPs from the active BP list
// for each BP in batch, set the corresponding bit in active_loc_list to zero if it matches a known SNP
template <target_system system>
void filter_known_snps(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    auto& snp = context.snp_filter;

    // compute the VCF ranges for each read
    snp.active_vcf_ranges.resize(batch.device.num_reads);
    parallel<system>::for_each(context.active_read_list.begin(),
                               context.active_read_list.end(),
                               compute_vcf_ranges<system>(context, batch.device));

    // build a list of reads with active VCF ranges
    snp.active_read_ids.resize(context.active_read_list.size());
    context.temp_u32.copy(context.active_read_list);

    uint32 num_active;
    num_active = parallel<system>::copy_if(context.temp_u32.begin(),
                                           context.temp_u32.size(),
                                           snp.active_read_ids.begin(),
                                           vcf_active_predicate<system>(snp.active_vcf_ranges),
                                           context.temp_storage);

    snp.active_read_ids.resize(num_active);

    // finally apply the VCF filter
    // this will create zeros in the active location list for each BP that matches a known variant
    parallel<system>::for_each(snp.active_read_ids.begin(),
                               snp.active_read_ids.end(),
                               filter_bps<system>(context, batch.device));
}
INSTANTIATE(filter_known_snps);

} // namespace firepony
