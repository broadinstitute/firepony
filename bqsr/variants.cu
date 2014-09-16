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
#include "variants.h"

#include "primitives/algorithms.h"
#include "primitives/cuda.h"
#include "primitives/parallel.h"

void SNPDatabase_refIDs::compute_sequence_offsets(const sequence_data& genome)
{
    variant_sequence_ref_ids.resize(reference_sequence_names.size());
    genome_start_positions.resize(reference_sequence_names.size());
    genome_stop_positions.resize(reference_sequence_names.size());

    for(unsigned int c = 0; c < reference_sequence_names.size(); c++)
    {
        uint32 id = genome.host.sequence_names.lookup(reference_sequence_names[c]);
        assert(id != uint32(-1));

        variant_sequence_ref_ids[c] = id;
        // store the genome offset for this VCF entry
        genome_start_positions[c] = genome.host.sequence_bp_start[id] + sequence_positions[c].x - 1; // sequence positions are 1-based, genome positions are 0-based
        genome_stop_positions[c] = genome.host.sequence_bp_start[id] + sequence_positions[c].y - 1;
    }
}

void DeviceSNPDatabase::load(const SNPDatabase_refIDs& ref)
{
    variant_sequence_ref_ids = ref.variant_sequence_ref_ids;
    sequence_positions = ref.sequence_positions;
    genome_start_positions = ref.genome_start_positions;
    genome_stop_positions = ref.genome_stop_positions;
    reference_sequences = ref.reference_sequences;
    variants = ref.variants;
    ref_variant_index = ref.ref_variant_index;
}

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
        const uint2& alignment_window = ctx.alignment_windows[read_index];
        uint2& vcf_range = ctx.snp_filter.active_vcf_ranges[read_index];

        // search for the starting range
        const uint32 *vcf_start;
        vcf_start = bqsr::lower_bound(alignment_window.x,
                                      ctx.db.genome_start_positions.begin(),
                                      ctx.db.genome_start_positions.size());

        // do a linear search to find the end of the VCF range
        // (there are generally very few VCF entries for an average read length --- and often none --- so this is expected to be faster than a binary search)
        const uint32 *vcf_end = vcf_start;
        while(vcf_end < ctx.db.genome_start_positions.begin() + ctx.db.genome_start_positions.size() &&
              *vcf_end < alignment_window.y)
        {
            vcf_end++;
        }

        if (vcf_end == vcf_start)
        {
            // emit an empty VCF range
            vcf_range = make_uint2(uint32(-1), uint32(-1));
        } else {
            if (vcf_end >= ctx.db.genome_start_positions.begin() + ctx.db.genome_start_positions.size())
                vcf_end--;

            vcf_range = make_uint2(vcf_start - ctx.db.genome_start_positions.begin(),
                                   vcf_end - ctx.db.genome_start_positions.begin());
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

private:
#if 0
    // xxxnsubtil: GATK doesn't actually test the read data, it only seems to look at the coordinates
    // test a VCF entry (identified by index in the db) against a read
    CUDA_HOST_DEVICE uint32 test_vcf(uint32 vcf_entry, const uint32 read_index, const uint32 read_bp_offset)
    {
        const BAM_CRQ_index& idx = batch.crq_index[read_index];
        const io::SNP_sequence_index& vcf_idx = ctx.db.ref_variant_index[vcf_entry];

        if (read_bp_offset + vcf_idx.variant_len > idx.read_len)
        {
            // if the variant ends past the end of the read, we can't test it
            return 0;
        }

        for(uint32 i = 0; i < vcf_idx.variant_len; i++)
        {
            if (batch.reads[idx.read_start + read_bp_offset + i] != ctx.db.variants[vcf_idx.variant_start + i])
            //if (batch.reads[idx.read_start + read_bp_offset + i] != ctx.db.reference_sequences[vcf_idx.reference_start + i])
                return 0;
        }

        return vcf_idx.variant_len;
    }
#endif

public:
    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index& idx = batch.crq_index(read_index);
        const uint2& alignment_window = ctx.alignment_windows[read_index];
        const uint16 *offset_list = &ctx.read_offset_list[0];

        uint2 vcf_db_range = ctx.snp_filter.active_vcf_ranges[read_index];
        uint2 vcf_range = make_uint2(ctx.db.genome_start_positions[vcf_db_range.x],
                                      ctx.db.genome_start_positions[vcf_db_range.y]);

        // traverse the list of reads together with the VCF list; when we find a match, stop
        for(uint32 bp = idx.read_start; bp < idx.read_start + idx.read_len; bp++)
        {
            if (offset_list[bp] == uint16(-1))
            {
                // skip bases that don't exist in the reference
                // (every starting point of a VCF entry must exist, even if the rest are insertions to the reference --- I think!)
                continue;
            }

            // compute the reference offset for the current BP
            const uint32 ref_offset = alignment_window.x + offset_list[bp];

            if (ref_offset < vcf_range.x)
            {
                // we're behind the known VCF range, skip this BP
                continue;
            }

            if (ref_offset > vcf_range.y)
            {
                // we've gone past the end of the range, stop the search
                return;
            }

            while (ref_offset > vcf_range.x)
            {
                // we've moved past the beginning of the current VCF entry
                // move forward in the VCF range
                vcf_db_range.x++;
                if (vcf_db_range.x > vcf_db_range.y)
                {
                    // no more VCF entries to test
                    return;
                }

                vcf_range.x = ctx.db.genome_start_positions[vcf_db_range.x];
            }

            // if we found the start of a variant, mark the corresponding BP range
            while (ref_offset == vcf_range.x)
            {
                uint32 vcf_len = ctx.db.genome_stop_positions[vcf_db_range.x];
                if (vcf_len)
                {
                    // turn off vcf_match_len BPs since they match the variant database
                    const uint32 start = idx.read_start;
                    const uint32 end = bqsr::min(start + vcf_len, start + idx.read_len);

                    for(uint32 dead_bp = start; dead_bp < end; dead_bp++)
                        ctx.active_location_list[dead_bp] = 0;

                    // move the BP counter forward
                    bp += end - start;
                }

                // move forward in the VCF range
                vcf_db_range.x++;
                if (vcf_db_range.x > vcf_db_range.y)
                {
                    // no more VCF entries to test
                    return;
                }

                vcf_range.x = ctx.db.genome_start_positions[vcf_db_range.x];
            }
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
                               vcf_active_predicate(snp.active_vcf_ranges));

    snp.active_read_ids.resize(num_active);

    // finally apply the VCF filter
    // this will create zeros in the active location list for each BP that matches a known variant
    thrust::for_each(snp.active_read_ids.begin(),
                     snp.active_read_ids.end(),
                     filter_bps(*context, batch.device));
}
