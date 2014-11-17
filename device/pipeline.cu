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

#include "pipeline.h"

#include "device/primitives/backends.h"
#include "device/primitives/timer.h"

#include "alignment_data_device.h"
#include "sequence_data_device.h"
#include "variant_data_device.h"

#include "baq.h"
#include "firepony_context.h"
#include "cigar.h"
#include "covariates.h"
#include "fractional_errors.h"
#include "read_filters.h"
#include "read_group_table.h"
#include "snp_filter.h"
#include "util.h"

namespace firepony {

template <target_system system>
void debug_read(firepony_context<system>& context, const alignment_batch<system>& batch, uint32 read_id);

template <target_system system>
void firepony_process_batch(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    timer<system> read_filter;
    timer<system> bp_filter;
    timer<system> snp_filter;
    timer<system> cigar_expansion;
    timer<system> baq;
    timer<system> fractional_error;
    timer<system> covariates;

    context.start_batch(batch);

    read_filter.start();

    // filter out invalid reads
    filter_invalid_reads(context, batch);

    if (context.active_read_list.size() > 0)
    {
        // build read offset and alignment window list (required by read filters)
        build_read_offset_list(context, batch);
        build_alignment_windows(context, batch);

        // filter malformed reads (uses the alignment windows)
        filter_malformed_reads(context, batch);
    }

    read_filter.stop();

    if (context.active_read_list.size() > 0)
    {
        // generate cigar events and coordinates
        // this will generate -1 read indices for events belonging to inactive reads, so it must happen after read filtering
        cigar_expansion.start();
        expand_cigars(context, batch);
        cigar_expansion.stop();

        // apply per-BP filters
        bp_filter.start();
        filter_bases(context, batch);
        bp_filter.stop();

        // filter known SNPs from active_loc_list
        snp_filter.start();
        filter_known_snps(context, batch);
        snp_filter.stop();

        // compute the base alignment quality for each read
        baq.start();
        baq_reads(context, batch);
        baq.stop();

        fractional_error.start();
        build_fractional_error_arrays(context, batch);
        fractional_error.stop();

        // build covariate tables
        covariates.start();
        gather_covariates(context, batch);
        covariates.stop();
    }

    if (context.options.debug)
    {
        for(uint32 read_id = 0; read_id < context.active_read_list.size(); read_id++)
        {
            const uint32 read_index = context.active_read_list[read_id];
            debug_read(context, batch, read_index);
        }
    }

    context.end_batch(batch);

    if (system == firepony::cuda)
    {
        cudaDeviceSynchronize();
    }

    context.stats.read_filter.add(read_filter);

    if (context.active_read_list.size() > 0)
    {
        context.stats.cigar_expansion.add(cigar_expansion);
        context.stats.bp_filter.add(bp_filter);
        context.stats.snp_filter.add(snp_filter);
        context.stats.baq.add(baq);
        context.stats.fractional_error.add(fractional_error);
        context.stats.covariates.add(covariates);
    }
}
INSTANTIATE(firepony_process_batch);

template <target_system system>
void firepony_postprocess(firepony_context<system>& context)
{
    timer<system> postprocessing;
    timer<host> output;

    postprocessing.start();
    postprocess_covariates(context);
    build_read_group_table(context);
    build_empirical_quality_score_table(context);
    postprocessing.stop();

    output.start();
    output_read_group_table(context);
    output_covariates(context);
    output.stop();

    if (system == firepony::cuda)
    {
        cudaDeviceSynchronize();
    }

    context.stats.postprocessing.add(postprocessing);
    context.stats.output.add(output);
}
INSTANTIATE(firepony_postprocess);

template <target_system system>
void debug_read(firepony_context<system>& context, const alignment_batch<system>& batch, uint32 read_id)
{
    const alignment_batch_host& h_batch = *batch.host;

    const uint32 read_index = context.active_read_list[read_id];
    const CRQ_index idx = h_batch.crq_index(read_index);

    fprintf(stderr, "== read %lu\n", context.stats.total_reads + read_id);

    fprintf(stderr, "name = [%s]\n", h_batch.name[read_index].c_str());

    fprintf(stderr, "  offset list = [ ");
    for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
    {
        uint16 off = context.read_offset_list[i];
        if (off == uint16(-1))
        {
            fprintf(stderr, "  - ");
        } else {
            fprintf(stderr, "% 3d ", off);
        }
    }
    fprintf(stderr, "]\n");

    debug_cigar(context, batch, read_index);
    debug_baq(context, batch, read_index);
    debug_fractional_error_arrays(context, batch, read_index);

    const uint2 alignment_window = context.alignment_windows[read_index];
    fprintf(stderr, "  sequence name [%s]\n  sequence base [%lu]\n  sequence offset [%u]\n  alignment window [%u, %u]\n",
            context.reference.host.sequence_names.lookup(h_batch.chromosome[read_index]).c_str(),
            context.reference.host.view.sequence_bp_start[h_batch.chromosome[read_index]],
            h_batch.alignment_start[read_index],
            alignment_window.x,
            alignment_window.y);

    const uint2 vcf_range = context.snp_filter.active_vcf_ranges[read_index];
    fprintf(stderr, "  active VCF range: [%u, %u[\n", vcf_range.x, vcf_range.y);

    fprintf(stderr, "\n");
}
INSTANTIATE(debug_read);

} // namespace firepony
