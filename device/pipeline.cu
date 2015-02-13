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

#include "pipeline.h"

#include "device/primitives/backends.h"
#include "device/primitives/timer.h"
#include "device/primitives/parallel.h"

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
        // GPU debugging output always goes to stdout; flush here to ensure it gets printed in the right place
        fflush(stdout);

        for(uint32 read_id = 0; read_id < context.active_read_list.size(); read_id++)
        {
            const uint32 read_index = context.active_read_list[read_id];
            debug_read(context, batch, read_index);
        }
    }

    context.end_batch(batch);

    parallel<system>::synchronize();

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
static void output_header(firepony_context<system>& context)
{
    // note: we only output 4 tables, as opposed to GATK's 5
    // this is because the quality quantization map doesn't seem to matter, so we omit that
    printf("%s", "#:GATKReport.v1.1:4\n");
    printf("%s", "#:GATKTable:2:17:%s:%s:;\n");
    printf("%s", "#:GATKTable:Arguments:Recalibration argument collection values used in this run\n");
    printf("%s", "Argument                    Value\n");
    printf("%s", "binary_tag_name             null\n");
    printf("%s", "covariate                   ReadGroupCovariate,QualityScoreCovariate,ContextCovariate,CycleCovariate\n");
    printf("%s", "default_platform            null\n");
    printf("%s", "deletions_default_quality   45\n");
    printf("%s", "force_platform              null\n");
    printf("%s", "indels_context_size         3\n");
    printf("%s", "insertions_default_quality  45\n");
    printf("%s", "low_quality_tail            2\n");
    printf("%s", "maximum_cycle_value         500\n");
    printf("%s", "mismatches_context_size     2\n");
    printf("%s", "mismatches_default_quality  -1\n");
    printf("%s", "no_standard_covs            false\n");
    printf("%s", "quantizing_levels           16\n");
    printf("%s", "recalibration_report        null\n");
    printf("%s", "run_without_dbsnp           false\n");
    printf("%s", "solid_nocall_strategy       THROW_EXCEPTION\n");
    printf("%s", "solid_recal_mode            SET_Q_ZERO\n");
    printf("\n");
}

template <target_system system>
void firepony_postprocess(firepony_context<system>& context)
{
    timer<system> postprocessing;
    timer<host> output;

    postprocessing.start();
    postprocess_covariates(context);
    build_read_group_table(context);
    compute_empirical_quality_scores(context);
    postprocessing.stop();

    output.start();
    output_header(context);
    output_read_group_table(context);
    output_covariates(context);
    output.stop();

    parallel<system>::synchronize();

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
    fprintf(stderr, "flags = %d\n", h_batch.flags[read_index]);

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

    const uint2 alignment_window = context.alignment_windows[read_index];
    fprintf(stderr, "  sequence name [%s]\n  sequence base [%lu]\n  sequence offset [%u]\n  alignment window [%u, %u]\n",
            context.reference.host.sequence_names.lookup(h_batch.chromosome[read_index]).c_str(),
            context.reference.host.sequence_bp_start[h_batch.chromosome[read_index]],
            h_batch.alignment_start[read_index],
            alignment_window.x,
            alignment_window.y);

    const uint2 vcf_range = context.snp_filter.active_vcf_ranges[read_index];
    fprintf(stderr, "  active VCF range: [%u, %u[\n", vcf_range.x, vcf_range.y);

    fprintf(stderr, "\n");
}
INSTANTIATE(debug_read);

} // namespace firepony
