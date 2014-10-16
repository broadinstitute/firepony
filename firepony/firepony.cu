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

#include "from_nvbio/dna.h"

#include <map>

#include "bqsr_types.h"
#include "gamgee_loader.h"
#include "alignment_data.h"
#include "sequence_data.h"
#include "variant_data.h"
#include "util.h"
#include "snp_filter.h"
#include "bqsr_context.h"
#include "read_filters.h"
#include "cigar.h"
#include "covariates.h"
#include "baq.h"
#include "fractional_errors.h"
#include "read_group_table.h"
#include "options.h"

void debug_read(bqsr_context *context, const alignment_batch& batch, int read_index);

void init_cuda(void)
{
    cudaDeviceProp prop;
    int dev;
    int runtime_version;

    // trigger runtime initialization
    printf("loading CUDA runtime...\n");
    cudaFree(0);

    cudaRuntimeGetVersion(&runtime_version);
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    printf("CUDA runtime version: %d.%d\n", runtime_version / 1000, runtime_version % 100);
    printf("device: %s (%lu MB)\n", prop.name, prop.totalGlobalMem / (1024 * 1024));
}

int main(int argc, char **argv)
{
    sequence_data reference;
    variant_database vcf;
    size_t num_bytes;
    bool ret;

    parse_command_line(argc, argv);

#ifndef RUN_ON_CPU
    init_cuda();
#endif

    // load the reference genome
    printf("loading reference from %s...\n", command_line_options.reference);
    ret = gamgee_load_sequences(&reference, command_line_options.reference,
                                SequenceDataMask::BASES |
                                SequenceDataMask::NAMES);
    if (ret == false)
    {
        printf("failed to load reference %s\n", command_line_options.reference);
        exit(1);
    }

    num_bytes = reference.download();
    printf("downloaded %lu MB of reference data\n", num_bytes / (1024 * 1024));

    printf("loading variant database %s...\n", command_line_options.snp_database);
    ret = gamgee_load_vcf(&vcf, reference, command_line_options.snp_database, VariantDataMask::CHROMOSOME |
                                                                              VariantDataMask::ALIGNMENT |
                                                                              VariantDataMask::REFERENCE);
    if (ret == false)
    {
        printf("failed to load variant database %s\n", command_line_options.snp_database);
        exit(1);
    }

    printf("%u variants\n", vcf.host.num_variants);

    num_bytes = vcf.download();
    printf("downloaded %lu MB of variant data\n", num_bytes / (1024 * 1024));

    printf("processing file %s...\n", command_line_options.input);
    gamgee_alignment_file bam(command_line_options.input);
    alignment_batch batch;

    bqsr_context context(bam.header, vcf, reference);

    uint32 data_mask = AlignmentDataMask::NAME |
                        AlignmentDataMask::CHROMOSOME |
                        AlignmentDataMask::ALIGNMENT_START |
                        AlignmentDataMask::CIGAR |
                        AlignmentDataMask::READS |
                        AlignmentDataMask::QUALITIES |
                        AlignmentDataMask::FLAGS |
                        AlignmentDataMask::MAPQ |
                        AlignmentDataMask::READ_GROUP;

    auto& stats = context.stats;
    cpu_timer wall_clock;
    cpu_timer io;

    gpu_timer read_filter;
    gpu_timer bp_filter;
    gpu_timer snp_filter;
    gpu_timer cigar_expansion;
    gpu_timer baq;
    gpu_timer fractional_error;
    gpu_timer covariates;

    wall_clock.start();

    for(;;)
    {
        // read in the next batch
        io.start();
        bool eof = !(bam.next_batch(&batch, data_mask, 80000));
        io.stop();
        stats.io.add(io);

        if (eof)
        {
            // no more data, stop
            break;
        }

        // load the next batch on the device
        batch.download();
        context.start_batch(batch);

        read_filter.start();

        // build read offset and alignment window list (required by read filters)
        build_read_offset_list(&context, batch);
        build_alignment_windows(&context, batch);
        // apply read filters
        filter_reads(&context, batch);

        read_filter.stop();

        // generate cigar events and coordinates
        // this will generate -1 read indices for events belonging to inactive reads, so it must happen after read filtering
        cigar_expansion.start();
        expand_cigars(&context, batch);
        cigar_expansion.stop();

        // apply per-BP filters
        bp_filter.start();
        filter_bases(&context, batch);
        bp_filter.stop();

        // filter known SNPs from active_loc_list
        snp_filter.start();
        filter_known_snps(&context, batch);
        snp_filter.stop();

        // compute the base alignment quality for each read
        baq.start();
        baq_reads(&context, batch);
        baq.stop();

        fractional_error.start();
        build_fractional_error_arrays(&context, batch);
        fractional_error.stop();

        // build covariate tables
        covariates.start();
        gather_covariates(&context, batch);
        covariates.stop();

#if 0
        for(uint32 read_id = 0; read_id < context.active_read_list.size(); read_id++)
        {
            const uint32 read_index = context.active_read_list[read_id];

            /*
            const char *name = &h_batch.names[h_batch.index[read_index].name];

            if (!strcmp(name, "SRR062635.1797528") ||
                !strcmp(name, "SRR062635.22970839") ||
                !strcmp(name, "SRR062641.22789430") ||
                !strcmp(name, "SRR062641.16264831"))
            {
                debug_read(&context, genome, h_batch, read_index);
            }*/

            debug_read(&context, batch, read_index);
        }
#endif

#if 0
        printf("active VCF ranges: %lu out of %lu reads (%f %%)\n",
                context.snp_filter.active_read_ids.size(),
                context.active_read_list.size(),
                100.0 * float(context.snp_filter.active_read_ids.size()) / context.active_read_list.size());

        H_ActiveLocationList h_bplist = context.active_location_list;
        uint32 zeros = 0;
        for(uint32 i = 0; i < h_bplist.size(); i++)
        {
            if (h_bplist[i] == 0)
                zeros++;
        }

        printf("active BPs: %u out of %u (%f %%)\n", h_bplist.size() - zeros, h_bplist.size(), 100.0 * float(h_bplist.size() - zeros) / float(h_bplist.size()));
#endif

        cudaDeviceSynchronize();
        stats.read_filter.add(read_filter);
        stats.cigar_expansion.add(cigar_expansion);
        stats.snp_filter.add(snp_filter);
        stats.bp_filter.add(bp_filter);
        stats.baq.add(baq);
        stats.fractional_error.add(fractional_error);
        stats.covariates.add(covariates);
    }

    gpu_timer postprocessing;
    cpu_timer output;

    postprocessing.start();
    build_read_group_table(&context);
    postprocessing.stop();

    output.start();
    output_read_group_table(&context);
    output_covariates(&context);
    output.stop();

    cudaDeviceSynchronize();
    wall_clock.stop();

    printf("%d reads filtered out of %d (%f%%)\n",
            context.stats.filtered_reads,
            context.stats.total_reads,
            float(context.stats.filtered_reads) / float(context.stats.total_reads) * 100.0);

    printf("computed base alignment quality for %d reads out of %d (%f%%)\n",
            context.stats.baq_reads,
            context.stats.total_reads - context.stats.filtered_reads,
            float(context.stats.baq_reads) / float(context.stats.total_reads - context.stats.filtered_reads) * 100.0);

    printf("\n");

    printf("wall clock time: %f\n", wall_clock.elapsed_time());
    printf("  io: %.4f (%.2f%%)\n", stats.io.elapsed_time, stats.io.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("  read filtering: %.4f (%.2f%%)\n", stats.read_filter.elapsed_time, stats.read_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("  cigar expansion: %.4f (%.2f%%)\n", stats.cigar_expansion.elapsed_time, stats.cigar_expansion.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("  snp filtering: %.4f (%.2f%%)\n", stats.snp_filter.elapsed_time, stats.snp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("  bp filtering: %.4f (%.2f%%)\n", stats.bp_filter.elapsed_time, stats.bp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("  baq: %.4f (%.2f%%)\n", stats.baq.elapsed_time, stats.baq.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("    setup: %.4f (%.2f%%)\n", stats.baq_setup.elapsed_time, stats.baq_setup.elapsed_time / stats.baq.elapsed_time * 100.0);
    printf("    hmm: %.4f (%.2f%%)\n", stats.baq_hmm.elapsed_time, stats.baq_hmm.elapsed_time / stats.baq.elapsed_time * 100.0);
    printf("    post: %.4f (%.2f%%)\n", stats.baq_postprocess.elapsed_time, stats.baq_postprocess.elapsed_time / stats.baq.elapsed_time * 100.0);
    printf("  fractional error: %.4f (%.2f%%)\n", stats.fractional_error.elapsed_time, stats.fractional_error.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("  covariates: %.4f (%.2f%%)\n", stats.covariates.elapsed_time, stats.covariates.elapsed_time / wall_clock.elapsed_time() * 100.0);
    printf("  post-processing: %.4f (%.2f%%)\n", postprocessing.elapsed_time(), postprocessing.elapsed_time() / wall_clock.elapsed_time() * 100.0);
    printf("  output: %.4f (%.2f%%)\n", output.elapsed_time(), output.elapsed_time() / wall_clock.elapsed_time() * 100.0);

    return 0;
}

void debug_read(bqsr_context *context, const alignment_batch& batch, int read_id)
{
    const alignment_batch_host& h_batch = batch.host;

    const uint32 read_index = context->active_read_list[read_id];
    const CRQ_index idx = h_batch.crq_index(read_index);

    printf("== read order %d read %d\n", read_id, read_index);

    printf("name = [%s]\n", h_batch.name[read_index].c_str());

    printf("  offset list = [ ");
    for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
    {
        uint16 off = context->read_offset_list[i];
        if (off == uint16(-1))
        {
            printf("  - ");
        } else {
            printf("% 3d ", off);
        }
    }
    printf("]\n");

    debug_cigar(context, batch, read_index);
    debug_baq(context, batch, read_index);
    debug_fractional_error_arrays(context, batch, read_index);

    const uint2 alignment_window = context->alignment_windows[read_index];
    printf("  sequence name [%s]\n  sequence base [%lu]\n  sequence offset [%u]\n  alignment window [%u, %u]\n",
            context->reference.sequence_names.lookup(h_batch.chromosome[read_index]).c_str(),
            context->reference.host.sequence_bp_start[h_batch.chromosome[read_index]],
            h_batch.alignment_start[read_index],
            alignment_window.x,
            alignment_window.y);

    const uint2 vcf_range = context->snp_filter.active_vcf_ranges[read_index];
    printf("  active VCF range: [%u, %u[\n", vcf_range.x, vcf_range.y);

    printf("\n");
}
