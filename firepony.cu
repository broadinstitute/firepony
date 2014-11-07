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

#include <map>

#include "device/alignment_data.h"
#include "device/baq.h"
#include "device/firepony_context.h"
#include "device/device_types.h"
#include "device/cigar.h"
#include "device/covariates.h"
#include "device/fractional_errors.h"
#include "device/read_filters.h"
#include "device/read_group_table.h"
#include "device/sequence_data.h"
#include "device/snp_filter.h"
#include "device/string_database.h"
#include "device/util.h"
#include "device/variant_data.h"
#include "command_line.h"
#include "gamgee_loader.h"

using namespace firepony;

void debug_read(firepony_context *context, const alignment_batch& batch, int read_index);

#include <thread>

struct io_thread
{
    static constexpr bool DISABLE_THREADING = false;
    static constexpr int NUM_BUFFERS = 3;

    alignment_batch batches[NUM_BUFFERS];
    volatile int put, get;
    volatile bool eof;

    gamgee_alignment_file file;
    uint32 data_mask;

    std::thread thread;

    io_thread(const char *fname, uint32 data_mask)
        : put(1),
          get(0),
          eof(false),
          file(fname),
          data_mask(data_mask)
    { }

    void start(void)
    {
        if (!DISABLE_THREADING)
        {
            thread = std::thread(&io_thread::run, this);
        }
    }

    void join(void)
    {
        if (!DISABLE_THREADING)
        {
            thread.join();
        }
    }

    int wrap(int val)
    {
        return val % NUM_BUFFERS;
    }

    alignment_batch& next_buffer(void)
    {
        if (DISABLE_THREADING)
        {
            return batches[0];
        } else {
            while(wrap(get + 1) == put)
                std::this_thread::yield();

            get = wrap(get + 1);
            return batches[get];
        }
    }

    bool done(void)
    {
        if (DISABLE_THREADING)
        {
            eof = !(file.next_batch(&batches[0], data_mask, command_line_options.batch_size));
            return eof;
        } else {
            if (!eof)
                return false;

            if (wrap(get + 1) != put)
                return false;

            return true;
        }
    }

private:
    void run(void)
    {
        while(!eof)
        {
            // wait for a slot
            while (put == get)
                std::this_thread::yield();

            eof = !(file.next_batch(&batches[put], data_mask, command_line_options.batch_size));
            if (!eof)
            {
                put = wrap(put + 1);
            }
        }
    }
};

void init_cuda(void)
{
    cudaDeviceProp prop;
    int dev;
    int runtime_version;

    // trigger runtime initialization
    fprintf(stderr, "loading CUDA runtime...\n");
    cudaFree(0);

    cudaRuntimeGetVersion(&runtime_version);
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    fprintf(stderr, "CUDA runtime version: %d.%d\n", runtime_version / 1000, runtime_version % 100);
    fprintf(stderr, "device: %s (%lu MB)\n", prop.name, prop.totalGlobalMem / (1024 * 1024));
}

int main(int argc, char **argv)
{
    firepony::sequence_data reference;
    variant_database vcf;
    size_t num_bytes;
    bool ret;

    parse_command_line(argc, argv);

#ifndef RUN_ON_CPU
    init_cuda();
#endif

    // load the reference genome
    fprintf(stderr, "loading reference from %s...\n", command_line_options.reference);
    ret = gamgee_load_sequences(&reference, command_line_options.reference,
                                SequenceDataMask::BASES |
                                SequenceDataMask::NAMES);
    if (ret == false)
    {
        fprintf(stderr, "failed to load reference %s\n", command_line_options.reference);
        exit(1);
    }

    num_bytes = reference.download();
    fprintf(stderr, "downloaded %lu MB of reference data\n", num_bytes / (1024 * 1024));

    fprintf(stderr, "loading variant database %s...\n", command_line_options.snp_database);
    ret = gamgee_load_vcf(&vcf, reference, command_line_options.snp_database, VariantDataMask::CHROMOSOME |
                                                                              VariantDataMask::ALIGNMENT);
    if (ret == false)
    {
        fprintf(stderr, "failed to load variant database %s\n", command_line_options.snp_database);
        exit(1);
    }

    fprintf(stderr, "%u variants\n", vcf.host.num_variants);

    num_bytes = vcf.download();
    fprintf(stderr, "downloaded %lu MB of variant data\n", num_bytes / (1024 * 1024));

    fprintf(stderr, "processing file %s...\n", command_line_options.input);

    const uint32 data_mask = AlignmentDataMask::NAME |
                             AlignmentDataMask::CHROMOSOME |
                             AlignmentDataMask::ALIGNMENT_START |
                             AlignmentDataMask::ALIGNMENT_STOP |
                             AlignmentDataMask::MATE_ALIGNMENT_START |
                             AlignmentDataMask::INFERRED_INSERT_SIZE |
                             AlignmentDataMask::CIGAR |
                             AlignmentDataMask::READS |
                             AlignmentDataMask::QUALITIES |
                             AlignmentDataMask::FLAGS |
                             AlignmentDataMask::MAPQ |
                             AlignmentDataMask::READ_GROUP;

    io_thread bam_thread(command_line_options.input, data_mask);
    bam_thread.start();

    firepony_context context(bam_thread.file.header, reference, vcf);

    auto& stats = context.stats;
    cpu_timer wall_clock;
    cpu_timer io;

    device_timer read_filter;
    device_timer bp_filter;
    device_timer snp_filter;
    device_timer cigar_expansion;
    device_timer baq;
    device_timer fractional_error;
    device_timer covariates;

    wall_clock.start();

    while(!bam_thread.done())
    {
        io.start();
        // fetch the next batch
        alignment_batch& batch = bam_thread.next_buffer();

        // load the next batch on the device
        batch.download();
        context.start_batch(batch);
        io.stop();

        read_filter.start();

        // build read offset and alignment window list (required by read filters)
        build_read_offset_list(context, batch);
        build_alignment_windows(context, batch);
        // apply read filters
        filter_reads(context, batch);

        read_filter.stop();

        // if all reads have been filtered out, skip the rest of the pipeline
        if (context.active_read_list.size() == 0)
            continue;

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

        if (command_line_options.debug)
        {
            for(uint32 read_id = 0; read_id < context.active_read_list.size(); read_id++)
            {
                const uint32 read_index = context.active_read_list[read_id];
                debug_read(&context, batch, read_index);
            }
        }

#if 0
        fprintf(stderr, "active VCF ranges: %lu out of %lu reads (%f %%)\n",
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

        fprintf(stderr, "active BPs: %u out of %u (%f %%)\n", h_bplist.size() - zeros, h_bplist.size(), 100.0 * float(h_bplist.size() - zeros) / float(h_bplist.size()));
#endif

        context.end_batch(batch);

        cudaDeviceSynchronize();
        stats.read_filter.add(read_filter);
        stats.cigar_expansion.add(cigar_expansion);
        stats.bp_filter.add(bp_filter);
        stats.snp_filter.add(snp_filter);
        stats.baq.add(baq);
        stats.fractional_error.add(fractional_error);
        stats.covariates.add(covariates);

        if (!command_line_options.debug)
        {
            fprintf(stderr, ".");
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n");

    device_timer postprocessing;
    cpu_timer output;

    postprocessing.start();
    build_read_group_table(context);
    postprocessing.stop();

    output.start();
    output_read_group_table(context);
    output_covariates(context);
    output.stop();

    cudaDeviceSynchronize();
    wall_clock.stop();

    fprintf(stderr, "%d reads filtered out of %d (%f%%)\n",
            context.stats.filtered_reads,
            context.stats.total_reads,
            float(context.stats.filtered_reads) / float(context.stats.total_reads) * 100.0);

    fprintf(stderr, "computed base alignment quality for %d reads out of %d (%f%%)\n",
            context.stats.baq_reads,
            context.stats.total_reads - context.stats.filtered_reads,
            float(context.stats.baq_reads) / float(context.stats.total_reads - context.stats.filtered_reads) * 100.0);

    fprintf(stderr, "\n");

    fprintf(stderr, "wall clock time: %f\n", wall_clock.elapsed_time());
    fprintf(stderr, "  blocked on io: %.4f (%.2f%%)\n", stats.io.elapsed_time, stats.io.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  read filtering: %.4f (%.2f%%)\n", stats.read_filter.elapsed_time, stats.read_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  cigar expansion: %.4f (%.2f%%)\n", stats.cigar_expansion.elapsed_time, stats.cigar_expansion.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  bp filtering: %.4f (%.2f%%)\n", stats.bp_filter.elapsed_time, stats.bp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  snp filtering: %.4f (%.2f%%)\n", stats.snp_filter.elapsed_time, stats.snp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  baq: %.4f (%.2f%%)\n", stats.baq.elapsed_time, stats.baq.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "    setup: %.4f (%.2f%%)\n", stats.baq_setup.elapsed_time, stats.baq_setup.elapsed_time / stats.baq.elapsed_time * 100.0);
    fprintf(stderr, "    hmm: %.4f (%.2f%%)\n", stats.baq_hmm.elapsed_time, stats.baq_hmm.elapsed_time / stats.baq.elapsed_time * 100.0);
    fprintf(stderr, "    post: %.4f (%.2f%%)\n", stats.baq_postprocess.elapsed_time, stats.baq_postprocess.elapsed_time / stats.baq.elapsed_time * 100.0);
    fprintf(stderr, "  fractional error: %.4f (%.2f%%)\n", stats.fractional_error.elapsed_time, stats.fractional_error.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  covariates: %.4f (%.2f%%)\n", stats.covariates.elapsed_time, stats.covariates.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "    gather: %.4f (%.2f%%)\n", stats.covariates_gather.elapsed_time, stats.covariates_gather.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "    filter: %.4f (%.2f%%)\n", stats.covariates_filter.elapsed_time, stats.covariates_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "    sort: %.4f (%.2f%%)\n", stats.covariates_sort.elapsed_time, stats.covariates_sort.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "    pack: %.4f (%.2f%%)\n", stats.covariates_pack.elapsed_time, stats.covariates_pack.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  post-processing: %.4f (%.2f%%)\n", postprocessing.elapsed_time(), postprocessing.elapsed_time() / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  output: %.4f (%.2f%%)\n", output.elapsed_time(), output.elapsed_time() / wall_clock.elapsed_time() * 100.0);

    bam_thread.join();

    return 0;
}

void debug_read(firepony_context *context, const alignment_batch& batch, int read_id)
{
    const alignment_batch_host& h_batch = batch.host;

    const uint32 read_index = context->active_read_list[read_id];
    const CRQ_index idx = h_batch.crq_index(read_index);

    fprintf(stderr, "== read %d\n", context->stats.total_reads + read_id);

    fprintf(stderr, "name = [%s]\n", h_batch.name[read_index].c_str());

    fprintf(stderr, "  offset list = [ ");
    for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
    {
        uint16 off = context->read_offset_list[i];
        if (off == uint16(-1))
        {
            fprintf(stderr, "  - ");
        } else {
            fprintf(stderr, "% 3d ", off);
        }
    }
    fprintf(stderr, "]\n");

    debug_cigar(*context, batch, read_index);
    debug_baq(*context, batch, read_index);
    debug_fractional_error_arrays(*context, batch, read_index);

    const uint2 alignment_window = context->alignment_windows[read_index];
    fprintf(stderr, "  sequence name [%s]\n  sequence base [%lu]\n  sequence offset [%u]\n  alignment window [%u, %u]\n",
            context->reference.sequence_names.lookup(h_batch.chromosome[read_index]).c_str(),
            context->reference.host.sequence_bp_start[h_batch.chromosome[read_index]],
            h_batch.alignment_start[read_index],
            alignment_window.x,
            alignment_window.y);

    const uint2 vcf_range = context->snp_filter.active_vcf_ranges[read_index];
    fprintf(stderr, "  active VCF range: [%u, %u[\n", vcf_range.x, vcf_range.y);

    fprintf(stderr, "\n");
}
