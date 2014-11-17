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

#include "alignment_data.h"
#include "sequence_data.h"
#include "variant_data.h"

#include "types.h"
#include "command_line.h"
#include "gamgee_loader.h"
#include "io_thread.h"
#include "string_database.h"

#include "device/pipeline.h"

#include "version.h"

using namespace firepony;

#if ENABLE_CUDA_BACKEND
static bool cuda_runtime_init(std::string& ret)
{
    cudaError_t err;
    int runtime_version;

    // force explicit runtime initialization
    err = cudaFree(0);
    if (err != cudaSuccess)
    {
        ret = std::string(cudaGetErrorString(err));
        return false;
    }

    err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess)
    {
        ret = std::string(cudaGetErrorString(err));
        return false;
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
             "%d.%d", runtime_version / 1000, runtime_version % 100);

    ret = std::string(buf);
    return true;
}

static void enumerate_gpus(std::vector<firepony_pipeline *>& ret)
{
    std::vector<firepony_pipeline *> gpus;

    if (!command_line_options.enable_cuda)
        return;

    cudaError_t err;
    int gpu_count;

    err = cudaGetDeviceCount(&gpu_count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "error enumerating CUDA devices: %s\n", cudaGetErrorString(err));
        return;
    }

    for(int dev = 0; dev < gpu_count; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        // sm 3.x or above is required
        if (prop.major < 3)
            continue;

        firepony_pipeline *pipeline = firepony_pipeline::create(firepony::cuda, dev);
        ret.push_back(pipeline);
    }
}
#endif

static std::vector<firepony_pipeline *> enumerate_compute_devices(void)
{
    std::vector<firepony_pipeline *> ret;
    int compute_device_count = 0;

#if ENABLE_CUDA_BACKEND
    enumerate_gpus(ret);
#endif

#if ENABLE_TBB_BACKEND
    compute_device_count = ret.size();
    if (command_line_options.enable_tbb)
    {
        firepony_pipeline *dev;
        dev = firepony_pipeline::create(firepony::intel_tbb, compute_device_count + 1);
        ret.push_back(dev);
    }
#endif

    return ret;
}

static void print_statistics(const timer<host>& wall_clock, const pipeline_statistics& stats)
{
    fprintf(stderr, "   blocked on io: %.4f (%.2f%%)\n", stats.io.elapsed_time, stats.io.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   read filtering: %.4f (%.2f%%)\n", stats.read_filter.elapsed_time, stats.read_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   cigar expansion: %.4f (%.2f%%)\n", stats.cigar_expansion.elapsed_time, stats.cigar_expansion.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   bp filtering: %.4f (%.2f%%)\n", stats.bp_filter.elapsed_time, stats.bp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   snp filtering: %.4f (%.2f%%)\n", stats.snp_filter.elapsed_time, stats.snp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   baq: %.4f (%.2f%%)\n", stats.baq.elapsed_time, stats.baq.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "     setup: %.4f (%.2f%%)\n", stats.baq_setup.elapsed_time, stats.baq_setup.elapsed_time / stats.baq.elapsed_time * 100.0);
    fprintf(stderr, "     hmm: %.4f (%.2f%%)\n", stats.baq_hmm.elapsed_time, stats.baq_hmm.elapsed_time / stats.baq.elapsed_time * 100.0);
    fprintf(stderr, "     post: %.4f (%.2f%%)\n", stats.baq_postprocess.elapsed_time, stats.baq_postprocess.elapsed_time / stats.baq.elapsed_time * 100.0);
    fprintf(stderr, "   fractional error: %.4f (%.2f%%)\n", stats.fractional_error.elapsed_time, stats.fractional_error.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   covariates: %.4f (%.2f%%)\n", stats.covariates.elapsed_time, stats.covariates.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "     gather: %.4f (%.2f%%)\n", stats.covariates_gather.elapsed_time, stats.covariates_gather.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "     filter: %.4f (%.2f%%)\n", stats.covariates_filter.elapsed_time, stats.covariates_filter.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "     sort: %.4f (%.2f%%)\n", stats.covariates_sort.elapsed_time, stats.covariates_sort.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "     pack: %.4f (%.2f%%)\n", stats.covariates_pack.elapsed_time, stats.covariates_pack.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   post-processing: %.4f (%.2f%%)\n", stats.postprocessing.elapsed_time, stats.postprocessing.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   output: %.4f (%.2f%%)\n", stats.output.elapsed_time, stats.output.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "   batches: %lu (%.2f batches/sec)\n", stats.num_batches, stats.num_batches / wall_clock.elapsed_time());
    fprintf(stderr, "   reads: %lu (%.2fM reads/sec)\n", stats.total_reads, stats.total_reads / 1000000.0 / wall_clock.elapsed_time());
}

int main(int argc, char **argv)
{
    std::vector<firepony_pipeline *> compute_devices;

    sequence_data_host h_reference;
    variant_database_host h_dbsnp;
    bool ret;

    fprintf(stderr, "Firepony v%d.%d.%d\n", FIREPONY_VERSION_MAJOR, FIREPONY_VERSION_MINOR, FIREPONY_VERSION_REV);
    parse_command_line(argc, argv);

#if ENABLE_CUDA_BACKEND
    std::string runtime_version;
    if (cuda_runtime_init(runtime_version) == false)
    {
        fprintf(stderr, "error loading CUDA runtime: %s\n", runtime_version.c_str());
        exit(1);
    }

    fprintf(stderr, "CUDA runtime version %s\n", runtime_version.c_str());
#endif

    compute_devices = enumerate_compute_devices();

    if (compute_devices.size() == 0)
    {
        fprintf(stderr, "failed to initialize compute backend\n");
        exit(1);
    }

    fprintf(stderr, "enabled compute devices:\n");
    for(auto d : compute_devices)
    {
        fprintf(stderr, "  %s\n", d->get_name().c_str());
    }
    fprintf(stderr, "\n");

    // load the reference genome
    fprintf(stderr, "loading reference from %s...\n", command_line_options.reference);
    ret = gamgee_load_sequences(&h_reference, command_line_options.reference,
                                SequenceDataMask::BASES |
                                SequenceDataMask::NAMES);
    if (ret == false)
    {
        fprintf(stderr, "failed to load reference %s\n", command_line_options.reference);
        exit(1);
    }

    fprintf(stderr, "loading variant database %s...\n", command_line_options.snp_database);
    ret = gamgee_load_vcf(&h_dbsnp, h_reference, command_line_options.snp_database,
                          VariantDataMask::CHROMOSOME | VariantDataMask::ALIGNMENT);

    if (ret == false)
    {
        fprintf(stderr, "failed to load variant database %s\n", command_line_options.snp_database);
        exit(1);
    }

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

    io_thread reader(command_line_options.input, data_mask, compute_devices.size());
    reader.start();

    for(auto d : compute_devices)
    {
        d->setup(&reader,
                 &command_line_options,
                 &reader.file.header,
                 &h_reference,
                 &h_dbsnp);
    }

    fprintf(stderr, "processing file %s...\n", command_line_options.input);

    timer<host> wall_clock;

    wall_clock.start();

    for(auto d : compute_devices)
    {
        d->start();
    }

    for(auto d : compute_devices)
    {
        d->join();
    }

    fprintf(stderr, "\n");

    // we pick the first device to do the postprocessing on
    auto d = compute_devices[0];
    if (compute_devices.size() > 1)
    {
        for(uint32 i = 1; i < compute_devices.size(); i++)
        {
            d->gather_intermediates(compute_devices[i]);
        }
    }

    d->postprocess();

    wall_clock.stop();

    // compute aggregate statistics
    pipeline_statistics aggregate_stats;
    for(auto d : compute_devices)
    {
        aggregate_stats += d->statistics();
    }

    fprintf(stderr, "%lu reads filtered out of %lu (%f%%)\n",
            aggregate_stats.filtered_reads,
            aggregate_stats.total_reads,
            float(aggregate_stats.filtered_reads) / float(aggregate_stats.total_reads) * 100.0);

    fprintf(stderr, "computed base alignment quality for %lu reads out of %lu (%f%%)\n",
            aggregate_stats.baq_reads,
            aggregate_stats.total_reads - aggregate_stats.filtered_reads,
            float(aggregate_stats.baq_reads) / float(aggregate_stats.total_reads - aggregate_stats.filtered_reads) * 100.0);

    fprintf(stderr, "\n");

    fprintf(stderr, "wall clock time: %f\n", wall_clock.elapsed_time());

    if (compute_devices.size() > 1)
        fprintf(stderr, "aggregate statistics:\n");

    print_statistics(wall_clock, aggregate_stats);
    if (compute_devices.size() > 1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "per-device statistics:\n");

        for(uint32 i = 0; i < compute_devices.size(); i++)
        {
            fprintf(stderr, " device %d: %s\n", i, compute_devices[i]->get_name().c_str());
            print_statistics(wall_clock, compute_devices[i]->statistics());
        }
    }
    fprintf(stderr, "\n");

    reader.join();

    return 0;
}
