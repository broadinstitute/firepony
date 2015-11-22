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

#include <map>

#include "alignment_data.h"
#include "sequence_database.h"
#include "types.h"
#include "command_line.h"
#include "io_thread.h"
#include "string_database.h"
#include "output.h"

#include "loader/alignments.h"
#include "loader/reference.h"
#include "loader/variants.h"

#include "device/pipeline.h"
#include "variant_database.h"

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

        // minimum of 4GB of memory required
        // (CUDA reports 4095MB for a K5000, so we use that instead of the full 4GB as the limit)
        if (prop.totalGlobalMem < size_t(4095) * 1024 * 1024)
            continue;

        firepony_pipeline *pipeline = firepony_pipeline::create(lift::cuda, dev);
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
        dev = firepony_pipeline::create(lift::host, compute_device_count + 1);
        ret.push_back(dev);
    }
#endif

    return ret;
}

static uint32 choose_batch_size(const std::vector<firepony_pipeline *>& devices)
{
    size_t min_gpu_memory = std::numeric_limits<size_t>::max();
    // max batch is 20k
    uint32 batch_size = 20000;
    uint32 num_gpus = 0;

#if ENABLE_CUDA_BACKEND
    for(const auto dev : devices)
    {
        if (dev->get_system() == firepony::cuda)
        {
            num_gpus++;
            min_gpu_memory = std::min(min_gpu_memory, dev->get_total_memory());
        }
    }

#define GBYTES(gb) (size_t(gb) * 1024u * 1024u * 1024u)
    if (min_gpu_memory <= GBYTES(11))
    {
        batch_size = 20000;
    }

    if (min_gpu_memory <= GBYTES(6))
    {
        // xxxnsubtil: confirm this
        batch_size = 18000;
    }

    if (min_gpu_memory <= GBYTES(4))
    {
        batch_size = 8000;
    }

    if (min_gpu_memory <= GBYTES(2))
    {
        // note: this will work, but is very slow compared to larger batch sizes
        // for testing such low memory GPUs you'll need to disable the memory check in enumerate_gpus
        batch_size = 8000;
    }
#undef GBYTES

#endif // if ENABLE_CUDA_BACKEND

    if (num_gpus == 0)
    {
        // CPUs strongly prefer small batches
        // override the batch size to 1000 for CPU-only runs
        batch_size = 1000;
    }

    return batch_size;
}

static void print_statistics(timer<host>& wall_clock, const pipeline_statistics& stats, int num_devices = 1)
{
    fprintf(stderr, "   blocked on io: %.4f (%.2f%%)\n", stats.io.elapsed_time, stats.io.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   read filtering: %.4f (%.2f%%)\n", stats.read_filter.elapsed_time, stats.read_filter.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   cigar expansion: %.4f (%.2f%%)\n", stats.cigar_expansion.elapsed_time, stats.cigar_expansion.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   bp filtering: %.4f (%.2f%%)\n", stats.bp_filter.elapsed_time, stats.bp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   snp filtering: %.4f (%.2f%%)\n", stats.snp_filter.elapsed_time, stats.snp_filter.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   baq: %.4f (%.2f%%)\n", stats.baq.elapsed_time, stats.baq.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "     setup: %.4f (%.2f%%)\n", stats.baq_setup.elapsed_time, stats.baq_setup.elapsed_time / stats.baq.elapsed_time * 100.0 / num_devices);
    fprintf(stderr, "     hmm: %.4f (%.2f%%)\n", stats.baq_hmm.elapsed_time, stats.baq_hmm.elapsed_time / stats.baq.elapsed_time * 100.0 / num_devices);

#if BAQ_HMM_SPLIT_PHASE
    fprintf(stderr, "       forward: %.4f (%.2f%%)\n", stats.baq_hmm_forward.elapsed_time, stats.baq_hmm_forward.elapsed_time / stats.baq.elapsed_time * 100.0 / num_devices);
    fprintf(stderr, "       backward: %.4f (%.2f%%)\n", stats.baq_hmm_backward.elapsed_time, stats.baq_hmm_backward.elapsed_time / stats.baq.elapsed_time * 100.0 / num_devices);
    fprintf(stderr, "       map: %.4f (%.2f%%)\n", stats.baq_hmm_map.elapsed_time, stats.baq_hmm_map.elapsed_time / stats.baq.elapsed_time * 100.0 / num_devices);
#endif

    fprintf(stderr, "     post: %.4f (%.2f%%)\n", stats.baq_postprocess.elapsed_time, stats.baq_postprocess.elapsed_time / stats.baq.elapsed_time * 100.0 / num_devices);
    fprintf(stderr, "   fractional error: %.4f (%.2f%%)\n", stats.fractional_error.elapsed_time, stats.fractional_error.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   covariates: %.4f (%.2f%%)\n", stats.covariates.elapsed_time, stats.covariates.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "     gather: %.4f (%.2f%%)\n", stats.covariates_gather.elapsed_time, stats.covariates_gather.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "     filter: %.4f (%.2f%%)\n", stats.covariates_filter.elapsed_time, stats.covariates_filter.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "     sort: %.4f (%.2f%%)\n", stats.covariates_sort.elapsed_time, stats.covariates_sort.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "     pack: %.4f (%.2f%%)\n", stats.covariates_pack.elapsed_time, stats.covariates_pack.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   post-processing: %.4f (%.2f%%)\n", stats.postprocessing.elapsed_time, stats.postprocessing.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   output: %.4f (%.2f%%)\n", stats.output.elapsed_time, stats.output.elapsed_time / wall_clock.elapsed_time() * 100.0 / num_devices);
    fprintf(stderr, "   batches: %lu (%.2f batches/sec)\n", stats.num_batches, stats.num_batches / wall_clock.elapsed_time());
    fprintf(stderr, "   reads: %lu (%.2fK reads/sec)\n", stats.total_reads, stats.total_reads / 1000.0 / wall_clock.elapsed_time());
}

int main(int argc, char **argv)
{
    std::vector<firepony_pipeline *> compute_devices;

    reference_file_handle *ref_h;
    variant_database_host h_dbsnp;
    bool ret;

    fprintf(stderr, "Firepony v%d.%d.%d\n", FIREPONY_VERSION_MAJOR, FIREPONY_VERSION_MINOR, FIREPONY_VERSION_REV);
    parse_command_line(argc, argv);

#if ENABLE_CUDA_BACKEND
    if (command_line_options.enable_cuda)
    {
        std::string runtime_version;
        if (cuda_runtime_init(runtime_version) == true)
        {
            fprintf(stderr, "CUDA runtime version %s\n", runtime_version.c_str());
        }
    }
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

    if (command_line_options.batch_size == uint32(-1))
    {
        command_line_options.batch_size = choose_batch_size(compute_devices);
    }

    if (command_line_options.output)
    {
        bool ret;
        ret = output_open_file(command_line_options.output);
        if (ret == false)
        {
            exit(1);
        }
    }

    if (command_line_options.verbose)
    {
        fprintf(stderr, "original command line: ");
        for(int i = 1; i < argc; i++)
        {
            fprintf(stderr, "%s ", argv[i]);
        }
        fprintf(stderr, "\n");

        fprintf(stderr, "computed command line: %s\n", canonical_command_line().c_str());
        fprintf(stderr, "\n");
    }

    timer<host> data_io;
    data_io.start();

    // load the reference genome
    ref_h = reference_file_handle::open(command_line_options.reference, compute_devices.size(), command_line_options.try_mmap);

    if (ref_h == nullptr)
    {
        fprintf(stderr, "failed to load reference %s\n", command_line_options.reference);
        exit(1);
    }

    fprintf(stderr, "loading variant database %s...", command_line_options.snp_database);
    fflush(stderr);
    ret = load_vcf(&h_dbsnp, ref_h, command_line_options.snp_database, command_line_options.try_mmap);
    fprintf(stderr, "\n");

    if (ret == false)
    {
        fprintf(stderr, "failed to load variant database %s\n", command_line_options.snp_database);
        exit(1);
    }

    data_io.stop();

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

    io_thread reader(command_line_options.input, data_mask, compute_devices.size(), ref_h);
    if (reader.start() == false)
    {
        exit(1);
    }

    for(auto d : compute_devices)
    {
        d->setup(&reader,
                 &command_line_options,
                 &reader.file.header,
                 &ref_h->sequence_data,
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

    fprintf(stderr, "wall clock times:\n");
    fprintf(stderr, " data I/O (reference + dbSNP): %f\n", data_io.elapsed_time());
    fprintf(stderr, " compute: %f\n", wall_clock.elapsed_time());
    fprintf(stderr, "\n");

    if (compute_devices.size() > 1)
        fprintf(stderr, "aggregate statistics:\n");

    print_statistics(wall_clock, aggregate_stats, compute_devices.size());

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
