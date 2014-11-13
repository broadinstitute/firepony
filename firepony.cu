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

using namespace firepony;

bool init_cuda(void)
{
    cudaError_t err;
    int device_count;

    // trigger runtime initialization
    fprintf(stderr, "loading CUDA runtime...\n");
    cudaFree(0);

    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        return false;
    }

    if (device_count == 0)
    {
        return false;
    }

    return true;
}

firepony_pipeline *choose_pipeline(void)
{
#if ENABLE_CUDA_BACKEND
    if (command_line_options.enable_cuda && init_cuda())
    {
        return firepony_pipeline::create(firepony::cuda);
    }
#endif

#if ENABLE_CPP_BACKEND
    if (command_line_options.enable_cpp)
    {
        return firepony_pipeline::create(firepony::cpp);
    }
#endif

#if ENABLE_OMP_BACKEND
    if (command_line_options.enable_omp)
    {
        return firepony_pipeline::create(firepony::omp);
    }
#endif

#if ENABLE_TBB_BACKEND
    if (command_line_options.enable_tbb)
    {
        return firepony_pipeline::create(firepony::intel_tbb);
    }
#endif

    return nullptr;
}

int main(int argc, char **argv)
{
    firepony_pipeline *pipeline;
    sequence_data_host h_reference;
    variant_database_host h_dbsnp;
    bool ret;

    parse_command_line(argc, argv);

    // choose a pipeline to use
    pipeline = choose_pipeline();
    if (pipeline == nullptr)
    {
        fprintf(stderr, "failed to initialize compute backend\n");
        exit(1);
    }

    fprintf(stderr, "compute device: %s\n", pipeline->get_name().c_str());

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

    fprintf(stderr, "%u variants\n", h_dbsnp.view.num_variants);


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

    pipeline->setup(&command_line_options,
                    &bam_thread.file.header,
                    &h_reference,
                    &h_dbsnp);

    fprintf(stderr, "processing file %s...\n", command_line_options.input);

    timer<host> wall_clock;
    timer<host> io;

    wall_clock.start();

    while(!bam_thread.done())
    {
        io.start();
        // fetch the next batch
        alignment_batch_host& h_batch = bam_thread.next_buffer();
        io.stop();

        pipeline->process_batch(&h_batch);

        if (!command_line_options.debug)
        {
            fprintf(stderr, ".");
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n");

    wall_clock.stop();

    pipeline->finish();

    auto& stats = pipeline->get_statistics();

    fprintf(stderr, "%lu reads filtered out of %lu (%f%%)\n",
            stats.filtered_reads,
            stats.total_reads,
            float(stats.filtered_reads) / float(stats.total_reads) * 100.0);

    fprintf(stderr, "computed base alignment quality for %lu reads out of %lu (%f%%)\n",
            stats.baq_reads,
            stats.total_reads - stats.filtered_reads,
            float(stats.baq_reads) / float(stats.total_reads - stats.filtered_reads) * 100.0);

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
    fprintf(stderr, "  post-processing: %.4f (%.2f%%)\n", stats.postprocessing.elapsed_time, stats.postprocessing.elapsed_time / wall_clock.elapsed_time() * 100.0);
    fprintf(stderr, "  output: %.4f (%.2f%%)\n", stats.output.elapsed_time, stats.output.elapsed_time / wall_clock.elapsed_time() * 100.0);

    bam_thread.join();

    return 0;
}
