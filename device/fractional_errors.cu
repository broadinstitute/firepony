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

#include "firepony_context.h"
#include "util.h"
#include "baq.h"
#include "primitives/parallel.h"

namespace firepony {

// implements GATK's BaseRecalibrator.calculateFractionalErrorArray
template <target_system system>
struct compute_fractional_errors : public lambda<system>
{
    LAMBDA_INHERIT_MEMBERS;

    typename d_packed_vector_1b<system>::const_view error_vector;
    typename d_vector<system, double>::view output_vector;

    compute_fractional_errors(typename firepony_context<system>::view ctx,
                              const typename alignment_batch_device<system>::const_view batch,
                              const typename d_packed_vector_1b<system>::const_view error_vector,
                              typename d_vector<system, double>::view output_vector)
        : lambda<system>(ctx, batch), error_vector(error_vector), output_vector(output_vector)
    { }

    CUDA_HOST_DEVICE void calculateAndStoreErrorsInBlock(const int iii,
                                                         const int blockStartIndex,
                                                         const typename d_packed_vector_1b<system>::const_view errorArray,
                                                         double *fractionalErrors)
    {
        int totalErrors = 0;
        for(int jjj = max(0, blockStartIndex - 1); jjj <= iii; jjj++)
        {
            totalErrors += errorArray[jjj];
        }

        for(int jjj = max(0, blockStartIndex - 1); jjj <= iii; jjj++)
        {
            fractionalErrors[jjj] = ((double) totalErrors) / ((double)(iii - max(0, blockStartIndex - 1) + 1));
        }
    }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const ushort2& read_window = ctx.cigar.read_window_clipped[read_index];
        const uint8 *baqArray = &ctx.baq.qualities[idx.qual_start] + read_window.x;

        // offset the error array by read_window.x to simulate hard clipping of soft clipped bases
        auto errorArray = error_vector + (idx.read_start + read_window.x);

        constexpr int BLOCK_START_UNSET = -1;

        // offset into output_vector to simulate hard clipping of soft clipped bases
        double *fractionalErrors = &output_vector[idx.qual_start] + read_window.x;
        const int fractionalErrors_length = read_window.y - read_window.x + 1;

        bool inBlock = false;
        int blockStartIndex = BLOCK_START_UNSET;
        int iii;

        for(iii = 0; iii < fractionalErrors_length; iii++)
        {
            if (baqArray[iii] == NO_BAQ_UNCERTAINTY)
            {
                if (!inBlock)
                {
                    fractionalErrors[iii] = (double) errorArray[iii];
                } else {
                    calculateAndStoreErrorsInBlock(iii, blockStartIndex, errorArray, fractionalErrors);
                    inBlock = false; // reset state variables
                    blockStartIndex = BLOCK_START_UNSET;
                }
            } else {
                inBlock = true;
                if (blockStartIndex == BLOCK_START_UNSET)
                {
                    blockStartIndex = iii;
                }
            }
        }

        if (inBlock)
        {
            calculateAndStoreErrorsInBlock(iii-1, blockStartIndex, errorArray, fractionalErrors);
        }
    }
};

template <target_system system>
void build_fractional_error_arrays(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    auto& frac = context.fractional_error;

    frac.snp_errors.resize(context.baq.qualities.size());
    frac.insertion_errors.resize(context.baq.qualities.size());
    frac.deletion_errors.resize(context.baq.qualities.size());

    thrust::fill(frac.snp_errors.begin(), frac.snp_errors.end(), 0.0);
    thrust::fill(frac.insertion_errors.begin(), frac.insertion_errors.end(), 0.0);
    thrust::fill(frac.deletion_errors.begin(), frac.deletion_errors.end(), 0.0);

    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_fractional_errors<system>(context, batch.device, context.cigar.is_snp, frac.snp_errors));
    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_fractional_errors<system>(context, batch.device, context.cigar.is_insertion, frac.insertion_errors));
    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_fractional_errors<system>(context, batch.device, context.cigar.is_deletion, frac.deletion_errors));
}
INSTANTIATE(build_fractional_error_arrays);

template <target_system system>
void debug_fractional_error_arrays(firepony_context<system>& context, const alignment_batch<system>& batch, int read_index)
{
    const alignment_batch_host& h_batch = *batch.host;

    fprintf(stderr, "  fractional error arrays:\n");

    const CRQ_index idx = h_batch.crq_index(read_index);

    fprintf(stderr, "    snp                        = [ ");
    for(uint32 i = idx.qual_start; i < idx.qual_start + idx.qual_len; i++)
    {
        double err = context.fractional_error.snp_errors[i];
        fprintf(stderr, " %.1f", err);
    }
    fprintf(stderr, " ]\n");

    fprintf(stderr, "    ins                        = [ ");
    for(uint32 i = idx.qual_start; i < idx.qual_start + idx.qual_len; i++)
    {
        double err = context.fractional_error.insertion_errors[i];
        fprintf(stderr, " %.1f", err);
    }
    fprintf(stderr, " ]\n");

    fprintf(stderr, "    del                        = [ ");
    for(uint32 i = idx.qual_start; i < idx.qual_start + idx.qual_len; i++)
    {
        double err = context.fractional_error.deletion_errors[i];
        fprintf(stderr, " %.1f", err);
    }
    fprintf(stderr, " ]\n");

    fprintf(stderr, "\n");
}
INSTANTIATE(debug_fractional_error_arrays);

} // namespace firepony

