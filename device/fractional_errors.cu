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

#include "firepony_context.h"
#include "util.h"
#include "baq.h"

namespace firepony {

// implements GATK's BaseRecalibrator.calculateFractionalErrorArray
template <target_system system>
struct compute_fractional_errors : public lambda<system>
{
    LAMBDA_INHERIT_MEMBERS;

    const packed_vector<system, 1> error_vector;
    pointer<system, double> output_vector;

    compute_fractional_errors(firepony_context<system> ctx,
                              const alignment_batch_device<system> batch,
                              const packed_vector<system, 1> error_vector,
                              pointer<system, double> output_vector)
        : lambda<system>(ctx, batch), error_vector(error_vector), output_vector(output_vector)
    { }

    CUDA_HOST_DEVICE void calculateAndStoreErrorsInBlock(const int iii,
                                                         const int blockStartIndex,
                                                         const typename packed_vector<system, 1>::const_stream_type errorArray,
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
        auto errorArray = error_vector.stream() + (idx.read_start + read_window.x);

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

    thrust::fill(lift::backend_policy<system>::execution_policy(), frac.snp_errors.begin(), frac.snp_errors.end(), 0.0);
    thrust::fill(lift::backend_policy<system>::execution_policy(), frac.insertion_errors.begin(), frac.insertion_errors.end(), 0.0);
    thrust::fill(lift::backend_policy<system>::execution_policy(), frac.deletion_errors.begin(), frac.deletion_errors.end(), 0.0);

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

} // namespace firepony
