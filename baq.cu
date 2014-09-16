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

// base alignment quality calculations (gatk: BAQ.java)

#include <nvbio/basic/types.h>
#include <nvbio/basic/dna.h>
#include <nvbio/basic/primitives.h>
#include <nvbio/basic/numbers.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include <stdlib.h>
#include <math.h>

#include "bqsr_types.h"
#include "bqsr_context.h"
#include "reference.h"
#include "bam_loader.h"

#if 0
#include "baq-cpu.h"
#endif

using namespace nvbio;

#define MAX_PHRED_SCORE 93
#define EM 0.33333333333
#define EI 0.25

#define MAX_BAND_WIDTH 7
#define MIN_BASE_QUAL 4

// all bases with q < minBaseQual are up'd to this value
#define MIN_BASE_QUAL 4

#define GAP_OPEN_PROBABILITY (pow(10.0, (-40.0)/10.))
#define GAP_EXTENSION_PROBABILITY 0.1
// ola
struct compute_hmm_windows : public bqsr_lambda_ref
{
    compute_hmm_windows(bqsr_context::view ctx,
                        const reference_genome_device::const_view reference,
                        const BAM_alignment_batch_device::const_view batch)
        : bqsr_lambda_ref(ctx, reference, batch)
    { }

    NVBIO_HOST_DEVICE void operator() (const uint32 read_index)
    {
        ushort2& out_read_window = ctx.baq.read_windows[read_index];
        uint2&   out_reference_window = ctx.baq.reference_windows[read_index];

        // grab reference sequence window in the genome
        const uint32 ref_ID = batch.alignment_sequence_IDs[read_index];
        const uint32 ref_base = reference.sequence_offsets[ref_ID];
        const uint32 ref_length = reference.sequence_offsets[ref_ID + 1] - ref_base;

        const uint32 seq_to_alignment_offset = batch.alignment_positions[read_index];

        const ushort2& read_window = ctx.cigar.read_window_clipped[read_index];
        const ushort2& read_window_no_insertions = ctx.cigar.read_window_clipped_no_insertions[read_index];
        const ushort2& reference_window = ctx.cigar.reference_window_clipped[read_index];

        const uint32 first_insertion_offset = read_window_no_insertions.x - read_window.x;
        const uint32 last_insertion_offset = read_window_no_insertions.y - read_window.y;

        const int offset = MAX_BAND_WIDTH / 2;
        uint32 readStart = reference_window.x + seq_to_alignment_offset; // always clipped

        // reference window for HMM
        uint32 start = nvbio::max(readStart - offset - first_insertion_offset, 0u);
        uint32 stop = reference_window.y + seq_to_alignment_offset + offset + last_insertion_offset;

        if (stop > ref_length)
        {
            out_read_window = make_ushort2(uint16(-1), uint16(-1));
            out_reference_window = make_uint2(uint32(-1), uint32(-1));
            return;
        }

        start += ref_base;
        stop += ref_base;

        // calcBAQFromHMM line 602 starts here
        int queryStart = read_window.x;
        int queryEnd = read_window.y;

        out_read_window = make_ushort2(queryStart, queryEnd);
        out_reference_window = make_uint2(start, stop);
    }
};

// encapsulates common state for the HMM algorithm
struct hmm_common : public bqsr_lambda_ref
{
    hmm_common(bqsr_context::view ctx,
               const reference_genome_device::const_view reference,
               const BAM_alignment_batch_device::const_view batch)
        : bqsr_lambda_ref(ctx, reference, batch)
    { }

    int bandWidth, bandWidth2;

    int referenceStart, referenceLength;
    int queryStart, queryEnd, queryLen;

    double *forwardMatrix;
    double *backwardMatrix;
    double *scalingFactors;

    double sM, sI, bM, bI;

    double m[9];

    D_PackedReference referenceBases;
    D_StreamDNA16 queryBases;
    const uint8 *inputQualities;

    uint8 *outputQualities;
    uint32 *outputState;

    template<typename Tuple>
    NVBIO_HOST_DEVICE void setup(const Tuple& hmm_index)
    {
        const uint32 read_index    = thrust::get<0>(hmm_index);
        const uint32 matrix_index  = thrust::get<1>(hmm_index);
        const uint32 scaling_index = thrust::get<2>(hmm_index);

        const BAM_CRQ_index& idx = batch.crq_index[read_index];

        // set up matrix and scaling factor pointers
        forwardMatrix = &ctx.baq.forward[matrix_index];
        backwardMatrix = &ctx.baq.backward[matrix_index];
        scalingFactors = &ctx.baq.scaling[scaling_index];

        // get the windows for the current read
        const uint2& reference_window = ctx.baq.reference_windows[read_index];
        const ushort2& read_window = ctx.baq.read_windows[read_index];

        referenceStart = reference_window.x;
        referenceLength = reference_window.y - reference_window.x + 1;

        queryStart = read_window.x;
        queryEnd = read_window.y;
        queryLen = read_window.y - read_window.x + 1;

        // compute band width
        if (referenceLength > queryLen)
            bandWidth = referenceLength;
        else
            bandWidth = queryLen;

        if (MAX_BAND_WIDTH < abs(referenceLength - queryLen))
        {
            bandWidth = abs(referenceLength - queryLen) + 3;
        }

        if (bandWidth > MAX_BAND_WIDTH)
            bandWidth = MAX_BAND_WIDTH;

        if (bandWidth < abs(referenceLength - queryLen))
        {
            bandWidth = abs(referenceLength - queryLen);
        }

        bandWidth2 = bandWidth * 2 + 1;

        // initialize transition probabilities
        sM = 1.0 / (2 * queryLen + 2);
        sI = sM;
        bM = (1 - GAP_OPEN_PROBABILITY) / referenceLength;
        bI = GAP_OPEN_PROBABILITY / referenceLength;

        m[0*3+0] = (1 - GAP_OPEN_PROBABILITY - GAP_OPEN_PROBABILITY) * (1 - sM);
        m[0*3+1] = GAP_OPEN_PROBABILITY * (1 - sM);
        m[0*3+2] = m[0*3+1];
        m[1*3+0] = (1 - GAP_EXTENSION_PROBABILITY) * (1 - sI);
        m[1*3+1] = GAP_EXTENSION_PROBABILITY * (1 - sI);
        m[1*3+2] = 0.0;
        m[2*3+0] = 1 - GAP_EXTENSION_PROBABILITY;
        m[2*3+1] = 0.0;
        m[2*3+2] = GAP_EXTENSION_PROBABILITY;

//        printf("referenceStart = %u\n", referenceStart);
//        printf("queryStart = %u queryLen = %u\n", queryStart, queryLen);

        queryBases = D_StreamDNA16(batch.reads.stream(), idx.read_start + queryStart);
        referenceBases = D_PackedReference(reference.genome_stream.m_sequence_stream, referenceStart);
        inputQualities = &batch.qualities[idx.qual_start] + queryStart;

        if (ctx.baq.qualities.size() > 0)
            outputQualities = &ctx.baq.qualities[idx.qual_start] + queryStart;
        else
            outputQualities = NULL;

        if (ctx.baq.state.size() > 0)
            outputState = &ctx.baq.state[idx.qual_start] + queryStart;
        else
            outputState = NULL;

        queryStart = 0;
    }

    NVBIO_HOST_DEVICE int set_u(const int b, const int i, const int k)
    {
        int x = i - b;
        x = x > 0 ? x : 0;
        return (k + 1 - x) * 3;
    }

    // computes a matrix offset for forwardMatrix or backwardMatrix
    NVBIO_HOST_DEVICE int off(int i, int j = 0)
    {
        return i * 6 * (2 * MAX_BAND_WIDTH + 1) + j;
    }

    // computes the required HMM matrix size for the given read length
    NVBIO_HOST_DEVICE static uint32 matrix_size(const uint32 read_len)
    {
        return (read_len + 1) * 6 * (2 * MAX_BAND_WIDTH + 1);
    }

    NVBIO_HOST_DEVICE static double qual2prob(uint8 q)
    {
        return pow(10.0, -q/10.0);
    }

    NVBIO_HOST_DEVICE static double calcEpsilon(uint8 ref, uint8 read, uint8 qualB)
    {
        double qual = qual2prob(qualB < MIN_BASE_QUAL ? MIN_BASE_QUAL : qualB);
        double e = (ref == read ? 1 - qual : qual * EM);
        return e;
    }
};

struct hmm_glocal_forward : public hmm_common
{
    hmm_glocal_forward(bqsr_context::view ctx,
                       const reference_genome_device::const_view reference,
                       const BAM_alignment_batch_device::const_view batch)
        : hmm_common(ctx, reference, batch)
    { }

    template<typename Tuple>
    NVBIO_HOST_DEVICE void operator() (const Tuple& hmm_index)
    {
        const uint32 read_index = thrust::get<0>(hmm_index);

        int i, k;

        hmm_common::setup(hmm_index);

//        printf("hmm_glocal(l_ref=%d qstart=%d, l_query=%d)\n", referenceLength, queryStart, queryLen);
//        printf("ref = { ");
//        for(int c = 0; c < referenceLength; c++)
//        {
//            printf("%c ", dna_to_char(referenceBases[c]));
//        }
//        printf("\n");
//
//        printf("que = { ");
//        for(int c = 0; c < queryLen; c++)
//        {
//            printf("%c ", iupac16_to_char(queryBases[c]));
//        }
//        printf("\n");
//
//        printf("_iqual = { % 3d % 3d % 3d % 3d % 3d ... % 3d % 3d % 3d % 3d % 3d }\n",
//                inputQualities[0], inputQualities[1], inputQualities[2], inputQualities[3], inputQualities[4],
//                inputQualities[queryLen - 5], inputQualities[queryLen - 4], inputQualities[queryLen - 3], inputQualities[queryLen - 2], inputQualities[queryLen - 1]);
//        printf("c->bw = %d, bw = %d, l_ref = %d, l_query = %d\n", maxBandWidth, bandWidth, referenceLength, queryLen);

        /*** forward ***/
        // f[0]
        forwardMatrix[off(0, set_u(bandWidth, 0, 0))] = 1.0;
        scalingFactors[0] = 1.0;
        { // f[1]
            double *fi = &forwardMatrix[off(1)];
            double sum;
            int beg = 1;
            int end = referenceLength < bandWidth + 1? referenceLength : bandWidth + 1;
            int _beg, _end;

            sum = 0.0;
            for (k = beg; k <= end; ++k)
            {
                int u;
                double e = calcEpsilon(dna_to_iupac16(referenceBases[k-1]), queryBases[queryStart], inputQualities[queryStart]);
//                printf("referenceBases[%d-1] = %c inputQualities[%d] = %d queryBases[%d] = %c -> e = %.4f\n", k, dna_to_char(referenceBases[k-1]), queryStart, inputQualities[queryStart], queryStart, iupac16_to_char(queryBases[queryStart]), e);

                u = set_u(bandWidth, 1, k);

                fi[u+0] = e * bM;
                fi[u+1] = EI * bI;

                sum += fi[u] + fi[u+1];
            }

            // rescale
            scalingFactors[1] = sum;
            _beg = set_u(bandWidth, 1, beg);
            _end = set_u(bandWidth, 1, end);
            _end += 2;

            for (int k = _beg; k <= _end; ++k)
                fi[k] /= sum;
        }

        // f[2..l_query]
        for (i = 2; i <= queryLen; ++i)
        {
            double *fi = &forwardMatrix[off(i)];
            double *fi1 = &forwardMatrix[off(i-1)];
            double sum;

            int beg = 1;
            int end = referenceLength;
            int x, _beg, _end;

            char qyi = queryBases[queryStart+i-1];

            x = i - bandWidth;
            beg = beg > x? beg : x; // band start

            x = i + bandWidth;
            end = end < x? end : x; // band end

            sum = 0.0;
            for (k = beg; k <= end; ++k)
            {
                int u, v11, v01, v10;
                double e = calcEpsilon(dna_to_iupac16(referenceBases[k-1]), qyi, inputQualities[queryStart+i-1]);
//                printf("referenceBases[%d-1] = %c inputQualities[%d+%d-1] = %d qyi = %c -> e = %.4f\n", k, dna_to_char(referenceBases[k-1]), queryStart, i, inputQualities[queryStart+i-1], iupac16_to_char(qyi), e);

                u = set_u(bandWidth, i, k);
                v11 = set_u(bandWidth, i-1, k-1);
                v10 = set_u(bandWidth, i-1, k);
                v01 = set_u(bandWidth, i, k-1);

                fi[u+0] = e * (m[0] * fi1[v11+0] + m[3] * fi1[v11+1] + m[6] * fi1[v11+2]);
                fi[u+1] = EI * (m[1] * fi1[v10+0] + m[4] * fi1[v10+1]);
                fi[u+2] = m[2] * fi[v01+0] + m[8] * fi[v01+2];

                sum += fi[u] + fi[u+1] + fi[u+2];

    //            printf("(%d,%d;%d): %.4f,%.4f,%.4f\n", i, k, u, fi[u], fi[u+1], fi[u+2]);
    //            printf(" .. u = %d v11 = %d v01 = %d v10 = %d e = %f\n", u, v11, v01, v10, e);
            }

            // rescale
            scalingFactors[i] = sum;

            _beg = set_u(bandWidth, i, beg);
            _end = set_u(bandWidth, i, end);
            _end += 2;

            for (k = _beg, sum = 1./sum; k <= _end; ++k)
                fi[k] *= sum;
        }

        { // f[l_query+1]
            double sum = 0.0;

            for (k = 1; k <= referenceLength; ++k)
            {
                int u = set_u(bandWidth, queryLen, k);

                if (u < 3 || u >= bandWidth2*3+3)
                    continue;

                sum += forwardMatrix[off(queryLen,u+0)] * sM + forwardMatrix[off(queryLen, u+1)] * sI;
            }

            scalingFactors[queryLen+1] = sum; // the last scaling factor
        }
    }
};

struct hmm_glocal_backward : public hmm_common
{
    hmm_glocal_backward(bqsr_context::view ctx,
                        const reference_genome_device::const_view reference,
                        const BAM_alignment_batch_device::const_view batch)
        : hmm_common(ctx, reference, batch)
    { }

    template<typename Tuple>
    NVBIO_HOST_DEVICE void operator() (const Tuple& hmm_index)
    {
        const uint32 read_index = thrust::get<0>(hmm_index);

        int i, k;

        hmm_common::setup(hmm_index);

        /*** backward ***/
        // b[l_query] (b[l_query+1][0]=1 and thus \tilde{b}[][]=1/s[l_query+1]; this is where s[l_query+1] comes from)
        for (k = 1; k <= referenceLength; ++k)
        {
            int u = set_u(bandWidth, queryLen, k);
            double *bi = &backwardMatrix[off(queryLen)];

            if (u < 3 || u >= bandWidth2*3+3)
                continue;

            bi[u+0] = sM / scalingFactors[queryLen] / scalingFactors[queryLen+1];
            bi[u+1] = sI / scalingFactors[queryLen] / scalingFactors[queryLen+1];
        }

        // b[l_query-1..1]
        for (i = queryLen - 1; i >= 1; --i)
        {
            int beg = 1;
            int end = referenceLength;
            int x, _beg, _end;

            double *bi = &backwardMatrix[off(i)];
            double *bi1 = &backwardMatrix[off(i+1)];
            double y = (i > 1)? 1. : 0.;

            char qyi1 = queryBases[queryStart+i];

            x = i - bandWidth;
            beg = beg > x? beg : x;

            x = i + bandWidth;
            end = end < x? end : x;

            for (k = end; k >= beg; --k)
            {
                int u, v11, v01, v10;

                u = set_u(bandWidth, i, k);
                v11 = set_u(bandWidth, i+1, k+1);
                v10 = set_u(bandWidth, i+1, k);
                v01 = set_u(bandWidth, i, k+1);

                /* const */ double e;
                if (k >= referenceLength)
                    e = 0;
                else
                    e = calcEpsilon(dna_to_iupac16(referenceBases[k]), qyi1, inputQualities[queryStart+i]) * bi1[v11];

                bi[u+0] = e * m[0] + EI * m[1] * bi1[v10+1] + m[2] * bi[v01+2]; // bi1[v11] has been folded into e.
                bi[u+1] = e * m[3] + EI * m[4] * bi1[v10+1];
                bi[u+2] = (e * m[6] + m[8] * bi[v01+2]) * y;
            }

            // rescale
            _beg = set_u(bandWidth, i, beg);
            _end = set_u(bandWidth, i, end);
            _end += 2;

            y = 1.0 / scalingFactors[i];
            for (k = _beg; k <= _end; ++k)
                bi[k] *= y;
        }

//        double pb = 0.0;
        { // b[0]
            int beg = 1;
            int end = referenceLength < bandWidth + 1? referenceLength : bandWidth + 1;

            double sum = 0.0;
            for (k = end; k >= beg; --k)
            {
                int u = set_u(bandWidth, 1, k);
                double e = calcEpsilon(dna_to_iupac16(referenceBases[k-1]), queryBases[queryStart], inputQualities[queryStart]);

                if (u < 3 || u >= bandWidth2*3+3)
                    continue;

                sum += e * backwardMatrix[off(1, u+0)] * bM + EI * backwardMatrix[off(1, u+1)] * bI;
            }

            backwardMatrix[off(0, set_u(bandWidth, 0, 0))] = sum / scalingFactors[0];
//            pb = backwardMatrix[off(0, set_u(bandWidth, 0, 0))]; // if everything works as is expected, pb == 1.0
        }
    }
};

struct hmm_glocal_map : public hmm_common
{
    hmm_glocal_map(bqsr_context::view ctx,
                   const reference_genome_device::const_view reference,
                   const BAM_alignment_batch_device::const_view batch)
        : hmm_common(ctx, reference, batch)
    { }

    template<typename Tuple>
    NVBIO_HOST_DEVICE void operator() (const Tuple& hmm_index)
    {
        const uint32 read_index = thrust::get<0>(hmm_index);

        int i, k;

        hmm_common::setup(hmm_index);

        /*** MAP ***/
        for (i = 1; i <= queryLen; ++i)
        {
            double sum = 0.0;
            double max = 0.0;

            const double *fi = &forwardMatrix[off(i)];
            const double *bi = &backwardMatrix[off(i)];

            int beg = 1;
            int end = referenceLength;
            int x, max_k = -1;

            x = i - bandWidth;
            beg = beg > x? beg : x;

            x = i + bandWidth;
            end = end < x? end : x;

            for (k = beg; k <= end; ++k)
            {
                const int u = set_u(bandWidth, i, k);
                double z = 0.0;

                z = fi[u+0] * bi[u+0];
                sum += z;
                if (z > max)
                {
                    max = z;
                    max_k = (k-1) << 2 | 0;
                }

                z = fi[u+1] * bi[u+1];
                sum += z;
                if (z > max)
                {
                    max = z;
                    max_k = (k-1) << 2 | 1;
                }
            }

            max /= sum;
            sum *= scalingFactors[i]; // if everything works as is expected, sum == 1.0

            if (outputState != NULL)
                outputState[queryStart+i-1] = max_k;

            if (outputQualities != NULL)
            {
                k = (int)(double(-4.343) * log(double(1.0) - double(max)) + double(.499)); // = 10*log10(1-max)
                outputQualities[queryStart+i-1] = (char)(k > 100? 99 : (k < MIN_BASE_QUAL ? MIN_BASE_QUAL : k));

    //            printf("outputQualities[%d]: max = %.16f l = %.4f dk = %.4f k = %d -> %d\n", i, max, l, dk, k, outputQualities[queryStart+i-1]);
            }

    //        printf("(%.4f,%.4f) (%d,%d,%d,%.4f)\n", pb, sum, (i-1), (max_k>>2), (max_k&3), max);
        }
    }
};

// functor to compute the size required for the forward/backward HMM matrix
// note that this computes the size required for *one* matrix only; we allocate the matrices on two separate vectors and use the same index for both
struct compute_hmm_matrix_size : public thrust::unary_function<uint32, uint32>, public bqsr_lambda
{
    compute_hmm_matrix_size(bqsr_context::view ctx,
                            const BAM_alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    NVBIO_HOST_DEVICE uint32 operator() (const uint32 read_index)
    {
        const BAM_CRQ_index& idx = batch.crq_index[read_index];
        return hmm_common::matrix_size(idx.read_len);
    }
};

struct compute_hmm_scaling_factor_size : public thrust::unary_function<uint32, uint32>, public bqsr_lambda
{
    compute_hmm_scaling_factor_size(bqsr_context::view ctx,
                                    const BAM_alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    NVBIO_HOST_DEVICE uint32 operator() (const uint32 read_index)
    {
        const BAM_CRQ_index& idx = batch.crq_index[read_index];
        return idx.read_len + 2;
    }
};

void baq_reads(bqsr_context *context, const reference_genome& reference, const BAM_alignment_batch_device& batch)
{
    struct baq_context& baq = context->baq;

    // compute the index and size of the HMM matrices
    baq.matrix_index.resize(context->active_read_list.size() + 1);
    // first offset is zero
    thrust::fill_n(baq.matrix_index.begin(), 1, 0);
    // do an inclusive scan to compute all offsets + the total size
    nvbio::inclusive_scan(context->active_read_list.size(),
                          thrust::make_transform_iterator(context->active_read_list.begin(),
                                                          compute_hmm_matrix_size(*context, batch)),
                          baq.matrix_index.begin() + 1,
                          thrust::plus<uint32>(),
                          context->temp_storage);

    // compute the index and size of the HMM scaling factors
    baq.scaling_index.resize(context->active_read_list.size() + 1);
    // first offset is zero
    thrust::fill_n(baq.scaling_index.begin(), 1, 0);
    nvbio::inclusive_scan(context->active_read_list.size(),
                          thrust::make_transform_iterator(context->active_read_list.begin(),
                                                          compute_hmm_scaling_factor_size(*context, batch)),
                          baq.scaling_index.begin() + 1,
                          thrust::plus<uint32>(),
                          context->temp_storage);

    // read back the last elements, which contain the size of the buffer required
    uint32 matrix_len = baq.matrix_index[context->active_read_list.size()];
    uint32 scaling_len = baq.scaling_index[context->active_read_list.size()];

//    printf("reads: %u\n", batch.num_reads);
//    printf("forward len = %u bytes = %lu\n", matrix_len, matrix_len * sizeof(double));
//    printf("expected len = %lu expected bytes = %lu\n",
//            hmm_common::matrix_size(100) * context->active_read_list.size(),
//            hmm_common::matrix_size(100) * context->active_read_list.size() * sizeof(double));
//    printf("per read matrix size = %u bytes = %lu\n", hmm_common::matrix_size(100), hmm_common::matrix_size(100) * sizeof(double));

    baq.forward.resize(matrix_len);
    baq.backward.resize(matrix_len);
    baq.scaling.resize(scaling_len);

//    printf("matrix index = [ ");
//    for(uint32 i = 0; i < 20; i++)
//    {
//        printf("%u, ", baq.matrix_index[i] + 0);
//    }
//    printf(" ... ");
//    for(uint32 i = baq.matrix_index.size() - 20; i < baq.matrix_index.size(); i++)
//    {
//        printf("%u, ", baq.matrix_index[i] + 0);
//    }
//    printf("]\n");
//    fflush(stdout);

    baq.read_windows.resize(batch.num_reads);
    baq.reference_windows.resize(batch.num_reads);

    baq.state.resize(batch.qualities.size());
    baq.qualities.resize(batch.qualities.size());

    thrust::fill(baq.state.begin(), baq.state.end(), uint32(-1));
    thrust::fill(baq.qualities.begin(), baq.qualities.end(), uint8(-1));

    // compute the alignment frames
    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     compute_hmm_windows(*context, reference.device, batch));

    // initialize matrices and scaling factors
    thrust::fill_n(baq.forward.begin(), baq.forward.size(), 0.0);
    thrust::fill_n(baq.backward.begin(), baq.backward.size(), 0.0);
    thrust::fill_n(baq.scaling.begin(), baq.scaling.size(), 0.0);

    // run the forward portion
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(context->active_read_list.begin(),
                                                                  baq.matrix_index.begin(),
                                                                  baq.scaling_index.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(context->active_read_list.end(),
                                                                  baq.matrix_index.end(),
                                                                  baq.scaling_index.end())),
                     hmm_glocal_forward(*context, reference.device, batch));

    // run the backward portion
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(context->active_read_list.begin(),
                                                                  baq.matrix_index.begin(),
                                                                  baq.scaling_index.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(context->active_read_list.end(),
                                                                  baq.matrix_index.end(),
                                                                  baq.scaling_index.end())),
                     hmm_glocal_backward(*context, reference.device, batch));

    // use the computed state to map qualities
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(context->active_read_list.begin(),
                                                                  baq.matrix_index.begin(),
                                                                  baq.scaling_index.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(context->active_read_list.end(),
                                                                  baq.matrix_index.end(),
                                                                  baq.scaling_index.end())),
                     hmm_glocal_map(*context, reference.device, batch));

#if 0
    for(uint32 i = 0; i < context->active_read_list.size(); i++)
    {
        uint32 read_index = context->active_read_list[i];

        ushort2 read_window;
        uint2 reference_window;

        cpu::frame_alignment(read_window, reference_window,
                             context, reference, batch, h_batch, read_index);
        //hengli::calcBAQFromHMM(NULL, NULL, NULL, context, reference, h_batch, read_index);
        printf("cpu frame_alignment: read = [ %u %u ] ref = [ %u %u ]\n", read_window.x, read_window.y, reference_window.x, reference_window.y);

        read_window = baq.read_windows[read_index];
        reference_window = baq.reference_windows[read_index];
        printf("gpu frame_alignment: read = [ %u %u ] ref = [ %u %u ]\n", read_window.x, read_window.y, reference_window.x, reference_window.y);
    }
#endif
}

void debug_baq(bqsr_context *context, const reference_genome& genome, const BAM_alignment_batch_host& batch, int read_index)
{
    printf("  BAQ info:\n");

    const BAM_CRQ_index& idx = batch.crq_index[read_index];

    ushort2 read_window = context->baq.read_windows[read_index];
    uint2 reference_window = context->baq.reference_windows[read_index];

    printf("    read window                 = [ %u %u ]\n", read_window.x, read_window.y);
    printf("    absolute reference window   = [ %u %u ]\n", reference_window.x, reference_window.y);
    //printf("    sequence base: %u\n", genome.sequence_offsets[batch.alignment_sequence_IDs[read_index]]);
    printf("    relative reference window   = [ %u %u ]\n",
            reference_window.x - genome.sequence_offsets[batch.alignment_sequence_IDs[read_index]],
            reference_window.y - genome.sequence_offsets[batch.alignment_sequence_IDs[read_index]]);

    printf("    BAQ state                   = [ ");
    for(uint32 i = 0; i < idx.read_len; i++)
    {
        uint32 s = context->baq.state[i];

        if (s == uint32(-1))
        {
            printf("  - ");
        } else {
            printf("% 3d ", s);
        }
    }
    printf(" ]\n");

    printf("    BAQ quals                   = [ ");
    for(uint32 i = 0; i < idx.read_len; i++)
    {
        uint8 q = context->baq.qualities[i];
        if (q == uint8(-1))
        {
            printf("  - ");
        } else {
            printf("% 3d ", q);
        }
    }
    printf(" ]\n");

    printf("\n");
}
