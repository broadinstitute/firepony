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

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include <stdlib.h>
#include <math.h>

#include "device_types.h"
#include "alignment_data_device.h"
#include "sequence_data_device.h"
#include "firepony_context.h"

#include "primitives/util.h"
#include "primitives/parallel.h"
#include "from_nvbio/dna.h"
#include "from_nvbio/alphabet.h"

namespace firepony {

#define MAX_PHRED_SCORE 93
#define EM 0.33333333333
#define EI 0.25

#define MAX_BAND_WIDTH 7
#define MIN_BASE_QUAL 4

// all bases with q < minBaseQual are up'd to this value
#define MIN_BASE_QUAL 4

#define GAP_OPEN_PROBABILITY (pow(10.0, (-40.0)/10.))
#define GAP_EXTENSION_PROBABILITY 0.1

// maximum read size for the lmem kernel
#define LMEM_MAX_READ_LEN 151
#define LMEM_MAT_SIZE ((LMEM_MAX_READ_LEN + 1) * 3 * (2 * MAX_BAND_WIDTH + 1))

template <target_system system>
struct compute_hmm_windows : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        uint2&   out_reference_window = ctx.baq.reference_windows[read_index];

        // grab reference sequence window in the genome
        const uint32 ref_ID = batch.chromosome[read_index];
        const uint32 ref_base = ctx.reference.sequence_bp_start[ref_ID];
        const uint32 ref_length = ctx.reference.sequence_bp_len[ref_ID];

        const uint32 seq_to_alignment_offset = batch.alignment_start[read_index];

        const ushort2& read_window = ctx.cigar.read_window_clipped[read_index];
        const ushort2& read_window_no_insertions = ctx.cigar.read_window_clipped_no_insertions[read_index];
        const ushort2& reference_window = ctx.cigar.reference_window_clipped[read_index];

        const uint32 first_insertion_offset = read_window_no_insertions.x - read_window.x;
        const uint32 last_insertion_offset = read_window_no_insertions.y - read_window.y;

        const int offset = MAX_BAND_WIDTH / 2;
        uint32 readStart = reference_window.x + seq_to_alignment_offset; // always clipped

        // reference window for HMM
        uint32 start = max(readStart - offset - first_insertion_offset, 0u);
        uint32 stop = reference_window.y + seq_to_alignment_offset + offset + last_insertion_offset;

        if (stop > ref_length)
        {
            out_reference_window = make_uint2(uint32(-1), uint32(-1));
            return;
        }

        start += ref_base;
        stop += ref_base;

        out_reference_window = make_uint2(start, stop);
    }
};

// runs the entire BAQ algorithm in a single kernel, storing forward and backward matrices in local memory
template <target_system system>
struct hmm_glocal_lmem : public lambda<system>
{
    typename d_vector<system, uint32>::view baq_state;

    hmm_glocal_lmem(typename firepony_context<system>::view ctx,
                    const typename alignment_batch_device<system>::const_view batch,
                    typename d_vector<system, uint32>::view baq_state)
        : lambda<system>(ctx, batch), baq_state(baq_state)
    { }

    int bandWidth, bandWidth2;

    int referenceStart, referenceLength;
    int queryStart, queryEnd, queryLen;

    double *scalingFactors;

    double sM, sI, bM, bI;

    double m[9];

    d_stream_dna16<system> referenceBases;
    d_stream_dna16<system> queryBases;
    const uint8 *inputQualities;

    uint8 *outputQualities;
    uint32 *outputState;

    template<typename Tuple>
    CUDA_HOST_DEVICE void setup(const Tuple& hmm_index)
    {
        auto& ctx = this->ctx;
        auto& batch = this->batch;

        const uint32 read_index    = thrust::get<0>(hmm_index);
        const uint32 scaling_index = thrust::get<2>(hmm_index);

        const CRQ_index idx = batch.crq_index(read_index);

        // set up scaling factor pointers
        scalingFactors = &ctx.baq.scaling[scaling_index];

        // get the windows for the current read
        const uint2& reference_window = ctx.baq.reference_windows[read_index];
        const ushort2& read_window = ctx.cigar.read_window_clipped[read_index];

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

//        fprintf(stderr, "referenceStart = %u\n", referenceStart);
//        fprintf(stderr, "queryStart = %u queryLen = %u\n", queryStart, queryLen);

        queryBases = batch.reads + idx.read_start + queryStart;
        referenceBases = ctx.reference.bases + referenceStart;
        inputQualities = &batch.qualities[idx.qual_start] + queryStart;

        if (ctx.baq.qualities.size() > 0)
            outputQualities = &ctx.baq.qualities[idx.qual_start] + queryStart;
        else
            outputQualities = NULL;

        if (baq_state.size() > 0)
            outputState = &baq_state[idx.qual_start] + queryStart;
        else
            outputState = NULL;

        queryStart = 0;
    }

    CUDA_HOST_DEVICE int set_u(const int b, const int i, const int k)
    {
        int x = i - b;
        x = x > 0 ? x : 0;
        return (k + 1 - x) * 3;
    }

    // computes a matrix offset for forwardMatrix or backwardMatrix
    CUDA_HOST_DEVICE int off(int i, int j = 0)
    {
        return i * 3 * (2 * MAX_BAND_WIDTH + 1) + j;
    }

    // computes the required HMM matrix size for the given read length
    CUDA_HOST_DEVICE static uint32 matrix_size(const uint32 read_len)
    {
        return (read_len + 1) * 3 * (2 * MAX_BAND_WIDTH + 1);
    }

    CUDA_HOST_DEVICE static double qual2prob(uint8 q)
    {
        return pow(10.0, -q/10.0);
    }

    CUDA_HOST_DEVICE static double calcEpsilon(uint8 ref, uint8 read, uint8 qualB)
    {
        if (ref == from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::N ||
            read == from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::N)
        {
            return 1.0;
        }

        double qual = qual2prob(qualB < MIN_BASE_QUAL ? MIN_BASE_QUAL : qualB);
        double e = (ref == read ? 1 - qual : qual * EM);
        return e;
    }

    template<typename Tuple>
    CUDA_HOST_DEVICE void operator() (const Tuple& hmm_index)
    {
        int i, k;

        double forwardMatrix[LMEM_MAT_SIZE];
        double backwardMatrix[LMEM_MAT_SIZE];

        memset(forwardMatrix, 0, sizeof(double) * LMEM_MAT_SIZE);
        memset(backwardMatrix, 0, sizeof(double) * LMEM_MAT_SIZE);

        setup(hmm_index);

//        const uint32 read_index    = thrust::get<0>(hmm_index);
//        fprintf(stderr, "read %d: hmm_glocal(l_ref=%d qstart=%d, l_query=%d)\n", read_index, referenceLength, queryStart, queryLen);
//        fprintf(stderr, "read %d: ref = { ", read_index);
//        for(int c = 0; c < referenceLength; c++)
//        {
//            fprintf(stderr, "%c ", from_nvbio::iupac16_to_char(referenceBases[c]));
//        }
//        fprintf(stderr, "\n");
//
//        fprintf(stderr, "read %d: que = { ", read_index);
//        for(int c = 0; c < queryLen; c++)
//        {
//            fprintf(stderr, "%c ", from_nvbio::iupac16_to_char(queryBases[c]));
//        }
//        fprintf(stderr, "\n");

//        fprintf(stderr, "read %d: _iqual = { % 3d % 3d % 3d % 3d % 3d ... % 3d % 3d % 3d % 3d % 3d }\n", read_index,
//                inputQualities[0], inputQualities[1], inputQualities[2], inputQualities[3], inputQualities[4],
//                inputQualities[queryLen - 5], inputQualities[queryLen - 4], inputQualities[queryLen - 3], inputQualities[queryLen - 2], inputQualities[queryLen - 1]);
//        fprintf(stderr, "read %d: c->bw = %d, bw = %d, l_ref = %d, l_query = %d\n", read_index, MAX_BAND_WIDTH, bandWidth, referenceLength, queryLen);

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
                double e = calcEpsilon(referenceBases[k-1], queryBases[queryStart], inputQualities[queryStart]);
//                fprintf(stderr, "read %d: referenceBases[%d-1] = %c inputQualities[%d] = %d queryBases[%d] = %c -> e = %.4f\n",
//                        read_index,
//                        k,
//                        from_nvbio::iupac16_to_char(referenceBases[k-1]),
//                        queryStart,
//                        inputQualities[queryStart],
//                        queryStart,
//                        from_nvbio::iupac16_to_char(queryBases[queryStart]), e);

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
                double e = calcEpsilon(referenceBases[k-1], qyi, inputQualities[queryStart+i-1]);
//                fprintf(stderr, "read %d: referenceBases[%d-1] = %c inputQualities[%d+%d-1] = %d qyi = %c -> e = %.4f\n",
//                        read_index,
//                        k,
//                        from_nvbio::iupac16_to_char(referenceBases[k-1]),
//                        queryStart,
//                        i,
//                        inputQualities[queryStart+i-1],
//                        from_nvbio::iupac16_to_char(qyi), e);

                u = set_u(bandWidth, i, k);
                v11 = set_u(bandWidth, i-1, k-1);
                v10 = set_u(bandWidth, i-1, k);
                v01 = set_u(bandWidth, i, k-1);

                fi[u+0] = e * (m[0] * fi1[v11+0] + m[3] * fi1[v11+1] + m[6] * fi1[v11+2]);
                fi[u+1] = EI * (m[1] * fi1[v10+0] + m[4] * fi1[v10+1]);
                fi[u+2] = m[2] * fi[v01+0] + m[8] * fi[v01+2];

                sum += fi[u] + fi[u+1] + fi[u+2];

    //            fprintf(stderr, "(%d,%d;%d): %.4f,%.4f,%.4f\n", i, k, u, fi[u], fi[u+1], fi[u+2]);
    //            fprintf(stderr, " .. u = %d v11 = %d v01 = %d v10 = %d e = %f\n", u, v11, v01, v10, e);
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
                    e = calcEpsilon(referenceBases[k], qyi1, inputQualities[queryStart+i]) * bi1[v11];

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
                double e = calcEpsilon(referenceBases[k-1], queryBases[queryStart], inputQualities[queryStart]);

                if (u < 3 || u >= bandWidth2*3+3)
                    continue;

                sum += e * backwardMatrix[off(1, u+0)] * bM + EI * backwardMatrix[off(1, u+1)] * bI;
            }

            backwardMatrix[off(0, set_u(bandWidth, 0, 0))] = sum / scalingFactors[0];
//            pb = backwardMatrix[off(0, set_u(bandWidth, 0, 0))]; // if everything works as is expected, pb == 1.0
        }

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

//            if (outputState != NULL)
                outputState[queryStart+i-1] = max_k;

//            if (outputQualities != NULL)
            {
                k = (int)(double(-4.343) * log(double(1.0) - double(max)) + double(.499)); // = 10*log10(1-max)
                outputQualities[queryStart+i-1] = (char)(k > 100? 99 : (k < MIN_BASE_QUAL ? MIN_BASE_QUAL : k));

//                fprintf(stderr, "read %d: outputQualities[%d]: max = %.16f k = %d -> %d\n", read_index, queryStart+i-1, max, k, outputQualities[queryStart+i-1]);
            }

    //        fprintf(stderr, "(%.4f,%.4f) (%d,%d,%d,%.4f)\n", pb, sum, (i-1), (max_k>>2), (max_k&3), max);
        }
    }
};

template <target_system system>
struct hmm_glocal : public lambda<system>
{
    typename vector<system, uint32>::view baq_state;

    hmm_glocal(typename firepony_context<system>::view ctx,
               const typename alignment_batch_device<system>::const_view batch,
               typename vector<system, uint32>::view baq_state)
        : lambda<system>(ctx, batch), baq_state(baq_state)
    { }

    int bandWidth, bandWidth2;

    int referenceStart, referenceLength;
    int queryStart, queryEnd, queryLen;

    double *forwardMatrix;
    double *backwardMatrix;
    double *scalingFactors;

    double sM, sI, bM, bI;

    double m[9];

    d_stream_dna16<system> referenceBases;
    d_stream_dna16<system> queryBases;
    const uint8 *inputQualities;

    uint8 *outputQualities;
    uint32 *outputState;

    template<typename Tuple>
    CUDA_HOST_DEVICE void setup(const Tuple& hmm_index)
    {
        auto& ctx = this->ctx;
        auto& batch = this->batch;

        const uint32 read_index    = thrust::get<0>(hmm_index);
        const uint32 matrix_index  = thrust::get<1>(hmm_index);
        const uint32 scaling_index = thrust::get<2>(hmm_index);

        const CRQ_index idx = batch.crq_index(read_index);

        // set up matrix and scaling factor pointers
        forwardMatrix = &ctx.baq.forward[matrix_index];
        backwardMatrix = &ctx.baq.backward[matrix_index];
        scalingFactors = &ctx.baq.scaling[scaling_index];

        // get the windows for the current read
        const uint2& reference_window = ctx.baq.reference_windows[read_index];
        const ushort2& read_window = ctx.cigar.read_window_clipped[read_index];

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

//        fprintf(stderr, "referenceStart = %u\n", referenceStart);
//        fprintf(stderr, "queryStart = %u queryLen = %u\n", queryStart, queryLen);

        queryBases = batch.reads + idx.read_start + queryStart;
        referenceBases = ctx.reference.bases + referenceStart;
        inputQualities = &batch.qualities[idx.qual_start] + queryStart;

        if (ctx.baq.qualities.size() > 0)
            outputQualities = &ctx.baq.qualities[idx.qual_start] + queryStart;
        else
            outputQualities = NULL;

        if (baq_state.size() > 0)
            outputState = &baq_state[idx.qual_start] + queryStart;
        else
            outputState = NULL;

        queryStart = 0;
    }

    CUDA_HOST_DEVICE int set_u(const int b, const int i, const int k)
    {
        int x = i - b;
        x = x > 0 ? x : 0;
        return (k + 1 - x) * 3;
    }

    // computes a matrix offset for forwardMatrix or backwardMatrix
    CUDA_HOST_DEVICE int off(int i, int j = 0)
    {
        return i * 6 * (2 * MAX_BAND_WIDTH + 1) + j;
    }

    // computes the required HMM matrix size for the given read length
    CUDA_HOST_DEVICE static uint32 matrix_size(const uint32 read_len)
    {
        return (read_len + 1) * 6 * (2 * MAX_BAND_WIDTH + 1);
    }

    CUDA_HOST_DEVICE static double qual2prob(uint8 q)
    {
        return pow(10.0, -q/10.0);
    }

    CUDA_HOST_DEVICE static double calcEpsilon(uint8 ref, uint8 read, uint8 qualB)
    {
        if (ref == from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::N ||
            read == from_nvbio::AlphabetTraits<from_nvbio::DNA_IUPAC>::N)
        {
            return 1.0;
        }

        double qual = qual2prob(qualB < MIN_BASE_QUAL ? MIN_BASE_QUAL : qualB);
        double e = (ref == read ? 1 - qual : qual * EM);
        return e;
    }

    template<typename Tuple>
    CUDA_HOST_DEVICE void operator() (const Tuple& hmm_index)
    {
        int i, k;

        setup(hmm_index);

//        const uint32 read_index    = thrust::get<0>(hmm_index);
//        fprintf(stderr, "read %d: hmm_glocal(l_ref=%d qstart=%d, l_query=%d)\n", read_index, referenceLength, queryStart, queryLen);
//        fprintf(stderr, "read %d: ref = { ", read_index);
//        for(int c = 0; c < referenceLength; c++)
//        {
//            fprintf(stderr, "%c ", from_nvbio::iupac16_to_char(referenceBases[c]));
//        }
//        fprintf(stderr, "\n");
//
//        fprintf(stderr, "read %d: que = { ", read_index);
//        for(int c = 0; c < queryLen; c++)
//        {
//            fprintf(stderr, "%c ", from_nvbio::iupac16_to_char(queryBases[c]));
//        }
//        fprintf(stderr, "\n");

//        fprintf(stderr, "read %d: _iqual = { % 3d % 3d % 3d % 3d % 3d ... % 3d % 3d % 3d % 3d % 3d }\n", read_index,
//                inputQualities[0], inputQualities[1], inputQualities[2], inputQualities[3], inputQualities[4],
//                inputQualities[queryLen - 5], inputQualities[queryLen - 4], inputQualities[queryLen - 3], inputQualities[queryLen - 2], inputQualities[queryLen - 1]);
//        fprintf(stderr, "read %d: c->bw = %d, bw = %d, l_ref = %d, l_query = %d\n", read_index, MAX_BAND_WIDTH, bandWidth, referenceLength, queryLen);

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
                double e = calcEpsilon(referenceBases[k-1], queryBases[queryStart], inputQualities[queryStart]);
//                fprintf(stderr, "read %d: referenceBases[%d-1] = %c inputQualities[%d] = %d queryBases[%d] = %c -> e = %.4f\n",
//                        read_index,
//                        k,
//                        from_nvbio::iupac16_to_char(referenceBases[k-1]),
//                        queryStart,
//                        inputQualities[queryStart],
//                        queryStart,
//                        from_nvbio::iupac16_to_char(queryBases[queryStart]), e);

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
                double e = calcEpsilon(referenceBases[k-1], qyi, inputQualities[queryStart+i-1]);
//                fprintf(stderr, "read %d: referenceBases[%d-1] = %c inputQualities[%d+%d-1] = %d qyi = %c -> e = %.4f\n",
//                        read_index,
//                        k,
//                        from_nvbio::iupac16_to_char(referenceBases[k-1]),
//                        queryStart,
//                        i,
//                        inputQualities[queryStart+i-1],
//                        from_nvbio::iupac16_to_char(qyi), e);

                u = set_u(bandWidth, i, k);
                v11 = set_u(bandWidth, i-1, k-1);
                v10 = set_u(bandWidth, i-1, k);
                v01 = set_u(bandWidth, i, k-1);

                fi[u+0] = e * (m[0] * fi1[v11+0] + m[3] * fi1[v11+1] + m[6] * fi1[v11+2]);
                fi[u+1] = EI * (m[1] * fi1[v10+0] + m[4] * fi1[v10+1]);
                fi[u+2] = m[2] * fi[v01+0] + m[8] * fi[v01+2];

                sum += fi[u] + fi[u+1] + fi[u+2];

    //            fprintf(stderr, "(%d,%d;%d): %.4f,%.4f,%.4f\n", i, k, u, fi[u], fi[u+1], fi[u+2]);
    //            fprintf(stderr, " .. u = %d v11 = %d v01 = %d v10 = %d e = %f\n", u, v11, v01, v10, e);
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
                    e = calcEpsilon(referenceBases[k], qyi1, inputQualities[queryStart+i]) * bi1[v11];

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
                double e = calcEpsilon(referenceBases[k-1], queryBases[queryStart], inputQualities[queryStart]);

                if (u < 3 || u >= bandWidth2*3+3)
                    continue;

                sum += e * backwardMatrix[off(1, u+0)] * bM + EI * backwardMatrix[off(1, u+1)] * bI;
            }

            backwardMatrix[off(0, set_u(bandWidth, 0, 0))] = sum / scalingFactors[0];
//            pb = backwardMatrix[off(0, set_u(bandWidth, 0, 0))]; // if everything works as is expected, pb == 1.0
        }

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

//                fprintf(stderr, "read %d: outputQualities[%d]: max = %.16f k = %d -> %d\n", read_index, queryStart+i-1, max, k, outputQualities[queryStart+i-1]);
            }

    //        fprintf(stderr, "(%.4f,%.4f) (%d,%d,%d,%.4f)\n", pb, sum, (i-1), (max_k>>2), (max_k&3), max);
        }
    }
};

// functor to compute the size required for the forward/backward HMM matrix
// note that this computes the size required for *one* matrix only; we allocate the matrices on two separate vectors and use the same index for both
template <target_system system>
struct compute_hmm_matrix_size : public thrust::unary_function<uint32, uint32>, public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE uint32 operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        return hmm_glocal<system>::matrix_size(idx.read_len);
    }
};

template <target_system system>
struct compute_hmm_scaling_factor_size : public thrust::unary_function<uint32, uint32>, public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE uint32 operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        return idx.read_len + 2;
    }
};

template <target_system system>
struct read_needs_baq : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE bool operator() (const uint32 read_index)
    {
        if (ctx.cigar.num_errors[read_index] != 0)
            return true;

        return false;
    }
};

template <target_system system>
struct read_flat_baq : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        if (ctx.baq.qualities.size() == 0)
        {
            return;
        }

        if (ctx.cigar.num_errors[read_index] != 0)
        {
            // reads with errors will have BAQ computed explicitly
            return;
        }

        const CRQ_index idx = batch.crq_index(read_index);
        const ushort2& read_window = ctx.cigar.read_window_clipped[read_index];
        const uint32 queryStart = read_window.x;
        const uint32 queryLen = read_window.y - read_window.x + 1;
        uint8 *outputQualities = &ctx.baq.qualities[idx.qual_start] + queryStart;

        memset(outputQualities, NO_BAQ_UNCERTAINTY, queryLen);
    }
};

// bottom half of BAQ.calcBAQFromHMM in GATK
template <target_system system>
struct cap_baq_qualities : public lambda<system>
{
    typename vector<system, uint32>::view baq_state;

    cap_baq_qualities(typename firepony_context<system>::view ctx,
                      const typename alignment_batch_device<system>::const_view batch,
                      typename vector<system, uint32>::view baq_state)
        : lambda<system>(ctx, batch), baq_state(baq_state)
    { }

    CUDA_HOST_DEVICE bool stateIsIndel(uint32 state)
    {
        return (state & 3) != 0;
    }

    CUDA_HOST_DEVICE uint32 stateAlignedPosition(uint32 state)
    {
        return state >> 2;
    }

    CUDA_HOST_DEVICE uint8 capBaseByBAQ(uint8 oq, uint8 bq, uint32 state, uint32 expectedPos)
    {
        uint8 b;
        bool isIndel = stateIsIndel(state);
        uint32 pos = stateAlignedPosition(state);

        if (isIndel || pos != expectedPos) // we are an indel or we don't algin to our best current position
        {
            b = MIN_BASE_QUAL; // just take b = minBaseQuality
        } else {
            b = min(bq, oq);
        }

        return b;
    }

    // xxxnsubtil: this could use some cleanup
    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        auto& ctx = this->ctx;
        auto& batch = this->batch;

        const CRQ_index idx = batch.crq_index(read_index);

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        const ushort2& read_window = ctx.cigar.read_window_clipped[read_index];
        const ushort2& read_window_no_insertions = ctx.cigar.read_window_clipped_no_insertions[read_index];
        const ushort2& reference_window = ctx.cigar.reference_window_clipped[read_index];

        const uint32 seq_to_alignment_offset = batch.alignment_start[read_index];
        const uint32 first_insertion_offset = read_window_no_insertions.x - read_window.x;

        const int offset = MAX_BAND_WIDTH / 2;

        const uint32 readStart = reference_window.x + seq_to_alignment_offset;
        const uint32 start = max(readStart - offset - first_insertion_offset, 0u);

        const int refOffset = (int)(start - readStart);

        uint32 readI = 0;
        uint32 refI = 0;
        uint32 current_op_offset = 0;
        uint32 numD = 0;

        // scan for the start of the baq region
        uint32 i;
        for(i = 0; i < cigar_end - cigar_start; i++)
        {
            const uint16 read_bp_idx = ctx.cigar.cigar_event_read_coordinates[cigar_start + i];
            if (read_bp_idx >= read_window.x)
                break;
        }

        const uint32 baq_start = i;

        for(; i < cigar_end - cigar_start; i++)
        {
            const uint16 read_bp_idx = ctx.cigar.cigar_event_read_coordinates[cigar_start + i];
            const uint32 qual_idx = idx.qual_start + read_bp_idx;

            switch(ctx.cigar.cigar_events[i + cigar_start])
            {
            case cigar_event::S:
                refI++;
                current_op_offset = 0;
                break;

            case cigar_event::I:
                ctx.baq.qualities[qual_idx] = batch.qualities[qual_idx];
                readI++;
                current_op_offset = 0;
                break;

            case cigar_event::D:
                refI++;
                numD++;
                current_op_offset = 0;
                break;

            case cigar_event::M:
                const uint32 expectedPos = refI - refOffset + (i - numD - baq_start - readI);
                ctx.baq.qualities[qual_idx] = capBaseByBAQ(batch.qualities[idx.qual_start + read_bp_idx],
                                                           ctx.baq.qualities[idx.qual_start + read_bp_idx],
                                                           baq_state[idx.qual_start + read_bp_idx],
                                                           expectedPos);
                readI++;
                refI++;
                current_op_offset++;

                break;
            }
        }
    }
};

// transforms BAQ scores the same way as GATK's encodeBQTag
template <target_system system>
struct recode_baq_qualities : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        for(uint32 i = idx.qual_start; i < idx.qual_start + idx.qual_len; i++)
        {
            const uint8 baq_i = ctx.baq.qualities[i];
            if (baq_i == uint8(-1))
            {
                continue;
            }

            const uint8 bq = batch.qualities[i] + 64;
            const uint8 tag = bq - baq_i;
            ctx.baq.qualities[i] = tag;
        }
    }
};

template <target_system system>
void baq_reads(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    struct baq_context<system>& baq = context.baq;
    vector<system, uint32>& active_baq_read_list = context.temp_u32;
    vector<system, uint32>& baq_state = context.temp_u32_2;

    // check if we can use the lmem kernel
    const bool baq_use_lmem = (batch.host->max_read_size <= LMEM_MAX_READ_LEN);
    static bool baq_lmem_warning_printed = false;

    if (!baq_use_lmem && !baq_lmem_warning_printed)
    {
        fprintf(stderr, "WARNING: read size exceeds LMEM_MAX_READ_LEN, using fallback path for BAQ\n");
        baq_lmem_warning_printed = true;
    }

    timer<system> baq_setup, baq_hmm, baq_postprocess;

    baq_setup.start();

    uint32 num_active;

    // collect the reads that we need to compute BAQ for
    active_baq_read_list.resize(context.active_read_list.size());

    num_active = parallel<system>::copy_if(context.active_read_list.begin(),
                                           context.active_read_list.size(),
                                           active_baq_read_list.begin(),
                                           read_needs_baq<system>(context, batch.device),
                                           context.temp_storage);

    active_baq_read_list.resize(num_active);

    if (!baq_use_lmem)
    {
        // compute the index and size of the HMM matrices
        baq.matrix_index.resize(num_active + 1);
        // first offset is zero
        thrust::fill_n(baq.matrix_index.begin(), 1, 0);
        // do an inclusive scan to compute all offsets + the total size
        parallel<system>::inclusive_scan(thrust::make_transform_iterator(active_baq_read_list.begin(),
                                                                         compute_hmm_matrix_size<system>(context, batch.device)),
                                         num_active,
                                         baq.matrix_index.begin() + 1,
                                         thrust::plus<uint32>());

        uint32 matrix_len = baq.matrix_index[num_active];

        baq.forward.resize(matrix_len);
        baq.backward.resize(matrix_len);
    }

    // compute the index and size of the HMM scaling factors
    baq.scaling_index.resize(num_active + 1);
    // first offset is zero
    thrust::fill_n(baq.scaling_index.begin(), 1, 0);
    parallel<system>::inclusive_scan(thrust::make_transform_iterator(active_baq_read_list.begin(),
                                                                     compute_hmm_scaling_factor_size<system>(context, batch.device)),
                                     num_active,
                                     baq.scaling_index.begin() + 1,
                                     thrust::plus<uint32>());

    uint32 scaling_len = baq.scaling_index[num_active];
    baq.scaling.resize(scaling_len);

//    fprintf(stderr, "reads: %u\n", batch.num_reads);
//    fprintf(stderr, "forward len = %u bytes = %lu\n", matrix_len, matrix_len * sizeof(double));
//    fprintf(stderr, "expected len = %lu expected bytes = %lu\n",
//            hmm_common::matrix_size(100) * context.active_read_list.size(),
//            hmm_common::matrix_size(100) * context.active_read_list.size() * sizeof(double));
//    fprintf(stderr, "per read matrix size = %u bytes = %lu\n", hmm_common::matrix_size(100), hmm_common::matrix_size(100) * sizeof(double));
//    fprintf(stderr, "matrix index = [ ");
//    for(uint32 i = 0; i < 20; i++)
//    {
//        fprintf(stderr, "%u, ", baq.matrix_index[i] + 0);
//    }
//    fprintf(stderr, " ... ");
//    for(uint32 i = baq.matrix_index.size() - 20; i < baq.matrix_index.size(); i++)
//    {
//        fprintf(stderr, "%u, ", baq.matrix_index[i] + 0);
//    }
//    fprintf(stderr, "]\n");
//    fflush(stdout);

    baq.reference_windows.resize(batch.device.num_reads);

    baq_state.resize(batch.device.qualities.size());
    baq.qualities.resize(batch.device.qualities.size());

    thrust::fill(baq_state.begin(), baq_state.end(), uint32(-1));
    thrust::fill(baq.qualities.begin(), baq.qualities.end(), uint8(-1));

    // compute the alignment frames
    // note: this is used both for real BAQ and flat BAQ, so we use the full active read list
    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_hmm_windows<system>(context, batch.device));

    // initialize matrices and scaling factors
    if (!baq_use_lmem)
    {
        thrust::fill_n(baq.forward.begin(), baq.forward.size(), 0.0);
        thrust::fill_n(baq.backward.begin(), baq.backward.size(), 0.0);
    }

    thrust::fill_n(baq.scaling.begin(), baq.scaling.size(), 0.0);

    baq_setup.stop();

    baq_hmm.start();

    if (baq_use_lmem)
    {
        // fast path: use local memory for the matrices
        parallel<system>::for_each(thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.begin(),
                                                                      baq.matrix_index.begin(),
                                                                      baq.scaling_index.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.end(),
                                                                      baq.matrix_index.end(),
                                                                      baq.scaling_index.end())),
                         hmm_glocal_lmem<system>(context, batch.device, baq_state));
    } else {
        // slow path: store the matrices in global memory
        parallel<system>::for_each(thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.begin(),
                                                                      baq.matrix_index.begin(),
                                                                      baq.scaling_index.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.end(),
                                                                      baq.matrix_index.end(),
                                                                      baq.scaling_index.end())),
                         hmm_glocal<system>(context, batch.device, baq_state));
    }

    baq_hmm.stop();

    baq_postprocess.start();

    // for any reads that we did *not* compute a BAQ, mark the base pairs as having no BAQ uncertainty
    parallel<system>::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     read_flat_baq<system>(context, batch.device));

    // transform quality scores
    parallel<system>::for_each(active_baq_read_list.begin(),
                     active_baq_read_list.end(),
                     cap_baq_qualities<system>(context, batch.device, baq_state));

    parallel<system>::for_each(active_baq_read_list.begin(),
                     active_baq_read_list.end(),
                     recode_baq_qualities<system>(context, batch.device));

    baq_postprocess.stop();

    context.stats.baq_reads += num_active;

    parallel<system>::synchronize();
    context.stats.baq_setup.add(baq_setup);
    context.stats.baq_hmm.add(baq_hmm);
    context.stats.baq_postprocess.add(baq_postprocess);
}
INSTANTIATE(baq_reads);

template <target_system system>
void debug_baq(firepony_context<system>& context, const alignment_batch<system>& batch, int read_index)
{
    const alignment_batch_host& h_batch = *batch.host;

    fprintf(stderr, "  BAQ info:\n");

    const CRQ_index idx = h_batch.crq_index(read_index);

    ushort2 read_window = context.cigar.read_window_clipped[read_index];
    uint2 reference_window = context.baq.reference_windows[read_index];

    fprintf(stderr, "    read window                 = [ %u %u ]\n", read_window.x, read_window.y);
    fprintf(stderr, "    absolute reference window   = [ %u %u ]\n", reference_window.x, reference_window.y);
    //fprintf(stderr, "    sequence base: %u\n", genome.sequence_offsets[batch.alignment_sequence_IDs[read_index]]);
    fprintf(stderr, "    relative reference window   = [ %lu %lu ]\n",
            reference_window.x - context.reference.host.sequence_bp_start[h_batch.chromosome[read_index]],
            reference_window.y - context.reference.host.sequence_bp_start[h_batch.chromosome[read_index]]);

    fprintf(stderr, "    BAQ quals                   = [ ");
    for(uint32 i = idx.qual_start; i < idx.qual_start + idx.qual_len; i++)
    {
        uint8 q = context.baq.qualities[i];
        if (q == uint8(-1))
        {
            fprintf(stderr, "  - ");
        } else {
            fprintf(stderr, "% 3d ", q);
        }
    }
    fprintf(stderr, " ]\n");

    fprintf(stderr, "\n");
}
INSTANTIATE(debug_baq);

} // namespace firepony

