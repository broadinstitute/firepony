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

// base alignment quality calculations (gatk: BAQ.java)

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include <lift/parallel.h>
#include <lift/memory/strided_iterator.h>

#include <stdlib.h>
#include <math.h>

#include "device_types.h"
#include "alignment_data_device.h"
#include "../sequence_database.h"
#include "firepony_context.h"

#include "primitives/util.h"
#include "from_nvbio/dna.h"
#include "from_nvbio/alphabet.h"

namespace firepony {

template <target_system system>
struct baq_stride
{ };

template <>
struct baq_stride<cuda>
{
    static constexpr uint32 stride = 32;
};

template <>
struct baq_stride<host>
{
    static constexpr uint32 stride = 1;
};

#define MAX_PHRED_SCORE 93
#define EM 0.33333333333
#define EI 0.25

#define MIN_BAND_WIDTH 7
#define MIN_BAND_WIDTH2 (MIN_BAND_WIDTH * 2 + 1)

// all bases with q < minBaseQual are up'd to this value
#define MIN_BASE_QUAL 4

#define GAP_OPEN_PROBABILITY (pow(10.0, (-40.0)/10.))
#define GAP_EXTENSION_PROBABILITY 0.1

static CUDA_HOST_DEVICE inline uint32 abs(int32 val)
{
    return (val < 0 ? uint32(-val) : uint32(val));
}

template <target_system system>
struct compute_hmm_windows : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        // read_needs_baq is used to avoid disabling reads that don't actually need BAQ if we determine that BAQ can't otherwise be computed
        // (we still compute HMM windows even for reads that do not need BAQ)
        const bool read_needs_baq = (ctx.cigar.num_errors[read_index] != 0);

        const CRQ_index idx = batch.crq_index(read_index);

        const ushort2& read_window_clipped = ctx.cigar.read_window_clipped[read_index];
        const ushort2& read_window_clipped_no_insertions = ctx.cigar.read_window_clipped_no_insertions[read_index];
        const ushort2& reference_window_clipped = ctx.cigar.reference_window_clipped[read_index];

        // note: the band width for any given read is not necessarily constant, but GATK always uses the min band width when computing the reference offset
        // this looks a lot like a bug in GATK, but we replicate the same behavior here
        constexpr int offset = MIN_BAND_WIDTH / 2;

        // xxxnsubtil: this is a hack --- GATK hard clips adapters and soft clipping regions, meaning that any clipped reads can *never* have leading/trail insertions
        // (the cigar operators are replaced with H, and lead/trail insertion detection tests for I at the very beginning/end of the read)
        // this effectively means that the no-insertions read window is not usable in this case

        // figure out if we applied clipping on either end of the read
        const bool left_clip = (read_window_clipped.x != 0);
        const bool right_clip = (read_window_clipped.y != idx.read_len - 1);

        // compute the left and right insertion offsets
        const short left_insertion = (left_clip ? 0 : read_window_clipped_no_insertions.x - read_window_clipped.x);
        const short right_insertion = (right_clip ? 0 : read_window_clipped.y - read_window_clipped_no_insertions.y);

        // compute the reference window in local read coordinates
        short2 hmm_reference_window;
        hmm_reference_window.x = reference_window_clipped.x - left_insertion - offset;
        hmm_reference_window.y = reference_window_clipped.y + right_insertion + offset;

        // figure out if the HMM window is contained inside the chromosome this read is aligned to
        auto& reference = ctx.reference_db.get_sequence(batch.chromosome[read_index]);

        int left_window_shift = 0;
        if (hmm_reference_window.x < 0 && batch.alignment_start[read_index] < abs(hmm_reference_window.x))
        {
            // this is very likely a problem in GATK: because their reference and read coordinates are off by 1, they end up
            // subtracting one base from the reference window when the start position is clamped
            left_window_shift = -1;
        }

        if (read_needs_baq && batch.alignment_start[read_index] + hmm_reference_window.y + left_window_shift >= reference.bases.size())
        {
            // HMM window lies beyond the end of the reference sequence
            // can't compute BAQ, filter out this read

            // clear the cigar event -> read bp offset mapping for this read
            const uint32 cigar_ev_start = ctx.cigar.cigar_offsets[idx.cigar_start];
            const uint32 cigar_ev_stop = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

            for(uint32 i = cigar_ev_start; i < cigar_ev_stop; i++)
            {
                ctx.cigar.cigar_event_read_index[i] = uint32(-1);
                ctx.cigar.cigar_event_read_coordinates[i] = uint16(-1);
            }

            // emit an invalid HMM window
            // this will cause this read to be removed from the active read list
            ctx.baq.hmm_reference_windows[read_index] = make_short2(short(-1), short(-1));

            return;
        }

        // write out the result
        ctx.baq.hmm_reference_windows[read_index] = hmm_reference_window;

        // compute the band width
        int referenceLen;
        int queryLen;

        referenceLen = hmm_reference_window.y - hmm_reference_window.x + 1;
        queryLen = read_window_clipped.y - read_window_clipped.x + 1;

        uint16 bandWidth = max(referenceLen, queryLen);

        if (MIN_BAND_WIDTH < abs(referenceLen - queryLen))
        {
            bandWidth = abs(referenceLen - queryLen) + 3;
        }

        if (bandWidth > MIN_BAND_WIDTH)
            bandWidth = MIN_BAND_WIDTH;

        if (bandWidth < abs(referenceLen - queryLen))
        {
            bandWidth = abs(referenceLen - queryLen);
        }

        ctx.baq.bandwidth[read_index] = bandWidth;
    }
};

template <target_system system>
struct hmm_window_valid : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE bool operator() (const uint32 read_index)
    {
        if (ctx.baq.hmm_reference_windows[read_index].x == short(-1) &&
            ctx.baq.hmm_reference_windows[read_index].y == short(-1))
        {
            return false;
        } else {
            return true;
        }
    }
};

typedef enum
{
    hmm_phase_forward = 1,
    hmm_phase_backward = 2,
    hmm_phase_map = 4,
    hmm_phase_all = 4 + 2 + 1,
} hmm_phase;

template <target_system system, uint32 phase>
struct hmm_glocal : public lambda<system>
{
    pointer<system, uint32> baq_state;

    hmm_glocal(firepony_context<system> ctx,
               const alignment_batch_device<system> batch,
               pointer<system, uint32> baq_state)
        : lambda<system>(ctx, batch), baq_state(baq_state)
    { }

    int bandWidth, bandWidth2;

    int referenceStart, referenceLength;
    int queryStart, queryEnd, queryLen;

    typedef strided_iterator<double, baq_stride<system>::stride> matrix_iterator;

    matrix_iterator forwardMatrix;
    matrix_iterator backwardMatrix;
    matrix_iterator scalingFactors;

    double sM, sI, bM, bI;

    double m[9];

    stream_dna16<system> referenceBases;
    stream_dna16<system> queryBases;
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

        const uint32 matrix_offset = matrix_index % baq_stride<system>::stride;
        const uint32 matrix_base = matrix_index - matrix_offset;

        const uint32 scaling_offset = scaling_index % baq_stride<system>::stride;
        const uint32 scaling_base = scaling_index - scaling_offset;

        // set up matrix and scaling factor pointers
        forwardMatrix = matrix_iterator(&ctx.baq.forward[matrix_base + matrix_offset]);
        backwardMatrix = matrix_iterator(&ctx.baq.backward[matrix_base + matrix_offset]);
        scalingFactors = matrix_iterator(&ctx.baq.scaling[scaling_base + scaling_offset]);

        // get the windows for the current read
        const auto& hmm_reference_window = ctx.baq.hmm_reference_windows[read_index];
        const ushort2& read_window_clipped = ctx.cigar.read_window_clipped[read_index];

        referenceStart = hmm_reference_window.x;
        referenceLength = hmm_reference_window.y - hmm_reference_window.x + 1;

        queryStart = read_window_clipped.x;
        queryEnd = read_window_clipped.y;
        queryLen = queryEnd - queryStart + 1;

        bandWidth = ctx.baq.bandwidth[read_index];
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

        queryBases = batch.reads.stream() + idx.read_start + queryStart;

        // compute the starting offset of the HMM window inside the reference
        // this handles the case where a read aligns to the start of a chromosome by clamping the resulting offset to zero
        uint32 hmm_reference_offset;
        if (hmm_reference_window.x < 0 && batch.alignment_start[read_index] < abs(hmm_reference_window.x))
        {
            hmm_reference_offset = 0;

            // this is very likely a problem in GATK: because their reference and read coordinates are off by 1, they end up
            // subtracting one base from the reference window when the start position is clamped
            referenceLength--;
        } else {
            hmm_reference_offset = batch.alignment_start[read_index] + hmm_reference_window.x;
        }

        referenceBases = ctx.reference_db.get_sequence_data(batch.chromosome[read_index], hmm_reference_offset);

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
        return i * (bandWidth2 * 3 + 6) + j;
    }

    // computes the required HMM matrix size for the given read length
    CUDA_HOST_DEVICE static uint32 matrix_size(const uint32 read_len, const int bandWidth)
    {
        const int bandWidth2 = bandWidth * 2 + 1;
        return (read_len + 1) * (bandWidth2 * 3 + 6);
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

    template <typename Tuple>
    CUDA_HOST_DEVICE void hmm_forward(const Tuple& hmm_index)
    {
        int i, k;

//        const uint32 read_index    = thrust::get<0>(hmm_index);
//        printf("read %d: hmm_glocal(l_ref=%d qstart=%d, l_query=%d)\n", read_index, referenceLength, queryStart, queryLen);
//        printf("read %d: ref = { ", read_index);
//        for(int c = 0; c < referenceLength; c++)
//        {
//            printf("%c ", from_nvbio::iupac16_to_char(referenceBases[c]));
//        }
//        printf("\n");
//
//        printf("read %d: que = { ", read_index);
//        for(int c = 0; c < queryLen; c++)
//        {
//            printf("%c ", from_nvbio::iupac16_to_char(queryBases[c]));
//        }
//        printf("\n");
//
//        printf("read %d: _iqual = { % 3d % 3d % 3d % 3d % 3d ... % 3d % 3d % 3d % 3d % 3d }\n", read_index,
//                inputQualities[0], inputQualities[1], inputQualities[2], inputQualities[3], inputQualities[4],
//                inputQualities[queryLen - 5], inputQualities[queryLen - 4], inputQualities[queryLen - 3], inputQualities[queryLen - 2], inputQualities[queryLen - 1]);
//        printf("read %d: c->bw = %d, bw = %d, l_ref = %d, l_query = %d\n", read_index, MIN_BAND_WIDTH, bandWidth, referenceLength, queryLen);

        /*** forward ***/
        // f[0]
        forwardMatrix[off(0, set_u(bandWidth, 0, 0))] = 1.0;
        scalingFactors[0] = 1.0;
        { // f[1]
            matrix_iterator fi(&forwardMatrix[off(1)]);
            double sum;
            int beg = 1;
            int end = referenceLength < bandWidth + 1? referenceLength : bandWidth + 1;
            int _beg, _end;

            sum = 0.0;
            for (k = beg; k <= end; ++k)
            {
                int u;
                double e = calcEpsilon(referenceBases[k-1], queryBases[queryStart], inputQualities[queryStart]);
//                printf("referenceBases[%d-1] = %c inputQualities[%d] = %d queryBases[%d] = %c -> e = %.4f\n",
////                       read_index,
//                       k,
//                       from_nvbio::iupac16_to_char(referenceBases[k-1]),
//                       queryStart,
//                       inputQualities[queryStart],
//                       queryStart,
//                       from_nvbio::iupac16_to_char(queryBases[queryStart]), e);

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
            matrix_iterator fi(&forwardMatrix[off(i)]);
            matrix_iterator fi1(&forwardMatrix[off(i-1)]);
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
//                printf("read %d: referenceBases[%d-1] = %c inputQualities[%d+%d-1] = %d qyi = %c -> e = %.4f\n",
//                       read_index,
//                       k,
//                       from_nvbio::iupac16_to_char(referenceBases[k-1]),
//                       queryStart,
//                       i,
//                       inputQualities[queryStart+i-1],
//                       from_nvbio::iupac16_to_char(qyi), e);

                u = set_u(bandWidth, i, k);
                v11 = set_u(bandWidth, i-1, k-1);
                v10 = set_u(bandWidth, i-1, k);
                v01 = set_u(bandWidth, i, k-1);

                fi[u+0] = e * (m[0] * fi1[v11+0] + m[3] * fi1[v11+1] + m[6] * fi1[v11+2]);
                fi[u+1] = EI * (m[1] * fi1[v10+0] + m[4] * fi1[v10+1]);
                fi[u+2] = m[2] * fi[v01+0] + m[8] * fi[v01+2];

                sum += fi[u] + fi[u+1] + fi[u+2];

//                printf("(%d,%d;%d): %.32f,%.32f,%.32f\n", i, k, u, fi[u], fi[u+1], fi[u+2]);
//                printf(" .. u = %d v11 = %d v01 = %d v10 = %d e = %f\n", u, v11, v01, v10, e);
//                printf("HMM: %d, %d, %d, %d, %d, %d, %d, %d, %d, %.32f, %.32f, %.32f, %.32f, %.32f, %.32f, %.32f, %.32f, %.32f, %.32f, %.32f\n",
//                       i, k, u,
//                       from_nvbio::iupac16_to_char(uint8(referenceBases[k-1])), from_nvbio::iupac16_to_char(qyi), uint8(inputQualities[queryStart+i-1]),
//                       v01, v10, v11,
//                       e,
//                       fi1[v01+0], fi1[v01+1],
//                       fi1[v10+0], fi1[v10+1],
//                       fi1[v11+0], fi1[v11+1], fi1[v11+2],
//                       fi[u+0], fi[u+1], fi[u+2]);
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

    template <typename Tuple>
    CUDA_HOST_DEVICE void hmm_backward(const Tuple& hmm_index)
    {
        int i, k;

        /*** backward ***/
        // b[l_query] (b[l_query+1][0]=1 and thus \tilde{b}[][]=1/s[l_query+1]; this is where s[l_query+1] comes from)
        for (k = 1; k <= referenceLength; ++k)
        {
            int u = set_u(bandWidth, queryLen, k);
            matrix_iterator bi(&backwardMatrix[off(queryLen)]);

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

            matrix_iterator bi(&backwardMatrix[off(i)]);
            matrix_iterator bi1(&backwardMatrix[off(i+1)]);

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
    }

    template <typename Tuple>
    CUDA_HOST_DEVICE void hmm_map(const Tuple& hmm_index)
    {
        int i, k;

        /*** MAP ***/
        for (i = 1; i <= queryLen; ++i)
        {
            double sum = 0.0;
            double max = 0.0;

            matrix_iterator fi(&forwardMatrix[off(i)]);
            matrix_iterator bi(&backwardMatrix[off(i)]);

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

//                printf("outputQualities[%d]: max = %.16f k = %d -> %d\n", queryStart+i-1, max, k, outputQualities[queryStart+i-1]);
            }

//            printf("(%.4f,%.4f) (%d,%d,%d,%.4f)\n", pb, sum, (i-1), (max_k>>2), (max_k&3), max);
        }
    }

    template <typename Tuple>
    CUDA_HOST_DEVICE void operator() (const Tuple& hmm_index)
    {
        setup(hmm_index);

        if (phase & hmm_phase_forward)
            hmm_forward(hmm_index);

        if (phase & hmm_phase_backward)
            hmm_backward(hmm_index);

        if (phase & hmm_phase_map)
            hmm_map(hmm_index);
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
        return hmm_glocal<system, hmm_phase_all>::matrix_size(idx.read_len, ctx.baq.bandwidth[read_index]);
    }
};

template <target_system system>
struct compute_hmm_matrix_size_strided : public lambda<system>
{
    LAMBDA_INHERIT_MEMBERS;

    pointer<system, uint32> active_read_list;
    pointer<system, uint32> matrix_size_output;

    compute_hmm_matrix_size_strided(firepony_context<system> ctx,
                                    const alignment_batch_device<system> batch,
                                    pointer<system, uint32> active_read_list,
                                    pointer<system, uint32> matrix_size_output)
        : lambda<system>(ctx, batch),
          active_read_list(active_read_list),
          matrix_size_output(matrix_size_output)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        uint32 ret = 0;

        // smear the max matrix size across the thread group so we can stride it later on
        constexpr uint32 stride = baq_stride<system>::stride;
        const uint32 start_range = index - (index % stride);
        const uint32 end_range = start_range + stride;

        for(uint32 i = start_range; i < end_range; i++)
        {
            if (i < active_read_list.size())
            {
                const uint32 read_index = active_read_list[i];
                const CRQ_index idx = batch.crq_index(read_index);
                const uint32 size = hmm_glocal<system, hmm_phase_all>::matrix_size(idx.read_len, ctx.baq.bandwidth[read_index]);
                ret = max(ret, size);
            }
        }

        matrix_size_output[index] = ret;
    }
};

template <target_system system>
struct stride_hmm_index
{
    pointer<system, uint32> index;

    stride_hmm_index(pointer<system, uint32> index)
        : index(index)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 matrix_group_index)
    {
        constexpr uint32 stride = baq_stride<system>::stride;

        auto initial_element = index[matrix_group_index * stride];
        for(uint32 i = 1; i < stride; i++)
        {
            index[matrix_group_index * stride + i] = initial_element + i;
        }
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
struct compute_hmm_scaling_factor_size_strided : public lambda<system>
{
    LAMBDA_INHERIT;

    const pointer<system, uint32> active_read_list;
    pointer<system, uint32> scaling_index_output;

    compute_hmm_scaling_factor_size_strided(firepony_context<system> ctx,
                                            const alignment_batch_device<system> batch,
                                            const pointer<system, uint32> active_read_list,
                                            pointer<system, uint32> scaling_index_output)
        : lambda<system>(ctx, batch),
          active_read_list(active_read_list),
          scaling_index_output(scaling_index_output)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        uint32 ret = 0;

        // smear the max scaling factor size across the thread group so we can stride it later on
        constexpr uint32 stride = baq_stride<system>::stride;
        const uint32 start_range = index - (index % stride);
        const uint32 end_range = start_range + stride;

        for(uint32 i = start_range; i < end_range; i++)
        {
            if (i < active_read_list.size())
            {
                const uint32 read_index = active_read_list[i];
                const CRQ_index idx = batch.crq_index(read_index);
                const uint32 size = idx.read_len + 2;
                ret = max(ret, size);
            }
        }

        scaling_index_output[index] = ret;
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
    LAMBDA_INHERIT;

    const pointer<system, uint32> baq_state;

    cap_baq_qualities(firepony_context<system> ctx,
                      const alignment_batch_device<system> batch,
                      const pointer<system, uint32> baq_state)
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

        if (isIndel || pos != expectedPos) // we are an indel or we don't align to our best current position
        {
            b = MIN_BASE_QUAL; // just take b = minBaseQuality
        } else {
            b = min(bq, oq);
        }

        return b;
    }

    template <bool update_qualities>
    CUDA_HOST_DEVICE bool cap_baq (const uint32 read_index,
                                   const CRQ_index idx,
                                   const uint32 cigar_start,
                                   const uint32 cigar_end,
                                   const uint32 baq_start,
                                   const ushort2 read_window_clipped,
                                   const ushort2 reference_window_clipped,
                                   const short2 hmm_reference_window)
    {
        uint32 readI = 0;
        uint32 refI = 0;
        uint32 numD = 0;

        // compute the shift applied to the reference during BAQ setup
        // this happens if the computed HMM window start lies before the chromosome start (i.e., the start coordinate is negative)
        // also computes the shift in expected read BP positions that is present in the BAQ state vector
        int16 reference_shift, read_shift;
        if (hmm_reference_window.x < 0 && batch.alignment_start[read_index] < abs(hmm_reference_window.x))
        {
            reference_shift = abs(hmm_reference_window.x);
            read_shift = batch.alignment_start[read_index];
        } else {
            reference_shift = 0;
            read_shift = 0;
        }

        // compute the reference window offset
        const int16 refOffset = hmm_reference_window.x - reference_window_clipped.x + reference_shift;

        for(uint32 i = baq_start; i < cigar_end - cigar_start; i++)
        {
            const uint16 read_bp_idx = ctx.cigar.cigar_event_read_coordinates[cigar_start + i];
            const uint32 qual_idx = idx.qual_start + read_bp_idx;

            if (read_bp_idx != uint16(-1))
            {
                if (read_bp_idx < read_window_clipped.x)
                    continue;

                if (read_bp_idx > read_window_clipped.y)
                    break;
            }

            switch(ctx.cigar.cigar_events[i + cigar_start])
            {
            case cigar_event::S:
                refI++;
                break;

            case cigar_event::I:
                if (update_qualities)
                {
                    ctx.baq.qualities[qual_idx] = batch.qualities[qual_idx];
                }

                readI++;
                break;

            case cigar_event::D:
                refI++;
                numD++;
                break;

            case cigar_event::M:
                if (update_qualities)
                {
                    const uint32 expectedPos = refI - refOffset + (i - numD - baq_start - readI) + read_shift;
                    ctx.baq.qualities[qual_idx] = capBaseByBAQ(batch.qualities[idx.qual_start + read_bp_idx],
                                                               ctx.baq.qualities[idx.qual_start + read_bp_idx],
                                                               baq_state[idx.qual_start + read_bp_idx],
                                                               expectedPos);
                }

                readI++;
                refI++;

                break;
            }
        }

        if (!update_qualities)
        {
            const uint32 read_len = read_window_clipped.y - read_window_clipped.x + 1;
            if (readI != read_len)
            {
                // odd cigar string, do not update
                return false;
            }
        }

        return true;
    }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        const auto read_window_clipped = ctx.cigar.read_window_clipped[read_index];
        const auto reference_window_clipped = ctx.cigar.reference_window_clipped[read_index];
        const auto hmm_reference_window = ctx.baq.hmm_reference_windows[read_index];

        // scan for the start of the baq region
        uint32 baq_start = 0;
        for(uint32 i = 0; i < cigar_end - cigar_start; i++)
        {
            const uint16 read_bp_idx = ctx.cigar.cigar_event_read_coordinates[cigar_start + i];
            if (read_bp_idx != uint16(-1) && read_bp_idx >= read_window_clipped.x)
            {
                baq_start = i;
                break;
            }
        }

        // check if we need to update the BAQ qualities...
        const bool need_update = cap_baq<false>(read_index,
                                                idx,
                                                cigar_start,
                                                cigar_end,
                                                baq_start,
                                                read_window_clipped,
                                                reference_window_clipped,
                                                hmm_reference_window);

        if (need_update)
        {
            // ... and update them
            cap_baq<true>(read_index,
                          idx,
                          cigar_start,
                          cigar_end,
                          baq_start,
                          read_window_clipped,
                          reference_window_clipped,
                          hmm_reference_window);
        } else {
            // ... or overwrite BAQ with the original qualities instead
            const uint32 base_off = idx.qual_start + read_window_clipped.x;
            const uint32 len = read_window_clipped.y - read_window_clipped.x + 1;

            memcpy(&ctx.baq.qualities[base_off], &batch.qualities[base_off], len);
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
    persistent_allocation<system, uint32>& active_baq_read_list = context.temp_u32;
    persistent_allocation<system, uint32>& baq_state = context.temp_u32_2;
    persistent_allocation<system, uint32>& temp_active_list = context.temp_u32_3;
    uint32 num_active;

    timer<system> baq_setup, baq_hmm, baq_postprocess;
#if BAQ_HMM_SPLIT_PHASE
    timer<system> baq_hmm_forward, baq_hmm_backward, baq_hmm_map;
#endif

    baq_setup.start();

    // allocate our output
    baq.qualities.resize(batch.device.qualities.size());
    thrust::fill(lift::backend_policy<system>::execution_policy(), baq.qualities.begin(), baq.qualities.end(), uint8(-1));

    // collect the reads that we need to compute BAQ for
    active_baq_read_list.resize(context.active_read_list.size());

    num_active = parallel<system>::copy_if(context.active_read_list.begin(),
                                           context.active_read_list.size(),
                                           active_baq_read_list.begin(),
                                           read_needs_baq<system>(context, batch.device),
                                           context.temp_storage);

    active_baq_read_list.resize(num_active);

    if (num_active)
    {
        baq.hmm_reference_windows.resize(batch.device.num_reads);
        baq.bandwidth.resize(batch.device.num_reads);

        // compute the HMM windows
        // note: this is used both for real BAQ and flat BAQ, so we use the full active read list
        parallel<system>::for_each(context.active_read_list.begin(),
                                   context.active_read_list.end(),
                                   compute_hmm_windows<system>(context, batch.device));

        // remove any reads for which we failed to compute the HMM window
        temp_active_list.resize(active_baq_read_list.size());
        num_active = parallel<system>::copy_if(active_baq_read_list.begin(),
                                               active_baq_read_list.size(),
                                               temp_active_list.begin(),
                                               hmm_window_valid<system>(context, batch.device),
                                               context.temp_storage);

        temp_active_list.resize(num_active);
        active_baq_read_list.copy(temp_active_list);
    }

    if (num_active)
    {
        temp_active_list.resize(context.active_read_list.size());
        uint32 temp_num_active = parallel<system>::copy_if(context.active_read_list.begin(),
                                                           context.active_read_list.size(),
                                                           temp_active_list.begin(),
                                                           hmm_window_valid<system>(context, batch.device),
                                                           context.temp_storage);
        temp_active_list.resize(temp_num_active);
        context.active_read_list.copy(temp_active_list);

        constexpr uint32 stride = baq_stride<system>::stride;

        if (stride == 1)
        {
            // compute the index and size of the HMM matrices
            baq.matrix_index.resize(num_active + 1);
            // first offset is zero
            thrust::fill_n(lift::backend_policy<system>::execution_policy(), baq.matrix_index.begin(), 1, 0);

            // do an inclusive scan to compute all offsets + the total size
            parallel<system>::inclusive_scan(thrust::make_transform_iterator(active_baq_read_list.begin(),
                                                                             compute_hmm_matrix_size<system>(context, batch.device)),
                                             num_active,
                                             baq.matrix_index.begin() + 1,
                                             thrust::plus<uint32>());

            // compute the index and size of the HMM scaling factors
            baq.scaling_index.resize(num_active + 1);
            // first offset is zero
            thrust::fill_n(lift::backend_policy<system>::execution_policy(), baq.scaling_index.begin(), 1, 0);
            parallel<system>::inclusive_scan(thrust::make_transform_iterator(active_baq_read_list.begin(),
                                                                             compute_hmm_scaling_factor_size<system>(context, batch.device)),
                                             num_active,
                                             baq.scaling_index.begin() + 1,
                                             thrust::plus<uint32>());
        } else {
            // round up the number of matrices to a multiple of baq_stride
            const uint32 num_matrices = ((num_active + stride - 1) / stride) * stride;
            temp_active_list.resize(num_matrices);

            // allocate and initialize the matrix and scaling indices
            baq.matrix_index.resize(num_matrices + 1);
            baq.scaling_index.resize(num_matrices + 1);
            // first offset is zero
            thrust::fill_n(lift::backend_policy<system>::execution_policy(), baq.matrix_index.begin(), 1, 0);
            thrust::fill_n(lift::backend_policy<system>::execution_policy(), baq.scaling_index.begin(), 1, 0);

            // compute matrix sizes as the maximum size of each baq_stride groups of reads
            parallel<system>::for_each(thrust::make_counting_iterator(0u),
                                       thrust::make_counting_iterator(0u) + num_matrices,
                                       compute_hmm_matrix_size_strided<system>(context, batch.device, active_baq_read_list, temp_active_list));

            parallel<system>::inclusive_scan(temp_active_list.begin(),
                                             num_matrices,
                                             baq.matrix_index.begin() + 1,
                                             thrust::plus<uint32>());

            parallel<system>::for_each(thrust::make_counting_iterator(0u),
                                       thrust::make_counting_iterator(0u) + baq.matrix_index.size() / stride,
                                       stride_hmm_index<system>(baq.matrix_index));

            // compute the scaling index
            parallel<system>::for_each(thrust::make_counting_iterator(0u),
                                       thrust::make_counting_iterator(0u) + num_matrices,
                                       compute_hmm_scaling_factor_size_strided<system>(context, batch.device, active_baq_read_list, temp_active_list));

            parallel<system>::inclusive_scan(temp_active_list.begin(),
                                             num_matrices,
                                             baq.scaling_index.begin() + 1,
                                             thrust::plus<uint32>());

            parallel<system>::for_each(thrust::make_counting_iterator(0u),
                                       thrust::make_counting_iterator(0u) + baq.scaling_index.size() / stride,
                                       stride_hmm_index<system>(baq.scaling_index));
        }

        uint32 matrix_len = baq.matrix_index.peek(baq.matrix_index.size() - 1);
        uint32 scaling_len = baq.scaling_index.peek(baq.scaling_index.size() - 1);

        baq.forward.resize(matrix_len);
        baq.backward.resize(matrix_len);
        baq.scaling.resize(scaling_len);

//        fprintf(stderr, "reads: %u\n", batch.num_reads);
//        fprintf(stderr, "forward len = %u bytes = %lu\n", matrix_len, matrix_len * sizeof(double));
//        fprintf(stderr, "expected len = %lu expected bytes = %lu\n",
//                hmm_common::matrix_size(100) * context.active_read_list.size(),
//                hmm_common::matrix_size(100) * context.active_read_list.size() * sizeof(double));
//        fprintf(stderr, "per read matrix size = %u bytes = %lu\n", hmm_common::matrix_size(100), hmm_common::matrix_size(100) * sizeof(double));
//        fprintf(stderr, "matrix index = [ ");
//        for(uint32 i = 0; i < 20; i++)
//        {
//            fprintf(stderr, "%u, ", baq.matrix_index[i] + 0);
//        }
//        fprintf(stderr, " ... ");
//        for(uint32 i = baq.matrix_index.size() - 20; i < baq.matrix_index.size(); i++)
//        {
//            fprintf(stderr, "%u, ", baq.matrix_index[i] + 0);
//        }
//        fprintf(stderr, "]\n");
//        fflush(stdout);


        baq_state.resize(batch.device.qualities.size());
        thrust::fill(lift::backend_policy<system>::execution_policy(), baq_state.begin(), baq_state.end(), uint32(-1));

        // initialize matrices and scaling factors
        thrust::fill_n(lift::backend_policy<system>::execution_policy(), baq.forward.begin(), baq.forward.size(), 0.0);
        thrust::fill_n(lift::backend_policy<system>::execution_policy(), baq.backward.begin(), baq.backward.size(), 0.0);
        thrust::fill_n(lift::backend_policy<system>::execution_policy(), baq.scaling.begin(), baq.scaling.size(), 0.0);

        baq_setup.stop();

        baq_hmm.start();

#if BAQ_HMM_SPLIT_PHASE
        // run the HMM split in 3 phases
        baq_hmm_forward.start();
        parallel<system>::for_each(thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.begin(),
                                                                                baq.matrix_index.begin(),
                                                                                baq.scaling_index.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.end(),
                                                                                baq.matrix_index.end(),
                                                                                baq.scaling_index.end())),
                                   hmm_glocal<system, hmm_phase_forward>(context, batch.device, baq_state));
        baq_hmm_forward.stop();

        baq_hmm_backward.start();
        parallel<system>::for_each(thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.begin(),
                                                                                baq.matrix_index.begin(),
                                                                                baq.scaling_index.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.end(),
                                                                                baq.matrix_index.end(),
                                                                                baq.scaling_index.end())),
                                   hmm_glocal<system, hmm_phase_backward>(context, batch.device, baq_state));
        baq_hmm_backward.stop();

        baq_hmm_map.start();
        parallel<system>::for_each(thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.begin(),
                                                                                baq.matrix_index.begin(),
                                                                                baq.scaling_index.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.end(),
                                                                                baq.matrix_index.end(),
                                                                                baq.scaling_index.end())),
                                   hmm_glocal<system, hmm_phase_map>(context, batch.device, baq_state));
        baq_hmm_map.stop();
#else
        // run the HMM itself
        parallel<system>::for_each(thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.begin(),
                                                                                baq.matrix_index.begin(),
                                                                                baq.scaling_index.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(active_baq_read_list.end(),
                                                                                baq.matrix_index.end(),
                                                                                baq.scaling_index.end())),
                                   hmm_glocal<system, hmm_phase_all>(context, batch.device, baq_state));
#endif

        baq_hmm.stop();
    }

    baq_postprocess.start();

#if PRESERVE_BAQ_STATE
    context.baq.state = baq_state;
#endif

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

    if (num_active)
    {
        context.stats.baq_setup.add(baq_setup);
        context.stats.baq_hmm.add(baq_hmm);
#if BAQ_HMM_SPLIT_PHASE
        context.stats.baq_hmm_forward.add(baq_hmm_forward);
        context.stats.baq_hmm_backward.add(baq_hmm_backward);
        context.stats.baq_hmm_map.add(baq_hmm_map);
#endif
    }

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
    short2 reference_window = context.baq.hmm_reference_windows[read_index];

    fprintf(stderr, "    read window                 = [ %u %u ]\n", read_window.x, read_window.y);
    fprintf(stderr, "    relative reference window   = [ %d %d ]\n", reference_window.x, reference_window.y);

    fprintf(stderr, "    BAQ quals                   = [ ");
    for(uint32 i = idx.qual_start; i < idx.qual_start + idx.qual_len; i++)
    {
        uint8 q = context.baq.qualities[i];
        if (q == uint8(-1))
        {
            fprintf(stderr, "   - ");
        } else {
            fprintf(stderr, "% 4d ", q);
        }
    }
    fprintf(stderr, " ]\n");

#if PRESERVE_BAQ_STATE
    fprintf(stderr, "    BAQ state                   = [ ");
    for(uint32 i = idx.qual_start; i < idx.qual_start + idx.qual_len; i++)
    {
        uint32 q = context.baq.state[i];
        if (q == uint8(-1))
        {
            fprintf(stderr, "   - ");
        } else {
            fprintf(stderr, "% 4d ", q);
        }
    }
    fprintf(stderr, " ]\n");
#endif

    fprintf(stderr, "\n");
}
INSTANTIATE(debug_baq);

} // namespace firepony

