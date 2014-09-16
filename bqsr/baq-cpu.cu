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

using namespace nvbio;

static double gapOpenProbability = -1;      // gap open probability [1e-3] (xxxnsubtil: this looks wrong? default is actually convertFromPhredScale(40.0) it seems)
static const double gapExtensionProbability = 0.1;    // gap extension probability [0.1]
static const int maxBandWidth = 7;         // band width [7]

// all bases with q < minBaseQual are up'd to this value
static const int minBaseQual = 4;

#define MAX_PHRED_SCORE 93
#define EM 0.33333333333
#define EI 0.25

#define MAX_BAND_WIDTH 7
#define MIN_BASE_QUAL 4

// all bases with q < minBaseQual are up'd to this value
#define MIN_BASE_QUAL 4

#define GAP_OPEN_PROBABILITY (pow(10.0, (-40.0)/10.))
#define GAP_EXTENSION_PROBABILITY 0.1

static double convertFromPhredScale(double x)
{
    return pow(10, (-x)/10.);
}

static int getBandWidth(void)
{
    return maxBandWidth;
}

namespace hengli {
static double EPSILONS[256][256][MAX_PHRED_SCORE+1];

static void initializeCachedData(void)

{
    gapOpenProbability = convertFromPhredScale(40.0);
    for(int i = 0; i < 256; i++)
    {
        for(int j = 0; j < 256; j++)
        {
            for(int q = 0; q <= MAX_PHRED_SCORE; q++)
            {
                EPSILONS[i][j][q] = 1.0;
            }
        }
    }

    double qual2prob[256];
    for(int i = 0; i < 256; i++)
        qual2prob[i] = pow(10, -i/10.0);

/*
    printf("qual2prob = [ ");
    for(int i = 0; i < 256; i++)
        printf("% .4f ", qual2prob[i]);
    printf("\n");
*/

    char bps[] = "ACGTacgt\0";

    for(int i1 = 0; i1 < strlen(bps); i1++)
    {
        for(int i2 = 0; i2 < strlen(bps); i2++)
        {
            for(int q = 0; q <= MAX_PHRED_SCORE; q++)
            {
                char b1 = bps[i1];
                char b2 = bps[i2];

                double qual = qual2prob[q < minBaseQual ? minBaseQual : q];
                double e = (tolower(b1) == tolower(b2) ? 1 - qual : qual * EM);
                EPSILONS[(int)b1][(int)b2][(int)q] = e;

                //printf("b1 = %c b2 = %c q = %d -> e = %.4f\n", b1, b2, q, e);
            }
        }
    }

/*
    printf("EPSILONS = [ ");
    for(int i = 0; i < sizeof(bps); i++)
    {
        for(int j = 0; j < sizeof(bps); j++)
        {
            for(int k = 0; k < MAX_PHRED_SCORE + 1; k++)
            {
                printf("% .4f ", EPSILONS[bps[i]][bps[j]][k]);
            }
        }
    } printf("\n");
*/


}

static double calcEpsilon(int ref, int read, int qualB)
{
    //printf("epsilon(%d, %d, %d) -> %.4f\n", ref, read, qualB, EPSILONS[ref][read][qualB]);
    return EPSILONS[ref][read][qualB];
}

static int set_u(const int b, const int i, const int k)
{
    int x = i - b;
    x = x > 0 ? x : 0;
    return (k + 1 - x) * 3;
}

static int hmm_glocal(const char *referenceBases, int referenceLength,
                      const char *queryBases,
                      int queryStart, int queryLen,
                      const char *inputQualities,
                      int *outputState,
                      char *outputQualities)
{
//    printf("hmm_glocal(l_ref=%d qstart=%d, l_query=%d)\n", referenceLength, queryStart, queryLen);
//    printf("ref = { %c %c %c %c %c ... %c %c %c %c %c }\n",
//            referenceBases[0], referenceBases[1], referenceBases[2], referenceBases[3], referenceBases[4],
//            referenceBases[referenceLength - 5], referenceBases[referenceLength - 4], referenceBases[referenceLength - 3], referenceBases[referenceLength - 2], referenceBases[referenceLength - 1]);
//    printf("query = { %c %c %c %c %c ... %c %c %c %c %c }\n",
//            queryBases[0], queryBases[1], queryBases[2], queryBases[3], queryBases[4],
//            queryBases[queryLen - 5], queryBases[queryLen - 4], queryBases[queryLen - 3], queryBases[queryLen - 2], queryBases[queryLen - 1]);
//    printf("_iqual = { % 3d % 3d % 3d % 3d % 3d ... % 3d % 3d % 3d % 3d % 3d }\n",
//            inputQualities[0], inputQualities[1], inputQualities[2], inputQualities[3], inputQualities[4],
//            inputQualities[queryLen - 5], inputQualities[queryLen - 4], inputQualities[queryLen - 3], inputQualities[queryLen - 2], inputQualities[queryLen - 1]);

    int i, k;

    /*** initialization ***/

    // set band width
    int bandWidth2, bandWidth;

    if (referenceLength > queryLen)
        bandWidth = referenceLength;
    else
        bandWidth = queryLen;

    if (maxBandWidth < abs(referenceLength - queryLen))
    {
        bandWidth = abs(referenceLength - queryLen) + 3;
//        printf("SC  cb=%d, bw=%d\n", maxBandWidth, bandWidth);
    }

    if (bandWidth > maxBandWidth)
        bandWidth = maxBandWidth;

    if (bandWidth < abs(referenceLength - queryLen))
    {
        int bwOld = bandWidth;
        bandWidth = abs(referenceLength - queryLen);
//        printf("old bw is %d, new is %d\n", bwOld, bandWidth);
    }

//    printf("c->bw = %d, bw = %d, l_ref = %d, l_query = %d\n", maxBandWidth, bandWidth, referenceLength, queryLen);
    bandWidth2 = bandWidth * 2 + 1;

    // allocate the forward and backward matrices f[][] and b[][] and the scaling array s[]
    double **forwardMatrix = new double*[queryLen+1];
    double **backwardMatrix = new double*[queryLen+1];
    for(int32 j = 0; j < queryLen+1; j++)
    {
        forwardMatrix[j] = new double[bandWidth2*3 + 6];
        for(int k = 0; k < bandWidth2*3 + 6; k++)
        {
            forwardMatrix[j][k] = 0.0;
        }

        backwardMatrix[j] = new double[bandWidth2*3 + 6];
        for(int k = 0; k < bandWidth2*3 + 6; k++)
        {
            backwardMatrix[j][k] = 0.0;
        }
    }

    double *scalingFactors = new double[queryLen+2];
    for(int k = 0; k < queryLen+2; k++)
    {
        scalingFactors[k] = 0.0;
    }

    // initialize transition probabilities
    double sM, sI, bM, bI;
    sM = 1.0 / (2 * queryLen + 2);
    sI = sM;
    bM = (1 - gapOpenProbability) / referenceLength;
    bI = gapOpenProbability / referenceLength; // (bM+bI)*l_ref==1

    double *m = new double[9];
    for(int k = 0; k < 9; k++)
    {
        m[k] = 0.0;
    }

    m[0*3+0] = (1 - gapOpenProbability - gapOpenProbability) * (1 - sM);
    m[0*3+1] = gapOpenProbability * (1 - sM);
    m[0*3+2] = m[0*3+1];
    m[1*3+0] = (1 - gapExtensionProbability) * (1 - sI);
    m[1*3+1] = gapExtensionProbability * (1 - sI);
    m[1*3+2] = 0.0;
    m[2*3+0] = 1 - gapExtensionProbability;
    m[2*3+1] = 0.0;
    m[2*3+2] = gapExtensionProbability;

//    printf("cd=%f ce=%f cb=%d\n", gapOpenProbability, gapExtensionProbability, maxBandWidth);
//    printf("sM=%f sI=%f bM=%f bI=%f m[0*3+0]=%f m[1*3+0]=%f m[2*3+0]=%f\n", sM, sI, bM, bI, m[0*3+0], m[1*3+0], m[2*3+0]);
//    printf("queryBases[queryStart] = %c inputQualities[queryStart] = %d\n", queryBases[queryStart], inputQualities[queryStart]);
//    printf("bandWidth = %d bandWidth2 = %d\n", bandWidth, bandWidth2);
//    printf("minBaseQual = %d\n", minBaseQual);

    /*** forward ***/
    // f[0]
    forwardMatrix[0][set_u(bandWidth, 0, 0)] = scalingFactors[0] = 1.;
    { // f[1]
        double *fi = forwardMatrix[1];
        double sum;
        int beg = 1;
        int end = referenceLength < bandWidth + 1? referenceLength : bandWidth + 1;
        int _beg, _end;

        for (k = beg, sum = 0.; k <= end; ++k)
        {
            int u;
            double e = calcEpsilon(referenceBases[k-1], queryBases[queryStart], inputQualities[queryStart]);

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

        for (k = _beg; k <= _end; ++k)
            fi[k] /= sum;
    }

    // f[2..l_query]
    for (i = 2; i <= queryLen; ++i)
    {
        double *fi = forwardMatrix[i];
        double *fi1 = forwardMatrix[i-1];
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
//            printf("referenceBases[%d-1] = %c inputQualities[%d+%d-1] = %d qyi = %c\n", k, referenceBases[k-1], queryStart, i, inputQualities[queryStart+i-1], qyi);
            double e = calcEpsilon(referenceBases[k-1], qyi, inputQualities[queryStart+i-1]);

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

            sum += forwardMatrix[queryLen][u+0] * sM + forwardMatrix[queryLen][u+1] * sI;
        }

        scalingFactors[queryLen+1] = sum; // the last scaling factor
    }

    /*** backward ***/
    // b[l_query] (b[l_query+1][0]=1 and thus \tilde{b}[][]=1/s[l_query+1]; this is where s[l_query+1] comes from)
    for (k = 1; k <= referenceLength; ++k)
    {
        int u = set_u(bandWidth, queryLen, k);
        double *bi = backwardMatrix[queryLen];

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

        double *bi = backwardMatrix[i];
        double *bi1 = backwardMatrix[i+1];
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

    double pb = 0.0;
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

            sum += e * backwardMatrix[1][u+0] * bM + EI * backwardMatrix[1][u+1] * bI;
        }

        backwardMatrix[0][set_u(bandWidth, 0, 0)] = sum / scalingFactors[0];
        pb = backwardMatrix[0][set_u(bandWidth, 0, 0)]; // if everything works as is expected, pb == 1.0
    }

    /*** MAP ***/
    for (i = 1; i <= queryLen; ++i)
    {
        double sum = 0.0;
        double max = 0.0;

        const double *fi = forwardMatrix[i];
        const double *bi = backwardMatrix[i];

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
            double l = log(double(1.0) - double(max));
            double dk = (double(-4.343) * log(double(1.0) - double(max)) + double(.499)); // = 10*log10(1-max)
            k = (int)(double(-4.343) * log(double(1.0) - double(max)) + double(.499)); // = 10*log10(1-max)
            outputQualities[queryStart+i-1] = (char)(k > 100? 99 : (k < minBaseQual ? minBaseQual : k));

//            printf("outputQualities[%d]: max = %.16f l = %.4f dk = %.4f k = %d -> %d\n", i, max, l, dk, k, outputQualities[queryStart+i-1]);
        }

//        printf("(%.4f,%.4f) (%d,%d,%d,%.4f)\n", pb, sum, (i-1), (max_k>>2), (max_k&3), max);
    }

    delete[] m;
    delete[] scalingFactors;

    for(int32 j = 0; j < queryLen+1; j++)
    {
        delete[] backwardMatrix[j];
        delete[] forwardMatrix[j];
    }

    delete[] backwardMatrix;
    delete[] forwardMatrix;

    return 0;
}

bool calcBAQFromHMM(uint8 *baq_output, int **state, char **bq,
                    bqsr_context *context, const reference_genome& reference,
                    const BAM_alignment_batch_host& batch, int read_index)
{
    io::SequenceData::const_plain_view_type ref = *reference.h_ref;
    H_PackedReference ref_stream(ref.m_sequence_stream);

    initializeCachedData();

    const uint32 ref_ID = batch.alignment_sequence_IDs[read_index];
    const uint32 ref_base = ref.m_sequence_index[ref_ID];
    const uint32 ref_length = ref_ID + 1 < ref.m_n_seqs ?
                                  ref.m_sequence_index[ref_ID + 1] - ref_base
                                : ref.m_sequence_stream_len - ref_base;

    const uint32 seq_to_alignment_offset = batch.alignment_positions[read_index];
    //const uint32 abs_alignment_start = ref_base + seq_to_alignment_offset;

    const BAM_CRQ_index& idx = batch.crq_index[read_index];

    const ushort2 read_window = context->cigar.read_window_clipped[read_index];
    const ushort2 read_window_no_insertions = context->cigar.read_window_clipped_no_insertions[read_index];
    const ushort2 reference_window = context->cigar.reference_window_clipped[read_index];

    const uint32 first_insertion_offset = read_window_no_insertions.x - read_window.x;
    const uint32 last_insertion_offset = read_window_no_insertions.y - read_window.y;

    int offset = getBandWidth() / 2;
    uint32 readStart = reference_window.x + seq_to_alignment_offset; // always clipped

    // reference window for HMM
    uint32 start = nvbio::max(readStart - offset - first_insertion_offset, 0u);
    uint32 stop = reference_window.y + seq_to_alignment_offset + offset + last_insertion_offset;

    if (stop > ref_length)
        return false;

    start += ref_base;
    stop += ref_base;

    // calcBAQFromHMM line 602 starts here
    int queryStart = read_window.x;
    int queryEnd = read_window.y;

    // calcBAQFromHMM line 543 starts here
    int queryLen = queryEnd - queryStart + 1;

    printf("calcBAQFromHMM: offset=%d readStart=%d start=%d stop=%d\n", offset, readStart, start - ref_base, stop - ref_base);
    printf("calcBAQFromHMM: first insertion offset = %d last insertion offset = %d\n", first_insertion_offset, last_insertion_offset);
    printf("calcBAQFromHMM: reference sequence window = [ %u, %u ]\n", start, stop);
    printf("calcBAQFromHMM: query window = [ %d, %d ]\n", queryStart, queryEnd);

#if 0
    *bq = new char[queryLen];
    *state = new int[queryLen];

    memset(*bq, 69, queryLen);
    memset(*state, 69, 4 * queryLen);

    char *refbases = new char[stop - start + 2];
    char *querybases = new char[queryLen + 1];
#endif

#if 0
    refbases[stop - start + 1] = '\0';
    for(uint32 c = start; c <= stop; c++)
    {
        if (c < start + 11)
            refbases[c - start] = 'N';
        else
            refbases[c - start] = dna_to_char(ref_stream[c]);
    }

    querybases[queryLen] = '\0';
    for(int c = 0; c < queryLen; c++)
    {
        querybases[c] = iupac16_to_char(batch.reads[idx.read_start + queryStart + c]);
    }

    const uint8 *quals = &batch.qualities[idx.qual_start + queryStart];

    printf(" refbases   = [ %s ]\n", refbases);
    printf(" querybases = [ %s ]\n", querybases);
    printf(" quals      = [ ");
    for(int i = 0; i < queryLen; i++)
    {
        printf("% 4d ", quals[i]);
    }
    printf("]\n");

    printf("hmm_glocal(refbases=%p, stop - start=%d, querybases=%p, queryStart=%d, queryLen=%d, qual=%p, state=%p, bq=%p): \n", refbases, stop - start, querybases, queryStart, queryLen, (char*)&batch.qualities[idx.qual_start], *state, *bq);
    hmm_glocal(refbases, stop - start + 1,
               querybases, 0, queryLen, (char *) quals, *state, *bq);


    printf(" state      = [ ");
    for(int c = 0; c < queryLen; c++)
    {
        printf("% 4d ", (*state)[c]);
    }
    printf("\n");

    printf(" bq         = [ ");
    for(int c = 0; c < queryLen; c++)
    {
        printf("% 4d ", (*bq)[c]);
    }
    printf("\n");

#endif

    return true;
}

void compute_baq(bqsr_context *context, const reference_genome& reference, const BAM_alignment_batch_host& batch)
{
    H_VectorU32 h_active_read_list = context->active_read_list;
    H_VectorU16_2 h_read_window_clipped = context->cigar.read_window_clipped;
    H_VectorU16_2 h_reference_window_clipped = context->cigar.reference_window_clipped;

    for(uint32 i = 0; i < h_active_read_list.size(); i++)
    {
        uint32 read_index = h_active_read_list[i];
        uint8 *out = new uint8[batch.crq_index[read_index].read_len];

        char *bq;
        int *state;

        hengli::calcBAQFromHMM(out, &state, &bq, context, reference, batch, read_index);

        delete[] out;
        delete[] state;
        delete[] bq;
    }

    exit(0);
}

} // namespace hengli

namespace cpu
{
static double EPSILONS[256][256][MAX_PHRED_SCORE+1];

static void initializeCachedData(void)

{
    gapOpenProbability = convertFromPhredScale(40.0);
    for(int i = 0; i < 256; i++)
    {
        for(int j = 0; j < 256; j++)
        {
            for(int q = 0; q <= MAX_PHRED_SCORE; q++)
            {
                EPSILONS[i][j][q] = 1.0;
            }
        }
    }

    double qual2prob[256];
    for(int i = 0; i < 256; i++)
        qual2prob[i] = pow(10, -i/10.0);

/*
    printf("qual2prob = [ ");
    for(int i = 0; i < 256; i++)
        printf("% .4f ", qual2prob[i]);
    printf("\n");
*/

    char bps[] = "ACGTacgt\0";

    for(int i1 = 0; i1 < strlen(bps); i1++)
    {
        for(int i2 = 0; i2 < strlen(bps); i2++)
        {
            for(int q = 0; q <= MAX_PHRED_SCORE; q++)
            {
                char b1 = bps[i1];
                char b2 = bps[i2];

                double qual = qual2prob[q < minBaseQual ? minBaseQual : q];
                double e = (tolower(b1) == tolower(b2) ? 1 - qual : qual * EM);
                EPSILONS[(int)b1][(int)b2][(int)q] = e;

                //printf("b1 = %c b2 = %c q = %d -> e = %.4f\n", b1, b2, q, e);
            }
        }
    }

/*
    printf("EPSILONS = [ ");
    for(int i = 0; i < sizeof(bps); i++)
    {
        for(int j = 0; j < sizeof(bps); j++)
        {
            for(int k = 0; k < MAX_PHRED_SCORE + 1; k++)
            {
                printf("% .4f ", EPSILONS[bps[i]][bps[j]][k]);
            }
        }
    } printf("\n");
*/


}

static double calcEpsilon(char ref, char read, uint8 qualB)
{
    //printf("epsilon(%d, %d, %d) -> %.4f\n", ref, read, qualB, EPSILONS[ref][read][qualB]);
    return EPSILONS[(int)ref][(int)read][qualB];
}

static int set_u(const int b, const int i, const int k)
{
    int x = i - b;
    x = x > 0 ? x : 0;
    return (k + 1 - x) * 3;
}

int hmm_glocal(const H_PackedReference& referenceBases,
               uint2 reference_window,
               const H_StreamDNA16& queryBases,
               uint2 query_window,
               const uint8 *inputQualities,
               uint32 *outputState,
               uint8 *outputQualities)
{
//    printf("hmm_glocal(l_ref=%d qstart=%d, l_query=%d)\n", referenceLength, queryStart, queryLen);
//    printf("ref = { %c %c %c %c %c ... %c %c %c %c %c }\n",
//            referenceBases[0], referenceBases[1], referenceBases[2], referenceBases[3], referenceBases[4],
//            referenceBases[referenceLength - 5], referenceBases[referenceLength - 4], referenceBases[referenceLength - 3], referenceBases[referenceLength - 2], referenceBases[referenceLength - 1]);
//    printf("query = { %c %c %c %c %c ... %c %c %c %c %c }\n",
//            queryBases[0], queryBases[1], queryBases[2], queryBases[3], queryBases[4],
//            queryBases[queryLen - 5], queryBases[queryLen - 4], queryBases[queryLen - 3], queryBases[queryLen - 2], queryBases[queryLen - 1]);
//    printf("_iqual = { % 3d % 3d % 3d % 3d % 3d ... % 3d % 3d % 3d % 3d % 3d }\n",
//            inputQualities[0], inputQualities[1], inputQualities[2], inputQualities[3], inputQualities[4],
//            inputQualities[queryLen - 5], inputQualities[queryLen - 4], inputQualities[queryLen - 3], inputQualities[queryLen - 2], inputQualities[queryLen - 1]);

    const int referenceStart = reference_window.x;
    const int referenceLength = reference_window.y;

    const int queryStart = query_window.x;
    const int queryLen = query_window.y;

    int i, k;

    /*** initialization ***/

    // set band width
    int bandWidth2, bandWidth;

    if (referenceLength > queryLen)
        bandWidth = referenceLength;
    else
        bandWidth = queryLen;

    if (maxBandWidth < abs(referenceLength - queryLen))
    {
        bandWidth = abs(referenceLength - queryLen) + 3;
//        printf("SC  cb=%d, bw=%d\n", maxBandWidth, bandWidth);
    }

    if (bandWidth > maxBandWidth)
        bandWidth = maxBandWidth;

    if (bandWidth < abs(referenceLength - queryLen))
    {
        int bwOld = bandWidth;
        bandWidth = abs(referenceLength - queryLen);
//        printf("old bw is %d, new is %d\n", bwOld, bandWidth);
    }

//    printf("c->bw = %d, bw = %d, l_ref = %d, l_query = %d\n", maxBandWidth, bandWidth, referenceLength, queryLen);
    bandWidth2 = bandWidth * 2 + 1;

    // allocate the forward and backward matrices f[][] and b[][] and the scaling array s[]
    double **forwardMatrix = new double*[queryLen+1];
    double **backwardMatrix = new double*[queryLen+1];
    for(int32 j = 0; j < queryLen+1; j++)
    {
        forwardMatrix[j] = new double[bandWidth2*3 + 6];
        for(int k = 0; k < bandWidth2*3 + 6; k++)
        {
            forwardMatrix[j][k] = 0.0;
        }

        backwardMatrix[j] = new double[bandWidth2*3 + 6];
        for(int k = 0; k < bandWidth2*3 + 6; k++)
        {
            backwardMatrix[j][k] = 0.0;
        }
    }

    double *scalingFactors = new double[queryLen+2];
    for(int k = 0; k < queryLen+2; k++)
    {
        scalingFactors[k] = 0.0;
    }

    // initialize transition probabilities
    double sM, sI, bM, bI;
    sM = 1.0 / (2 * queryLen + 2);
    sI = sM;
    bM = (1 - gapOpenProbability) / referenceLength;
    bI = gapOpenProbability / referenceLength; // (bM+bI)*l_ref==1

    double *m = new double[9];
    for(int k = 0; k < 9; k++)
    {
        m[k] = 0.0;
    }

    m[0*3+0] = (1 - gapOpenProbability - gapOpenProbability) * (1 - sM);
    m[0*3+1] = gapOpenProbability * (1 - sM);
    m[0*3+2] = m[0*3+1];
    m[1*3+0] = (1 - gapExtensionProbability) * (1 - sI);
    m[1*3+1] = gapExtensionProbability * (1 - sI);
    m[1*3+2] = 0.0;
    m[2*3+0] = 1 - gapExtensionProbability;
    m[2*3+1] = 0.0;
    m[2*3+2] = gapExtensionProbability;

//    printf("cd=%f ce=%f cb=%d\n", gapOpenProbability, gapExtensionProbability, maxBandWidth);
//    printf("sM=%f sI=%f bM=%f bI=%f m[0*3+0]=%f m[1*3+0]=%f m[2*3+0]=%f\n", sM, sI, bM, bI, m[0*3+0], m[1*3+0], m[2*3+0]);
//    printf("queryBases[queryStart] = %c inputQualities[queryStart] = %d\n", queryBases[queryStart], inputQualities[queryStart]);
//    printf("bandWidth = %d bandWidth2 = %d\n", bandWidth, bandWidth2);
//    printf("minBaseQual = %d\n", minBaseQual);

    /*** forward ***/
    // f[0]
    forwardMatrix[0][set_u(bandWidth, 0, 0)] = scalingFactors[0] = 1.;
    { // f[1]
        double *fi = forwardMatrix[1];
        double sum;
        int beg = 1;
        int end = referenceLength < bandWidth + 1? referenceLength : bandWidth + 1;
        int _beg, _end;

        for (k = beg, sum = 0.; k <= end; ++k)
        {
            int u;
            char refBase = dna_to_char(referenceBases[referenceStart + k-1]);
            char queryBase = iupac16_to_char(queryBases[queryStart]);

            double e = calcEpsilon(refBase, queryBase, inputQualities[queryStart]);

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

        for (k = _beg; k <= _end; ++k)
            fi[k] /= sum;
    }

    // f[2..l_query]
    for (i = 2; i <= queryLen; ++i)
    {
        double *fi = forwardMatrix[i];
        double *fi1 = forwardMatrix[i-1];
        double sum;

        int beg = 1;
        int end = referenceLength;
        int x, _beg, _end;

        char qyi = iupac16_to_char(queryBases[queryStart+i-1]);

        x = i - bandWidth;
        beg = beg > x? beg : x; // band start

        x = i + bandWidth;
        end = end < x? end : x; // band end

        sum = 0.0;
        for (k = beg; k <= end; ++k)
        {
            int u, v11, v01, v10;
//            printf("referenceBases[%d-1] = %c inputQualities[%d+%d-1] = %d qyi = %c\n", k, referenceBases[k-1], queryStart, i, inputQualities[queryStart+i-1], qyi);

            char refBase = dna_to_char(referenceBases[referenceStart + k-1]);

            double e = calcEpsilon(refBase, qyi, inputQualities[queryStart+i-1]);

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

            sum += forwardMatrix[queryLen][u+0] * sM + forwardMatrix[queryLen][u+1] * sI;
        }

        scalingFactors[queryLen+1] = sum; // the last scaling factor
    }

    /*** backward ***/
    // b[l_query] (b[l_query+1][0]=1 and thus \tilde{b}[][]=1/s[l_query+1]; this is where s[l_query+1] comes from)
    for (k = 1; k <= referenceLength; ++k)
    {
        int u = set_u(bandWidth, queryLen, k);
        double *bi = backwardMatrix[queryLen];

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

        double *bi = backwardMatrix[i];
        double *bi1 = backwardMatrix[i+1];
        double y = (i > 1)? 1. : 0.;

        char qyi1 = iupac16_to_char(queryBases[queryStart+i]);

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
            else {
                char refBase = dna_to_char(referenceBases[referenceStart + k]);
                e = calcEpsilon(refBase, qyi1, inputQualities[queryStart+i]) * bi1[v11];
            }

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

    double pb = 0.0;
    { // b[0]
        int beg = 1;
        int end = referenceLength < bandWidth + 1? referenceLength : bandWidth + 1;

        double sum = 0.0;
        for (k = end; k >= beg; --k)
        {
            int u = set_u(bandWidth, 1, k);

            char refBase = dna_to_char(referenceBases[referenceStart + k-1]);
            char queryBase = iupac16_to_char(queryBases[queryStart]);

            double e = calcEpsilon(refBase, queryBase, inputQualities[queryStart]);

            if (u < 3 || u >= bandWidth2*3+3)
                continue;

            sum += e * backwardMatrix[1][u+0] * bM + EI * backwardMatrix[1][u+1] * bI;
        }

        backwardMatrix[0][set_u(bandWidth, 0, 0)] = sum / scalingFactors[0];
        pb = backwardMatrix[0][set_u(bandWidth, 0, 0)]; // if everything works as is expected, pb == 1.0
    }

    /*** MAP ***/
    for (i = 1; i <= queryLen; ++i)
    {
        double sum = 0.0;
        double max = 0.0;

        const double *fi = forwardMatrix[i];
        const double *bi = backwardMatrix[i];

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

        //if (outputState != NULL)
            outputState[queryStart+i-1] = max_k;

        //if (outputQualities != NULL)
        {
            double l = log(double(1.0) - double(max));
            double dk = (double(-4.343) * log(double(1.0) - double(max)) + double(.499)); // = 10*log10(1-max)
            k = (int)(double(-4.343) * log(double(1.0) - double(max)) + double(.499)); // = 10*log10(1-max)
            outputQualities[queryStart+i-1] = (char)(k > 100? 99 : (k < minBaseQual ? minBaseQual : k));

//            printf("outputQualities[%d]: max = %.16f l = %.4f dk = %.4f k = %d -> %d\n", i, max, l, dk, k, outputQualities[queryStart+i-1]);
        }

//        printf("(%.4f,%.4f) (%d,%d,%d,%.4f)\n", pb, sum, (i-1), (max_k>>2), (max_k&3), max);
    }

    delete[] m;
    delete[] scalingFactors;

    for(int32 j = 0; j < queryLen+1; j++)
    {
        delete[] backwardMatrix[j];
        delete[] forwardMatrix[j];
    }

    delete[] backwardMatrix;
    delete[] forwardMatrix;

    return 0;
}

// compute the read and reference windows to run BAQ on for a given read
void frame_alignment(ushort2& out_read_window, uint2& out_reference_window,
                     bqsr_context *context,
                     const reference_genome& reference,
                     const BAM_alignment_batch_device& batch,
                     const BAM_alignment_batch_host& h_batch,
                     uint32 read_index)
{
    const BAM_CRQ_index& idx = batch.crq_index[read_index];

    io::SequenceData::const_plain_view_type ref = *reference.h_ref;

    // grab reference sequence window in the full genome
    const uint32 ref_ID = batch.alignment_sequence_IDs[read_index];
    const uint32 ref_base = ref.m_sequence_index[ref_ID];
    const uint32 ref_length = ref_ID + 1 < ref.m_n_seqs ?
                                ref.m_sequence_index[ref_ID + 1] - ref_base
                              : ref.m_sequence_stream_len - ref_base;

    const uint32 seq_to_alignment_offset = batch.alignment_positions[read_index];

    const ushort2 read_window = context->cigar.read_window_clipped[read_index];
    const ushort2 read_window_no_insertions = context->cigar.read_window_clipped_no_insertions[read_index];
    const ushort2 reference_window = context->cigar.reference_window_clipped[read_index];

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

//    printf("frame_alignment: offset=%d readStart=%d start=%d stop=%d\n", offset, readStart, start - ref_base, stop - ref_base);
//    printf("frame_alignment: first insertion offset = %d last insertion offset = %d\n", first_insertion_offset, last_insertion_offset);
//    printf("frame_alignment: reference sequence window = [ %u, %u ]\n", out_reference_window.x, out_reference_window.y);
//    printf("frame_alignment: query window = [ %d, %d ]\n", out_read_window.x, out_read_window.y);
}

}
