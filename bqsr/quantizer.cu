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

#include <set>

#include "bqsr_context.h"
#include "covariates_table.h"
#include "quantizer.h"

#include "primitives/parallel.h"

#include "covariates/table_quality_scores.h"

static constexpr int MAX_SAM_QUAL_SCORE = 93;
static constexpr int QUANTIZING_LEVELS = 16;
static constexpr int MIN_USABLE_Q_SCORE = 6;
static constexpr int SMOOTHING_CONSTANT = 1;
static constexpr int MAX_RECALIBRATED_Q_SCORE = 93;
static constexpr int MAX_REASONABLE_Q_SCORE = 60;
static constexpr double RESOLUTION_BINS_PER_QUAL = 1.0;
static constexpr int MAX_GATK_USABLE_Q_SCORE = 40;

static CUDA_HOST_DEVICE uint8 boundQual(const int qual, const uint8 maxQual)
{
    return uint8(max(min(qual, maxQual & 0xff), 1) & 0xff);
}

static CUDA_HOST_DEVICE uint8 errorProbToQual(const double errorRate, const uint8 maxQual)
{
    const double d = round(-10.0 * log10(errorRate));
    return boundQual((int)d, maxQual);
}

static CUDA_HOST_DEVICE uint8 errorProbToQual(const double errorRate)
{
    return errorProbToQual(errorRate, MAX_SAM_QUAL_SCORE);
}

static CUDA_HOST_DEVICE double qualToErrorProb(uint8 qual)
{
    return pow(10.0, qual / -10.0);
}


static CUDA_HOST_DEVICE double gaussian(double value)
{
    // f(x) = a + b*exp(-((x - c)^2 / (2*d^2)))
    // Note that b is the height of the curve's peak, c is the position of the center of the peak, and d controls the width of the "bell".
    constexpr double GF_a = 0.0;
    constexpr double GF_b = 0.9;
    constexpr double GF_c = 0.0;
    constexpr double GF_d = 0.5; // with these parameters, deltas can shift at most ~20 Q points

    return GF_a + GF_b * pow(double(M_E), -(pow(value - GF_c, 2.0) / (2 * GF_d * GF_d)));
}

static CUDA_HOST_DEVICE double log10QempPrior(const double Qempirical, const double Qreported)
{
    int difference = min(abs(int(Qempirical - Qreported)), MAX_GATK_USABLE_Q_SCORE);
    return log10(gaussian(double(difference)));
}

static CUDA_HOST_DEVICE double lnToLog10(const double ln)
{
    return ln * log10(M_E);
}

static CUDA_HOST_DEVICE double log10Gamma(const double x)
{
    return lnToLog10(lgammaf(x));
}

static CUDA_HOST_DEVICE double log10Factorial(const uint32 x)
{
    return log10Gamma(x + 1);
}

static CUDA_HOST_DEVICE double log10BinomialCoefficient(const uint32 n, const uint32 k)
{
    return log10Factorial(n) - log10Factorial(k) - log10Factorial(n - k);
}

static CUDA_HOST_DEVICE double log10BinomialProbability(const uint32 n, const uint32 k, const double log10p)
{
    double log10OneMinusP = log10(1 - pow(10.0, log10p));
    return log10BinomialCoefficient(n, k) + log10p * k + log10OneMinusP* (n - k);
}


static CUDA_HOST_DEVICE double qualToErrorProbLog10(double qual)
{
    return qual / -10.0;
}

static CUDA_HOST_DEVICE double log10QempLikelihood(const double Qempirical, uint32 nObservations, uint32 nErrors)
{
    if (nObservations == 0)
        return 0.0;

    return log10BinomialProbability(nObservations, nErrors, qualToErrorProbLog10(Qempirical));
}

static CUDA_HOST_DEVICE void normalizeFromLog10(double *normalized, const double *array, int array_len)
{
    double maxValue = array[0];
    for(int i = 1; i < array_len; i++)
    {
        if (array[i] > maxValue)
            maxValue = array[i];
    }

    double sum = 0.0;
    for(int i = 0; i < array_len; i++)
    {
        normalized[i] = pow(10.0, array[i] - maxValue);
        sum += normalized[i];
    }

    for(int i = 0; i < array_len; i++)
    {
        double x = normalized[i] / sum;
        normalized[i] = x;
    }
}

static CUDA_HOST_DEVICE double bayesianEstimateOfEmpiricalQuality(const uint32 nObservations, const uint32 nErrors, const double QReported)
{
    constexpr int numBins = (MAX_REASONABLE_Q_SCORE + 1) * int(RESOLUTION_BINS_PER_QUAL);
    double log10Posteriors[numBins];

    for(int bin = 0; bin < numBins; bin++)
    {
        const double QEmpOfBin = bin / RESOLUTION_BINS_PER_QUAL;
        log10Posteriors[bin] = log10QempPrior(QEmpOfBin, QReported) + log10QempLikelihood(QEmpOfBin, nObservations, nErrors);
    }

    double normalizedPosteriors[numBins];
    normalizeFromLog10(normalizedPosteriors, log10Posteriors, numBins);

    int MLEbin = 0;
    for(int i = 1; i < numBins; i++)
    {
        if (log10Posteriors[i] > log10Posteriors[MLEbin])
            MLEbin = i;
    }

    double Qemp = MLEbin / RESOLUTION_BINS_PER_QUAL;
    return Qemp;
}

static CUDA_HOST_DEVICE double calcEmpiricalQuality(const covariate_empirical_value& val)
{
    // smoothing is one error and one non-error observation
    const uint32 mismatches = uint32(val.mismatches + 0.5) + SMOOTHING_CONSTANT;
    const uint32 observations = val.observations + SMOOTHING_CONSTANT + SMOOTHING_CONSTANT;

    double empiricalQual = bayesianEstimateOfEmpiricalQuality(observations, mismatches, val.estimated_quality);
    return min(empiricalQual, double(MAX_RECALIBRATED_Q_SCORE));
}

static CUDA_HOST_DEVICE double calcExpectedErrors(const covariate_value& val, uint8 qual)
{
    return val.observations * qualToErrorProb(qual);
}

struct generate_read_group_table : public bqsr_lambda_context
{
    using bqsr_lambda_context::bqsr_lambda_context;

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        const auto& key_in = ctx.covariates.quality.keys[index];
        const auto& value_in = ctx.covariates.quality.values[index];
        const auto qual = covariate_table_quality::decode(key_in, covariate_table_quality::QualityScore);

        auto& key_out = ctx.quantizer.read_group_keys[index];
        auto& value_out = ctx.quantizer.read_group_values[index];

        // remove the quality from the key
        key_out = key_in & ~covariate_table_quality::chain::key_mask(covariate_table_quality::QualityScore);

        value_out.observations = value_in.observations;
        value_out.mismatches = value_in.mismatches;

        value_out.expected_errors = calcExpectedErrors(value_in, qual);
    }
};

struct covariate_empirical_value_sum
{
    CUDA_HOST_DEVICE covariate_empirical_value operator() (const covariate_empirical_value& a, const covariate_empirical_value& b)
    {
        return { a.observations + b.observations,
                 a.mismatches + b.mismatches,
                 a.expected_errors + b.expected_errors,
                 0.0f,
                 0.0f };
    }
};

struct covariate_compute_empirical_quality : public bqsr_lambda_context
{
    D_Vector<covariate_empirical_value>::view output;

    covariate_compute_empirical_quality(bqsr_context::view ctx,
                                        D_Vector<covariate_empirical_value>::view output)
        : bqsr_lambda_context(ctx), output(output)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        covariate_empirical_value& val = output[index];

        val.estimated_quality = double(-10.0 * log10(val.expected_errors / double(val.observations)));
        val.empirical_quality = calcEmpiricalQuality(val);
    }
};

void build_read_group_table(bqsr_context *context)
{
    const auto& cv = context->covariates;
    auto& quant = context->quantizer;

    auto& rg_keys = quant.read_group_keys;
    auto& rg_values = quant.read_group_values;

    // convert the quality table into the read group table
    rg_keys.resize(cv.quality.size());
    rg_values.resize(cv.quality.size());
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + cv.quality.size(),
                     generate_read_group_table(*context));

    D_Vector<covariate_key>& temp_keys = context->temp_u32;
    D_Vector<covariate_empirical_value> temp_values;
    D_VectorU8& temp_storage = context->temp_storage;

    bqsr::sort_by_key(rg_keys, rg_values, temp_keys, temp_values, temp_storage, covariate_table_quality::chain::bits_used);

    // reduce the read group table by key
    thrust::pair<D_Vector<covariate_key>::iterator, D_Vector<covariate_empirical_value>::iterator> out;
    out = thrust::reduce_by_key(rg_keys.begin(),
                                rg_keys.end(),
                                rg_values.begin(),
                                temp_keys.begin(),
                                temp_values.begin(),
                                thrust::equal_to<covariate_key>(),
                                covariate_empirical_value_sum());

    uint32 new_size = out.first - temp_keys.begin();

    temp_keys.resize(new_size);
    temp_values.resize(new_size);

    rg_keys = temp_keys;
    rg_values = temp_values;

    // compute empirical qualities
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + new_size,
                     covariate_compute_empirical_quality(*context, rg_values));
}

void output_read_group_table(bqsr_context *context)
{
    auto& quant = context->quantizer;
    auto& rg_keys = quant.read_group_keys;
    auto& rg_values = quant.read_group_values;

    printf("#:GATKTable:6:3:%%s:%%s:%%.4f:%%.4f:%%d:%%.2f:;\n");
    printf("#:GATKTable:RecalTable0:\n");
    printf("ReadGroup\tEventType\tEmpiricalQuality\tEstimatedQReported\tObservations\tErrors\n");

    for(uint32 i = 0; i < rg_keys.size(); i++)
    {
        uint32 rg_id = covariate_table_quality::decode(rg_keys[i], covariate_table_quality::ReadGroup);
        const std::string& rg_name = context->bam_header.read_groups_db.lookup(rg_id);

        covariate_empirical_value val = rg_values[i];

        // ReadGroup, EventType, EmpiricalQuality, EstimatedQReported, Observations, Errors
        printf("%s\t%c\t\t%.4f\t\t\t%.4f\t\t\t%d\t\t%.2f\n",
                rg_name.c_str(),
                cigar_event::ascii(covariate_table_quality::decode(rg_keys[i], covariate_table_quality::EventTracker)),
                val.empirical_quality,
                val.estimated_quality,
                val.observations,
                val.mismatches);
    }

    printf("\n");
}

struct QualInterval
{
    int qStart, qEnd, fixedQual, level;
    uint32 nObservations, nErrors;
    std::set<QualInterval> subIntervals;

    QualInterval()
        : qStart(-1), fixedQual(-1)
    { }

    QualInterval(int qStart, int qEnd, uint32 nObservations, uint32 nErrors, int level, std::set<QualInterval> subIntervals)
        : qStart(qStart), qEnd(qEnd), fixedQual(-1), level(level), nObservations(nObservations), nErrors(nErrors), subIntervals(subIntervals)
    { }

    QualInterval(int qStart, int qEnd, uint32 nObservations, uint32 nErrors, int level, int fixedQual)
        : qStart(qStart), qEnd(qEnd), nObservations(nObservations), nErrors(nErrors), level(level), fixedQual(fixedQual)
    { }

    std::string as_string(void) const
    {
        char buf[256];

        if (qStart == -1)
            snprintf(buf, 256, "null");
        else
            snprintf(buf, 256, "QQ:%d-%d fixedQual=%d", qStart, qEnd, fixedQual);

        return std::string(buf);
    }

    double getErrorRate(void) const
    {
        if (hasFixedQual())
        {
            return qualToErrorProb(uint8(fixedQual));
        } else {
            if (nObservations == 0)
            {
                return 0.0;
            } else {
                return (nErrors + 1) / (1.0 * (nObservations + 1));
            }
        }
    }

    uint8 getQual(void) const
    {
        if (!hasFixedQual())
        {
            return errorProbToQual(getErrorRate());
        } else {
            return uint8(fixedQual);
        }
    }

    bool hasFixedQual(void) const
    {
        return fixedQual != -1;
    }

    bool operator==(const QualInterval& qualInterval) const
    {
        return qStart == qualInterval.qStart;
    }

    bool operator<(const QualInterval& qualInterval) const
    {
        return qStart < qualInterval.qStart;
    }

    bool operator>(const QualInterval& qualInterval) const
    {
        return qStart > qualInterval.qStart;
    }

    QualInterval merge(const QualInterval& toMerge) const
    {
        const QualInterval left = *this < toMerge ? *this : toMerge;
        const QualInterval right = *this < toMerge ? toMerge : *this;

        const uint32 nCombinedObs = left.nObservations + right.nObservations;
        const uint32 nCombinedErr = left.nErrors + right.nErrors;

        const int level = max(left.level, right.level) + 1;

        std::set<QualInterval> new_subIntervals;
        new_subIntervals.insert(left);
        new_subIntervals.insert(right);
        return QualInterval(left.qStart, right.qEnd, nCombinedObs, nCombinedErr, level, new_subIntervals);
    }

    double calcPenalty(const double globalErrorRate) const
    {
        if (globalErrorRate == 0.0)
        {
            return 0.0;
        }

        if (subIntervals.empty())
        {
            if (qEnd <= MIN_USABLE_Q_SCORE)
            {
                return 0.0;
            } else {
                return abs(log10(getErrorRate()) - log10(globalErrorRate)) * nObservations;
            }
        } else {
            double sum = 0;
            for(auto& interval : subIntervals)
            {
                sum += interval.calcPenalty(globalErrorRate);
            }

            return sum;
        }
    }

    double getPenalty(void) const
    {
        return calcPenalty(getErrorRate());
    }
};

void mergeLowestPenaltyIntervals(std::set<QualInterval>& intervals)
{
    auto it1 = intervals.begin();
    auto it1p = intervals.begin();
    it1p++;

    QualInterval minMerge;

    while(it1p != intervals.end())
    {
        const QualInterval left = *it1;
        const QualInterval right = *it1p;
        const QualInterval merged = left.merge(right);
        if (minMerge.qStart == -1 || merged.getPenalty() < minMerge.getPenalty())
        {
            minMerge = merged;
        }

        it1++;
        it1p++;
    }

    // remove all minMerge.subIntervals from intervals
    for (const auto& i : minMerge.subIntervals)
    {
        intervals.erase(i);
    }

    intervals.insert(minMerge);
}

static std::set<QualInterval> quantize(H_VectorU32& nObservationsPerQual, const int nLevels)
{
    std::set<QualInterval> intervals;

    for(int qStart = 0; size_t(qStart) < nObservationsPerQual.size(); qStart++)
    {
        const uint32 nObs = nObservationsPerQual[qStart];
        const double errorRate = qualToErrorProb(uint8(qStart));
        const double nErrors = nObs * errorRate;
        const QualInterval qi(qStart, qStart, nObs, uint32(floor(nErrors)), 0, qStart);
        intervals.insert(qi);
    }

    while (intervals.size() > size_t(nLevels))
    {
        mergeLowestPenaltyIntervals(intervals);
    }

    return intervals;
}

struct generate_empirical_qual_table : public bqsr_lambda_context
{
    using bqsr_lambda_context::bqsr_lambda_context;

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        const auto& key_in = ctx.covariates.quality.keys[index];
        const auto& value_in = ctx.covariates.quality.values[index];
        const auto qual = covariate_table_quality::decode(key_in, covariate_table_quality::QualityScore);

        auto& value_out = ctx.quantizer.empirical_quality_values[index];

        value_out.observations = value_in.observations;
        value_out.mismatches = value_in.mismatches;

        value_out.expected_errors = calcExpectedErrors(value_in, qual);
    }
};

void build_quality_quantization_table(bqsr_context *context)
{
    const auto& cv = context->covariates;
    auto& quant = context->quantizer;

    // build a table of empirical qualities
    // xxxnsubtil: this repeats some of build_read_group_table(), should merge
    auto& values = quant.empirical_quality_values;

    // convert the quality table into the empirical quality table
    values.resize(cv.quality.size());
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + cv.quality.size(),
                     generate_empirical_qual_table(*context));

    // compute empirical qualities
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + values.size(),
                     covariate_compute_empirical_quality(*context, values));

    // read values back to the host
    H_Vector<covariate_empirical_value> h_values = values;

    // compute a histogram of quality values
    auto& histogram = context->covariates.quality_histogram;
    histogram.resize(MAX_SAM_QUAL_SCORE + 1);
    thrust::fill(histogram.begin(), histogram.end(), uint32(0));

    for(uint32 i = 0; i < h_values.size(); i++)
    {
        int empiricalQual = round(h_values[i].empirical_quality);
        assert(empiricalQual >= 0 && empiricalQual <= MAX_SAM_QUAL_SCORE);
        histogram[empiricalQual] += h_values[i].observations;
    }

    // create the quantization map
    std::set<QualInterval> quantized = quantize(histogram, QUANTIZING_LEVELS);

    auto& map = context->covariates.quantization_map;
    map.resize(MAX_SAM_QUAL_SCORE + 1);
    thrust::fill(map.begin(), map.end(), uint8(-1));

    for(auto& interval : quantized)
    {
        for(int q = interval.qStart; q <= interval.qEnd; q++)
        {
            map[q] = interval.getQual();
        }
    }
}

void output_quality_quantization_table(bqsr_context *context)
{
    H_CovariateTable quality_table;
    quality_table.copyfrom(context->covariates.quality);

    auto& histogram = context->covariates.quality_histogram;
    auto& map = context->covariates.quantization_map;

    printf("#:GATKTable:3:94:%%s:%%s:%%s:;\n");
    printf("#:GATKTable:Quantized:Quality quantization map\n");
    printf("QualityScore\tCount\tQuantizedScore\n");
    for(uint32 i = 0; i < histogram.size(); i++)
    {
        printf("%d\t\t%d\t\t%d\n", i, histogram[i], map[i]);
    }
    printf("\n");
}

void debug_quality_quantization_table(bqsr_context *context)
{
    auto& quant = context->quantizer;

    printf("quality histogram = [ ");
    for(uint32 i = 0; i < quant.quality_histogram.size(); i++)
    {
        printf("% 3d ", uint32(quant.quality_histogram[i]));
    }
    printf("\n");
}
