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

#include "primitives/parallel.h"
#include "firepony_context.h"
#include "covariate_table.h"

namespace firepony {

static constexpr int SMOOTHING_CONSTANT = 1;
static constexpr int MAX_RECALIBRATED_Q_SCORE = 93;
static constexpr int MAX_REASONABLE_Q_SCORE = 60;
static constexpr double RESOLUTION_BINS_PER_QUAL = 1.0;
static constexpr int MAX_GATK_USABLE_Q_SCORE = 40;

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

CUDA_HOST_DEVICE double calcEmpiricalQuality(const covariate_empirical_value& val)
{
    // smoothing is one error and one non-error observation
    const uint32 mismatches = uint32(val.mismatches + 0.5) + SMOOTHING_CONSTANT;
    const uint32 observations = val.observations + SMOOTHING_CONSTANT + SMOOTHING_CONSTANT;

    double empiricalQual = bayesianEstimateOfEmpiricalQuality(observations, mismatches, val.estimated_quality);
    return min(empiricalQual, double(MAX_RECALIBRATED_Q_SCORE));
}

template <target_system system>
struct calc_empirical_quality
{
    typename covariate_empirical_table<system>::view table;

    calc_empirical_quality(typename covariate_empirical_table<system>::view table)
        : table(table)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        covariate_empirical_value& val = table.values[index];

        val.estimated_quality = double(-10.0 * log10(val.expected_errors / double(val.observations)));
        val.empirical_quality = calcEmpiricalQuality(val);
    }
};

template <target_system system>
void compute_empirical_quality(firepony_context<system>& context, covariate_empirical_table<system>& table)
{
    parallel<system>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + table.size(),
                               calc_empirical_quality<system>(table));
}
INSTANTIATE(compute_empirical_quality)

} // namespace firepony
