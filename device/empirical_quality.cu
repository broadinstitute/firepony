/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <lift/parallel.h>

#include "firepony_context.h"
#include "covariate_table.h"

namespace firepony {

static constexpr int SMOOTHING_CONSTANT = 1;
static constexpr int MAX_RECALIBRATED_Q_SCORE = 93;
static constexpr int MAX_REASONABLE_Q_SCORE = 60;
static constexpr double RESOLUTION_BINS_PER_QUAL = 1.0;
static constexpr int MAX_GATK_USABLE_Q_SCORE = 40;
// note: MAX_NUMBER_OF_OBSERVATIONS is unsigned but contains a signed limit
static constexpr uint32 MAX_NUMBER_OF_OBSERVATIONS = std::numeric_limits<int32>::max() - 1;

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

static CUDA_HOST_DEVICE double log10QempPrior(const double Qempirical, const double Qreported, bool need_rounding)
{
    double delta = Qempirical - Qreported;
    if (need_rounding)
    {
        delta = round(delta);
    }

    int difference = min(abs(int(delta)), MAX_GATK_USABLE_Q_SCORE);
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

static CUDA_HOST_DEVICE double log10Factorial(const uint64 x)
{
    return log10Gamma(x + 1);
}

static CUDA_HOST_DEVICE double log10BinomialCoefficient(const uint64 n, const uint64 k)
{
    return log10Factorial(n) - log10Factorial(k) - log10Factorial(n - k);
}

static CUDA_HOST_DEVICE double log10BinomialProbability(const uint64 n, const uint64 k, const double log10p)
{
    double log10OneMinusP = log10(1 - pow(10.0, log10p));
    return log10BinomialCoefficient(n, k) + log10p * k + log10OneMinusP * (n - k);
}

static CUDA_HOST_DEVICE double qualToErrorProbLog10(double qual)
{
    return qual / -10.0;
}

static CUDA_HOST_DEVICE double log10QempLikelihood(const double Qempirical, uint64 nObservations, uint64 nErrors)
{
    if (nObservations == 0)
        return 0.0;

    // mimic GATK's strange behavior
    if (nObservations > MAX_NUMBER_OF_OBSERVATIONS)
    {
        double fraction = double(MAX_NUMBER_OF_OBSERVATIONS) / double(nObservations);
        nErrors = round(double(nErrors) * fraction);
        nObservations = MAX_NUMBER_OF_OBSERVATIONS;
    }

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

static CUDA_HOST_DEVICE double bayesianEstimateOfEmpiricalQuality(const uint64 nObservations, const uint64 nErrors, const double QReported, bool need_rounding)
{
    constexpr int numBins = (MAX_REASONABLE_Q_SCORE + 1) * int(RESOLUTION_BINS_PER_QUAL);
    double log10Posteriors[numBins];

    for(int bin = 0; bin < numBins; bin++)
    {
        const double QEmpOfBin = bin / RESOLUTION_BINS_PER_QUAL;
        log10Posteriors[bin] = log10QempPrior(QEmpOfBin, QReported, need_rounding) + log10QempLikelihood(QEmpOfBin, nObservations, nErrors);
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

static CUDA_HOST_DEVICE double calcEmpiricalQuality(const covariate_empirical_value& val, bool need_rounding)
{
    // smoothing is one error and one non-error observation
    const uint64 mismatches = uint64(val.mismatches + 0.5) + SMOOTHING_CONSTANT;
    const uint64 observations = val.observations + SMOOTHING_CONSTANT + SMOOTHING_CONSTANT;

    double empiricalQual = bayesianEstimateOfEmpiricalQuality(observations, mismatches, val.estimated_quality, need_rounding);
    return min(empiricalQual, double(MAX_RECALIBRATED_Q_SCORE));
}

template <target_system system>
struct calc_empirical_quality
{
    typename covariate_empirical_table<system>::view table;
    bool need_rounding;

    calc_empirical_quality(typename covariate_empirical_table<system>::view table, bool need_rounding)
        : table(table), need_rounding(need_rounding)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        covariate_empirical_value& val = table.values[index];

        val.estimated_quality = double(-10.0 * log10(val.expected_errors / double(val.observations)));
        val.empirical_quality = calcEmpiricalQuality(val, need_rounding);
    }
};

template <target_system system>
void compute_empirical_quality(firepony_context<system>& context, covariate_empirical_table<system>& table, bool need_rounding)
{
    parallel<system>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + table.size(),
                               calc_empirical_quality<system>(table, need_rounding));
}
INSTANTIATE(compute_empirical_quality)

} // namespace firepony
