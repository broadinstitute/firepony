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
#include "covariate_table.h"
#include "read_group_table.h"

#include "primitives/parallel.h"

#include "covariates/packer_quality_score.h"

#include <thrust/reduce.h>

namespace firepony {

template <target_system system>
struct generate_read_group_table : public lambda_context<system>
{
    LAMBDA_CONTEXT_INHERIT;

    CUDA_HOST_DEVICE double qualToErrorProb(uint8 qual)
    {
        return pow(10.0, qual / -10.0);
    }

    CUDA_HOST_DEVICE double calcExpectedErrors(const covariate_empirical_value& val, uint8 qual)
    {
        return val.observations * qualToErrorProb(qual);
    }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        typedef covariate_packer_quality_score<system> packer;

        auto& rg = ctx.read_group_table.read_group_table;

        auto& key = rg.keys[index];
        auto& value = rg.values[index];

        // decode the quality
        const auto qual = packer::decode(key, packer::QualityScore);

        // remove the quality from the key
        key &= ~packer::chain::key_mask(packer::QualityScore);

        // compute the expected error rate
        value.expected_errors = calcExpectedErrors(value, qual);
    }
};

template <target_system system>
struct covariate_compute_empirical_quality : public lambda_context<system>
{
    LAMBDA_CONTEXT_INHERIT;

    static constexpr int SMOOTHING_CONSTANT = 1;
    static constexpr int MAX_RECALIBRATED_Q_SCORE = 93;
    static constexpr int MAX_REASONABLE_Q_SCORE = 60;
    static constexpr double RESOLUTION_BINS_PER_QUAL = 1.0;
    static constexpr int MAX_GATK_USABLE_Q_SCORE = 40;

    CUDA_HOST_DEVICE double gaussian(double value)
    {
        // f(x) = a + b*exp(-((x - c)^2 / (2*d^2)))
        // Note that b is the height of the curve's peak, c is the position of the center of the peak, and d controls the width of the "bell".
        constexpr double GF_a = 0.0;
        constexpr double GF_b = 0.9;
        constexpr double GF_c = 0.0;
        constexpr double GF_d = 0.5; // with these parameters, deltas can shift at most ~20 Q points

        return GF_a + GF_b * pow(double(M_E), -(pow(value - GF_c, 2.0) / (2 * GF_d * GF_d)));
    }

    CUDA_HOST_DEVICE double log10QempPrior(const double Qempirical, const double Qreported)
    {
        int difference = min(abs(int(Qempirical - Qreported)), MAX_GATK_USABLE_Q_SCORE);
        return log10(gaussian(double(difference)));
    }

    CUDA_HOST_DEVICE double lnToLog10(const double ln)
    {
        return ln * log10(M_E);
    }

    CUDA_HOST_DEVICE double log10Gamma(const double x)
    {
        return lnToLog10(lgammaf(x));
    }

    CUDA_HOST_DEVICE double log10Factorial(const uint32 x)
    {
        return log10Gamma(x + 1);
    }

    CUDA_HOST_DEVICE double log10BinomialCoefficient(const uint32 n, const uint32 k)
    {
        return log10Factorial(n) - log10Factorial(k) - log10Factorial(n - k);
    }

    CUDA_HOST_DEVICE double log10BinomialProbability(const uint32 n, const uint32 k, const double log10p)
    {
        double log10OneMinusP = log10(1 - pow(10.0, log10p));
        return log10BinomialCoefficient(n, k) + log10p * k + log10OneMinusP* (n - k);
    }


    CUDA_HOST_DEVICE double qualToErrorProbLog10(double qual)
    {
        return qual / -10.0;
    }

    CUDA_HOST_DEVICE double log10QempLikelihood(const double Qempirical, uint32 nObservations, uint32 nErrors)
    {
        if (nObservations == 0)
            return 0.0;

        return log10BinomialProbability(nObservations, nErrors, qualToErrorProbLog10(Qempirical));
    }

    CUDA_HOST_DEVICE void normalizeFromLog10(double *normalized, const double *array, int array_len)
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

    CUDA_HOST_DEVICE double bayesianEstimateOfEmpiricalQuality(const uint32 nObservations, const uint32 nErrors, const double QReported)
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

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        covariate_empirical_value& val = ctx.read_group_table.read_group_table.values[index];

        val.estimated_quality = double(-10.0 * log10(val.expected_errors / double(val.observations)));
        val.empirical_quality = calcEmpiricalQuality(val);
    }
};

template <target_system system>
void build_read_group_table(firepony_context<system>& context)
{
    const auto& cv = context.covariates;
    auto& rg = context.read_group_table.read_group_table;

    if (cv.quality.size() == 0)
    {
        // if we didn't gather any entries in the table, there's nothing to do
        return;
    }

    // convert the quality table into the read group table
    covariate_observation_to_empirical_table(context, cv.quality, rg);
    // transform the read group table in place to remove the quality value from the keys and compute the estimated error
    parallel<system>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + cv.quality.size(),
                               generate_read_group_table<system>(context));

    // sort and pack the read group table
    auto& temp_keys = context.temp_u32;
    firepony::vector<system, covariate_empirical_value> temp_values;
    auto& temp_storage = context.temp_storage;

    rg.sort(temp_keys, temp_values, temp_storage, covariate_packer_quality_score<system>::chain::bits_used);
    rg.pack(temp_keys, temp_values);

    // compute empirical qualities
    parallel<system>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + rg.size(),
                               covariate_compute_empirical_quality<system>(context));
}
INSTANTIATE(build_read_group_table);

template <target_system system>
void output_read_group_table(firepony_context<system>& context)
{
    typedef covariate_packer_quality_score<system> packer;

    covariate_empirical_table<host> table;
    table.copyfrom(context.read_group_table.read_group_table);

    printf("#:GATKTable:6:3:%%s:%%s:%%.4f:%%.4f:%%d:%%.2f:;\n");
    printf("#:GATKTable:RecalTable0:\n");
    printf("ReadGroup\tEventType\tEmpiricalQuality\tEstimatedQReported\tObservations\tErrors\n");

    for(uint32 i = 0; i < table.size(); i++)
    {
        uint32 rg_id = packer::decode(table.keys[i], packer::ReadGroup);
        const std::string& rg_name = context.bam_header.host.read_groups_db.lookup(rg_id);

        covariate_empirical_value val = table.values[i];

        // ReadGroup, EventType, EmpiricalQuality, EstimatedQReported, Observations, Errors
        printf("%s\t%c\t\t%.4f\t\t\t%.4f\t\t\t%d\t\t%.2f\n",
                rg_name.c_str(),
                cigar_event::ascii(packer::decode(table.keys[i], packer::EventTracker)),
                round_n(val.empirical_quality, 4),
                round_n(val.estimated_quality, 4),
                val.observations,
                round_n(val.mismatches, 2));
    }

    printf("\n");
}
INSTANTIATE(output_read_group_table);

} // namespace firepony

