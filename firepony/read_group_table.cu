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

#include "bqsr_context.h"
#include "covariates_table.h"
#include "read_group_table.h"

#include "primitives/parallel.h"

#include "covariates/table_quality_scores.h"

#include <thrust/reduce.h>

struct generate_read_group_table : public bqsr_lambda_context
{
    using bqsr_lambda_context::bqsr_lambda_context;

    CUDA_HOST_DEVICE double qualToErrorProb(uint8 qual)
    {
        return pow(10.0, qual / -10.0);
    }

    CUDA_HOST_DEVICE double calcExpectedErrors(const covariate_value& val, uint8 qual)
    {
        return val.observations * qualToErrorProb(qual);
    }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        const auto& key_in = ctx.covariates.quality.keys[index];
        const auto& value_in = ctx.covariates.quality.values[index];
        const auto qual = covariate_table_quality::decode(key_in, covariate_table_quality::QualityScore);

        auto& key_out = ctx.read_group_table.read_group_keys[index];
        auto& value_out = ctx.read_group_table.read_group_values[index];

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
    using bqsr_lambda_context::bqsr_lambda_context;

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
        covariate_empirical_value& val = ctx.read_group_table.read_group_values[index];

        val.estimated_quality = double(-10.0 * log10(val.expected_errors / double(val.observations)));
        val.empirical_quality = calcEmpiricalQuality(val);
    }
};

void build_read_group_table(bqsr_context *context)
{
    const auto& cv = context->covariates;
    auto& rg = context->read_group_table;

    auto& rg_keys = rg.read_group_keys;
    auto& rg_values = rg.read_group_values;

    // convert the quality table into the read group table
    rg_keys.resize(cv.quality.size());
    rg_values.resize(cv.quality.size());
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + cv.quality.size(),
                     generate_read_group_table(*context));

    D_Vector<covariate_key>& temp_keys = context->temp_u32;
    D_Vector<covariate_empirical_value> temp_values;
    D_VectorU8& temp_storage = context->temp_storage;

    temp_keys.resize(rg_keys.size());
    temp_values.resize(rg_keys.size());

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
                     covariate_compute_empirical_quality(*context));
}

void output_read_group_table(bqsr_context *context)
{
    auto& rg = context->read_group_table;
    auto& rg_keys = rg.read_group_keys;
    auto& rg_values = rg.read_group_values;

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

