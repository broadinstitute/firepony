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
#include "empirical_quality.h"

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

    compute_empirical_quality(context, rg);
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

