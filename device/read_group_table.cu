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

#include "firepony_context.h"
#include "covariate_table.h"
#include "read_group_table.h"
#include "empirical_quality.h"

#include "primitives/parallel.h"
#include "covariates/packer_quality_score.h"
#include "expected_error.h"

#include <thrust/reduce.h>

namespace firepony {

template <target_system system>
struct remove_quality_from_key : public lambda_context<system>
{
    LAMBDA_CONTEXT_INHERIT;

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        auto& key = ctx.covariates.read_group.keys[index];

        // remove the quality from the key
        key &= ~covariate_packer_quality_score<system>::chain::key_mask(covariate_packer_quality_score<system>::QualityScore);
    }
};

// computes the read group table based on the quality score table
template <target_system system>
void build_read_group_table(firepony_context<system>& context)
{
    auto& cv = context.covariates;

    if (cv.quality.size() == 0)
    {
        // if we didn't gather any entries in the table, there's nothing to do
        return;
    }

    // convert the quality table into the read group table
    covariate_observation_to_empirical_table(context, cv.quality, cv.read_group);
    // compute the expected error for each entry
    compute_expected_error<system, covariate_packer_quality_score<system> >(context, cv.read_group);

    // transform the read group table in place to remove the quality value from the key
    parallel<system>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + cv.quality.size(),
                               remove_quality_from_key<system>(context));

    // sort and pack the read group table
    auto& temp_keys = context.temp_u32;
    firepony::vector<system, covariate_empirical_value> temp_values;
    auto& temp_storage = context.temp_storage;

    cv.read_group.sort(temp_keys, temp_values, temp_storage, covariate_packer_quality_score<system>::chain::bits_used);
    cv.read_group.pack(temp_keys, temp_values, temp_storage);

    // finally compute the empirical quality for this table
    compute_empirical_quality(context, cv.read_group, false);
}
INSTANTIATE(build_read_group_table);

template <target_system system>
void output_read_group_table(firepony_context<system>& context)
{
    typedef covariate_packer_quality_score<system> packer;

    covariate_empirical_table<host> table;
    table.copyfrom(context.covariates.read_group);

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

