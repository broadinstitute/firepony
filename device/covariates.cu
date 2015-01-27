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

#include "../types.h"
#include "alignment_data_device.h"
#include "firepony_context.h"
#include "covariates.h"
#include "expected_error.h"
#include "empirical_quality.h"

#include "covariates/packer_context.h"
#include "covariates/packer_cycle_illumina.h"
#include "covariates/packer_quality_score.h"

#include "primitives/util.h"
#include "primitives/parallel.h"

#include <thrust/functional.h>

namespace firepony {

// accumulates events from a read batch into a given covariate table
template <target_system system, typename covariate_packer>
struct covariate_gatherer : public lambda<system>
{
    LAMBDA_INHERIT;

    CUDA_HOST_DEVICE void operator()(const uint32 cigar_event_index)
    {
        const uint32 read_index = ctx.cigar.cigar_event_read_index[cigar_event_index];

        if (read_index == uint32(-1))
        {
            // inactive read
            return;
        }

        const CRQ_index idx = batch.crq_index(read_index);
        const uint16 read_bp_offset = ctx.cigar.cigar_event_read_coordinates[cigar_event_index];
        if (read_bp_offset == uint16(-1))
        {
            return;
        }

        if (read_bp_offset < ctx.cigar.read_window_clipped[read_index].x ||
            read_bp_offset > ctx.cigar.read_window_clipped[read_index].y)
        {
            return;
        }

        if (ctx.active_location_list[idx.read_start + read_bp_offset] == 0)
        {
            return;
        }

        if (ctx.cigar.cigar_events[cigar_event_index] == cigar_event::S)
        {
            return;
        }

        covariate_key_set keys = covariate_packer::chain::encode(ctx, batch, read_index, read_bp_offset, cigar_event_index, covariate_key_set{0, 0, 0});

        ctx.covariates.scratch_table_space.keys  [cigar_event_index * 3 + 0] = keys.M;
        ctx.covariates.scratch_table_space.values[cigar_event_index * 3 + 0].observations = 1;
        ctx.covariates.scratch_table_space.values[cigar_event_index * 3 + 0].mismatches = ctx.fractional_error.snp_errors[idx.qual_start + read_bp_offset];

        ctx.covariates.scratch_table_space.keys  [cigar_event_index * 3 + 1] = keys.I;
        ctx.covariates.scratch_table_space.values[cigar_event_index * 3 + 1].observations = 1;
        ctx.covariates.scratch_table_space.values[cigar_event_index * 3 + 1].mismatches = ctx.fractional_error.insertion_errors[idx.qual_start + read_bp_offset];

        ctx.covariates.scratch_table_space.keys  [cigar_event_index * 3 + 2] = keys.D;
        ctx.covariates.scratch_table_space.values[cigar_event_index * 3 + 2].observations = 1;
        ctx.covariates.scratch_table_space.values[cigar_event_index * 3 + 2].mismatches = ctx.fractional_error.deletion_errors[idx.qual_start + read_bp_offset];
    }
};

// functor that determines if a key is valid
// note: operator() returns a uint32 to allow for composition of this functor with reduction operators
template <target_system system, typename covariate_packer>
struct is_key_valid : public thrust::unary_function<covariate_key, uint32>
{
    CUDA_HOST_DEVICE uint32 operator() (const covariate_key key)
    {
        constexpr bool sparse = covariate_packer::chain::is_sparse(covariate_packer::TargetCovariate);

        if (key == covariate_key(-1) ||
            (sparse && covariate_packer::decode(key, covariate_packer::TargetCovariate) == covariate_packer::chain::invalid_key(covariate_packer::TargetCovariate)))
        {
            return 0;
        } else {
            return 1;
        }
    }
};

template <target_system system, typename covariate_packer, typename Tuple>
struct is_key_value_pair_valid : public thrust::unary_function<Tuple, bool>
{
    CUDA_HOST_DEVICE uint32 operator() (const Tuple& T)
    {
        covariate_key key = thrust::get<0>(T);
        return is_key_valid<system, covariate_packer>()(key) ? true : false;
    }
};

// processes a batch of reads and updates covariate table data for a given table
template <typename covariate_packer, target_system system>
static void build_covariates_table(covariate_observation_table<system>& table, firepony_context<system>& context, const alignment_batch<system>& batch)
{
    auto& cv = context.covariates;
    auto& scratch_table = cv.scratch_table_space;

    firepony::vector<system, covariate_observation_value> temp_values;
    firepony::vector<system, covariate_key> temp_keys;

    timer<system> covariates_gather, covariates_filter, covariates_sort, covariates_pack;

    covariates_gather.start();

    // set up a scratch table space with enough room for 3 keys per cigar event
    scratch_table.resize(context.cigar.cigar_events.size() * 3);

    // mark all keys as invalid
    thrust::fill(scratch_table.keys.begin(),
                 scratch_table.keys.end(),
                 covariate_key(-1));

    // generate keys into the scratch table
    parallel<system>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + context.cigar.cigar_event_read_coordinates.size(),
                               covariate_gatherer<system, covariate_packer>(context, batch.device));

    covariates_gather.stop();

    covariates_filter.start();

    // count valid keys
    uint32 valid_keys = parallel<system>::sum(thrust::make_transform_iterator(scratch_table.keys.begin(),
                                                                              is_key_valid<system, covariate_packer>()),
                                              scratch_table.keys.size(),
                                              context.temp_storage);

    if (valid_keys)
    {
        // concatenate valid keys to the end of the output table
        size_t off = table.size();
        table.resize(table.size() + valid_keys);

        parallel<system>::copy_if(thrust::make_zip_iterator(thrust::make_tuple(scratch_table.keys.begin(),
                                                                               scratch_table.values.begin())),
                                  scratch_table.keys.size(),
                                  thrust::make_zip_iterator(thrust::make_tuple(table.keys.begin() + off,
                                                                               table.values.begin() + off)),
                                  is_key_value_pair_valid<system,
                                                          covariate_packer,
                                                          thrust::tuple<const covariate_key&, const typename covariate_observation_table<system>::value_type&> >(),
                                  context.temp_storage);
    }

    covariates_filter.stop();

    if (valid_keys)
    {
        // sort and reduce the table by key
        covariates_sort.start();
        table.sort(temp_keys, temp_values, context.temp_storage, covariate_packer::chain::bits_used);
        covariates_sort.stop();

        covariates_pack.start();
        table.pack(temp_keys, temp_values, context.temp_storage);
        covariates_pack.stop();
    }

    parallel<system>::synchronize();

    context.stats.covariates_gather.add(covariates_gather);
    context.stats.covariates_filter.add(covariates_filter);

    if (valid_keys)
    {
        context.stats.covariates_sort.add(covariates_sort);
        context.stats.covariates_pack.add(covariates_pack);
    }
}

template <target_system system>
struct compute_high_quality_windows : public lambda<system>
{
    LAMBDA_INHERIT;

    enum {
        // any bases with q <= LOW_QUAL_TAIL are considered low quality
        LOW_QUAL_TAIL = 2
    };

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const auto& window = ctx.cigar.read_window_clipped[read_index];
        auto& low_qual_window = ctx.covariates.high_quality_window[read_index];

        low_qual_window = window;

        while(batch.qualities[idx.qual_start + low_qual_window.x] <= LOW_QUAL_TAIL &&
                low_qual_window.x < low_qual_window.y)
        {
            low_qual_window.x++;
        }

        while(batch.qualities[idx.qual_start + low_qual_window.y] <= LOW_QUAL_TAIL &&
                low_qual_window.y > low_qual_window.x)
        {
            low_qual_window.y--;
        }
    }
};

template <target_system system>
void gather_covariates(firepony_context<system>& context, const alignment_batch<system>& batch)
{
    auto& cv = context.covariates;

    // compute the "high quality" windows (i.e., clip off low quality ends from each read)
    cv.high_quality_window.resize(batch.device.num_reads);
    parallel<system>::for_each(context.active_read_list.begin(),
                               context.active_read_list.end(),
                               compute_high_quality_windows<system>(context, batch.device));

    build_covariates_table<covariate_packer_quality_score<system> >(cv.quality, context, batch);
    build_covariates_table<covariate_packer_cycle_illumina<system> >(cv.cycle, context, batch);
    build_covariates_table<covariate_packer_context<system> >(cv.context, context, batch);
}
INSTANTIATE(gather_covariates);

template <target_system system> void postprocess_covariates(firepony_context<system>& context)
{
    auto& cv = context.covariates;

    // sort and pack all tables
    // this is required because we may have collected results from different devices
    vector<system, covariate_observation_value> temp_values;
    vector<system, covariate_key> temp_keys;

    cv.quality.sort(temp_keys, temp_values, context.temp_storage, covariate_packer_quality_score<system>::chain::bits_used);
    cv.quality.pack(temp_keys, temp_values, context.temp_storage);

    cv.cycle.sort(temp_keys, temp_values, context.temp_storage, covariate_packer_cycle_illumina<system>::chain::bits_used);
    cv.cycle.pack(temp_keys, temp_values, context.temp_storage);

    cv.context.sort(temp_keys, temp_values, context.temp_storage, covariate_packer_context<system>::chain::bits_used);
    cv.context.pack(temp_keys, temp_values, context.temp_storage);
}
INSTANTIATE(postprocess_covariates);

template <target_system system>
void output_covariates(firepony_context<system>& context)
{
    covariate_packer_quality_score<system>::dump_table(context, context.covariates.empirical_quality);

    printf("#:GATKTable:8:2386:%%s:%%s:%%s:%%s:%%s:%%.4f:%%d:%%.2f:;\n");
    printf("#:GATKTable:RecalTable2:\n");
    printf("ReadGroup\tQualityScore\tCovariateValue\tCovariateName\tEventType\tEmpiricalQuality\tObservations\tErrors\n");
    covariate_packer_context<system>::dump_table(context, context.covariates.context);
    covariate_packer_cycle_illumina<system>::dump_table(context, context.covariates.cycle);
    printf("\n");
}
INSTANTIATE(output_covariates);

template <target_system system>
void build_empirical_quality_score_table(firepony_context<system>& context)
{
    auto& cv = context.covariates;
    auto& table = cv.empirical_quality;

    if (cv.quality.size() == 0)
    {
        // if we didn't gather any entries in the table, there's nothing to do
        return;
    }

    // convert the quality table into the read group table
    covariate_observation_to_empirical_table(context, cv.quality, table);
    // compute the expected error for each entry
    compute_expected_error<system, covariate_packer_quality_score<system> >(context, table);
    // finally compute the empirical quality for this table
    compute_empirical_quality(context, table, true);
}
INSTANTIATE(build_empirical_quality_score_table);

} // namespace firepony

