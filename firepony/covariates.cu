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


#include "bqsr_types.h"
#include "alignment_data.h"
#include "bqsr_context.h"
#include "covariates.h"

#include "covariates/table_context.h"
#include "covariates/table_cycle_illumina.h"
#include "covariates/table_quality_scores.h"

#include "primitives/util.h"
#include "primitives/parallel.h"

#include <thrust/functional.h>

// accumulates events from a read batch into a given covariate table
template <typename covariate_table>
struct covariate_gatherer : public bqsr_lambda
{
    using bqsr_lambda::bqsr_lambda;

    CUDA_DEVICE void operator()(const uint32 cigar_event_index)
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

        if (ctx.active_location_list[idx.read_start + read_bp_offset] == 0)
        {
            return;
        }

        if (ctx.cigar.cigar_events[cigar_event_index] == cigar_event::S)
        {
            return;
        }

        covariate_key_set keys = covariate_table::chain::encode(ctx, batch, read_index, read_bp_offset, cigar_event_index, covariate_key_set{0, 0, 0});

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

// functor used for filtering out invalid keys in a table
template <typename covariate_table>
struct flag_valid_keys : public bqsr_lambda
{
    D_VectorU8::view flags;

    flag_valid_keys(bqsr_context::view ctx,
                    const alignment_batch_device::const_view batch,
                    D_VectorU8::view flags)
        : bqsr_lambda(ctx, batch), flags(flags)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 key_index)
    {
        constexpr bool sparse = covariate_table::chain::is_sparse(covariate_table::TargetCovariate);

        const covariate_key& key = ctx.covariates.scratch_table_space.keys[key_index];

        if (key == covariate_key(-1) ||
            (sparse && covariate_table::decode(key, covariate_table::TargetCovariate) == covariate_table::chain::invalid_key(covariate_table::TargetCovariate)))
        {
            flags[key_index] = 0;
        } else {
            flags[key_index] = 1;
        }
    }
};

// processes a batch of reads and updates covariate table data for a given table
template <typename covariate_table>
static void build_covariates_table(D_CovariateTable& table, bqsr_context *context, const alignment_batch& batch)
{
    covariates_context& cv = context->covariates;
    auto& scratch_table = cv.scratch_table_space;

    D_VectorU8& flags = context->temp_u8;

    D_Vector<covariate_value> temp_values;
    D_Vector<covariate_key> temp_keys;

    // set up a scratch table space with enough room for 3 keys per cigar event
    scratch_table.resize(context->cigar.cigar_events.size() * 3);
    table.resize(context->cigar.cigar_events.size() * 3);
    flags.resize(context->cigar.cigar_events.size() * 3);

    // mark all keys as invalid
    thrust::fill(scratch_table.keys.begin(),
                 scratch_table.keys.end(),
                 covariate_key(-1));

    // generate keys into the scratch table
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + context->cigar.cigar_event_read_coordinates.size(),
                     covariate_gatherer<covariate_table>(*context, batch.device));

    // flag valid keys
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + cv.scratch_table_space.keys.size(),
                     flag_valid_keys<covariate_table>(*context, batch.device, flags));

    // copy valid keys into the output table
    bqsr::copy_flagged(scratch_table.keys.begin(),
                       flags.size(),
                       table.keys.begin(),
                       flags.begin(),
                       context->temp_storage);

    bqsr::copy_flagged(scratch_table.values.begin(),
                       flags.size(),
                       table.values.begin(),
                       flags.begin(),
                       context->temp_storage);

    // count valid keys
    uint32 valid_keys = thrust::reduce(flags.begin(), flags.end(), uint32(0));

    // resize the table
    table.resize(valid_keys);

    // sort and reduce the table by key
    table.sort(temp_keys, temp_values, context->temp_storage, covariate_table::chain::bits_used);
    table.pack(temp_keys, temp_values);
}

struct compute_high_quality_windows : public bqsr_lambda
{
    using bqsr_lambda::bqsr_lambda;

    enum {
        // any bases with q <= LOW_QUAL_TAIL are considered low quality
        LOW_QUAL_TAIL = 2
    };

    CUDA_DEVICE void operator() (const uint32 read_index)
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
                low_qual_window.y >= window.x)
        {
            low_qual_window.y--;
        }
    }
};

void gather_covariates(bqsr_context *context, const alignment_batch& batch)
{
    auto& cv = context->covariates;

    // compute the "high quality" windows (i.e., clip off low quality ends from each read)
    cv.high_quality_window.resize(batch.device.num_reads);
    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     compute_high_quality_windows(*context, batch.device));

    build_covariates_table<covariate_table_quality>(cv.quality, context, batch);
    build_covariates_table<covariate_table_cycle_illumina>(cv.cycle, context, batch);
    build_covariates_table<covariate_table_context>(cv.context, context, batch);
}

void output_covariates(bqsr_context *context)
{
    covariate_table_quality::dump_table(context, context->covariates.quality);

    printf("#:GATKTable:8:2386:%%s:%%s:%%s:%%s:%%s:%%.4f:%%d:%%.2f:;\n");
    printf("#:GATKTable:RecalTable2:\n");
    printf("ReadGroup\tQualityScore\tCovariateValue\tCovariateName\tEventType\tEmpiricalQuality\tObservations\tErrors\n");
    covariate_table_context::dump_table(context, context->covariates.context);
    covariate_table_cycle_illumina::dump_table(context, context->covariates.cycle);
    printf("\n");
}
