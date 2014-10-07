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
#include "covariates_bit_packing.h"

#include "primitives/util.h"
#include "primitives/parallel.h"

#include <thrust/functional.h>

// defines a covariate chain equivalent to GATK's RecalTable1
struct covariates_quality_table
{
    // the type that represents the chain of covariates
    typedef covariate_ReadGroup<
             covariate_QualityScore<
              covariate_EventTracker<> > > chain;

    // the index of each covariate in the chain
    // (used when decoding a key)
    // the order is defined by the typedef above
    typedef enum {
        ReadGroup = 3,
        QualityScore = 2,
        EventTracker = 1,

        // target covariate is mostly meaningless for recaltable1
        TargetCovariate = QualityScore,
    } CovariateID;

    // extract a given covariate value from a key
    static CUDA_HOST_DEVICE uint32 decode(covariate_key key, CovariateID id)
    {
        return chain::decode(key, id);
    }

    static void dump_table(bqsr_context *context, D_CovariateTable& d_table)
    {
        H_CovariateTable table;
        table.copyfrom(d_table);

        printf("#:GATKTable:6:138:%%s:%%s:%%s:%%.4f:%%d:%%.2f:;\n");
        printf("#:GATKTable:RecalTable1:\n");
        printf("ReadGroup\tQualityScore\tEventType\tEmpiricalQuality\tObservations\tErrors\n");
        for(uint32 i = 0; i < table.size(); i++)
        {
            uint32 rg_id = decode(table.keys[i], ReadGroup);
            const std::string& rg_name = context->bam_header.read_groups_db.lookup(rg_id);

            printf("%s\t%d\t\t%c\t\t%.4f\t\t\t%d\t\t%.2f\n",
                    rg_name.c_str(),
                    decode(table.keys[i], QualityScore),
                    cigar_event::ascii(decode(table.keys[i], EventTracker)),
                    float(decode(table.keys[i], QualityScore)),
                    table.values[i].observations,
                    table.values[i].mismatches);
        }
        printf("\n");
    }
};

// the cycle portion of GATK's RecalTable2
struct covariate_table_cycle_illumina
{
    // the type that represents the chain of covariates
    typedef covariate_ReadGroup<
             covariate_QualityScore<
              covariate_Cycle_Illumina<
               covariate_EventTracker<> > > > chain;

    // the index of each covariate in the chain
    // (used when decoding a key)
    // the order is defined by the typedef above
    typedef enum {
        ReadGroup = 4,
        QualityScore = 3,
        Cycle = 2,
        EventTracker = 1,

        // defines which covariate is the "target" for this table
        // used when checking for invalid keys
        TargetCovariate = Cycle,
    } CovariateID;

    // extract a given covariate value from a key
    static CUDA_HOST_DEVICE uint32 decode(covariate_key key, CovariateID id)
    {
        return chain::decode(key, id);
    }

    static void dump_table(bqsr_context *context, D_CovariateTable& d_table)
    {
        H_CovariateTable table;
        table.copyfrom(d_table);

        for(uint32 i = 0; i < table.size(); i++)
        {
            uint32 rg_id = decode(table.keys[i], ReadGroup);
            const std::string& rg_name = context->bam_header.read_groups_db.lookup(rg_id);

            // decode the group separately
            uint32 raw_group = decode(table.keys[i], Cycle);
            int group = raw_group >> 1;

            // apply the "sign" bit
            if (raw_group & 1)
                group = -group;

            // ReadGroup, QualityScore, CovariateValue, CovariateName, EventType, EmpiricalQuality, Observations, Errors
            printf("%s\t%d\t\t%d\t\t%s\t\t%c\t\t%.4f\t\t%d\t\t%.2f\n",
                    rg_name.c_str(),
                    decode(table.keys[i], QualityScore),
                    group,
                    "Cycle",
                    cigar_event::ascii(decode(table.keys[i], EventTracker)),
                    float(decode(table.keys[i], QualityScore)),
                    table.values[i].observations,
                    table.values[i].mismatches);
        }
    }
};

// the context portion of GATK's RecalTable2
struct covariate_table_context
{
    enum {
        num_bases_mismatch = 2,
        num_bases_indel = 3,

        // xxxnsubtil: this is duplicated from covariate_Context
        num_bases_in_context = constexpr_max(num_bases_mismatch, num_bases_indel),

        base_bits = num_bases_in_context * 2,
        base_bits_mismatch = num_bases_mismatch * 2,
        base_bits_indel = num_bases_indel * 2,

        length_bits = 4,
    };

    // the type that represents the chain of covariates
    typedef covariate_ReadGroup<
             covariate_QualityScore<
              covariate_Context<num_bases_mismatch, num_bases_indel,
               covariate_EventTracker<> > > > chain;

    // the index of each covariate in the chain
    // (used when decoding a key)
    // the order is defined by the typedef above
    typedef enum {
        ReadGroup = 4,
        QualityScore = 3,
        Context = 2,
        EventTracker = 1,

        // defines which covariate is the "target" for this table
        // used when checking for invalid keys
        TargetCovariate = Context,
    } CovariateID;

    // extract a given covariate value from a key
    static CUDA_HOST_DEVICE uint32 decode(covariate_key key, CovariateID id)
    {
        return chain::decode(key, id);
    }

    static void dump_table(bqsr_context *context, D_CovariateTable& d_table)
    {
        H_CovariateTable table;
        table.copyfrom(d_table);

        for(uint32 i = 0; i < table.size(); i++)
        {
            uint32 rg_id = decode(table.keys[i], ReadGroup);
            const std::string& rg_name = context->bam_header.read_groups_db.lookup(rg_id);

            // decode the context separately
            covariate_key raw_context = decode(table.keys[i], Context);
            int size = (raw_context & BITMASK(length_bits));
            covariate_key context = raw_context >> length_bits;

            char sequence[size + 1];

            for(int j = 0; j < size; j++)
            {
                const int offset = j * 2;
                sequence[j] = from_nvbio::dna_to_char((context >> offset) & 3);
            }
            sequence[size] = 0;

            // ReadGroup, QualityScore, CovariateValue, CovariateName, EventType, EmpiricalQuality, Observations, Errors
            printf("%s\t%d\t\t%s\t\t%s\t\t%c\t\t%.4f\t\t%d\t\t%.2f\n",
                    rg_name.c_str(),
                    decode(table.keys[i], QualityScore),
                    sequence,
                    "Context",
                    cigar_event::ascii(decode(table.keys[i], EventTracker)),
                    float(decode(table.keys[i], QualityScore)),
                    table.values[i].observations,
                    table.values[i].mismatches);
        }
    }
};

template <typename covariate_packer>
struct covariate_gatherer : public bqsr_lambda
{
    using bqsr_lambda::bqsr_lambda;

    CUDA_DEVICE void operator()(const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        for(uint32 i = cigar_start; i < cigar_end; i++)
        {
            uint16 read_bp_offset = ctx.cigar.cigar_event_read_coordinates[i];
            if (read_bp_offset == uint16(-1))
            {
                continue;
            }

            if (ctx.active_location_list[idx.read_start + read_bp_offset] == 0)
            {
                continue;
            }

            if (ctx.cigar.cigar_events[i] == cigar_event::S)
            {
                continue;
            }

            covariate_key_set keys = covariate_packer::chain::encode(ctx, batch, read_index, read_bp_offset, i, covariate_key_set{0, 0, 0});

            ctx.covariates.scratch_table_space.keys  [i * 3 + 0] = keys.M;
            ctx.covariates.scratch_table_space.values[i * 3 + 0].observations = 1;
            ctx.covariates.scratch_table_space.values[i * 3 + 0].mismatches = ctx.fractional_error.snp_errors[idx.qual_start + read_bp_offset];

            ctx.covariates.scratch_table_space.keys  [i * 3 + 1] = keys.I;
            ctx.covariates.scratch_table_space.values[i * 3 + 1].observations = 1;
            ctx.covariates.scratch_table_space.values[i * 3 + 1].mismatches = ctx.fractional_error.insertion_errors[idx.qual_start + read_bp_offset];

            ctx.covariates.scratch_table_space.keys  [i * 3 + 2] = keys.D;
            ctx.covariates.scratch_table_space.values[i * 3 + 2].observations = 1;
            ctx.covariates.scratch_table_space.values[i * 3 + 2].mismatches = ctx.fractional_error.deletion_errors[idx.qual_start + read_bp_offset];
        }
    }
};

template <typename covariate_packer>
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
        constexpr bool sparse = covariate_packer::chain::is_sparse(covariate_packer::TargetCovariate);

        const covariate_key& key = ctx.covariates.scratch_table_space.keys[key_index];

        if (key == covariate_key(-1) ||
            (sparse && covariate_packer::decode(key, covariate_packer::TargetCovariate) == covariate_packer::chain::invalid_key(covariate_packer::TargetCovariate)))
        {
            flags[key_index] = 0;
        } else {
            flags[key_index] = 1;
        }
    }
};

template <typename covariate_packer>
void build_covariates_table(D_CovariateTable& table, bqsr_context *context, const alignment_batch& batch)
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
    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     covariate_gatherer<covariate_packer>(*context, batch.device));

    // flag valid keys
    thrust::for_each(thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(0u) + cv.scratch_table_space.keys.size(),
                     flag_valid_keys<covariate_packer>(*context, batch.device, flags));

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
    table.sort(temp_keys, temp_values, context->temp_storage, covariate_packer::chain::next_offset);
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

    build_covariates_table<covariates_quality_table>(cv.quality, context, batch);
    build_covariates_table<covariate_table_cycle_illumina>(cv.cycle, context, batch);
    build_covariates_table<covariate_table_context>(cv.context, context, batch);
}

void output_covariates(bqsr_context *context)
{
    covariates_quality_table::dump_table(context, context->covariates.quality);

    printf("#:GATKTable:8:2386:%%s:%%s:%%s:%%s:%%s:%%.4f:%%d:%%.2f:;\n");
    printf("#:GATKTable:RecalTable2:\n");
    printf("ReadGroup\tQualityScore\tCovariateValue\tCovariateName\tEventType\tEmpiricalQuality\tObservations\tErrors\n");
    covariate_table_cycle_illumina::dump_table(context, context->covariates.cycle);
    covariate_table_context::dump_table(context, context->covariates.context);
    printf("\n");
}
