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

#pragma once

#include "bit_packers/read_group.h"
#include "bit_packers/quality_score.h"
#include "bit_packers/event_tracker.h"
#include "bit_packers/cycle_illumina.h"

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
                    round_n(double(decode(table.keys[i], QualityScore)), 4),
                    table.values[i].observations,
                    round_n(table.values[i].mismatches, 2));
        }
    }
};
