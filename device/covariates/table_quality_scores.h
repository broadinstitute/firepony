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

namespace firepony {

// defines a covariate chain equivalent to GATK's RecalTable1
template <target_system system>
struct covariate_table_quality
{
    // the type that represents the chain of covariates
    typedef covariate_ReadGroup<system,
             covariate_QualityScore<system,
              covariate_EventTracker<system> > > chain;

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

    static void dump_table(firepony_context<system>& context, d_covariate_table<system>& d_table)
    {
        h_covariate_table table;
        table.copyfrom(d_table);

        printf("#:GATKTable:6:138:%%s:%%s:%%s:%%.4f:%%d:%%.2f:;\n");
        printf("#:GATKTable:RecalTable1:\n");
        printf("ReadGroup\tQualityScore\tEventType\tEmpiricalQuality\tObservations\tErrors\n");
        for(uint32 i = 0; i < table.size(); i++)
        {
            // skip null entries in the table
            if (table.values[i].observations == 0)
                continue;

            uint32 rg_id = decode(table.keys[i], ReadGroup);
            const std::string& rg_name = context.bam_header.host.read_groups_db.lookup(rg_id);

            printf("%s\t%d\t\t%c\t\t%.4f\t\t\t%d\t\t%.2f\n",
                    rg_name.c_str(),
                    decode(table.keys[i], QualityScore),
                    cigar_event::ascii(decode(table.keys[i], EventTracker)),
                    round_n(double(decode(table.keys[i], QualityScore)), 4),
                    table.values[i].observations,
                    round_n(table.values[i].mismatches, 2));
        }
        printf("\n");
    }
};

} // namespace firepony
