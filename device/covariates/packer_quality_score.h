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

#pragma once

#include "bit_packers/read_group.h"
#include "bit_packers/quality_score.h"
#include "bit_packers/event_tracker.h"

namespace firepony {

// defines a covariate chain equivalent to GATK's RecalTable1
template <target_system system>
struct covariate_packer_quality_score
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

    static void dump_table(firepony_context<system>& context, covariate_empirical_table<system>& d_table)
    {
        covariate_empirical_table<host> table;
        table.copyfrom(d_table);

        const char *fmt_string_header =
#if DISABLE_OUTPUT_ROUNDING
               "#:GATKTable:6:%lu:%%s:%%s:%%s:%%.64f:%%d:%%.64f:;\n";
#else
               "#:GATKTable:6:%lu:%%s:%%s:%%s:%%.4f:%%d:%%.2f:;\n";
#endif
        printf(fmt_string_header, table.size());

        printf("#:GATKTable:RecalTable1:\n");
        printf("ReadGroup\tQualityScore\tEventType\tEmpiricalQuality\tObservations\tErrors\n");
        for(uint32 i = 0; i < table.size(); i++)
        {
            // skip null entries in the table
            if (table.values[i].observations == 0)
                continue;

            uint32 rg_id = decode(table.keys[i], ReadGroup);
            const std::string& rg_name = context.bam_header.host.read_groups_db.lookup(rg_id);

            const char *fmt_string =
#if DISABLE_OUTPUT_ROUNDING
                   "%s\t%d\t\t%c\t\t%.64f\t\t\t%d\t\t%.64f\n";
#else
                   "%s\t%d\t\t%c\t\t%.4f\t\t\t%d\t\t%.2f\n";
#endif

            printf(fmt_string,
                   rg_name.c_str(),
                   decode(table.keys[i], QualityScore),
                   cigar_event::ascii(decode(table.keys[i], EventTracker)),
                   table.values[i].empirical_quality,
                   table.values[i].observations,
                   round_n(table.values[i].mismatches, 2));
        }
        printf("\n");
    }
};

} // namespace firepony
