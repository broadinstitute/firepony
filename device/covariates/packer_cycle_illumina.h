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

#include "../../table_formatter.h"
#include "bit_packers/read_group.h"
#include "bit_packers/quality_score.h"
#include "bit_packers/event_tracker.h"
#include "bit_packers/cycle_illumina.h"

namespace firepony {

// the cycle portion of GATK's RecalTable2
template <target_system system>
struct covariate_packer_cycle_illumina
{
    // the type that represents the chain of covariates
    typedef covariate_ReadGroup<system,
             covariate_QualityScore<system,
              covariate_Cycle_Illumina<system,
               covariate_EventTracker<system> > > > chain;

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

    static void dump_table(firepony_context<system>& context,
                           covariate_empirical_table<system>& d_table,
                           table_formatter& fmt)
    {
        covariate_empirical_table<host> table;
        table.copyfrom(d_table);

        for(uint32 i = 0; i < table.size(); i++)
        {
            // skip null entries in the table
            if (table.values[i].observations == 0)
                continue;

            uint32 rg_id = decode(table.keys[i], ReadGroup);
            const std::string& rg_name = context.bam_header.host.read_groups_db.lookup(rg_id);

            const char ev = cigar_event::ascii(decode(table.keys[i], EventTracker));
            const covariate_empirical_value& val = table.values[i];

            uint8 qual = decode(table.keys[i], QualityScore);

            // decode the group separately
            uint32 raw_group = decode(table.keys[i], Cycle);
            int group = raw_group >> 1;

            // apply the "sign" bit
            if (raw_group & 1)
                group = -group;

            fmt.start_row();

            fmt.data(rg_name);
            fmt.data_int_as_string(qual);
            fmt.data_int_as_string(group);
            fmt.data(std::string("Cycle"));
            fmt.data(ev);
            fmt.data(val.empirical_quality);
            fmt.data(val.observations);
            fmt.data(val.mismatches);

            fmt.end_row();
        }
    }
};

} // namespace firepony
