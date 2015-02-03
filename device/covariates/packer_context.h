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
#include "bit_packers/context.h"

namespace firepony {

// the context portion of GATK's RecalTable2
template <target_system system>
struct covariate_packer_context
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
    typedef covariate_ReadGroup<system,
             covariate_QualityScore<system,
              covariate_Context<system, num_bases_mismatch, num_bases_indel,
               covariate_EventTracker<system> > > > chain;

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

    static void dump_table(firepony_context<system>& context, covariate_empirical_table<system>& d_table)
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
            const char *fmt_string =
#if DISABLE_OUTPUT_ROUNDING
                   "%s\t%d\t\t%s\t\t%s\t\t%c\t\t%.64f\t\t%d\t\t%.64f\n";
#else
                   "%s\t%d\t\t%s\t\t%s\t\t%c\t\t%.4f\t\t%d\t\t%.2f\n";
#endif

            printf(fmt_string,
                   rg_name.c_str(),
                   decode(table.keys[i], QualityScore),
                   sequence,
                   "Context",
                   cigar_event::ascii(decode(table.keys[i], EventTracker)),
                   table.values[i].empirical_quality,
                   table.values[i].observations,
                   round_n(table.values[i].mismatches, 2));
        }
    }
};

} // namespace firepony
