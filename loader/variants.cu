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

#include <gamgee/variant/variant_reader.h>

#include <string>

#include "../command_line.h"
#include "reference.h"
#include "../variant_database.h"

namespace firepony {

bool load_vcf(variant_database_host *output, reference_file_handle *reference_handle, const char *filename)
{
    std::vector<std::string> gamgee_chromosomes(0);

    for(auto& record : gamgee::VariantReader<gamgee::VariantIterator>{std::string(filename)})
    {
        // header.chromosomes() and record.chromosome_name() are *very* slow
        // unfortunately, gamgee does not really expose a better way of implementing this
        auto gamgee_chromosome_id = record.chromosome();
        if (gamgee_chromosome_id >= gamgee_chromosomes.size())
        {
            // reload the list of chromosomes from gamgee
            gamgee_chromosomes = record.header().chromosomes();
        }

        const std::string& chromosome_name = gamgee_chromosomes[gamgee_chromosome_id];
        reference_handle->make_sequence_available(chromosome_name);

        uint32 id = reference_handle->sequence_data.sequence_names.lookup(chromosome_name);
        if (id == uint32(-1))
        {
            fprintf(stderr, "WARNING: chromosome %s not found in reference data, skipping\n", chromosome_name.c_str());
            continue;
        }

        // make sure we have storage for the chromosome
        output->new_entry(id);
        auto& chromosome = output->get_sequence(id);

        // note: VCF positions are 1-based, but we convert to 0-based
        chromosome.feature_start.push_back(record.alignment_start() - 1);
        chromosome.feature_stop.push_back(record.alignment_stop() - 1);
    }

    // generate the search index for each chromosome
    for(auto *chromosome : output->storage)
    {
        chromosome->max_end_point_left.resize(chromosome->feature_stop.size());
        thrust::inclusive_scan(chromosome->feature_stop.begin(),
                               chromosome->feature_stop.end(),
                               chromosome->max_end_point_left.begin(),
                               thrust::maximum<uint32>());
    }

    return true;
}

} // namespace firepony
