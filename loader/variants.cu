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

#include "../variant_data.h"
#include "../command_line.h"
#include "reference.h"

#include "../device/util.h"
#include "../device/from_nvbio/dna.h"

namespace firepony {

#include <thrust/iterator/transform_iterator.h>

struct iupac16 : public thrust::unary_function<char, uint8>
{
    uint8 operator() (char in)
    {
        return from_nvbio::char_to_iupac16(in);
    }
};

bool load_vcf(variant_database_host *output, reference_file_handle *reference_handle, const char *filename, uint32 data_mask)
{
    auto& h = *output;
    h.data_mask = data_mask;

    std::vector<std::string> gamgee_chromosomes(0);

    for(auto& record : gamgee::VariantReader<gamgee::VariantIterator>{std::string(filename)})
    {
        // collect all the common variant data
        struct
        {
            uint32 chromosome;
            uint32 feature_start;
            uint32 feature_stop;
            uint32 reference_sequence_start;
            uint32 reference_sequence_len;
            uint32 id;
            float qual;
            uint32 n_samples;
            uint32 n_alleles;
        } variant_data;

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

        if (data_mask & VariantDataMask::CHROMOSOME)
        {
            uint32 id = reference_handle->sequence_data.sequence_names.lookup(chromosome_name);
            if (id == uint32(-1))
            {
                fprintf(stderr, "WARNING: chromosome %s not found in reference data, skipping\n", chromosome_name.c_str());
                continue;
            }

            variant_data.chromosome = id;
        }

        // note: VCF positions are 1-based, but we convert to 0-based
        variant_data.feature_start = record.alignment_start() + reference_handle->sequence_data.sequence_bp_start[variant_data.chromosome] - 1;
        variant_data.feature_stop = record.alignment_stop() + reference_handle->sequence_data.sequence_bp_start[variant_data.chromosome] - 1;

        if (data_mask & VariantDataMask::ID)
        {
            variant_data.id = output->id_db.insert(record.id());
        }

        variant_data.qual = record.qual();
        variant_data.n_samples = record.n_samples();
        variant_data.n_alleles = record.n_alleles();

        // for each possible alternate, add one entry to the variant database
        // xxxnsubtil: this is wrong if alternate data is masked out in data_mask!
        for(const std::string& alt : record.alt())
        {
            h.num_variants++;

            if (data_mask & VariantDataMask::CHROMOSOME)
            {
                h.chromosome.push_back(variant_data.chromosome);
            }

            if (data_mask & VariantDataMask::ALIGNMENT)
            {
                h.feature_start.push_back(variant_data.feature_start);
                h.feature_stop.push_back(variant_data.feature_stop);
            }

            if (data_mask & VariantDataMask::ID)
            {
                h.id.push_back(variant_data.id);
            }

            if (data_mask & VariantDataMask::REFERENCE)
            {
                const std::string& ref = record.ref();
                const uint32 ref_start = h.reference_sequence.size();
                const uint32 ref_len = ref.size();

                h.reference_sequence_start.push_back(ref_start);
                h.reference_sequence_len.push_back(ref_len);

                // make sure we have enough memory, then read in the sequence
                // xxxnsubtil: we don't pad reference data to a dword multiple since variants are
                // meant to be read-only, so RMW hazards should never happen
                h.reference_sequence.resize(ref_start + ref_len);
                assign(ref_len,
                       thrust::make_transform_iterator(ref.begin(), iupac16()),
                       h.reference_sequence.stream_at_index(ref_start));
            }

            if (data_mask & VariantDataMask::ALTERNATE)
            {
                const uint32 alt_start = h.alternate_sequence.size();
                const uint32 alt_len = alt.size();

                h.alternate_sequence_start.push_back(alt_start);
                h.alternate_sequence_len.push_back(alt_len);

                // xxxnsubtil: same as above, this is not padded to dword size
                h.alternate_sequence.resize(alt_start + alt_len);
                assign(alt_len,
                       thrust::make_transform_iterator(alt.begin(), iupac16()),
                       h.alternate_sequence.stream_at_index(alt_start));
            }

            if (data_mask & VariantDataMask::QUAL)
            {
                h.qual.push_back(variant_data.qual);
            }

            if (data_mask & VariantDataMask::N_SAMPLES)
            {
                h.n_samples.push_back(variant_data.n_samples);
            }

            if (data_mask & VariantDataMask::N_ALLELES)
            {
                h.n_alleles.push_back(variant_data.n_alleles);
            }
        }
    }

    return true;
}

} // namespace firepony
