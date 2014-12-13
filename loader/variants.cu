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

#include <gamgee/variant/variant_reader.h>

#include <string>

#include "../variant_data.h"
#include "../command_line.h"
#include "reference.h"

#include "../device/util.h"
#include "../device/from_nvbio/dna.h"

namespace firepony {

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
            uint32 chromosome_window_start;
            uint32 reference_window_start;
            uint32 alignment_window_len;
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

        variant_data.chromosome_window_start = record.alignment_start();
        // note: VCF positions are 1-based, but we convert to 0-based in the reference window
        variant_data.reference_window_start = record.alignment_start() + reference_handle->sequence_data.sequence_bp_start[variant_data.chromosome] - 1;
        variant_data.alignment_window_len = record.alignment_stop() - record.alignment_start();

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
                h.chromosome_window_start.push_back(variant_data.chromosome_window_start);
                h.reference_window_start.push_back(variant_data.reference_window_start);
                h.alignment_window_len.push_back(variant_data.alignment_window_len);
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
                for(uint32 i = 0; i < ref_len; i++)
                {
                    h.reference_sequence[ref_start + i] = from_nvbio::char_to_iupac16(ref[i]);
                }
            }

            if (data_mask & VariantDataMask::ALTERNATE)
            {
                const uint32 alt_start = h.alternate_sequence.size();
                const uint32 alt_len = alt.size();

                h.alternate_sequence_start.push_back(alt_start);
                h.alternate_sequence_len.push_back(alt_len);

                // xxxnsubtil: same as above, this is not padded to dword size
                h.alternate_sequence.resize(alt_start + alt_len);
                for(uint32 i = 0; i < alt_len; i++)
                {
                    h.alternate_sequence[alt_start + i] = from_nvbio::char_to_iupac16(alt[i]);
                }
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
