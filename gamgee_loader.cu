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

#include <gamgee/sam.h>
#include <gamgee/sam_iterator.h>
#include <gamgee/sam_tag.h>
#include <gamgee/fastq.h>
#include <gamgee/fastq_iterator.h>
#include <gamgee/fastq_reader.h>
#include <gamgee/variant_reader.h>

#include <string>

#include "alignment_data.h"
#include "sequence_data.h"
#include "variant_data.h"
#include "gamgee_loader.h"

#include "device/util.h"
#include "device/from_nvbio/dna.h"

namespace firepony {

gamgee_alignment_file::gamgee_alignment_file(const char *fname)
    : file(std::string(fname))
{
    gamgee_header = file.header();

    // build the read group ID map
    auto read_groups = gamgee_header.read_groups();
    for(const auto& rg : read_groups)
    {
        std::string name;
        if (rg.platform_unit.size() != 0)
        {
            name = rg.platform_unit;
        } else {
            name = rg.id;
        }

        read_group_id_to_name[rg.id] = name;
    }

    for(uint32 i = 0; i < gamgee_header.n_sequences(); i++)
    {
        header.chromosome_lengths.push_back(gamgee_header.sequence_length(i));
    }

    iterator = file.begin();
}

gamgee_alignment_file::~gamgee_alignment_file()
{
}

static uint8 gamgee_to_firepony_cigar_op(gamgee::CigarElement e)
{
    if (gamgee::Cigar::cigar_op(e) == gamgee::CigarOperator::B)
    {
        // xxxnsubtil: firepony does not have B; return X instead
        return cigar_op::OP_X;
    } else {
        return uint8(gamgee::Cigar::cigar_op(e));
    }
}

static uint32 extract_gamgee_flags(gamgee::Sam& record)
{
    uint32 flags = 0;

    if (record.paired())
        flags |= AlignmentFlags::PAIRED;

    if (record.properly_paired())
        flags |= AlignmentFlags::PROPER_PAIR;

    if (record.unmapped())
        flags |= AlignmentFlags::UNMAP;

    if (record.mate_unmapped())
        flags |= AlignmentFlags::MATE_UNMAP;

    if (record.reverse())
        flags |= AlignmentFlags::REVERSE;

    if (record.mate_reverse())
        flags |= AlignmentFlags::MATE_REVERSE;

    if (record.first())
        flags |= AlignmentFlags::READ1;

    if (record.last())
        flags |= AlignmentFlags::READ2;

    if (record.secondary())
        flags |= AlignmentFlags::SECONDARY;

    if (record.fail())
        flags |= AlignmentFlags::QC_FAIL;

    if (record.duplicate())
        flags |= AlignmentFlags::DUPLICATE;

    if (record.supplementary())
        flags |= AlignmentFlags::SUPPLEMENTARY;

    return flags;
}

bool gamgee_alignment_file::next_batch(alignment_batch_host *batch, uint32 data_mask, const uint32 batch_size)
{
    uint32 read_id;

    batch->reset(data_mask, batch_size);

    for(read_id = 0; read_id < batch_size; read_id++, ++iterator)
    {
        gamgee::Sam record = *iterator;
        if (record.empty())
        {
            break;
        }

        batch->num_reads++;
        batch->name.push_back(record.name());

        if (data_mask & AlignmentDataMask::CHROMOSOME)
        {
            batch->chromosome.push_back(record.chromosome());
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            batch->alignment_start.push_back(record.alignment_start() - 1);
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            batch->alignment_stop.push_back(record.alignment_stop() - 1);
        }

        if (data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            batch->mate_chromosome.push_back(record.mate_chromosome());
        }

        if (data_mask & AlignmentDataMask::MATE_ALIGNMENT_START)
        {
            batch->mate_alignment_start.push_back(record.mate_alignment_start() - 1);
        }

        if (data_mask & AlignmentDataMask::INFERRED_INSERT_SIZE)
        {
            batch->inferred_insert_size.push_back(record.insert_size());
        }

        if (data_mask & AlignmentDataMask::CIGAR)
        {
            gamgee::Cigar cigar = record.cigar();

            batch->cigar_start.push_back(batch->cigars.size());
            batch->cigar_len.push_back(cigar.size());

            for(uint32 i = 0; i < cigar.size(); i++)
            {
                cigar_op op;

                op.op = gamgee_to_firepony_cigar_op(cigar[i]);
                op.len = gamgee::Cigar::cigar_oplen(cigar[i]);

                batch->cigars.push_back(op);
            }
        }

        if (data_mask & AlignmentDataMask::READS)
        {
            gamgee::ReadBases read = record.bases();

            batch->read_start.push_back(batch->reads.size());
            batch->read_len.push_back(read.size());

            // figure out the length of the sequence data,
            // rounded up to reach a dword boundary
            const uint32 padded_read_len_bp = ((read.size() + 7) / 8) * 8;

            // make sure we have enough memory, then read in the sequence
            batch->reads.resize(batch->reads.size() + padded_read_len_bp);

            // xxxnsubtil: this is going to be really slow
            for(uint32 i = 0; i < read.size(); i++)
            {
                batch->reads[batch->read_start[read_id] + i] = uint32(read[i]);
            }

            if (batch->max_read_size < read.size())
            {
                batch->max_read_size = read.size();
            }
        }

        if (data_mask & AlignmentDataMask::QUALITIES)
        {
            gamgee::BaseQuals quals = record.base_quals();

            batch->qual_start.push_back(batch->qualities.size());
            batch->qual_len.push_back(quals.size());

            batch->qualities.resize(batch->qualities.size() + quals.size());
            memcpy(&batch->qualities[batch->qual_start[read_id]], &quals[0], quals.size());
        }

        if (data_mask & AlignmentDataMask::FLAGS)
        {
            batch->flags.push_back(extract_gamgee_flags(record));
        }

        if (data_mask & AlignmentDataMask::MAPQ)
        {
            batch->mapq.push_back(record.mapping_qual());
        }

        if (data_mask & AlignmentDataMask::READ_GROUP)
        {
            // locate the RG tag
            gamgee::SamTag<std::string> tag = record.string_tag("RG");
            if (tag.missing())
            {
                // invalid read group
                batch->read_group.push_back(uint32(-1));
            } else {
                auto iter = read_group_id_to_name.find(tag.value());
                if (iter == read_group_id_to_name.end())
                {
                    fprintf(stderr, "WARNING: found read with invalid read group identifier\n");
                    batch->read_group.push_back(uint32(-1));
                } else {
                    uint32 rg_id = header.read_groups_db.insert(iter->second);
                    batch->read_group.push_back(rg_id);
                }
            }
        }
    }

    if (read_id == 0)
        return false;

    return true;
}

const char *gamgee_alignment_file::get_sequence_name(uint32 id)
{
    return gamgee_header.sequence_name(id).c_str();
}

#include <thrust/iterator/transform_iterator.h>

struct iupac16 : public thrust::unary_function<char, uint8>
{
    uint8 operator() (char in)
    {
        return from_nvbio::char_to_iupac16(in);
    }
};

// loader for sequence data
bool gamgee_load_sequences(sequence_data_host *output, const char *filename, uint32 data_mask)
{
    auto& h = output->host_malloc_container;
    h.data_mask = data_mask;

    for (gamgee::Fastq& record : gamgee::FastqReader(std::string(filename)))
    {
        h.num_sequences++;

        if (data_mask & SequenceDataMask::BASES)
        {
            std::string sequence = record.sequence();

            const size_t seq_start = h.bases.size();
            const size_t seq_len = sequence.size();

            h.sequence_bp_start.push_back(seq_start);
            h.sequence_bp_len.push_back(seq_len);

            h.bases.resize(seq_start + seq_len);

            assign(sequence.size(),
                   thrust::make_transform_iterator(sequence.begin(), iupac16()),
                   h.bases.stream_at_index(seq_start));
        }

        if (data_mask & SequenceDataMask::QUALITIES)
        {
            assert(!"unimplemented");
            return false;
        }

        if (data_mask & SequenceDataMask::NAMES)
        {
            uint32 seq_id = output->sequence_names.insert(record.name());
            h.sequence_id.push_back(seq_id);
        }
    }

    return true;
}

bool gamgee_load_vcf(variant_database_host *output, const sequence_data_host& reference, const char *filename, uint32 data_mask)
{
    auto& h = output->host_malloc_container;
    h.data_mask = data_mask;

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

        if (data_mask & VariantDataMask::CHROMOSOME)
        {
            uint32 gamgee_chromosome_id = record.chromosome();
            std::string chromosome_name = record.header().chromosomes()[gamgee_chromosome_id];

            uint32 id = reference.sequence_names.lookup(chromosome_name);
            if (id == uint32(-1))
            {
                fprintf(stderr, "WARNING: chromosome %s not found in reference data, skipping\n", chromosome_name.c_str());
                continue;
            }

            variant_data.chromosome = id;
        }

        variant_data.chromosome_window_start = record.alignment_start();
        // note: VCF positions are 1-based, but we convert to 0-based in the reference window
        variant_data.reference_window_start = record.alignment_start() + reference.sequence_bp_start[variant_data.chromosome] - 1;
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
        for(std::string& alt : record.alt())
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
