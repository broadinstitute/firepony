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

#include <string>

#include "gamgee_loader.h"
#include "util.h"
#include "alignment_data.h"
#include "sequence_data.h"

#include "from_nvbio/dna.h"

gamgee_alignment_file::gamgee_alignment_file(const char *fname)
    : file(std::string(fname))
{
    gamgee_header = file.header();

    for(uint32 i = 0; i < gamgee_header.n_sequences(); i++)
    {
        header.chromosome_lengths.push_back(gamgee_header.sequence_length(i));
    }
    header.d_chromosome_lengths = header.chromosome_lengths;

    iterator = file.begin();
}

gamgee_alignment_file::~gamgee_alignment_file()
{
}

static uint8 gamgee_to_bqsr_cigar_op(gamgee::CigarElement e)
{
    if (gamgee::Cigar::cigar_op(e) == gamgee::CigarOperator::B)
    {
        // xxxnsubtil: bqsr does not have B; return X instead
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

bool gamgee_alignment_file::next_batch(alignment_batch *batch, uint32 data_mask, const uint32 batch_size)
{
    alignment_batch_host *h_batch = &batch->host;
    uint32 read_id;

    batch->data_mask = data_mask;
    h_batch->reset(data_mask, batch_size);

    for(read_id = 0; read_id < batch_size; read_id++, ++iterator)
    {
        gamgee::Sam record = *iterator;
        if (record.empty())
        {
            break;
        }

        h_batch->num_reads++;
        h_batch->name.push_back(record.name());

        if (data_mask & AlignmentDataMask::CHROMOSOME)
        {
            h_batch->chromosome.push_back(record.chromosome());
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            h_batch->alignment_start.push_back(record.alignment_start() - 1);
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            h_batch->alignment_stop.push_back(record.alignment_stop() - 1);
        }

        if (data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            h_batch->mate_chromosome.push_back(record.mate_chromosome());
        }

        if (data_mask & AlignmentDataMask::MATE_ALIGNMENT_START)
        {
            h_batch->mate_alignment_start.push_back(record.mate_alignment_start() - 1);
        }

        if (data_mask & AlignmentDataMask::CIGAR)
        {
            gamgee::Cigar cigar = record.cigar();

            h_batch->cigar_start.push_back(h_batch->cigars.size());
            h_batch->cigar_len.push_back(cigar.size());

            for(uint32 i = 0; i < cigar.size(); i++)
            {
                cigar_op op;

                op.op = gamgee_to_bqsr_cigar_op(cigar[i]);
                op.len = gamgee::Cigar::cigar_oplen(cigar[i]);

                h_batch->cigars.push_back(op);
            }
        }

        if (data_mask & AlignmentDataMask::READS)
        {
            gamgee::ReadBases read = record.bases();

            h_batch->read_start.push_back(h_batch->reads.size());
            h_batch->read_len.push_back(read.size());

            // figure out the length of the sequence data,
            // rounded up to reach a dword boundary
            const uint32 padded_read_len_bp = ((read.size() + 7) / 8) * 8;

            // make sure we have enough memory, then read in the sequence
            h_batch->reads.resize(h_batch->reads.size() + padded_read_len_bp);

            // xxxnsubtil: this is going to be really slow
            for(uint32 i = 0; i < read.size(); i++)
            {
                h_batch->reads[h_batch->read_start[read_id] + i] = uint32(read[i]);
            }
        }

        if (data_mask & AlignmentDataMask::QUALITIES)
        {
            gamgee::BaseQuals quals = record.base_quals();

            h_batch->qual_start.push_back(h_batch->qualities.size());
            h_batch->qual_len.push_back(quals.size());

            h_batch->qualities.resize(h_batch->qualities.size() + quals.size());
            memcpy(&h_batch->qualities[h_batch->qual_start[read_id]], &quals[0], quals.size());
        }

        if (data_mask & AlignmentDataMask::FLAGS)
        {
            h_batch->flags.push_back(extract_gamgee_flags(record));
        }

        if (data_mask & AlignmentDataMask::MAPQ)
        {
            h_batch->mapq.push_back(record.mapping_qual());
        }

        if (data_mask & AlignmentDataMask::READ_GROUP)
        {
            // locate the RG tag
            gamgee::SamTag<std::string> tag = record.string_tag("RG");
            if (tag.missing())
            {
                // invalid read group
                h_batch->read_group.push_back(uint32(-1));
            } else {
                uint32 rg_id = header.read_groups_db.insert(tag.value());
                h_batch->read_group.push_back(rg_id);
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
bool gamgee_load_sequences(sequence_data *output, const char *filename, uint32 data_mask)
{
    sequence_data_host& h = output->host;
    output->data_mask = data_mask;

    printf("loading %s...\n", filename);

    for (gamgee::Fastq& record : gamgee::FastqReader(std::string(filename)))
    {
        printf("... %s (%lu bases)\n", record.name().c_str(), record.sequence().size());

        h.num_sequences++;

        if (data_mask & SequenceDataMask::BASES)
        {
            std::string sequence = record.sequence();

            const size_t seq_start = h.bases.size();
            const size_t seq_len = sequence.size();

            h.sequence_bp_start.push_back(seq_start);
            h.sequence_bp_len.push_back(seq_len);

            h.bases.resize(seq_start + seq_len);

            bqsr::assign(sequence.size(),
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
            uint32 seq_id = h.sequence_names.insert(record.name());
            h.sequence_id.push_back(seq_id);
        }
    }

    return true;
}
