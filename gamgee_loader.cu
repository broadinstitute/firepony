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

#include "gamgee_loader.h"
#include "util.h"

gamgee_file::gamgee_file(const char *fname)
    : file(std::string(fname))
{
    gamgee_header = file.header();
    iterator = file.begin();
}

gamgee_file::~gamgee_file()
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

bool gamgee_file::next_batch(alignment_batch *batch, uint32 data_mask, const uint32 batch_size)
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

        if (data_mask & CHROMOSOME)
        {
            h_batch->chromosome.push_back(record.chromosome());
        }

        if (data_mask & ALIGNMENT_START)
        {
            h_batch->alignment_start.push_back(record.alignment_start());
        }

        if (data_mask & ALIGNMENT_STOP)
        {
            h_batch->alignment_stop.push_back(record.alignment_stop());
        }

        if (data_mask & MATE_CHROMOSOME)
        {
            h_batch->mate_chromosome.push_back(record.mate_chromosome());
        }

        if (data_mask & MATE_ALIGNMENT_START)
        {
            h_batch->mate_alignment_start.push_back(record.mate_alignment_start());
        }

        if (data_mask & CIGAR)
        {
            gamgee::Cigar cigar = record.cigar();

            h_batch->cigar_start.push_back(h_batch->cigar.size());
            h_batch->cigar_len.push_back(cigar.size());

            for(uint32 i = 0; i < cigar.size(); i++)
            {
                cigar_op op;

                op.op = gamgee_to_bqsr_cigar_op(cigar[i]);
                op.len = gamgee::Cigar::cigar_oplen(cigar[i]);

                h_batch->cigar.push_back(op);
            }
        }

        if (data_mask & READS)
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

        if (data_mask & QUALITIES)
        {
            gamgee::BaseQuals quals = record.base_quals();

            h_batch->qual_start.push_back(h_batch->qualities.size());
            h_batch->qual_len.push_back(quals.size());

            h_batch->qualities.resize(h_batch->qualities.size() + quals.size());
            memcpy(&h_batch->qualities[h_batch->qual_start[read_id]], &quals[0], quals.size());
        }

        if (data_mask & FLAGS)
        {
            h_batch->flags.push_back(extract_gamgee_flags(record));
        }

        if (data_mask & MAPQ)
        {
            h_batch->mapq.push_back(record.mapq());
        }

        if (data_mask & READ_GROUP)
        {
            // locate the RG tag
            gamgee::SamTag<std::string> tag = record.string_tag("RG");
            if (tag.missing())
            {
                // invalid read group
                h_batch->read_group.push_back(uint32(-1));
            } else {
                uint32 h = bqsr_string_hash(tag.value().c_str());
                uint32 rg_id;

                // xxxnsubtil: note that we don't validate the read groups against the header
                if (header.rg_name_to_id.find(h) == header.rg_name_to_id.end())
                {
                    // new read group, assign an ID and store in the header
                    rg_id = header.read_group_names.size();

                    header.rg_name_to_id[h] = rg_id;
                    header.read_group_names.push_back(tag.value());
                } else {
                    // we've seen this read group before, reuse the same ID
                    rg_id = header.rg_name_to_id[h];
                }

                h_batch->read_group.push_back(rg_id);
            }
        }
    }

    if (read_id == 0)
        return false;

    return true;
}

const char *gamgee_file::get_sequence_name(uint32 id)
{
    return gamgee_header.sequence_name(id).c_str();
}
