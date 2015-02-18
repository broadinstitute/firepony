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

#include <gamgee/sam/sam.h>
#include <gamgee/sam/sam_iterator.h>
#include <gamgee/sam/sam_tag.h>

#include <string>

#include "../alignment_data.h"
#include "alignments.h"
#include "reference.h"

#include "../device/util.h"
#include "../device/from_nvbio/dna.h"

namespace firepony {

alignment_file::alignment_file(const char *fname)
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

alignment_file::~alignment_file()
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

bool alignment_file::next_batch(alignment_batch_host *batch, uint32 data_mask, reference_file_handle *reference, const uint32 batch_size)
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
            std::string sequence_name = get_sequence_name(record.chromosome());
            const bool seq_valid = reference->make_sequence_available(sequence_name);

            if (seq_valid)
            {
                batch->chromosome.push_back(reference->sequence_data.sequence_names.lookup(sequence_name));
            } else {
                batch->chromosome.push_back(uint32(-1));
            }
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
            std::string sequence_name = get_sequence_name(record.mate_chromosome());
            const bool seq_valid = reference->make_sequence_available(sequence_name);

            if (seq_valid)
            {
                batch->mate_chromosome.push_back(reference->sequence_data.sequence_names.lookup(sequence_name));
            } else {
                batch->mate_chromosome.push_back(uint32(-1));
            }
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

    batch->reference_generation = reference->sequence_data.generation;
    return true;
}

const char *alignment_file::get_sequence_name(uint32 id)
{
    if (id >= gamgee_header.n_sequences())
    {
        // if the reference sequence is not noted in the file, return a special token
        // this will cause an invalid reference sequence ID to be inserted in the data stream
        return "<firepony-invalid-reference-sequence>";
    }

    return gamgee_header.sequence_name(id).c_str();
}

} // namespace firepony
