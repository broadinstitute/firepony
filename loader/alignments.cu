/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string>

#include <htslib/hfile.h>
#include <htslib/bgzf.h>

#include "../alignment_data.h"
#include "alignments.h"
#include "reference.h"

#include "../device/util.h"
#include "../device/from_nvbio/dna.h"

namespace firepony {

struct read_group
{
    std::string id;
    std::string center;
    std::string description;
    std::string date_time;
    std::string flow_order;
    std::string key_sequence;
    std::string library;
    std::string programs;
    std::string median_insert_size;
    std::string platform;
    std::string platform_unit;
    std::string sample;

    bool match_tag(const std::string& token, const char tag[2])
    {
        return token[0] == tag[0] && token[1] == tag[1];
    }

    read_group(const std::string& header_line)
    {
        // split string at each tab character
        // note that we don't use std::regex because it seems to be unavailable on common Linux installations
        const std::string delimiter = "\t";
        std::string temp = header_line;
        size_t pos = 0;
        std::string token;

        pos = temp.find(delimiter);
        while(pos != std::string::npos)
        {
            token = temp.substr(0, pos);
            temp.erase(0, pos + delimiter.length());

            // skip TG:
            auto value = token.substr(3, token.length() - 3);

            if (match_tag(token, "ID"))
                id = value;

            if (match_tag(token, "CN"))
                center = value;

            if (match_tag(token, "DS"))
                description = value;

            if (match_tag(token, "DT"))
                date_time = value;

            if (match_tag(token, "FO"))
                flow_order = value;

            if (match_tag(token, "KS"))
                key_sequence = value;

            if (match_tag(token, "LB"))
                library = value;

            if (match_tag(token, "PG"))
                programs = value;

            if (match_tag(token, "PI"))
                median_insert_size = value;

            if (match_tag(token, "PL"))
                platform = value;

            if (match_tag(token, "PU"))
                platform_unit = value;

            if (match_tag(token, "SM"))
                sample = value;

            pos = temp.find(delimiter);
        }
    }
};

alignment_file::alignment_file(const char *fname)
    : fname(fname),
      fp(nullptr),
      bam_header(nullptr),
      data(nullptr)
{
}

alignment_file::~alignment_file()
{
}

bool alignment_file::init(void)
{
    // determine the file size using stdio
    FILE *myfp;
    myfp = fopen(fname, "rb");
    if (myfp == 0)
    {
        fprintf(stderr, "error opening %s\n", fname);
        return false;
    }

    fseek(myfp, 0, SEEK_END);
    file_size = ftell(myfp);
    fclose(myfp);

    // open file for reading through htslib
    fp = hts_open(fname, "r");
    if (fp == 0)
    {
        fprintf(stderr, "error opening %s\n", fname);
        return false;
    }

    bam_header = sam_hdr_read(fp);
    header_text = std::string(bam_header->text, bam_header->l_text);
    data = bam_init1();

    // build the read group ID map, loosely based on gamgee
    // nobody should ever have to do this...
    size_t rg_start = 0;
    for(;;)
    {
        rg_start = header_text.find("@RG", rg_start);
        if (rg_start == std::string::npos)
        {
            break;
        }


        size_t rg_end = header_text.find("\n", rg_start + 1);
        std::string rg_record = header_text.substr(rg_start, rg_end - rg_start);
        read_group rg(rg_record);

        std::string name;
        if (rg.platform_unit.size() != 0)
        {
            name = rg.platform_unit;
        } else {
            name = rg.id;
        }

        read_group_id_to_name[rg.id] = name;

        rg_start = rg_end + 1;
    }

    for(int32 i = 0; i < bam_header->n_targets; i++)
    {
        header.chromosome_lengths.push_back(bam_header->target_len[i]);
    }

    return true;
}

static uint8 htslib_to_firepony_cigar_op(uint32 e)
{
    // lowest 4 bits contain cigar op
    uint32 op = (e & 0xf);

    if (op == BAM_CBACK)
    {
        // xxxnsubtil: firepony does not have B; return X instead
        return cigar_op::OP_X;
    } else {
        return uint8(op);
    }
}

static uint32 extract_htslib_flags(bam1_t *data)
{
    uint32 flags = 0;

    if (data->core.flag & BAM_FPAIRED)
        flags |= AlignmentFlags::PAIRED;

    if (data->core.flag & BAM_FPROPER_PAIR)
        flags |= AlignmentFlags::PROPER_PAIR;

    if (data->core.flag & BAM_FUNMAP)
        flags |= AlignmentFlags::UNMAP;

    if (data->core.flag & BAM_FMUNMAP)
        flags |= AlignmentFlags::MATE_UNMAP;

    if (data->core.flag & BAM_FREVERSE)
        flags |= AlignmentFlags::REVERSE;

    if (data->core.flag & BAM_FMREVERSE)
        flags |= AlignmentFlags::MATE_REVERSE;

    if (data->core.flag & BAM_FREAD1)
        flags |= AlignmentFlags::READ1;

    if (data->core.flag & BAM_FREAD2)
        flags |= AlignmentFlags::READ2;

    if (data->core.flag & BAM_FSECONDARY)
        flags |= AlignmentFlags::SECONDARY;

    if (data->core.flag & BAM_FQCFAIL)
        flags |= AlignmentFlags::QC_FAIL;

    if (data->core.flag & BAM_FDUP)
        flags |= AlignmentFlags::DUPLICATE;

    if (data->core.flag & BAM_FSUPPLEMENTARY)
        flags |= AlignmentFlags::SUPPLEMENTARY;

    return flags;
}

bool alignment_file::next_batch(alignment_batch_host *batch, uint32 data_mask, reference_file_handle *reference, const uint32 batch_size)
{
    uint32 read_id;

    batch->reset(data_mask, batch_size, reference->sequence_data);

    for(read_id = 0; read_id < batch_size; read_id++)
    {
        int ret;
        ret = sam_read1(fp, bam_header, data);
        if (ret < 0)
        {
            break;
        }

        batch->num_reads++;
        batch->name.push_back(bam_get_qname(data));

        if (data_mask & AlignmentDataMask::CHROMOSOME)
        {
            std::string sequence_name = get_sequence_name(data->core.tid);
            const bool seq_valid = reference->make_sequence_available(sequence_name);

            if (seq_valid)
            {
                uint16 seq_id = reference->sequence_data.sequence_names.lookup(sequence_name);
                batch->chromosome.push_back(seq_id);
                batch->chromosome_map.mark_resident(seq_id);
            } else {
                batch->chromosome.push_back(uint16(-1));
            }
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            batch->alignment_start.push_back(data->core.pos);
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            batch->alignment_stop.push_back(bam_endpos(data));
        }

        if (data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            std::string sequence_name = get_sequence_name(data->core.mtid);
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
            batch->mate_alignment_start.push_back(data->core.mpos);
        }

        if (data_mask & AlignmentDataMask::INFERRED_INSERT_SIZE)
        {
            batch->inferred_insert_size.push_back(data->core.isize);
        }

        if (data_mask & AlignmentDataMask::CIGAR)
        {
            // "In the CIGAR array, each element is a 32-bit integer. The
            // lower 4 bits gives a CIGAR operation and the higher 28 bits keep the
            // length of a CIGAR."
            uint32 *cigar = bam_get_cigar(data);
            uint32 cigar_len = data->core.n_cigar;

            batch->cigar_start.push_back(batch->cigars.size());
            batch->cigar_len.push_back(cigar_len);

            for(uint32 i = 0; i < cigar_len; i++)
            {
                cigar_op op;

                op.op = htslib_to_firepony_cigar_op(cigar[i]);
                op.len = cigar[i] >> 4;

                batch->cigars.push_back(op);
            }
        }

        if (data_mask & AlignmentDataMask::READS)
        {
            // let the compiler infer the type from this absurd mess,
            // bam_seqi assumes we know the type but it's not documented
            auto seq = bam_get_seq(data);
            uint32 seq_len = data->core.l_qseq;

            batch->read_start.push_back(batch->reads.size());
            batch->read_len.push_back(seq_len);

            // figure out the length of the sequence data,
            // rounded up to reach a dword boundary
            const uint32 padded_read_len_bp = ((seq_len + 7) / 8) * 8;

            // make sure we have enough memory, then read in the sequence
            batch->reads.resize(batch->reads.size() + padded_read_len_bp);

            // xxxnsubtil: this is going to be really slow
            // can we use assign here instead?
            for(uint32 i = 0; i < seq_len; i++)
            {
                batch->reads[batch->read_start[read_id] + i] = uint32(bam_seqi(seq, i));
            }

            if (batch->max_read_size < seq_len)
            {
                batch->max_read_size = seq_len;
            }
        }

        if (data_mask & AlignmentDataMask::QUALITIES)
        {
            auto quals = bam_get_qual(data);
            uint32 qual_len = data->core.l_qseq;

            batch->qual_start.push_back(batch->qualities.size());
            batch->qual_len.push_back(qual_len);

            batch->qualities.resize(batch->qualities.size() + qual_len);
            memcpy(&batch->qualities[batch->qual_start[read_id]], &quals[0], qual_len);
        }

        if (data_mask & AlignmentDataMask::FLAGS)
        {
            batch->flags.push_back(extract_htslib_flags(data));
        }

        if (data_mask & AlignmentDataMask::MAPQ)
        {
            batch->mapq.push_back(data->core.qual);
        }

        if (data_mask & AlignmentDataMask::READ_GROUP)
        {
            // locate the RG tag
            uint8 *tag = bam_aux_get(data, "RG");

            if (tag == nullptr)
            {
                // invalid read group
                batch->read_group.push_back(uint32(-1));
            } else {
                const char *rgid = bam_aux2Z(tag);
                auto iter = read_group_id_to_name.find(rgid);
                if (iter == read_group_id_to_name.end())
                {
                    fprintf(stderr, "WARNING: found read with invalid read group identifier [%s]\n", rgid);
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

const char *alignment_file::get_sequence_name(uint32 id)
{
    if (id >= uint32(bam_header->n_targets))
    {
        // if the reference sequence is not noted in the file, return a special token
        // this will cause an invalid reference sequence ID to be inserted in the data stream
        return "<firepony-invalid-reference-sequence>";
    }

    return bam_header->target_name[id];
}

// htslib makes it unclear whether we can rely on this interface not breaking
// if this ever fails to compile, they probably changed their data structures
static off_t htsfile_ftell(htsFile *fp)
{
    struct hFILE *hfp;

    switch(fp->format.format)
    {
    case htsExactFormat::bam:
        hfp = fp->fp.bgzf->fp;
        break;

    case htsExactFormat::sam:
        hfp = fp->fp.hfile;
        break;

    default:
        return off_t(-1);
    }

    return htell(hfp);
}

float alignment_file::progress(void)
{
    return (float)htsfile_ftell(fp) / (float)file_size;
}

} // namespace firepony
