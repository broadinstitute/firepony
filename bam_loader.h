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

#pragma once

#include <zlib/zlib.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/packed_vector.h>
#include <vector>
#include <string>
#include <map>

#include "bqsr_types.h"

using namespace nvbio;

// the BAM header section
struct BAM_header
{
    // header section
    uint8 magic[4];  // BAM magic string
    int32 l_text;    // length of the header text
    std::string text; // the header text itself

    // reference sequence section
    int32 n_ref;     // number of reference sequences

    // reference sequence names and lengths
    std::vector<std::string> sq_names;
    H_VectorI32 sq_lengths;
    // xxxnsubtil: ugly hack, fix!
    D_VectorI32 d_sq_lengths;

    // key is a hash of the read group name, value is a sequential ID for the read group
    std::map<uint32, uint32> rg_name_to_id;
    // number of read groups in the map
    uint32 n_read_groups;

    struct view
    {
        int32 n_ref;
        D_VectorI32::plain_view_type sq_lengths;
        uint32 n_read_groups;
    };

    operator view()
    {
        // xxxnsubtil: ugly hack, fix and revert to const_view!
        d_sq_lengths = sq_lengths;

        view v = {
                n_ref,
                plain_view(d_sq_lengths),
                n_read_groups,
        };

        return v;
    }
};

// BAM alignment section
struct BAM_alignment_header
{
    int32  block_size; // length of the remainder of the alignment record
    int32  refID;      // reference sequence ID, -1 <= refID < n_ref (-1 for a read without a mapping position)
    int32  pos;        // 0-based leftmost coordinate
    uint32 bin_mq_nl;  // bin << 16 | MAPQ << 8 | l_read_name
    uint32 flag_nc;    // FLAG << 16 | n_cigar_op
    int32  l_seq;      // length of the sequence
    int32  next_refID; // refID of the next segment (-1 <= next_refID < n_ref)
    int32  next_pos;   // 0-based leftmost pos of the next segment
    int32  tlen;       // template length

    // followed by a BAM_alignment_data block
    // and then followed by block_size - ftell() bytes of auxiliary data

    uint32 bin(void) const
    {
        return bin_mq_nl >> 16;
    }

    uint32 mapq(void) const
    {
        return (bin_mq_nl & 0xff00) >> 8;
    }

    uint32 l_read_name(void) const
    {
        return bin_mq_nl & 0xff;
    }

    uint32 flags(void) const
    {
        return flag_nc >> 16;
    }

    uint32 num_cigar_ops(void) const
    {
        return flag_nc & 0xffff;
    }
};

struct BAM_alignment_tag
{
    char tag[2];
    char val_type;
    union
    {
        int8 i8;
        uint8 u8;
        int16 i16;
        uint16 u16;
        int32 i32;
        uint32 u32;
        float f32;
    } d;
} __attribute__((packed));

/*
struct BAM_alignment_data
{
    const char name[1024];  // read name, NULL terminated
    uint32 cigar[1024]; // CIGAR string, encoded as op_len << 4 | op ; 'MIDNSHP=X' -> 012345678
    uint8 seq[1024];   // 4-bit encoded read: '=ACMGRSVTWYHKDBN' -> [0, 15]; other characters mapped to 'N'
                       //   high nibble first: 1st base in the highest 4-bit of the 1st byte
    uint8 qual[1024];  // Phred-base quality (a sequence of 0xFF if absent)
};
*/

struct BAM_cigar_op
{
    uint32 len:24, op:4;

    enum
    {
        OP_M     = 0,
        OP_I     = 1,
        OP_D     = 2,
        OP_N     = 3,
        OP_S     = 4,
        OP_H     = 5,
        OP_P     = 6,
        OP_MATCH = 7,
        OP_X     = 8,
    };

    NVBIO_HOST_DEVICE char ascii_op(void) const
    {
        return op == 0 ? 'M' :
               op == 1 ? 'I' :
               op == 2 ? 'D' :
               op == 3 ? 'N' :
               op == 4 ? 'S' :
               op == 5 ? 'H' :
               op == 6 ? 'P' :
               op == 7 ? '=' :
                         'X';
    }
};

typedef nvbio::vector<device_tag, BAM_cigar_op> D_VectorCigarOp;
typedef nvbio::vector<host_tag, BAM_cigar_op> H_VectorCigarOp;

// CRQ: cigars, reads, qualities
struct BAM_CRQ_index
{
    uint32 cigar_start, cigar_len;
    uint32 read_start, read_len;
    uint32 qual_start, qual_len; // qual_len is always the same as read_len for BAM, but not necessarily for SAM

    NVBIO_HOST_DEVICE BAM_CRQ_index()
        : cigar_start(0), cigar_len(0),
          read_start(0), read_len(0),
          qual_start(0), qual_len(0)
    { }

    NVBIO_HOST_DEVICE BAM_CRQ_index(uint32 cigar_start, uint32 cigar_len,
                                    uint32 read_start, uint32 read_len,
                                    uint32 qual_start, uint32 qual_len)
        : cigar_start(cigar_start), cigar_len(cigar_len),
          read_start(read_start), read_len(read_len),
          qual_start(qual_start), qual_len(qual_len)
    { }
};

typedef nvbio::vector<device_tag, BAM_CRQ_index> D_VectorCRQIndex;
typedef nvbio::vector<host_tag, BAM_CRQ_index> H_VectorCRQIndex;

template <typename system_tag>
struct BAM_alignment_batch_storage
{
    uint32 num_reads;

    nvbio::vector<system_tag, BAM_cigar_op> cigars;
    nvbio::PackedVector<system_tag, 4> reads; //VectorDNA16<system_tag> reads;
    nvbio::vector<system_tag, uint8> qualities;
    nvbio::vector<system_tag, uint16> flags;
    nvbio::vector<system_tag, uint32> read_groups;
    nvbio::vector<system_tag, uint32> alignment_positions; // relative to the sequence!
    nvbio::vector<system_tag, uint32> alignment_sequence_IDs;
    nvbio::vector<system_tag, uint8> mapq;

    nvbio::vector<system_tag, BAM_CRQ_index> crq_index; // CRQ: cigars, reads, qualities
};

struct BAM_alignment_index
{
    uint32 aux_data_start, aux_data_len;
    uint32 name;

    BAM_alignment_index() : aux_data_start(0), aux_data_len(0), name(0) { }
    BAM_alignment_index(uint32 aux_data_start, uint32 aux_data_len, uint32 name)
        : aux_data_start(aux_data_start), aux_data_len(aux_data_len), name(name) { }
};

struct BAM_alignment_batch_host : public BAM_alignment_batch_storage<host_tag>
{
    // host-only data that we never move to the GPU
    nvbio::vector<host_tag, BAM_alignment_header> align_headers;
    nvbio::vector<host_tag, char> aux_data;
    nvbio::vector<host_tag, char> names;

    nvbio::vector<host_tag, BAM_alignment_index> index;

    void reset(uint32 batch_size, bool skip_headers)
    {
        num_reads = 0;

        align_headers.clear();
        index.clear();
        crq_index.clear();
        aux_data.clear();
        names.clear();
        cigars.clear();
        reads.clear();
        qualities.clear();
        flags.clear();
        alignment_positions.clear();
        alignment_sequence_IDs.clear();
        mapq.clear();

        // read groups are special: they're initialized to zero
        read_groups.assign(batch_size, 0);

        if (align_headers.size() < batch_size)
        {
            if (!skip_headers)
            {
                align_headers.reserve(batch_size);
                index.reserve(batch_size);
                aux_data.reserve(batch_size * 1024);
                names.reserve(batch_size * 512);
            }

            crq_index.reserve(batch_size);
            cigars.reserve(batch_size * 32);
            reads.reserve(batch_size * 350);
            qualities.reserve(batch_size * 350);
            flags.reserve(batch_size);
            alignment_positions.reserve(batch_size);
            alignment_sequence_IDs.reserve(batch_size);
            mapq.reserve(batch_size);
        }
    }
};

// device version of alignment data
struct BAM_alignment_batch_device : public BAM_alignment_batch_storage<device_tag>
{
    void load(BAM_alignment_batch_host& batch)
    {
        num_reads = batch.num_reads;

        cigars = batch.cigars;
        reads = batch.reads;
        qualities = batch.qualities;
        flags = batch.flags;
        alignment_positions = batch.alignment_positions;
        alignment_sequence_IDs = batch.alignment_sequence_IDs;
        mapq = batch.mapq;
        read_groups = batch.read_groups;
        crq_index = batch.crq_index;

    }

    NVBIO_HOST_DEVICE BAM_alignment_batch_device() { }

    struct const_view
    {
        uint32 num_reads;

        D_VectorCigarOp::const_plain_view_type cigars;
        D_VectorDNA16::const_plain_view_type reads;
        D_VectorU8::const_plain_view_type qualities;
        D_VectorU16::const_plain_view_type flags;
        D_VectorU32::const_plain_view_type read_groups;
        D_VectorU32::const_plain_view_type alignment_positions;
        D_VectorU32::const_plain_view_type alignment_sequence_IDs;
        D_VectorU8::const_plain_view_type mapq;
        D_VectorCRQIndex::const_plain_view_type crq_index;
    };

    operator const_view() const
    {
        const_view v = {
                num_reads,

                plain_view(cigars),
                plain_view(reads),
                plain_view(qualities),
                plain_view(flags),
                plain_view(read_groups),
                plain_view(alignment_positions),
                plain_view(alignment_sequence_IDs),
                plain_view(mapq),
                plain_view(crq_index)
        };

        return v;
    }
};
/*
struct BAM_alignment_batch_device_view
{
    uint32 num_reads;

    uint32 *read_order;
    BAM_cigar_op *cigars;
    uint32 *reads;
    uint8 *qualities;
    uint32 *read_groups;
    uint32 *alignment_positions;
    uint32 *alignment_sequence_IDs;
    BAM_CRQ_index *crq_index;
};

static BAM_alignment_batch_device_view plain_view(BAM_alignment_batch_device& dev)
{
    BAM_alignment_batch_device_view v;

    v.num_reads = dev.crq_index.size();
    v.read_order = thrust::raw_pointer_cast(&dev.read_order[0]);
    v.cigars = thrust::raw_pointer_cast(&dev.cigars[0]);
    v.reads = thrust::raw_pointer_cast(&dev.reads.m_storage[0]);
    v.qualities = thrust::raw_pointer_cast(&dev.qualities[0]);
    v.read_groups = thrust::raw_pointer_cast(&dev.read_groups[0]);
    v.alignment_positions = thrust::raw_pointer_cast(&dev.alignment_positions[0]);
    v.alignment_sequence_IDs = thrust::raw_pointer_cast(&dev.alignment_sequence_IDs[0]);
    v.crq_index = thrust::raw_pointer_cast(&dev.crq_index[0]);

    return v;
}

static const BAM_alignment_batch_device_view plain_view(const BAM_alignment_batch_device& dev)
{
    return plain_view((BAM_alignment_batch_device&)dev);
}
*/

struct BAMfile
{
public:
    BAM_header header;

private:
    gzFile fp;
    z_off_t data_start;
    bool eof;

public:
    BAMfile(const char *fname);
    ~BAMfile();

    bool next_batch(BAM_alignment_batch_host *batch, bool skip_headers = false, const uint32 batch_size = 100000);

private:
    bool readData(void *output, unsigned int len, int line);
    bool init(void);
    char *parse_header_tag(uint32 *len, const char *tag, char *start_ptr, char *end_ptr);
};
