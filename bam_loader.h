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

using namespace nvbio;

//template <typename system_tag> using Vector_DNA16 = nvbio::PackedVector<system_tag, 4>;

typedef nvbio::PackedVector<host_tag, 4> HostVector_DNA16;
typedef nvbio::PackedVector<device_tag, 4> DeviceVector_DNA16;
typedef HostVector_DNA16::plain_view_type HostStream_DNA16;
typedef DeviceVector_DNA16::plain_view_type DeviceStream_DNA16;

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
    std::vector<int32> sq_lengths;

    // key is a hash of the read group name, value is a sequential ID for the read group
    std::map<uint32, uint32> rg_name_to_id;
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

struct BAM_tag
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

    char ascii_op(void)
    {
        static const char op_table[] = "MIDNSHP=X";
        assert(op < sizeof(op_table));
        return op_table[op];
    }
};

// CRQ: cigars, reads, qualities
struct BAM_CRQ_index
{
    uint32 cigar_start, cigar_len;
    uint32 read_start, read_len;
    uint32 qual_start;

    BAM_CRQ_index()
        : cigar_start(0), cigar_len(0),
          read_start(0), read_len(0),
          qual_start(0)
    { }

    BAM_CRQ_index(uint32 cigar_start, uint32 cigar_len,
                  uint32 read_start, uint32 read_len,
                  uint32 qual_start)
        : cigar_start(cigar_start), cigar_len(cigar_len),
          read_start(read_start), read_len(read_len),
          qual_start(qual_start) { }
};

template <typename system_tag>
struct BAM_alignment_batch_storage
{
    nvbio::vector<system_tag, BAM_cigar_op> cigars;
    nvbio::PackedVector<system_tag, 4> reads; //Vector_DNA16<system_tag> reads;
    nvbio::vector<system_tag, uint8> qualities;
    nvbio::vector<system_tag, uint32> read_groups;

    nvbio::vector<system_tag, BAM_CRQ_index> crq_index; // CRQ: cigars, reads, qual_start
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
};

// device version of alignment data
typedef struct BAM_alignment_batch_storage<device_tag> BAM_alignment_batch_device;

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
};

