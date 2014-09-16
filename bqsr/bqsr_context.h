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

#include "bqsr_types.h"
#include "alignment_data.h"
#include "sequence_data.h"
#include "variants.h"
#include "covariates.h"
#include "cigar.h"
#include "alignment_data.h"
#include "baq.h"

struct bqsr_statistics // host-only
{
    uint32 total_reads;        // total number of reads processed
    uint32 filtered_reads;     // number of reads filtered out in pre-processing
    uint32 baq_reads;          // number of reads for which BAQ was computed

    bqsr_statistics()
        : total_reads(0),
          filtered_reads(0),
          baq_reads(0)
    { }
};

struct bqsr_context
{
    alignment_header& bam_header;
    const DeviceSNPDatabase& db;
    const sequence_data& reference;

    // sorted list of active reads
    D_VectorU32 active_read_list;
    // alignment windows for each read in reference coordinates
    D_VectorU32_2 alignment_windows;
    // alignment windows for each read in local sequence coordinates
    D_VectorU16_2 sequence_alignment_windows;

    // list of active BP locations
    D_ActiveLocationList active_location_list;
    // list of read offsets in the reference for each BP (relative to the alignment start position)
    D_VectorU16 read_offset_list;

    // temporary storage for CUB calls
    D_VectorU8 temp_storage;

    // and more temporary storage
    D_VectorU32 temp_u32;
    D_VectorU32 temp_u32_2;
    D_VectorU32 temp_u32_3;
    D_VectorU32 temp_u32_4;

    // various pipeline states go here
    snp_filter_context snp_filter;
    cigar_context cigar;
    baq_context baq;
    covariates_context covariates;

    // --- everything below this line is host-only and not available on the device
    bqsr_statistics stats;

    bqsr_context(alignment_header& bam_header,
                 const DeviceSNPDatabase& db,
                 const sequence_data& reference)
        : bam_header(bam_header),
          db(db),
          reference(reference)
    { }

    struct view
    {
        alignment_header::const_view            bam_header;
        DeviceSNPDatabase::const_view           db;
        sequence_data_device::const_view        reference;
        D_VectorU32::view                       active_read_list;
        D_VectorU32_2::view                     alignment_windows;
        D_VectorU16_2::view                     sequence_alignment_windows;
        D_ActiveLocationList::view              active_location_list;
        D_VectorU16::view                       read_offset_list;
        D_VectorU8::view                        temp_storage;
        D_VectorU32::view                       temp_u32;
        D_VectorU32::view                       temp_u32_2;
        D_VectorU32::view                       temp_u32_3;
        D_VectorU32::view                       temp_u32_4;
        snp_filter_context::view                snp_filter;
        cigar_context::view                     cigar;
        baq_context::view                       baq;
        covariates_context::view                covariates;
    };

    operator view()
    {
        view v = {
            bam_header,
            db,
            reference.device,
            active_read_list,
            alignment_windows,
            sequence_alignment_windows,
            active_location_list,
            read_offset_list,
            temp_storage,
            temp_u32,
            temp_u32_2,
            temp_u32_3,
            temp_u32_4,
            snp_filter,
            cigar,
            baq,
            covariates,
        };

        return v;
    }

    void start_batch(alignment_batch& batch);
#if 0
    void compact_active_read_list(void);
#endif
};

// encapsulates common state for our thrust functors to save a little typing
struct bqsr_lambda
{
    bqsr_context::view ctx;
    const alignment_batch_device::const_view batch;

    bqsr_lambda(bqsr_context::view ctx,
                const alignment_batch_device::const_view batch)
        : ctx(ctx),
          batch(batch)
    { }
};

void debug_read(bqsr_context *context, const alignment_batch& batch, int read_id);