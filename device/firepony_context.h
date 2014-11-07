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

#include "device_types.h"
#include "alignment_data.h"
#include "sequence_data.h"
#include "variant_data.h"
#include "snp_filter.h"
#include "covariates.h"
#include "cigar.h"
#include "alignment_data.h"
#include "baq.h"
#include "fractional_errors.h"
#include "read_group_table.h"
#include "util.h"

namespace firepony {

struct pipeline_statistics // host-only
{
    uint32 total_reads;        // total number of reads processed
    uint32 filtered_reads;     // number of reads filtered out in pre-processing
    uint32 baq_reads;          // number of reads for which BAQ was computed

    time_series io;
    time_series read_filter;
    time_series snp_filter;
    time_series bp_filter;
    time_series cigar_expansion;
    time_series baq;
    time_series fractional_error;
    time_series covariates;

    time_series baq_setup;
    time_series baq_hmm;
    time_series baq_postprocess;

    time_series covariates_gather;
    time_series covariates_filter;
    time_series covariates_sort;
    time_series covariates_pack;

    pipeline_statistics()
        : total_reads(0),
          filtered_reads(0),
          baq_reads(0)
    { }
};

struct firepony_context
{
    const alignment_header& bam_header;
    const variant_database& variant_db;
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
    D_VectorF32 temp_f32;
    D_VectorU8  temp_u8;

    // various pipeline states go here
    snp_filter_context snp_filter;
    cigar_context cigar;
    baq_context baq;
    covariates_context covariates;
    fractional_error_context fractional_error;
    read_group_table_context read_group_table;

    // --- everything below this line is host-only and not available on the device
    pipeline_statistics stats;

    firepony_context(const alignment_header& bam_header,
                     const sequence_data& reference,
                     const variant_database& variant_db)
        : bam_header(bam_header),
          reference(reference),
          variant_db(variant_db)
    { }

    struct view
    {
        alignment_header::const_view            bam_header;
        variant_database_device::const_view     variant_db;
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
        D_VectorU8::view                        temp_u8;
        snp_filter_context::view                snp_filter;
        cigar_context::view                     cigar;
        baq_context::view                       baq;
        covariates_context::view                covariates;
        fractional_error_context::view          fractional_error;
        read_group_table_context::view          read_group_table;
    };

    operator view()
    {
        view v = {
            bam_header,
            variant_db.device,
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
            temp_u8,
            snp_filter,
            cigar,
            baq,
            covariates,
            fractional_error,
            read_group_table,
        };

        return v;
    }

    void start_batch(const alignment_batch& batch);
    void end_batch(const alignment_batch& batch);
};

// encapsulates common state for our thrust functors to save a little typing
struct lambda
{
    firepony_context::view ctx;
    const alignment_batch_device::const_view batch;

    lambda(firepony_context::view ctx,
           const alignment_batch_device::const_view batch)
        : ctx(ctx),
          batch(batch)
    { }
};

struct lambda_context
{
    firepony_context::view ctx;

    lambda_context(firepony_context::view ctx)
        : ctx(ctx)
    { }
};

} // namespace firepony

