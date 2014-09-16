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
#include "variants.h"
#include "covariates.h"
#include "bam_loader.h"

using namespace nvbio;

struct snp_filter_context
{
    // active reads for the VCF search
    D_VectorU32 active_read_ids;
    // active VCF range for each read
    D_VectorU32_2 active_vcf_ranges;

    struct view
    {
        D_VectorU32::plain_view_type active_read_ids;
        D_VectorU32_2::plain_view_type active_vcf_ranges;
    };

    operator view()
    {
        view v = {
            plain_view(active_read_ids),
            plain_view(active_vcf_ranges)
        };
        return v;
    }
};

struct covariate_context
{

};

struct bqsr_context
{
    BAM_header& bam_header;
    const DeviceSNPDatabase& db;
    const reference_genome_device& reference;

    // sorted list of active reads
    D_VectorU32 active_read_list;
    // alignment windows for each read in reference coordinates
    D_VectorU32_2 alignment_windows;
    // alignment windows for each read in local sequence coordinates
    D_VectorU16_2 sequence_alignment_windows;

    // list of active BP locations
    D_ActiveLocationList active_location_list;
    // list of read offsets in the reference for each BP (relative to the alignment start position)
    D_ReadOffsetList read_offset_list;

    // temporary storage for CUB calls
    D_VectorU8 temp_storage;
    // temporary storage for compacting U32 vectors
    D_VectorU32 temp_u32;

    // various pipeline states go here
    snp_filter_context snp_filter;


    bqsr_context(BAM_header& bam_header,
                 const DeviceSNPDatabase& db,
                 const reference_genome_device& reference)
        : bam_header(bam_header),
          db(db),
          reference(reference)
    { }

    struct view
    {
        BAM_header::view                        bam_header;
        DeviceSNPDatabase::const_view           db;
        reference_genome_device::const_view     reference;
        D_VectorU32::plain_view_type            active_read_list;
        D_VectorU32_2::plain_view_type          alignment_windows;
        D_VectorU16_2::plain_view_type          sequence_alignment_windows;
        D_ActiveLocationList::plain_view_type   active_location_list;
        D_ReadOffsetList::plain_view_type       read_offset_list;
        D_VectorU8::plain_view_type             temp_storage;
        D_VectorU32::plain_view_type            temp_u32;
        snp_filter_context::view                snp_filter;
    };

    operator view()
    {
        view v = {
            bam_header,
            db,
            reference,
            plain_view(active_read_list),
            plain_view(alignment_windows),
            plain_view(sequence_alignment_windows),
            plain_view(active_location_list),
            plain_view(read_offset_list),
            plain_view(temp_storage),
            plain_view(temp_u32),
            snp_filter,
        };
        return v;
    }

    void start_batch(BAM_alignment_batch_device& batch);
    void compact_active_read_list(void);
};

// encapsulates common state for our thrust functors to save a little typing
struct bqsr_lambda
{
    bqsr_context::view ctx;
    const BAM_alignment_batch_device::const_view batch;

    bqsr_lambda(bqsr_context::view ctx,
                       const BAM_alignment_batch_device::const_view batch)
        : ctx(ctx),
          batch(batch)
    { }
};
