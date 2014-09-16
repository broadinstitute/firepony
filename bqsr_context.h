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

using namespace nvbio;

struct snp_filter_context
{
    // list of active BP locations
    D_ActiveLocationList active_location_list;
    // list of read offsets in the reference for each BP (relative to the alignment start position)
    D_ReadOffsetList read_offset_list;

    // active reads for the VCF search
    D_VectorU32 active_read_ids;
    D_VectorU32 active_read_ids_temp;

    // active VCF range for each read
    D_VectorU2 active_vcf_ranges;

    struct view
    {
        D_ActiveLocationList::plain_view_type active_location_list;
        D_ReadOffsetList::plain_view_type read_offset_list;
        D_VectorU32::plain_view_type active_read_ids;
        D_VectorU32::plain_view_type active_read_ids_temp;
        D_VectorU2::plain_view_type active_vcf_ranges;
    };

    operator view()
    {
        view v = {
            plain_view(active_location_list),
            plain_view(read_offset_list),
            plain_view(active_read_ids),
            plain_view(active_read_ids_temp),
            plain_view(active_vcf_ranges)
        };
        return v;
    }
};

struct bqsr_context
{
    const DeviceSNPDatabase& db;
    const reference_genome_device& reference;

    // the read order for processing reads
    D_VectorU32 read_order;
    // alignment windows for each read in reference coordinates
    D_VectorU2 alignment_windows;
    // temporary storage for CUB calls
    D_VectorU8 temp_storage;

    snp_filter_context snp_filter;


    bqsr_context(const DeviceSNPDatabase& db,
                 const reference_genome_device& reference)
        : db(db),
          reference(reference)
    { }

    struct view
    {
        DeviceSNPDatabase::const_view db;
        reference_genome_device::const_view reference;
        D_VectorU32::plain_view_type read_order;
        D_VectorU2::plain_view_type alignment_windows;
        D_VectorU8::plain_view_type temp_storage;
        snp_filter_context::view snp_filter;
    };

    operator view()
    {
        view v = {
            db,
            reference,
            plain_view(read_order),
            plain_view(alignment_windows),
            plain_view(temp_storage),
            snp_filter,
        };
        return v;
    }
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
