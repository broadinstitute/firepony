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

#pragma once

#include "../types.h"
#include "../runtime_options.h"

#include "alignment_data_device.h"
#include "../sequence_database.h"
#include "../variant_database.h"
#include "snp_filter.h"
#include "covariates.h"
#include "cigar.h"
#include "baq.h"
#include "fractional_errors.h"
#include "util.h"

#include "device/primitives/timer.h"

namespace firepony {

struct pipeline_statistics // host-only
{
    uint64 total_reads;        // total number of reads processed
    uint64 filtered_reads;     // number of reads filtered out in pre-processing
    uint64 baq_reads;          // number of reads for which BAQ was computed
    uint64 num_batches;        // number of batches processed

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

    time_series postprocessing;
    time_series output;

    pipeline_statistics()
        : total_reads(0),
          filtered_reads(0),
          baq_reads(0),
          num_batches(0)
    { }

    pipeline_statistics& operator+=(const pipeline_statistics& other)
    {
        total_reads += other.total_reads;
        filtered_reads += other.filtered_reads;
        baq_reads += other.baq_reads;
        num_batches += other.num_batches;

        io += other.io;
        read_filter += other.read_filter;
        snp_filter += other.snp_filter;
        bp_filter += other.bp_filter;
        cigar_expansion += other.cigar_expansion;
        baq += other.baq;
        fractional_error += other.fractional_error;
        covariates += other.covariates;

        baq_setup += other.baq_setup;
        baq_hmm += other.baq_hmm;
        baq_postprocess += other.baq_postprocess;

        covariates_gather += other.covariates_gather;
        covariates_filter += other.covariates_filter;
        covariates_sort += other.covariates_sort;
        covariates_pack += other.covariates_pack;

        postprocessing += other.postprocessing;
        output += other.output;

        return *this;
    }
};

template <target_system system>
struct firepony_context
{
    // identifies the compute device we're using on this context
    // note that the meaning depends on the target system
    const int compute_device;

    const runtime_options& options;

    const alignment_header<system>& bam_header;
    const variant_database<system>& variant_db;
    const sequence_database<system>& reference_db;

    // sorted list of active reads
    vector<system, uint32> active_read_list;
    // alignment windows for each read in chromosome coordinates
    vector<system, uint2> alignment_windows;

    // list of active BP locations
    // we match the BP representation size to avoid RMW hazards at the edges of reads
    vector_dna16<system> active_location_list;
    // list of read offsets in the reference for each BP (relative to the alignment start position)
    vector<system, uint16> read_offset_list;

    // temporary storage for CUB calls
    vector<system, uint8> temp_storage;

    // and more temporary storage
    vector<system, uint32> temp_u32;
    vector<system, uint32> temp_u32_2;
    vector<system, uint32> temp_u32_3;
    vector<system, uint32> temp_u32_4;
    vector<system, uint8>  temp_u8;

    // various pipeline states go here
    snp_filter_context<system> snp_filter;
    cigar_context<system> cigar;
    baq_context<system> baq;
    covariates_context<system> covariates;
    fractional_error_context<system> fractional_error;

    // --- everything below this line is host-only and not available on the device
    pipeline_statistics stats;

    firepony_context(const int compute_device,
                     const runtime_options& options,
                     const alignment_header<system>& bam_header,
                     const sequence_database<system>& reference_db,
                     const variant_database<system>& variant_db)
        : compute_device(compute_device),
          options(options),
          bam_header(bam_header),
          reference_db(reference_db),
          variant_db(variant_db)
    { }

    struct view
    {
        typename alignment_header_device<system>::const_view    bam_header;
        typename variant_database_device<system>::const_view    variant_db;
        typename sequence_database_device<system>::const_view   reference_db;
        typename vector<system, uint32>::view                   active_read_list;
        typename vector<system, uint2>::view                    alignment_windows;
        typename vector_dna16<system>::view                     active_location_list;
        typename vector<system, uint16>::view                   read_offset_list;
        typename vector<system, uint8>::view                    temp_storage;
        typename vector<system, uint32>::view                   temp_u32;
        typename vector<system, uint32>::view                   temp_u32_2;
        typename vector<system, uint32>::view                   temp_u32_3;
        typename vector<system, uint32>::view                   temp_u32_4;
        typename vector<system, uint8>::view                    temp_u8;
        typename snp_filter_context<system>::view               snp_filter;
        typename cigar_context<system>::view                    cigar;
        typename baq_context<system>::view                      baq;
        typename covariates_context<system>::view               covariates;
        typename fractional_error_context<system>::view         fractional_error;
    };

    operator view()
    {
        view v = {
            bam_header.device,
            variant_db.device,
            reference_db.device,
            active_read_list,
            alignment_windows,
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
        };

        return v;
    }

    void start_batch(const alignment_batch<system>& batch);
    void end_batch(const alignment_batch<system>& batch);
};

// encapsulates common state for our thrust functors to save a little typing
template <target_system system>
struct lambda
{
    typename firepony_context<system>::view ctx;
    const typename alignment_batch_device<system>::const_view batch;

    lambda(typename firepony_context<system>::view ctx,
           const typename alignment_batch_device<system>::const_view batch)
        : ctx(ctx),
          batch(batch)
    { }
};
#define LAMBDA_INHERIT_MEMBERS using lambda<system>::ctx; using lambda<system>::batch
#define LAMBDA_INHERIT using lambda<system>::lambda; LAMBDA_INHERIT_MEMBERS

template <target_system system>
struct lambda_context
{
    typename firepony_context<system>::view ctx;

    lambda_context(typename firepony_context<system>::view ctx)
        : ctx(ctx)
    { }
};
#define LAMBDA_CONTEXT_INHERIT_MEMBERS using lambda_context<system>::ctx
#define LAMBDA_CONTEXT_INHERIT using lambda_context<system>::lambda_context; LAMBDA_CONTEXT_INHERIT_MEMBERS

} // namespace firepony

