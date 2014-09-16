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
#include "bqsr_context.h"
#include "covariates.h"

#define BITMASK(bits) ((1 << bits) - 1)

// base covariate class
template<typename PrevCovariate, uint32 bits>
struct covariate
{
    // the previous covariate in the chain
    // (first covariate will have this set to struct covariate_null)
    typedef PrevCovariate PreviousCovariate;

    enum {
        // our index in the chain
        index = PreviousCovariate::index + 1,
        // our own offset
        offset = PreviousCovariate::next_offset,
        // our bit mask
        mask = BITMASK(bits) << PreviousCovariate::next_offset,
        // the offset for the next covariate in the chain
        next_offset = PreviousCovariate::next_offset + bits
    };

protected:
    template<typename K>
    static NVBIO_HOST_DEVICE uint32 build_key(uint32 input_key, K data,
                                              bqsr_context::view ctx,
                                              const BAM_alignment_batch_device::const_view batch,
                                              uint32 read_index, uint16 bp_offset, uint32 cigar_event_index)
    {
        // add in our bits
        uint32 out = input_key | (((data) & BITMASK(bits)) << offset);
        // pass along to next in chain
        return PreviousCovariate::encode(ctx, batch, read_index, bp_offset, cigar_event_index, out);
    }

public:
    static NVBIO_HOST_DEVICE uint32 decode(uint32 input_key, uint32 target_index)
    {
        if (target_index == index)
        {
            // grab the bits we put in input_key
            uint32 out = (input_key >> offset) & BITMASK(bits);
            return out;
        } else {
            return PreviousCovariate::decode(input_key, target_index);
        }
    }
};

// chain terminator
struct covariate_null
{
    typedef covariate_null PreviousCovariate;

    enum {
        index = 0,
        mask = 0,
        next_offset = 0
    };

    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                           uint32 input_key = 0)
    {
        return input_key;
    }

    static NVBIO_HOST_DEVICE uint32 decode(uint32 input_key, uint32 target_index)
    {
        return 0;
    }

#if 0
    static const char *whoami(uint32 target_index)
    {
        return "<null>";
    }
#endif
};

template<typename PreviousCovariate = covariate_null>
struct covariate_ReadGroup : public covariate<PreviousCovariate, 8>
{
    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                           uint32 input_key = 0)
    {
        uint8 key = (uint8) batch.read_groups[read_index];
        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }

#if 0
    static const char *whoami(uint32 target_index)
    {
        if (covariate<PreviousCovariate, 8>::index == target_index)
        {
            return "ReadGroup";
        } else {
            return PreviousCovariate::whoami(target_index);
        }
    }
#endif
};

template <typename PreviousCovariate = covariate_null>
struct covariate_EventType : public covariate<PreviousCovariate, 2>
{
    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                           uint32 input_key = 0)
    {
        uint8 key = ctx.cigar.cigar_events[cigar_event_index];
        return covariate<PreviousCovariate, 2>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }

#if 0
    static const char *whoami(uint32 target_index)
    {
        if (covariate<PreviousCovariate, 2>::index == target_index)
        {
            return "EventType";
        } else {
            return PreviousCovariate::whoami(target_index);
        }
    }
#endif
};

template <typename PreviousCovariate = covariate_null>
struct covariate_QualityScore : public covariate<PreviousCovariate, 8>
{
    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                           uint32 input_key = 0)
    {
        const BAM_CRQ_index& idx = batch.crq_index[read_index];
        uint8 key = batch.qualities[idx.qual_start + bp_offset];

        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }

#if 0
    static const char *whoami(uint32 target_index)
    {
        if (covariate<PreviousCovariate, 8>::index == target_index)
        {
            return "QualityScore";
        } else {
            return PreviousCovariate::whoami(target_index);
        }
    }
#endif
};

template <typename PreviousCovariate = covariate_null>
struct covariate_EmpiricalQuality : public covariate<PreviousCovariate, 8>
{
    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                           uint32 input_key = 0)
    {
        const BAM_CRQ_index& idx = batch.crq_index[read_index];
        uint8 key = ctx.baq.qualities[idx.qual_start + bp_offset];

        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }

#if 0
    static const char *whoami(uint32 target_index)
    {
        if (covariate<PreviousCovariate, 8>::index == target_index)
        {
            return "EmpiricalQuality";
        } else {
            return PreviousCovariate::whoami(target_index);
        }
    }
#endif
};
