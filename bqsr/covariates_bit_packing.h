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
#include "covariates_table.h"
#include "cigar.h"

#include "primitives/cuda.h"

#define BITMASK(bits) ((1 << bits) - 1)

struct covariate_key_set
{
    covariate_key M;
    covariate_key I;
    covariate_key D;
};

typedef enum {
    EVENT_M = 0,
    EVENT_I = 1,
    EVENT_D = 2
} CovariateEventIndex;

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
    static CUDA_HOST_DEVICE covariate_key_set build_key(covariate_key_set input_key, covariate_key_set data,
                                                        bqsr_context::view ctx,
                                                        const alignment_batch_device::const_view batch,
                                                        uint32 read_index, uint16 bp_offset, uint32 cigar_event_index)
    {
        // add in our bits
        input_key.M = input_key.M | (((data.M) & BITMASK(bits)) << offset);
        input_key.I = input_key.I | (((data.I) & BITMASK(bits)) << offset);
        input_key.D = input_key.D | (((data.D) & BITMASK(bits)) << offset);

        // pass along to next in chain
        return PreviousCovariate::encode(ctx, batch, read_index, bp_offset, cigar_event_index, input_key);
    }

public:
    static CUDA_HOST_DEVICE uint32 decode(covariate_key input, uint32 target_index)
    {
        if (target_index == index)
        {
            covariate_key out = (input >> offset) & BITMASK(bits);
            return out;
        } else {
            return PreviousCovariate::decode(input, target_index);
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

    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        return input_key;
    }

    static CUDA_HOST_DEVICE uint32 decode(covariate_key input, uint32 target_index)
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
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        covariate_key_set key = { batch.read_group[read_index],
                                  batch.read_group[read_index],
                                  batch.read_group[read_index] };
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

// this is a little weird, since it doesn't actually change
template <typename PreviousCovariate = covariate_null>
struct covariate_EventTracker : public covariate<PreviousCovariate, 2>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        return covariate<PreviousCovariate, 2>::build_key(input_key,
                                                          {cigar_event::M, cigar_event::I, cigar_event::D},
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }

#if 0
    static const char *whoami(uint32 target_index)
    {
        if (covariate<PreviousCovariate, 2>::index == target_index)
        {
            return "EventTracker";
        } else {
            return PreviousCovariate::whoami(target_index);
        }
    }
#endif
};

//template <typename PreviousCovariate = covariate_null>
//struct covariate_EventType : public covariate<PreviousCovariate, 2>
//{
//    static CUDA_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
//                                          const alignment_batch_device::const_view batch,
//                                          uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
//                                          covariate_key_set input_key = {0, 0, 0})
//    {
//        uint8 key = ctx.cigar.cigar_events[cigar_event_index];
//        return covariate<PreviousCovariate, 2>::build_key(input_key, key,
//                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
//    }
//
//#if 0
//    static const char *whoami(uint32 target_index)
//    {
//        if (covariate<PreviousCovariate, 2>::index == target_index)
//        {
//            return "EventType";
//        } else {
//            return PreviousCovariate::whoami(target_index);
//        }
//    }
//#endif
//};

//template <cigar_event::Event EV, typename PreviousCovariate = covariate_null>
//struct covariate_FixedEvent : public covariate<PreviousCovariate, 2>
//{
//    static CUDA_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
//                                          const alignment_batch_device::const_view batch,
//                                          uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
//                                          uint32 input_key = 0)
//    {
//        return covariate<PreviousCovariate, 2>::build_key(input_key, EV,
//                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
//    }
//
//#if 0
//    static const char *whoami(uint32 target_index)
//    {
//        if (covariate<PreviousCovariate, 2>::index == target_index)
//        {
//            switch(EV)
//            {
//            case cigar_event::M:
//                return "FixedEvent<M>";
//
//            case cigar_event::I:
//                return "FixedEvent<I>";
//
//            case cigar_event::D:
//                return "FixedEvent<D>";
//
//            case cigar_event::S:
//                return "FixedEvent<S>";
//            }
//        }
//    }
//#endif
//};

template <typename PreviousCovariate = covariate_null>
struct covariate_QualityScore : public covariate<PreviousCovariate, 8>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        const CRQ_index idx = batch.crq_index(read_index);
        // xxxnsubtil: 45 is the "default" base quality for when insertion/deletion qualities are not available in the read
        // we should eventually grab these from the alignment data itself if they're present
        covariate_key_set key = { batch.qualities[idx.qual_start + bp_offset],
                                  45,
                                  45 };

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

//template <typename PreviousCovariate = covariate_null>
//struct covariate_EmpiricalQuality : public covariate<PreviousCovariate, 8>
//{
//    static CUDA_HOST_DEVICE encode(bqsr_context::view ctx,
//                                          const alignment_batch_device::const_view batch,
//                                          uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
//                                          uint32 input_key = 0)
//    {
//        const CRQ_index idx = batch.crq_index(read_index);
//        uint8 key = ctx.baq.qualities[idx.qual_start + bp_offset];
//
//        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
//                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
//    }
//
//#if 0
//    static const char *whoami(uint32 target_index)
//    {
//        if (covariate<PreviousCovariate, 8>::index == target_index)
//        {
//            return "EmpiricalQuality";
//        } else {
//            return PreviousCovariate::whoami(target_index);
//        }
//    }
//#endif
//};
