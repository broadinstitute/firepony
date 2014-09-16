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

#include <nvbio/basic/primitives.h>

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
                                              uint32 read_index, uint16 bp, uint32 cigar_ev)
    {
        // add in our bits
        uint32 out = input_key | (((data) & BITMASK(bits)) << offset);
        // pass along to next in chain
        return PreviousCovariate::encode(ctx, batch, read_index, bp, cigar_ev, out);
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
                                           uint32 read_index, uint16 bp, uint32 cigar_ev,
                                           uint32 input_key = 0)
    {
        return input_key;
    }

    static NVBIO_HOST_DEVICE uint32 decode(uint32 input_key, uint32 target_index)
    {
        return 0;
    }
};

template<typename PreviousCovariate = covariate_null>
struct covariate_ReadGroup : public covariate<PreviousCovariate, 8>
{
    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp, uint32 cigar_ev,
                                           uint32 input_key = 0)
    {
        uint8 key = batch.read_groups[read_index];
        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp, cigar_ev);
    }
};

template <typename PreviousCovariate = covariate_null>
struct covariate_EventType : public covariate<PreviousCovariate, 2>
{
    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp, uint32 cigar_ev,
                                           uint32 input_key = 0)
    {
        uint8 key = ctx.cigar.cigar_ops[cigar_ev];
        return covariate<PreviousCovariate, 2>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp, cigar_ev);
    }
};

template <typename PreviousCovariate = covariate_null>
struct covariate_BaseQuality : public covariate<PreviousCovariate, 8>
{
    static NVBIO_HOST_DEVICE uint32 encode(bqsr_context::view ctx,
                                           const BAM_alignment_batch_device::const_view batch,
                                           uint32 read_index, uint16 bp, uint32 cigar_ev,
                                           uint32 input_key = 0)
    {
        const BAM_CRQ_index& idx = batch.crq_index[read_index];
        uint8 key = batch.qualities[idx.qual_start + bp];

        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp, cigar_ev);
    }
};

struct covariate_entry
{
    uint32 key;
    uint32 count;
};

typedef nvbio::vector<device_tag, covariate_entry> D_CovariateTable;
typedef nvbio::vector<host_tag, covariate_entry> H_CovariateTable;

struct covariate_context
{
    D_CovariateTable table;

    struct view
    {
        D_CovariateTable::plain_view_type table;
    };

    operator view()
    {
        view v = {
            plain_view(table),
        };

        return v;
    }
};

template<typename covariate_chain>
__global__ void covariates_kernel(covariate_chain covariates,
                                  bqsr_context::view ctx,
                                  covariate_context::view cv_ctx,
                                  const BAM_alignment_batch_device::const_view batch)
{
    int tid = blockIdx.x + threadIdx.x * blockDim.x;

    printf("tid = %d\n", tid);
    if(tid < ctx.active_read_list.size())
    {
        uint32 read_index = ctx.active_read_list[tid];
        const BAM_CRQ_index& idx = batch.crq_index[read_index];

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        for(uint32 ev = cigar_start; ev < cigar_end; ev++)
        {
            uint16 bp = ctx.cigar.cigar_op_read_coordinates[ev];
            if (bp != uint16(-1))
            {
                uint32 key = covariates.encode(ctx, batch, read_index, bp, ev);
                printf("read %d bp %d: key = %x\n", read_index, bp, key);
            }
        }
    }
}

// defines a covariate chain equivalent to GATK's RecalTable1
struct covariates_recaltable1
{
    // the type that represents the chain of covariates
    typedef covariate_EventType<
                covariate_BaseQuality<
                    covariate_ReadGroup<> > > chain;

    // the index of each covariate in the chain
    // (used when decoding a key)
    typedef enum {
        EventType = 0,
        BaseQuality = 1,
        ReadGroup = 2,
    } CovariateID;

    // extract a given covariate value from a key
    static uint32 decode(CovariateID id, uint32 key) { return chain::decode(key, id); }
};

void gather_covariates(bqsr_context *context, const BAM_alignment_batch_device& batch)
{
    covariate_context cv_ctx;

    covariates_kernel<<<1, 1>>>(covariates_recaltable1::chain(), *context, cv_ctx, batch);

    cudaDeviceSynchronize();
}
