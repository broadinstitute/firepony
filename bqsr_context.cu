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

#include "bqsr_context.h"

void bqsr_context::start_batch(BAM_alignment_batch_device& batch)
{
    // initialize the read order with 0..N
    active_read_list.resize(batch.num_reads);
    thrust::copy(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(0) + batch.crq_index.size(),
                 active_read_list.begin());

    // set up the active location list to cover all of the current batch
    active_location_list.resize(batch.reads.size());
    // mark all BPs as active
    thrust::fill(active_location_list.m_storage.begin(),
                 active_location_list.m_storage.end(),
                 0xffffffff);

    stats.total_reads += batch.num_reads;
}
#if 0

struct read_active_predicate
{
    NVBIO_HOST_DEVICE bool operator() (const uint32 index)
    {
        return index != uint32(-1);
    }
};

// remove reads that have been killed (index set to -1) from the active read list
void bqsr_context::compact_active_read_list(void)
{
    uint32 num_active;

    temp_u32 = active_read_list;
    num_active = nvbio::copy_if(temp_u32.size(),
                                temp_u32.begin(),
                                active_read_list.begin(),
                                read_active_predicate(),
                                temp_storage);

    active_read_list.resize(num_active);
}

#endif
