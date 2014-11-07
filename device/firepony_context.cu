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

#include "firepony_context.h"
#include "util.h"

namespace firepony {

void firepony_context::start_batch(const alignment_batch& batch)
{
    // initialize the read order with 0..N
    active_read_list.resize(batch.host.num_reads);
    thrust::copy(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(0) + batch.host.num_reads,
                 active_read_list.begin());

    // set up the active location list to cover all of the current batch
    active_location_list.resize(batch.host.reads.size());
    // mark all BPs as active
    thrust::fill(active_location_list.m_storage.begin(),
                 active_location_list.m_storage.end(),
                 0xffffffff);
}

void firepony_context::end_batch(const alignment_batch& batch)
{
    stats.total_reads += batch.host.num_reads;
}

} // namespace firepony

