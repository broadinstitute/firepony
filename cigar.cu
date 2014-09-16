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
#include <nvbio/basic/numbers.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

#include "cigar.h"
#include "bqsr_types.h"
#include "bqsr_context.h"
#include "bam_loader.h"

// compute the length of a given cigar operator
struct cigar_op_len : public thrust::unary_function<const cigar_op&, uint32>
{
    NVBIO_HOST_DEVICE uint32 operator() (const cigar_op& op) const
    {
        return op.len;
    }
};

// expand cigar ops
struct cigar_op_expand : public bqsr_lambda
{
    cigar_op_expand(bqsr_context::view ctx,
                    const BAM_alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    NVBIO_HOST_DEVICE void operator() (const uint32 op_index)
    {
        const cigar_op& op = batch.cigars[op_index];
        const uint32 out_base = ctx.cigar.cigar_offsets[op_index];

        for(uint32 i = 0; i < op.len; i++)
        {
            switch(op.op)
            {
            case cigar_op::OP_M:
            case cigar_op::OP_MATCH:
            case cigar_op::OP_X:
                ctx.cigar.cigar_events[out_base + i] = cigar_event::M;
                break;

            case cigar_op::OP_I:
            case cigar_op::OP_N:
                ctx.cigar.cigar_events[out_base + i] = cigar_event::I;
                break;

            case cigar_op::OP_D:
            case cigar_op::OP_H:
            case cigar_op::OP_P:
                ctx.cigar.cigar_events[out_base + i] = cigar_event::D;
                break;
            }
        }
    }
};

// expand cigar coordinates for a read
// xxxnsubtil: this is very similar to compute_alignment_window, should merge
struct cigar_coordinates_expand : public bqsr_lambda
{
    cigar_coordinates_expand(bqsr_context::view ctx,
                             const BAM_alignment_batch_device::const_view batch)
        : bqsr_lambda(ctx, batch)
    { }

    NVBIO_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const BAM_CRQ_index& idx = batch.crq_index[read_index];
        const cigar_op *cigar = &batch.cigars[idx.cigar_start];

        uint32 base = ctx.cigar.cigar_offsets[idx.cigar_start];
        uint16 *output_read_coordinates = &ctx.cigar.cigar_op_read_coordinates[base];
        uint16 *output_reference_coordinates = &ctx.cigar.cigar_op_reference_coordinates[base];

        uint16 read_offset = 0;
        uint16 reference_offset = 0;

        for(uint32 c = 0; c < idx.cigar_len; c++)
        {
            switch(cigar[c].op)
            {
            case cigar_op::OP_M:
            case cigar_op::OP_MATCH:
            case cigar_op::OP_X:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *output_read_coordinates++ = read_offset;
                    *output_reference_coordinates++ = reference_offset;

                    read_offset++;
                    reference_offset++;
                }

                break;

            case cigar_op::OP_S:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *output_read_coordinates++ = uint16(-1);
                    *output_reference_coordinates++ = uint16(-1);

                    read_offset++;
                }

                break;

            case cigar_op::OP_N: // xxxnsubtil: N is really not supported and shouldn't be here
            case cigar_op::OP_I:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *output_read_coordinates++ = read_offset;
                    *output_reference_coordinates++ = uint16(-1);

                    read_offset++;
                }

                break;

            case cigar_op::OP_D:
            case cigar_op::OP_H:
            case cigar_op::OP_P: // xxxnsubtil: not sure how to handle P
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *output_read_coordinates++ = uint16(-1);
                    *output_reference_coordinates++ = reference_offset;

                    reference_offset++;
                }

                break;
            }
        }
    }
};

void expand_cigars(bqsr_context *context, const BAM_alignment_batch_device& batch)
{
    cigar_context& ctx = context->cigar;

    // compute the offsets of each expanded cigar op
    // xxxnsubtil: we ignore the active read list here, so we do unnecessary work
    // might want to revisit this
    ctx.cigar_offsets.resize(batch.cigars.size() + 1);

    // mark the first offset as 0
    thrust::fill_n(ctx.cigar_offsets.begin(), 1, 0);
    // do an inclusive scan to compute all offsets + the total size
    nvbio::inclusive_scan(batch.cigars.size(),
                          thrust::make_transform_iterator(batch.cigars.begin(), cigar_op_len()),
                          ctx.cigar_offsets.begin() + 1,    // the first output is 0
                          thrust::plus<uint32>(),
                          context->temp_storage);

    // read back the last element, which contains the size of the buffer required
    uint32 expanded_cigar_len = ctx.cigar_offsets[batch.cigars.size()];

    // make sure we have enough room for the expanded cigars
    ctx.cigar_ops.resize(expanded_cigar_len);
    ctx.cigar_op_reference_coordinates.resize(expanded_cigar_len);
    ctx.cigar_op_read_coordinates.resize(expanded_cigar_len);

    // expand the cigar ops first (xxxnsubtil: same as above, active read list is ignored)
    thrust::for_each(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(0) + batch.cigars.size(),
                     cigar_op_expand(*context, batch));

    // now expand the coordinates per read
    // this avoids having to deal with boundary condtions within reads
    thrust::for_each(context->active_read_list.begin(),
                     context->active_read_list.end(),
                     cigar_coordinates_expand(*context, batch));
}
