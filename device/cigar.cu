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

#include "from_nvbio/dna.h"

#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

#include "cigar.h"
#include "types.h"
#include "firepony_context.h"
#include "alignment_data.h"
#include "util.h"

#include "primitives/cuda.h"
#include "primitives/parallel.h"

namespace firepony {

// compute the length of a given cigar operator
struct cigar_op_len : public thrust::unary_function<const cigar_op&, uint32>
{
    CUDA_HOST_DEVICE uint32 operator() (const cigar_op& op) const
    {
        return op.len;
    }
};

// expand cigar ops into temp storage
struct cigar_op_expand : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE void operator() (const uint32 op_index)
    {
        const cigar_op& op = batch.cigars[op_index];
        const uint32 out_base = ctx.cigar.cigar_offsets[op_index];

        uint8 *out = &ctx.temp_storage[0] + out_base;

        for(uint32 i = 0; i < op.len; i++)
        {
            switch(op.op)
            {
            case cigar_op::OP_M:
            case cigar_op::OP_MATCH:
            case cigar_op::OP_X:
                out[i] = cigar_event::M;
                break;

            case cigar_op::OP_I:
            case cigar_op::OP_N:
                out[i] = cigar_event::I;
                break;

            case cigar_op::OP_D:
            case cigar_op::OP_H:
            case cigar_op::OP_P:
                out[i] = cigar_event::D;
                break;

            case cigar_op::OP_S:
                out[i] = cigar_event::S;
                break;
            }
        }
    }
};

// compact the cigar events from temporary storage into a 2-bit packed vector
struct cigar_op_compact : public lambda
{
    cigar_op_compact(firepony_context::view ctx,
                     const alignment_batch_device::const_view batch)
        : lambda(ctx, batch)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 word_index)
    {
        D_PackedVector_2b::view& events = ctx.cigar.cigar_events;
        const uint8 *input = &ctx.temp_storage[word_index * D_PackedVector_2b::SYMBOLS_PER_WORD];

        for(uint32 i = 0; i < D_PackedVector_2b::SYMBOLS_PER_WORD; i++)
        {
            events[word_index * D_PackedVector_2b::SYMBOLS_PER_WORD + i] = input[i];
        }
    }
};

// initialize read windows
// note: this does not initialize the reference window, as it needs to be computed once all clipping has been done
struct read_window_init : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        ctx.cigar.read_window_clipped[read_index] = make_ushort2(0, idx.read_len - 1);
    }
};

// clips sequencing adapters from the reads
struct remove_adapters : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE bool hasWellDefinedFragmentSize(const uint32 read_index)
    {
        const auto flags = batch.flags[read_index];
        const auto inferred_insert_size = batch.inferred_insert_size[read_index];

        if (inferred_insert_size == 0)
            // no adaptors in reads with mates in another chromosome or unmapped pairs
            return false;

        if (!(flags & AlignmentFlags::PAIRED))
            // only reads that are paired can be adaptor trimmed
            return false;

        if ((flags & AlignmentFlags::UNMAP) ||
            (flags & AlignmentFlags::MATE_UNMAP))
            // only reads when both reads are mapped can be trimmed
            return false;

        if ((flags & AlignmentFlags::REVERSE) ==
            (flags & AlignmentFlags::MATE_REVERSE))
            // sanity check to ensure that read1 and read2 aren't on the same strand
            return false;

        if (flags & AlignmentFlags::REVERSE)
        {
            // we're on the negative strand, so our read runs right to left
            return batch.alignment_stop[read_index] > batch.mate_alignment_start[read_index];
        } else {
            // we're on the positive strand, so our mate should be to our right (his start + insert size should be past our start)
            return batch.alignment_start[read_index] <= batch.mate_alignment_start[read_index] + inferred_insert_size;
        }
    }

    static constexpr uint32 CANNOT_COMPUTE_ADAPTOR_BOUNDARY = 0xffffffff;

    CUDA_HOST_DEVICE uint32 getAdaptorBoundary(const uint32 read_index)
    {
        if (!hasWellDefinedFragmentSize(read_index))
        {
            return CANNOT_COMPUTE_ADAPTOR_BOUNDARY;
        }

        if (batch.flags[read_index] & AlignmentFlags::REVERSE)
        {
            return uint32(batch.mate_alignment_start[read_index] - 1);
        } else {
            const int insertSize = (batch.inferred_insert_size[read_index] < 0 ? -batch.inferred_insert_size[read_index] : batch.inferred_insert_size[read_index]);
            return uint32(batch.alignment_start[read_index] + insertSize + 1);
        }
    }

    CUDA_HOST_DEVICE int getReadCoordinateForReferenceCoordinate(const uint32 read_index, uint32 ref_coord)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_stop = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        for(uint32 ev = cigar_start; ev < cigar_stop; ev++)
        {
            if (batch.alignment_start[read_index] + ctx.cigar.cigar_event_reference_coordinates[ev] == ref_coord)
            {
                return ctx.cigar.cigar_event_read_coordinates[ev];
            }
        }

        return -1;
    }

    CUDA_HOST_DEVICE ushort2 hardClipByReferenceCoordinates(const uint32 read_index, int refStart, int refStop)
    {
        int start;
        int stop;

        if (refStart < 0)
        {
            start = 0;
            stop = getReadCoordinateForReferenceCoordinate(read_index, uint32(refStop));
        } else {
            start = getReadCoordinateForReferenceCoordinate(read_index, uint32(refStart));
            stop = ctx.cigar.read_window_clipped[read_index].y - ctx.cigar.read_window_clipped[read_index].x - 1;
        }

        return make_ushort2(start, stop);
    }

    CUDA_HOST_DEVICE void hardClipByReferenceCoordinates_LeftTail(const uint32 read_index, int refStop)
    {
        auto& read_window_clipped = ctx.cigar.read_window_clipped[read_index];

        ushort2 adapter = hardClipByReferenceCoordinates(read_index, -1, refStop);
        read_window_clipped.x = max<uint16>(read_window_clipped.x, adapter.y + 1);
    }

    CUDA_HOST_DEVICE void hardClipByReferenceCoordinates_RightTail(const uint32 read_index, int refStart)
    {
        auto& read_window_clipped = ctx.cigar.read_window_clipped[read_index];

        ushort2 adapter = hardClipByReferenceCoordinates(read_index, refStart, -1);
        read_window_clipped.y = min<uint16>(read_window_clipped.y, adapter.x - 1);
    }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const auto& read_window_clipped = ctx.cigar.read_window_clipped[read_index];
        uint32 adaptorBoundary = getAdaptorBoundary(read_index);

        if (adaptorBoundary == CANNOT_COMPUTE_ADAPTOR_BOUNDARY)
            return;

        // xxxnsubtil: i think this might be wrong, doesn't account for indels (do we need to?)
        if (adaptorBoundary < batch.alignment_start[read_index] + read_window_clipped.x || adaptorBoundary > batch.alignment_start[read_index] + read_window_clipped.y)
            return;

        if (batch.flags[read_index] & AlignmentFlags::REVERSE)
        {
            hardClipByReferenceCoordinates_LeftTail(read_index, adaptorBoundary);
        } else {
            hardClipByReferenceCoordinates_RightTail(read_index, adaptorBoundary);
        }
    }
};

// remove soft-clip regions from the active read window
struct remove_soft_clips : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        auto read_window_clipped = ctx.cigar.read_window_clipped[read_index];

        uint32 cigar_index = idx.cigar_start;
        uint32 read_offset = 0;

        // note: we assume that the leading/trailing clip regions have been validated by the read filtering stage!

        // iterate forward through the leading clip region
        while(cigar_index < idx.cigar_start + idx.cigar_len &&
                read_offset < idx.read_len)
        {
            const auto& op = batch.cigars[cigar_index];

            if (op.op == cigar_op::OP_H || op.op == cigar_op::OP_S)
            {
                if (read_offset + op.len > read_window_clipped.x)
                {
                    read_window_clipped.x = min<uint16>(read_offset + op.len, read_window_clipped.y);
                }

                read_offset += op.len;
                cigar_index++;
            } else {
                break;
            }
        }

        // iterate backward through the trailing clip region
        cigar_index = idx.cigar_start + idx.cigar_len - 1;
        read_offset = read_window_clipped.y;

        while(cigar_index >= idx.cigar_start &&
                read_offset > read_window_clipped.x)
        {
            const auto& op = batch.cigars[cigar_index];

            if (op.op == cigar_op::OP_H || op.op == cigar_op::OP_S)
            {
                if (read_offset - op.len < read_window_clipped.y)
                {
                    read_window_clipped.y = max<uint16>(read_offset - op.len, read_window_clipped.x);
                }

                read_offset -= op.len;
                cigar_index--;
            } else {
                break;
            }
        }

        ctx.cigar.read_window_clipped[read_index] = read_window_clipped;
    }
};

// compute clipped read window without leading/trailing insertions
struct compute_no_insertions_window : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const auto& read_window_clipped = ctx.cigar.read_window_clipped[read_index];
        auto read_window_clipped_no_insertions = ctx.cigar.read_window_clipped_no_insertions[read_index];

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        read_window_clipped_no_insertions = read_window_clipped;

        // iterate forward at the start
        uint32 ev = cigar_start;
        uint32 read_offset = read_window_clipped.x;

        while(ev < cigar_start + read_window_clipped.y &&
              read_offset < read_window_clipped.y &&
              ctx.cigar.cigar_events[ev] == cigar_event::I)
        {
            read_window_clipped_no_insertions.x++;
            read_offset++;
            ev++;
        }

        ev = cigar_end - 1;
        read_offset = read_window_clipped.y;

        while(ev > cigar_start + read_window_clipped_no_insertions.x &&
                read_offset > read_window_clipped_no_insertions.x &&
                ctx.cigar.cigar_events[ev] == cigar_event::I)
        {
            read_window_clipped_no_insertions.y--;
            read_offset--;
            ev--;
        }

        ctx.cigar.read_window_clipped_no_insertions[read_index] = read_window_clipped_no_insertions;
    }
};

struct compute_reference_window : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const auto& read_window_clipped_no_insertions = ctx.cigar.read_window_clipped_no_insertions[read_index];
        auto& reference_window_clipped = ctx.cigar.reference_window_clipped[read_index];

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len];

        // do a linear search for the read offset
        // (this could be smarter, but it doesn't seem to matter)
        for(uint32 i = cigar_start; i < cigar_end; i++)
        {
            if (ctx.cigar.cigar_event_read_coordinates[i] == read_window_clipped_no_insertions.x)
            {
                while(ctx.cigar.cigar_event_reference_coordinates[i] == uint16(-1) &&
                        i < cigar_end)
                {
                    i++;
                }

                if (i == cigar_end)
                {
                    // should never happen
                    reference_window_clipped = make_ushort2(uint16(-1), uint16(-1));
                    return;
                }

                reference_window_clipped.x = ctx.cigar.cigar_event_reference_coordinates[i];
                break;
            }
        }

        for(uint32 i = cigar_end - 1; i >= cigar_start; i--)
        {
            if (ctx.cigar.cigar_event_read_coordinates[i] == read_window_clipped_no_insertions.y)
            {
                while(ctx.cigar.cigar_event_reference_coordinates[i] == uint16(-1) &&
                        i > cigar_start)
                {
                    i--;
                }

                if (i == cigar_start)
                {
                    // should never happen
                    reference_window_clipped = make_ushort2(uint16(-1), uint16(-1));
                    return;
                }

                reference_window_clipped.y = ctx.cigar.cigar_event_reference_coordinates[i];
                break;
            }
        }
    }
};

// expand cigar coordinates for a read
// xxxnsubtil: this is very similar to compute_alignment_window, should merge
struct cigar_coordinates_expand : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const cigar_op *cigar = &batch.cigars[idx.cigar_start];

        uint32 base = ctx.cigar.cigar_offsets[idx.cigar_start];
        uint32 *output_read_index = &ctx.cigar.cigar_event_read_index[base];
        uint16 *output_read_coordinates = &ctx.cigar.cigar_event_read_coordinates[base];
        uint16 *output_reference_coordinates = &ctx.cigar.cigar_event_reference_coordinates[base];

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
                    *output_read_index++ = read_index;
                    *output_read_coordinates++ = read_offset;
                    *output_reference_coordinates++ = reference_offset;

                    read_offset++;
                    reference_offset++;
                }

                break;

            case cigar_op::OP_S:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *output_read_index++ = read_index;
                    *output_read_coordinates++ = read_offset;
                    *output_reference_coordinates++ = uint16(-1);

                    read_offset++;
                }

                break;

            case cigar_op::OP_N: // xxxnsubtil: N is really not supported and shouldn't be here
            case cigar_op::OP_I:
                for(uint32 i = 0; i < cigar[c].len; i++)
                {
                    *output_read_index++ = read_index;
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
                    *output_read_index++ = read_index;
                    *output_read_coordinates++ = uint16(-1);
                    *output_reference_coordinates++ = reference_offset;

                    reference_offset++;
                }
            }
        }
    }
};

struct compute_error_vectors : public lambda
{
    using lambda::lambda;

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const bool negative_strand = batch.flags[read_index] & AlignmentFlags::REVERSE;

        // fetch the alignment base in reference coordinates
        const uint32 seq_id = batch.chromosome[read_index];
        const uint32 seq_base = ctx.reference.sequence_bp_start[seq_id];
        const uint32 align_offset = batch.alignment_start[read_index];
        const uint32 reference_alignment_start = seq_base + align_offset;

        const auto& read_window_clipped = ctx.cigar.read_window_clipped[read_index];

        uint16 current_bp_idx = 0;
        uint16 num_errors = 0;

        // go through the cigar events looking for the event we're interested in
        for(uint32 event = idx.cigar_start; event < idx.cigar_start + idx.cigar_len; event++)
        {
            // figure out the cigar event range for this event
            const uint32 cigar_start = ctx.cigar.cigar_offsets[event];
            const uint32 cigar_end = ctx.cigar.cigar_offsets[event+1];

            switch(batch.cigars[event].op)
            {
            case cigar_op::OP_M:

                for(uint32 i = cigar_start; i < cigar_end; i++)
                {
                    // update the current read bp index
                    current_bp_idx = ctx.cigar.cigar_event_read_coordinates[i];
                    // load the read bp
                    const uint8 read_bp = batch.reads[idx.read_start + current_bp_idx];

                    // load the corresponding sequence bp
                    const uint32 reference_bp_idx = reference_alignment_start + ctx.cigar.cigar_event_reference_coordinates[i];
                    const uint8 reference_bp = ctx.reference.bases[reference_bp_idx];

                    if (reference_bp != read_bp)
                    {
                        ctx.cigar.is_snp[idx.read_start + current_bp_idx] = 1;

                        // if we are inside the clipped read window, count this error
                        if (current_bp_idx >= read_window_clipped.x && current_bp_idx <= read_window_clipped.y)
                            num_errors++;
                    }
                }

                break;

            case cigar_event::I:
                // mark the read bp where an insertion begins
                current_bp_idx = ctx.cigar.cigar_event_read_coordinates[cigar_start];

                if (!negative_strand)
                {
                    ctx.cigar.is_insertion[idx.read_start + current_bp_idx - 1] = 1;
                } else {
                    ctx.cigar.is_insertion[idx.read_start + current_bp_idx + batch.cigars[event].len - 1] = 1;
                }

                // if we are inside the clipped read window, count this error
                // xxxnsubtil: unclear what happens if only part of the insertion is inside the clipped read window --- can this happen?
                if (current_bp_idx >= read_window_clipped.x && current_bp_idx <= read_window_clipped.y)
                    num_errors++;

                break;

            case cigar_event::D:
                // note: deletions do not exist in the read, so current_bp_idx is not updated here
                // mark the read bp where a deletion begins
                if (!negative_strand)
                {
                    ctx.cigar.is_deletion[idx.read_start + current_bp_idx] = 1;
                } else {
                    ctx.cigar.is_deletion[idx.read_start + current_bp_idx + 1] = 1;
                }

                // if we are inside the clipped read window, count this error
                if (current_bp_idx >= read_window_clipped.x && current_bp_idx <= read_window_clipped.y)
                    num_errors++;

                break;
            }
        }

        ctx.cigar.num_errors[read_index] = num_errors;
    }
};

#ifdef CUDA_DEBUG
// debug aid: sanity check that the expanded cigar events match what we expect
struct sanity_check_cigar_events : public lambda
{
    sanity_check_cigar_events(firepony_context::view ctx,
                              const alignment_batch_device::const_view batch)
        : lambda(ctx, batch)
    { }

    CUDA_HOST_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const cigar_op *cigar = &batch.cigars[idx.cigar_start];

        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        uint32 cigar_event_idx = 0;

        for(uint32 c = 0; c < idx.cigar_len; c++)
        {
            for(uint32 i = 0; i < cigar[c].len; i++)
            {
                switch(cigar[c].op)
                {
                case cigar_op::OP_M:
                case cigar_op::OP_MATCH:
                case cigar_op::OP_X:
                    if (ctx.cigar.cigar_events[cigar_start + cigar_event_idx] != cigar_event::M)
                    {
                        printf("*** failed sanity check: read %u cigar op %u %u event offset %u: expected M, got %c\n",
                                read_index, c, i, cigar_start + cigar_event_idx, cigar_event::ascii(ctx.cigar.cigar_events[cigar_start + cigar_event_idx]));
                        return;
                    }

                    cigar_event_idx++;
                    break;

                case cigar_op::OP_N: // xxxnsubtil: N is really not supported and shouldn't be here
                case cigar_op::OP_I:
                    if (ctx.cigar.cigar_events[cigar_start + cigar_event_idx] != cigar_event::I)
                    {
                        printf("*** failed sanity check: read %u cigar op %u %u event offset %u: expected I, got %c\n",
                                read_index, c, i, cigar_start + cigar_event_idx, cigar_event::ascii(ctx.cigar.cigar_events[cigar_start + cigar_event_idx]));
                        return;
                    }

                    cigar_event_idx++;
                    break;

                case cigar_op::OP_D:
                case cigar_op::OP_H:
                case cigar_op::OP_P: // xxxnsubtil: not sure how to handle P
                    if (ctx.cigar.cigar_events[cigar_start + cigar_event_idx] != cigar_event::D)
                    {
                        printf("*** failed sanity check: read %u cigar op %u %u event offset %u: expected D, got %c\n",
                                read_index, c, i, cigar_start + cigar_event_idx, cigar_event::ascii(ctx.cigar.cigar_events[cigar_start + cigar_event_idx]));
                        return;
                    }

                    cigar_event_idx++;
                    break;

                case cigar_op::OP_S:
                    if (ctx.cigar.cigar_events[cigar_start + cigar_event_idx] != cigar_event::S)
                    {
                        printf("*** failed sanity check: read %u cigar op %u %u event offset %u: expected S, got %c\n",
                                read_index, c, i, cigar_start + cigar_event_idx, cigar_event::ascii(ctx.cigar.cigar_events[cigar_start + cigar_event_idx]));
                        return;
                    }

                    cigar_event_idx++;
                    break;
                }
            }
        }
    }
};
#endif

void expand_cigars(firepony_context& context, const alignment_batch& batch)
{
    cigar_context& ctx = context.cigar;

    // compute the offsets of each expanded cigar op
    // xxxnsubtil: we ignore the active read list here, so we do unnecessary work
    // might want to revisit this
    ctx.cigar_offsets.resize(batch.device.cigars.size() + 1);

    // mark the first offset as 0
    thrust::fill_n(ctx.cigar_offsets.begin(), 1, 0);
    // do an inclusive scan to compute all offsets + the total size
    inclusive_scan(thrust::make_transform_iterator(batch.device.cigars.begin(), cigar_op_len()),
                   batch.device.cigars.size(),
                   ctx.cigar_offsets.begin() + 1,
                   thrust::plus<uint32>());

    // read back the last element, which contains the size of the buffer required
    uint32 expanded_cigar_len = ctx.cigar_offsets[batch.device.cigars.size()];

    // make sure we have enough room for the expanded cigars
    // note: temporary storage must be padded to a multiple of the word size, since we'll pack whole words at a time
    pack_prepare_storage_2bit(context.temp_storage, expanded_cigar_len);
    ctx.cigar_events.resize(expanded_cigar_len);

    ctx.cigar_event_read_index.resize(expanded_cigar_len);
    ctx.cigar_event_reference_coordinates.resize(expanded_cigar_len);
    ctx.cigar_event_read_coordinates.resize(expanded_cigar_len);

    ctx.read_window_clipped.resize(batch.device.num_reads);
    ctx.read_window_clipped_no_insertions.resize(batch.device.num_reads);
    ctx.reference_window_clipped.resize(batch.device.num_reads);

    ctx.is_snp.resize(batch.device.reads.size());
    ctx.is_insertion.resize(batch.device.reads.size());
    ctx.is_deletion.resize(batch.device.reads.size());
    ctx.num_errors.resize(batch.device.num_reads);

    // the following require zero initialization
    thrust::fill(ctx.is_snp.m_storage.begin(), ctx.is_snp.m_storage.end(), 0);
    thrust::fill(ctx.is_insertion.m_storage.begin(), ctx.is_insertion.m_storage.end(), 0);
    thrust::fill(ctx.is_deletion.m_storage.begin(), ctx.is_deletion.m_storage.end(), 0);
    thrust::fill(ctx.num_errors.begin(), ctx.num_errors.end(), 0);

    // cigar_events_read_index is initialized to -1; this means that all reads are considered inactive
    // it will be filled in during cigar coordinate expansion to mark active reads
    thrust::fill(ctx.cigar_event_read_index.begin(), ctx.cigar_event_read_index.end(), uint32(-1));

    // expand the cigar ops into temp storage (xxxnsubtil: same as above, active read list is ignored)
    thrust::for_each(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(0) + batch.device.cigars.size(),
                     cigar_op_expand(context, batch.device));

    // pack the cigar into a 2-bit vector
    pack_to_2bit(ctx.cigar_events, context.temp_storage);

#ifdef CUDA_DEBUG
    thrust::for_each(firepony_context.active_read_list.begin(),
                     firepony_context.active_read_list.end(),
                     sanity_check_cigar_events(firepony_context, batch.device));
#endif

    // now expand the coordinates per read
    // this avoids having to deal with boundary conditions within reads
    thrust::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     cigar_coordinates_expand(context, batch.device));

    // initialize read windows
    thrust::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     read_window_init(context, batch.device));

    // remove sequencing adapters
    thrust::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     remove_adapters(context, batch.device));

    // remove soft clip regions
    thrust::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     remove_soft_clips(context, batch.device));

    // compute the no insertions window based on the clipping window
    thrust::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_no_insertions_window(context, batch.device));

    // finally, compute the reference window (using the no insertions window)
    thrust::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_reference_window(context, batch.device));

    // compute the error bit vectors
    // this also counts the number of errors in each read
    thrust::for_each(context.active_read_list.begin(),
                     context.active_read_list.end(),
                     compute_error_vectors(context, batch.device));
}

void debug_cigar(firepony_context& context, const alignment_batch& batch, int read_index)
{
    const alignment_batch_host& h_batch = batch.host;

    const CRQ_index idx = h_batch.crq_index(read_index);
    const cigar_context& ctx = context.cigar;

    fprintf(stderr, "  cigar info:\n");

    fprintf(stderr, "    cigar                       = [");
    for(uint32 i = idx.cigar_start; i < idx.cigar_start + idx.cigar_len; i++)
    {
        cigar_op op = h_batch.cigars[i];
        fprintf(stderr, "%d%c", op.len, op.ascii_op());
    }
    fprintf(stderr, "]\n");

    uint32 cigar_start = ctx.cigar_offsets[idx.cigar_start];
    uint32 cigar_end = ctx.cigar_offsets[idx.cigar_start + idx.cigar_len];
    fprintf(stderr, "    offset range                = [% 3d, % 3d]\n", cigar_start, cigar_end);

    fprintf(stderr, "                                    ");
    for(uint32 i = 0; i < cigar_end - cigar_start; i++)
    {
        fprintf(stderr, "% 3d ", i);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "    event list                  = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        fprintf(stderr, "  %c ", cigar_event::ascii(ctx.cigar_events[i]));
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    event idx -> read coords    = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        fprintf(stderr, "% 3d ", (int16) ctx.cigar_event_read_coordinates[i]);
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    event reference coordinates = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        fprintf(stderr, "% 3d ", (int16) ctx.cigar_event_reference_coordinates[i]);
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    is snp                      = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        uint16 read_bp_idx = ctx.cigar_event_read_coordinates[i];
        if (read_bp_idx == uint16(-1))
        {
            fprintf(stderr, "  - ");
        } else {
            fprintf(stderr, "% 3d ", (uint8) ctx.is_snp[idx.read_start + read_bp_idx]);
        }
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    is insertion                = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        uint16 read_bp_idx = ctx.cigar_event_read_coordinates[i];
        if (read_bp_idx == uint16(-1))
        {
            fprintf(stderr, "  - ");
        } else {
            fprintf(stderr, "% 3d ", (uint8) ctx.is_insertion[read_bp_idx]);
        }
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    is deletion                 = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        uint16 read_bp_idx = ctx.cigar_event_read_coordinates[i];
        if (read_bp_idx == uint16(-1))
        {
            fprintf(stderr, "  - ");
        } else {
            fprintf(stderr, "% 3d ", (uint8) ctx.is_deletion[read_bp_idx]);
        }
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    active location list        = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        uint16 bp_offset = ctx.cigar_event_read_coordinates[i];
        if (bp_offset == uint16(-1))
        {
            fprintf(stderr, "  - ");
        } else {
            fprintf(stderr, "% 3d ", context.active_location_list[idx.read_start + bp_offset] ? 1 : 0);
        }
    }
    fprintf(stderr, "]\n");

    const uint32 ref_sequence_id = h_batch.chromosome[read_index];
    const uint32 ref_sequence_base = context.reference.host.sequence_bp_start[ref_sequence_id];
    const uint32 ref_sequence_offset = ref_sequence_base + h_batch.alignment_start[read_index];

    fprintf(stderr, "    reference sequence data     = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        const uint16 ref_bp = ctx.cigar_event_reference_coordinates[i];
        fprintf(stderr, "  %c ", ref_bp == uint16(-1) ? '-' : from_nvbio::iupac16_to_char(context.reference.host.bases[ref_sequence_offset + ref_bp]));
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    read sequence data          = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        const uint16 read_bp = ctx.cigar_event_read_coordinates[i];
        char base;

        if (read_bp == uint16(-1))
        {
            base = '-';
        } else {
            base = from_nvbio::iupac16_to_char(h_batch.reads[idx.read_start + read_bp]);
            if (ctx.cigar_events[i] == cigar_event::S)
            {
                // display soft-clipped bases in lowercase
                base = tolower(base);
            }
        }

        fprintf(stderr, "  %c ", base);
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    read quality data           = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        const uint16 read_bp = ctx.cigar_event_read_coordinates[i];

        if (read_bp == uint16(-1))
        {
            fprintf(stderr, "    ");
        } else {
            fprintf(stderr, "% 3d ", h_batch.qualities[idx.qual_start + read_bp]);
        }
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "    ... in ascii                = [ ");
    for(uint32 i = cigar_start; i < cigar_end; i++)
    {
        const uint16 read_bp = ctx.cigar_event_read_coordinates[i];

        if (read_bp == uint16(-1))
        {
            fprintf(stderr, " -  ");
        } else {
            fprintf(stderr, "  %c ", h_batch.qualities[idx.qual_start + read_bp] + '!');
        }
    }
    fprintf(stderr, "]\n");

    ushort2 read_window_clipped = ctx.read_window_clipped[read_index];
    fprintf(stderr, "    clipped read window         = [ % 3d, % 3d ]\n", read_window_clipped.x, read_window_clipped.y);

    ushort2 read_window_clipped_no_insertions = ctx.read_window_clipped_no_insertions[read_index];
    fprintf(stderr, "    ... lead/trail insertions   = [ % 3d, % 3d ]\n",
                read_window_clipped_no_insertions.x, read_window_clipped_no_insertions.y);

    ushort2 reference_window_clipped = ctx.reference_window_clipped[read_index];
    fprintf(stderr, "    clipped reference window    = [ % 3d, % 3d ]\n",
                reference_window_clipped.x, reference_window_clipped.y);

    uint16 err = ctx.num_errors[read_index];
    fprintf(stderr, "    errors in clipped region    = [ % 3d ]\n", err);

    fprintf(stderr, "\n");
}

} // namespace firepony
