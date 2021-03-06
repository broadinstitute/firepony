/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "device_types.h"
#include "alignment_data_device.h"
#include "../sequence_database.h"

namespace firepony
{

struct cigar_event
{
    // note that this is stored in a packed 2-bit vector
    typedef enum {
        M = 0,  // match or mismatch
        I = 1,  // insertion to the reference
        D = 2,  // deletion from the reference
        S = 3,  // soft clipping (bases exist in read but not in reference)
    } Event;

    static CUDA_HOST_DEVICE char ascii(Event e)
    {
        return e == M ? 'M' :
               e == I ? 'I' :
               e == D ? 'D' :
                        'S';
    }

    static CUDA_HOST_DEVICE char ascii(uint32 e)
    {
        return e == M ? 'M' :
               e == I ? 'I' :
               e == D ? 'D' :
               e == S ? 'S' :
                        '?';
    }
};

template <target_system system>
struct cigar_context
{
    // prefix sum of cigar offsets
    persistent_allocation<system, uint32> cigar_offsets;

    // a vector of cigar "events"
    packed_vector<system, 2> cigar_events;
    // the read index for each cigar event
    persistent_allocation<system, uint32> cigar_event_read_index;
    // the read coordinate for each cigar event
    persistent_allocation<system, uint16> cigar_event_read_coordinates;
    // the reference coordinate for each cigar event, relative to the start of the alignment window
    persistent_allocation<system, uint16> cigar_event_reference_coordinates;

    // alignment window in the read, not including clipped bases
    persistent_allocation<system, ushort2> read_window_clipped;
    // alignment window in the read, not including clipped bases or leading/trailing insertions
    persistent_allocation<system, ushort2> read_window_clipped_no_insertions;
    // alignment window in the reference, not including clipped bases (relative to base alignment position)
    persistent_allocation<system, ushort2> reference_window_clipped;

    // bit vector representing SNPs, one per read bp
    // (1 means reference mismatch, 0 means match or non-M cigar event)
    packed_vector<system, 1> is_snp;
    // bit vector representing insertions, one per read bp (similar to is_snp)
    packed_vector<system, 1> is_insertion;
    // bit vector representing deletions, one per read bp
    // note that this only flags the starting base for a deletion and is independent of the other bit vectors
    // in other words, the same bp can have is_deletion and one of is_insertion or is_snp set
    packed_vector<system, 1> is_deletion;

    // number of errors for each read
    persistent_allocation<system, uint16> num_errors;
};

template <target_system system> void expand_cigars(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void debug_cigar(firepony_context<system>& context, const alignment_batch<system>& batch, int read_index);

} // namespace firepony
