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
#include "baq.h"

namespace hengli {
  bool calcBAQFromHMM(uint8 *baq_output, int **state, char **bq,
                      bqsr_context *context, const reference_genome& reference,
                      const BAM_alignment_batch_host& batch, int read_index);
  void compute_baq(bqsr_context *context, const reference_genome& reference, const BAM_alignment_batch_host& batch);
}

namespace cpu {
  // compute the read and reference windows to run BAQ on for a given read
  void frame_alignment(ushort2& out_read_window, uint2& out_reference_window,
                       bqsr_context *context,
                       const reference_genome& reference,
                       const BAM_alignment_batch_device& batch,
                       const BAM_alignment_batch_host& h_batch,
                       uint32 read_index);
  int hmm_glocal(const H_PackedReference& referenceBases,
                 uint2 reference_window,
                 const H_StreamDNA16& queryBases,
                 uint2 query_window,
                 const uint8 *inputQualities,
                 uint32 *outputState,
                 uint8 *outputQualities);
}
