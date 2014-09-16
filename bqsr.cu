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

#include <nvbio/basic/types.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/dna.h>
#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/vcf.h>
#include <nvbio/io/sequence/sequence_pac.h>

#include <map>

#include "bam_loader.h"
#include "reference.h"
#include "util.h"
#include "variants.h"
#include "bqsr_context.h"

/*
// sort batch by increasing alignment position
void device_sort_batch(BAM_alignment_batch_device *batch)
{
    D_VectorU32 temp_pos = batch->alignment_positions;

    thrust::sort_by_key(temp_pos.begin(),
                        temp_pos.begin() + temp_pos.size(),
                        batch->read_order.begin());
}
*/

int main(int argc, char **argv)
{
    // load the reference genome
    const char *ref_name = "hs37d5";
    const char *vcf_name = "/home/nsubtil/hg96/ALL.chr20.integrated_phase1_v3.20101123.snps_indels_svs.genotypes-stripped.vcf";

    struct reference_genome genome;

    printf("loading reference %s...\n", ref_name);

    if (genome.load(ref_name) == false)
    {
        printf("failed to load reference %s\n", ref_name);
        exit(1);
    }

    genome.download();

    SNPDatabase_refIDs db;
    printf("loading variant database %s...\n", vcf_name);
    io::loadVCF(db, vcf_name);
    db.compute_sequence_offsets(genome);

    DeviceSNPDatabase dev_db;
    dev_db.load(db);

    bqsr_context ctx(dev_db, genome.device);

    printf("%lu variants\n", db.genome_positions.size());
    printf("reading BAM...\n");

    BAMfile bam("/home/nsubtil/hg96/HG00096.chrom20.ILLUMINA.bwa.GBR.low_coverage.20120522.bam");

    BAM_alignment_batch_host h_batch;
    BAM_alignment_batch_device batch;

    uint64 alignments = 0;

    while(bam.next_batch(&h_batch, true, 2000000))
    {
        // load the next batch on the device
        batch.load(h_batch);

        // initialize the read order with 0..N
        ctx.read_order.resize(batch.crq_index.size());
        thrust::copy(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(0) + batch.crq_index.size(),
                     ctx.read_order.begin());

        // sort read_order by decreasing len
        //device_sort_batch(&batch);

        // transform the alignment coordinates to genome coordinates
        //genome.device.transform_alignment_start_positions(&batch);


        // build read offset list
        build_read_offset_list(&ctx, batch);
        // build read alignment window list
        build_alignment_windows(&ctx, batch);

        // filter known SNPs from active_loc_list
        filter_snps(&ctx, batch);

#if 0
        H_ReadOffsetList h_read_offset_list = ctx.snp_filter.read_offset_list;
        H_VectorU2 h_alignment_windows = ctx.alignment_windows;
        H_VectorU2 h_vcf_ranges = ctx.snp_filter.active_vcf_ranges;

        H_VectorU32 h_read_order = ctx.read_order;
        for(uint32 read_id = 0; read_id < 50 && read_id < h_read_order.size(); read_id++)
        {
            const uint32 read_index = h_read_order[read_id];

            const BAM_CRQ_index& idx = h_batch.crq_index[read_index];

            printf("read order %d read %d: cigar = [", read_id, read_index);
            for(uint32 i = idx.cigar_start; i < idx.cigar_start + idx.cigar_len; i++)
            {
                printf("%d%c", h_batch.cigars[i].len, h_batch.cigars[i].ascii_op());
            }
            printf("] offset list = [ ");

            for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
            {
                printf("%d ", h_read_offset_list[i]);
            }
            printf("]\n  sequence base [%u] sequence offset [%u] alignment window [%u, %u]\n",
                    genome.ref_sequence_offsets[h_batch.alignment_sequence_IDs[read_index]],
                    h_batch.alignment_positions[read_index],
                    h_alignment_windows[read_index].x,
                    h_alignment_windows[read_index].y);
            printf("  active VCF range: [%u, %u]\n", h_vcf_ranges[read_index].x, h_vcf_ranges[read_index].y);
        }
#endif

#if 1
        printf("active VCF ranges: %d out of %d reads (%f %)\n",
                ctx.snp_filter.active_read_ids.size(),
                ctx.read_order.size(),
                100.0 * float(ctx.snp_filter.active_read_ids.size()) / ctx.read_order.size());
#endif
        alignments += h_batch.crq_index.size();
    }

    printf("%llu alignments\n", alignments);

    return 0;
}
