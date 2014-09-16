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
#include "from_nvbio/vcf.h"

#include <map>

#include "bqsr_types.h"
#include "gamgee_loader.h"
#include "alignment_data.h"
#include "sequence_data.h"
#include "util.h"
#include "variants.h"
#include "bqsr_context.h"
#include "filters.h"
#include "cigar.h"
#include "covariates.h"
#include "baq.h"

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

void debug_read(bqsr_context *context, const alignment_batch& batch, int read_index);

void init_cuda(void)
{
    cudaDeviceProp prop;
    int dev;
    int runtime_version;

    // trigger runtime initialization
    printf("loading CUDA runtime...\n");
    cudaFree(0);

    cudaRuntimeGetVersion(&runtime_version);
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    printf("CUDA runtime version: %d.%d\n", runtime_version / 1000, runtime_version % 100);
    printf("device: %s (%lu MB)\n", prop.name, prop.totalGlobalMem / (1024 * 1024));
}

int main(int argc, char **argv)
{
    // load the reference genome
    const char *ref_name = "/home/nsubtil/hg96/hs37d5.fa";
    //const char *ref_name = "/home/nsubtil/hg96/test";
    const char *vcf_name = "/home/nsubtil/hg96/ALL.chr20.integrated_phase1_v3.20101123.snps_indels_svs.genotypes-stripped.vcf";
    //const char *vcf_name = "/home/nsubtil/hg96/ALL.chr20.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf";
    //const char *vcf_name = "/home/nsubtil/hg96/one-variant.vcf";
    const char *bam_name = "/home/nsubtil/hg96/HG00096.chrom20.ILLUMINA.bwa.GBR.low_coverage.20120522.bam";
//    const char *bam_name = "/home/nsubtil/hg96/one-read.bam";

    init_cuda();

    sequence_data reference;

    if (gamgee_load_sequences(&reference, ref_name, SequenceDataMask::BASES | SequenceDataMask::NAMES) == false)
    {
        printf("failed to load reference %s\n", ref_name);
        exit(1);
    }

    reference.download();

    SNPDatabase_refIDs db;
    printf("loading variant database %s...\n", vcf_name);
    loadVCF(db, vcf_name);
    db.compute_sequence_offsets(reference);

    DeviceSNPDatabase dev_db;
    dev_db.load(db);


    printf("%lu variants\n", db.genome_start_positions.size());
    printf("reading BAM %s...\n", bam_name);

    gamgee_alignment_file bam(bam_name);
    alignment_batch batch;

    bqsr_context context(bam.header, dev_db, reference);

    uint32 data_mask = AlignmentDataMask::NAME |
                        AlignmentDataMask::CHROMOSOME |
                        AlignmentDataMask::ALIGNMENT_START |
                        AlignmentDataMask::CIGAR |
                        AlignmentDataMask::READS |
                        AlignmentDataMask::QUALITIES |
                        AlignmentDataMask::FLAGS |
                        AlignmentDataMask::MAPQ |
                        AlignmentDataMask::READ_GROUP;

    while(bam.next_batch(&batch, data_mask, 100000))
//    while(bam.next_batch(&batch, 500))
    {
        // load the next batch on the device
        batch.download();
        context.start_batch(batch);

        // build read offset list
        build_read_offset_list(&context, batch);
        // build read alignment window list
        build_alignment_windows(&context, batch);

        // apply read filters
        filter_reads(&context, batch);

        // apply per-BP filters
        filter_bases(&context, batch);

        // filter known SNPs from active_loc_list
        filter_known_snps(&context, batch);

        // generate cigar events and coordinates
        expand_cigars(&context, batch);

        // compute the base alignment quality for each read
        baq_reads(&context, batch);

        // build covariate tables
        gather_covariates(&context, batch);

#if 0
        for(uint32 read_id = 0; read_id < context.active_read_list.size(); read_id++)
        {
            const uint32 read_index = context.active_read_list[read_id];

            /*
            const char *name = &h_batch.names[h_batch.index[read_index].name];

            if (!strcmp(name, "SRR062635.1797528") ||
                !strcmp(name, "SRR062635.22970839") ||
                !strcmp(name, "SRR062641.22789430") ||
                !strcmp(name, "SRR062641.16264831"))
            {
                debug_read(&context, genome, h_batch, read_index);
            }*/

            debug_read(&context, batch, read_index);
        }
#endif

#if 0
        printf("active VCF ranges: %lu out of %lu reads (%f %%)\n",
                context.snp_filter.active_read_ids.size(),
                context.active_read_list.size(),
                100.0 * float(context.snp_filter.active_read_ids.size()) / context.active_read_list.size());

        H_ActiveLocationList h_bplist = context.active_location_list;
        uint32 zeros = 0;
        for(uint32 i = 0; i < h_bplist.size(); i++)
        {
            if (h_bplist[i] == 0)
                zeros++;
        }

        printf("active BPs: %u out of %u (%f %%)\n", h_bplist.size() - zeros, h_bplist.size(), 100.0 * float(h_bplist.size() - zeros) / float(h_bplist.size()));
#endif

        break;
    }

    output_covariates(&context);

    printf("%d reads filtered out of %d (%f%%)\n",
            context.stats.filtered_reads,
            context.stats.total_reads,
            float(context.stats.filtered_reads) / float(context.stats.total_reads) * 100.0);

    printf("computed base alignment quality for %d reads out of %d (%f%%)\n",
            context.stats.baq_reads,
            context.stats.total_reads - context.stats.filtered_reads,
            float(context.stats.baq_reads) / float(context.stats.total_reads - context.stats.filtered_reads) * 100.0);

    return 0;
}

void debug_read(bqsr_context *context, const alignment_batch& batch, int read_id)
{
    const alignment_batch_host& h_batch = batch.host;

    const uint32 read_index = context->active_read_list[read_id];
    const CRQ_index idx = h_batch.crq_index(read_index);

    printf("== read order %d read %d\n", read_id, read_index);

    printf("name = [%s]\n", h_batch.name[read_index].c_str());

    printf("  offset list = [ ");
    for(uint32 i = idx.read_start; i < idx.read_start + idx.read_len; i++)
    {
        uint16 off = context->read_offset_list[i];
        printf("%d ", off);
    }
    printf("]\n");

    debug_cigar(context, batch, read_index);
    debug_baq(context, batch, read_index);

    const uint2 alignment_window = context->alignment_windows[read_index];
    printf("  sequence name [%s]\n  sequence base [%lu]\n  sequence offset [%u]\n  alignment window [%u, %u]\n",
            context->reference.host.sequence_names.lookup(h_batch.chromosome[read_index]).c_str(),
            context->reference.host.sequence_bp_start[h_batch.chromosome[read_index]],
            h_batch.alignment_start[read_index],
            alignment_window.x,
            alignment_window.y);

    const uint2 vcf_range = context->snp_filter.active_vcf_ranges[read_index];
    printf("  active VCF range: [%u, %u[\n", vcf_range.x, vcf_range.y);

    printf("\n");
}
