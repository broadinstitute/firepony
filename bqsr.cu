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
#include <nvbio/io/reads/reads.h>
#include <nvbio/io/vcf.h>
#include <nvbio/io/fmi.h>

#include <map>

#include "bam_loader.h"
#include "reference.h"
#include "util.h"

struct SNPDatabase_refIDs : public nvbio::io::SNPDatabase
{
    // maps a variant ID to a reference sequence ID
    nvbio::vector<host_tag, uint32> variant_sequence_ref_ids;

    void build_variant_sequence_ref_ids(const reference_genome& genome)
    {
        variant_sequence_ref_ids.resize(reference_sequence_names.size());
        for(unsigned int c = 0; c < reference_sequence_names.size(); c++)
        {
            uint32 h = bqsr_string_hash(reference_sequence_names[c].c_str());

            assert(genome.ref_sequence_id_map.find(h) != genome.ref_sequence_id_map.end());

            // use find() to avoid touching the map --- operator[] can insert, which is invalid on a const object
            uint32 id = (*genome.ref_sequence_id_map.find(h)).second;
            variant_sequence_ref_ids[c] = id;
        }
    }
};


struct DeviceSNPDatabase
{
    // reference sequence ID for each variant
    nvbio::vector<device_tag, uint32> variant_sequence_ref_ids;
    // position of the variant in the reference sequence (first base in the sequence is position 1)
    nvbio::vector<device_tag, uint64> positions;

    // packed reference sequences
    nvbio::PackedVector<device_tag, 4> reference_sequences;
    // packed variant sequences
    nvbio::PackedVector<device_tag, 4> variants;
    // an index for both references and variants
    nvbio::vector<device_tag, io::SNP_sequence_index> ref_variant_index;

    void load(const SNPDatabase_refIDs& ref)
    {
        variant_sequence_ref_ids = ref.variant_sequence_ref_ids;
        positions = ref.positions;
        reference_sequences = ref.reference_sequences;
        variants = ref.variants;
        ref_variant_index = ref.ref_variant_index;
    }
};

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
    //io::loadVCF(db, "/home/nsubtil/hg96/ALL.chr20.integrated_phase1_v3.20101123.snps_indels_svs.genotypes-stripped.vcf");
    //io::loadVCF(db, "/home/nsubtil/test.vcf");
    //io::loadVCF(db, "/home/nsubtil/hg96/1k.chr20.vcf");
    io::loadVCF(db, vcf_name);
    db.build_variant_sequence_ref_ids(genome);

    DeviceSNPDatabase dev_db;
    dev_db.load(db);

    printf("%lu variants\n", db.positions.size());

#if 0
    uint64 size = 0;
    for(uint32 c = 0; c < db.positions.size(); c++)
    {
#if 0
        const struct io::SNP_sequence_index& idx = db.ref_variant_index[c];
        char reference[256], variant[256];

        nvbio::dna16_to_string(db.reference_sequences.begin() + idx.reference_start, idx.reference_len, reference);
        nvbio::dna16_to_string(db.variants.begin() + idx.variant_start, idx.variant_len, variant);

        printf("variant %d: chrom [%s] pos %llu reference[%s] variant[%s] quality %d\n",
                c,
                db.reference_sequence_names[c].c_str(),
                db.positions[c],
                reference,
                variant,
                db.variant_qualities[c]);
#endif

        size += db.reference_sequence_names[c].size();
        size += 9;
        size += 1;
    }

    size += db.reference_sequences.size() / 2;
    size += db.variants.size() / 2;
    printf("VCF size: %llu\n", size);
#endif

    BAMfile bam("/home/nsubtil/hg96/HG00096.chrom20.ILLUMINA.bwa.GBR.low_coverage.20120522.bam");

    BAM_alignment_batch_host batch;

    printf("reading BAM...\n");

    uint64 alignments = 0;
    while(bam.next_batch(&batch, false, 200000 / 4))
    {
        alignments += batch.crq_index.size();
#if 0
        for(unsigned int c = 0; c < batch.crq_index.size(); c++)
        {
            const BAM_alignment_header& align = batch.align_headers[c];
            const BAM_alignment_index& idx = batch.index[c];
            const BAM_CRQ_index& crq_idx = batch.crq_index[c];

            // QNAME
            printf("%s\t", &batch.names[idx.name]);
            // FLAG
            printf("%d\t", align.flags());
            // RNAME
            printf("%s\t", align.refID == -1 ? "*" : bam.header.sq_names[align.refID].c_str());
            // POS
            printf("%d\t", align.pos == -1 ? 0 : align.pos + 1);
            // MAPQ
            printf("%d\t", align.mapq());

            // CIGAR
            if (crq_idx.cigar_len == 0)
            {
                printf("*\t");
            } else {
                for(unsigned int i = crq_idx.cigar_start; i < crq_idx.cigar_start + crq_idx.cigar_len; i++)
                {
                    printf("%d%c", batch.cigars[i].len, batch.cigars[i].ascii_op());
                }

                printf("\t");
            }

            // RNEXT
            printf("%s\t", align.next_refID == -1 ? "*" : bam.header.sq_names[align.next_refID].c_str());
            // PNEXT
            printf("%d\t", align.next_pos == -1 ? 0 : align.next_pos + 1);
            // TLEN
            printf("%d\t", align.tlen);

            // SEQ + QUAL
            if (crq_idx.read_len == 0)
            {
                printf("*\t");
            } else {
                for(unsigned int i = crq_idx.read_start; i < crq_idx.read_start + crq_idx.read_len; i++)
                {
                    printf("%c", bam_to_bp(batch.reads[i]));
                }

                printf("\t");
            }

            // QUAL
            if (crq_idx.read_len == 0)
            {
                printf("*");
            } else {
                for(unsigned int i = crq_idx.read_start; i < crq_idx.read_start + crq_idx.read_len; i++)
                {
                    printf("%c", batch.qualities[i] + 33);
                }
            }

            printf("\n");
/*
            // AUX
            if (idx.aux_data_len == 0)
            {
                printf("\n");
            } else {
                printf("\t%s\n", &batch.aux_data[idx.aux_data_start]);
            }
            */
        }

        break;
#endif
    }

    printf("%llu alignments\n", alignments);

    return 0;
}
