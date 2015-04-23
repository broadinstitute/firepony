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

#include <unistd.h>
#include <getopt.h>

#include <string>

#include "loader/reference.h"
#include "loader/variants.h"
#include "sequence_database.h"
#include "variant_database.h"
#include "mmap.h"
#include "serialization.h"

using namespace firepony;

template <typename T>
void create_shmem_segment(const char *fname, const T& data)
{
    shared_memory_file shmem;
    size_t size;
    bool ret;

    size = serialization::serialized_size(data);

    fprintf(stderr, "%s: allocating %lu MB of shared memory...\n", fname, size / (1024 * 1024));
    ret = shared_memory_file::create(&shmem, fname, size);
    if(ret == false)
    {
        fprintf(stderr, "failed to create shared memory segment for %s\n", fname);
        exit(1);
    }

    serialization::serialize(shmem.data, data);
    shmem.unmap();
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s <reference.fa> <variants.vcf>\n", argv[0]);
        exit(1);
    }

    const char *fasta = argv[1];
    const char *vcf = argv[2];

    // open the reference file
    reference_file_handle *ref_h = reference_file_handle::open(fasta, 1, false);
    if (ref_h == nullptr)
    {
        fprintf(stderr, "could not load %s\n", fasta);
        exit(1);
    }

    // load the variant database
    // this will populate the reference database as well
    variant_database_host h_dbsnp;
    bool ret;

    fprintf(stderr, "loading variant database %s...", vcf);
    fflush(stderr);

    ret = load_vcf(&h_dbsnp, ref_h, vcf, false);
    fprintf(stderr, "\n");
    if (ret == false)
    {
        fprintf(stderr, "could not load %s\n", vcf);
        exit(1);
    }

    create_shmem_segment(fasta, ref_h->sequence_data);
    create_shmem_segment(vcf, h_dbsnp);

    return 0;
}
