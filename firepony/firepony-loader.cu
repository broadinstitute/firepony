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

#include "gamgee_loader.h"
#include "sequence_data.h"
#include "variant_data.h"
#include "mmap.h"
#include "serialization.h"

using namespace firepony;

firepony::sequence_data reference;

void load_fasta(const char *fname)
{
    shared_memory_file shmem;
    size_t size;
    bool ret;

    printf("loading %s...\n", fname);

    // load the sequence data
    ret = gamgee_load_sequences(&reference, fname,
                                SequenceDataMask::BASES |
                                SequenceDataMask::NAMES,
                                false);
    if (ret == false)
    {
        printf("could not load %s\n", fname);
        exit(1);
    }

    size = reference.serialized_size();

    printf("allocating %lu MB of shared memory...\n", size / (1024 * 1024));
    ret = shared_memory_file::create(&shmem, fname, size);
    if (ret == false)
    {
        printf("failed to create shared memory segment for %s\n", fname);
        exit(1);
    }

    reference.serialize(shmem.data);
    shmem.unmap();

    printf("%s loaded\n", fname);
}


void load_vcf(const char *fname)
{
    shared_memory_file shmem;
    variant_database data;
    size_t size;
    bool ret;

    printf("loading %s...\n", fname);

    // load the variant data
    ret = gamgee_load_vcf(&data, reference, fname,
                          VariantDataMask::CHROMOSOME |
                          VariantDataMask::ALIGNMENT,
                          false);

    if (ret == false)
    {
        printf("could not load %s\n", fname);
        exit(1);
    }

    size = data.serialized_size();

    printf("allocating %lu MB of shared memory...\n", size / (1024 * 1024));
    ret = shared_memory_file::create(&shmem, fname, size);
    if (ret == false)
    {
        printf("failed to create shared memory segment for %s\n", fname);
        exit(1);
    }

    data.serialize(shmem.data);
    shmem.unmap();

    printf("%s loaded\n", fname);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: %s <reference.fa> <variants.vcf>\n", argv[0]);
        exit(1);
    }

    const char *fasta = argv[1];
    const char *vcf = argv[2];

    load_fasta(fasta);
    load_vcf(vcf);

    return 0;
}
