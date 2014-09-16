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

enum FileFormat {
    FILE_UNKNOWN,
    FILE_VCF,
    FILE_FASTA,
};

bool test_suffix(const char *name, const char *suffix)
{
    const char *ptr;

    ptr = strstr(name, suffix);
    if (ptr && ptr[strlen(suffix)] == 0)
    {
        return true;
    }

    // try the same suffix with ".gz" appended
    std::string suffix_gz = std::string(suffix) + std::string(".gz");
    ptr = strstr(name, suffix_gz.c_str());
    if (ptr && ptr[suffix_gz.size()] == 0)
    {
        return true;
    }

    return false;
}

FileFormat detect_file_format(char *name)
{
    if (test_suffix(name, ".vcf"))
    {
        return FILE_VCF;
    }

    if (test_suffix(name, ".fasta") ||
        test_suffix(name, ".fa") ||
        test_suffix(name, ".fastq") ||
        test_suffix(name, ".fq"))
    {
        return FILE_FASTA;
    }

    return FILE_UNKNOWN;
}

void load_vcf(char *fname)
{
    assert(!"not implemented yet");
}

void load_fasta(char *fname)
{
    shared_memory_file shmem;
    sequence_data data;
    size_t size;
    bool ret;

    // load the sequence data
    ret = gamgee_load_sequences(&data, fname,
                                SequenceDataMask::BASES | SequenceDataMask::NAMES,
                                false);
    if (ret == false)
    {
        printf("could not load %s, skipping\n", fname);
        return;
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
    if (argc == 1)
    {
        printf("usage: %s <file> [file] ...\n", argv[0]);
    }

    for(int i = 1; i < argc; i++)
    {
        switch(detect_file_format(argv[i]))
        {
        case FILE_VCF:
            load_vcf(argv[i]);
            break;

        case FILE_FASTA:
            load_fasta(argv[i]);
            break;

        case FILE_UNKNOWN:
            printf("unknown file format for file %s, skipping\n", argv[i]);
            break;
        }
    }

    return 0;
}
