/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string>

#include <htslib/vcf.h>

#include "../command_line.h"
#include "reference.h"
#include "../variant_database.h"

#include "../mmap.h"
#include "../serialization.h"

namespace firepony {

bool load_vcf(variant_database_host *output, reference_file_handle *reference_handle, const char *filename, bool try_mmap)
{
    bool loaded = false;

    if (try_mmap)
    {
        shared_memory_file shmem;

        loaded = shared_memory_file::open(&shmem, filename);
        if (loaded == true)
        {
            serialization::unserialize(output, shmem.data);
            shmem.unmap();
        }
    }

    if (!loaded)
    {
        htsFile *fp;
        bcf_hdr_t *bcf_header;
        bcf1_t *data;

        fp = bcf_open(filename, "r");
        if (fp == NULL)
        {
            fprintf(stderr, "error opening %s\n", filename);
            return false;
        }

        bcf_header = bcf_hdr_read(fp);
        bcf_hdr_set_samples(bcf_header, nullptr, 0);

        // build a list of chromosomes in the header
        std::vector<std::string> chromosomes;
        for(int32 i = 0; i < bcf_header->nhrec; i++)
        {
            bcf_hrec_t *r = bcf_header->hrec[i];

            if (r->type == BCF_HL_CTG)
            {
                chromosomes.push_back(*r->vals);
            }
        }

        data = bcf_init();

        for(;;)
        {
            // read the next thing
            int ret;
            ret = bcf_read(fp, bcf_header, data);

            if (ret == -1)
            {
                break;
            }

            // "unpack" up to ALT inclusive
            bcf_unpack(data, BCF_UN_STR);

            // grab the data we need
            uint32 chromosome_id = data->rid;
            const std::string& chromosome_name = chromosomes[chromosome_id];
            reference_handle->make_sequence_available(chromosome_name);

            uint32 id = reference_handle->sequence_data.sequence_names.lookup(chromosome_name);
            if (id == uint32(-1))
            {
                fprintf(stderr, "WARNING: chromosome %s not found in reference data, skipping\n", chromosome_name.c_str());
                continue;
            }

            // make sure we have storage for the chromosome
            output->new_entry(id);
            auto& chromosome = output->get_sequence(id);

            chromosome.feature_start.push_back(data->pos);
            chromosome.feature_stop.push_back(data->pos + data->rlen - 1);
        }
    }

    // generate the search index for each chromosome
    for(auto *chromosome : output->storage)
    {
        chromosome->max_end_point_left.resize(chromosome->feature_stop.size());
        thrust::inclusive_scan(lift::backend_policy<host>::execution_policy(),
                               chromosome->feature_stop.begin(),
                               chromosome->feature_stop.end(),
                               chromosome->max_end_point_left.begin(),
                               thrust::maximum<uint32>());
    }

    return true;
}

} // namespace firepony
