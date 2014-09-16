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


#include "bqsr_types.h"
#include "alignment_data.h"
#include "bqsr_context.h"
#include "covariates.h"
#include "covariates_bit_packing.h"

#include <nvbio/basic/primitives.h>
#include <thrust/functional.h>

// defines a covariate chain equivalent to GATK's RecalTable1
struct covariates_recaltable1
{
    // the type that represents the chain of covariates
    typedef covariate_ReadGroup<
             covariate_QualityScore<
               covariate_EventType<> > > chain;

    // the index of each covariate in the chain
    // (used when decoding a key)
    // the order is defined by the typedef above
    typedef enum {
        ReadGroup = 3,
        QualityScore = 2,
        EventType = 1,
    } CovariateID;

    // extract a given covariate value from a key
    static NVBIO_HOST_DEVICE uint32 decode(CovariateID id, uint32 key)
    {
        return chain::decode(key, id);
    }

    static void dump_table(bqsr_context *context, D_CovariateTable& d_table)
    {
        H_CovariateTable table;
        table.copyfrom(d_table);

        printf("ReadGroup\tQualityScore\tEventType\t\tObservations\n");
        for(uint32 i = 0; i < table.size(); i++)
        {
            uint32 rg_id = decode(ReadGroup, table.keys[i]);
            const std::string& rg_name = context->bam_header.read_groups_db.lookup(rg_id);

            printf("%s\t%d\t\t%c\t\t%d\n",
                    rg_name.c_str(),
                    decode(QualityScore, table.keys[i]),
                    cigar_event::ascii(decode(EventType, table.keys[i])),
                    table.observations[i]);
        }
    }
};


template <typename covariate_chain>
struct covariate_gatherer
{
    // covariate table in local memory
    D_LocalCovariateTable table;

    // state
    bqsr_context::view& ctx;
    const alignment_batch_device::const_view& batch;

    NVBIO_HOST_DEVICE covariate_gatherer(bqsr_context::view& ctx,
                                         const alignment_batch_device::const_view& batch)
        : ctx(ctx), batch(batch)
    {
    }

private:
    // gather_covariates_for_read can run in two modes: gather (where table data is updated normally) and rollback (reverts all updates up to a given cigar event index)
    typedef enum {
        Gather,
        Rollback,
    } GatherCovariatesMode;

    uint32 last_cigar_event_index;

    // returns false if we ran out of space
    template<GatherCovariatesMode MODE>
    NVBIO_HOST_DEVICE bool process_read(bqsr_context::view ctx,
                                        const alignment_batch_device::const_view batch,
                                        const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);
        const uint32 cigar_start = ctx.cigar.cigar_offsets[idx.cigar_start];
        const uint32 cigar_end = (MODE == Rollback ? last_cigar_event_index : ctx.cigar.cigar_offsets[idx.cigar_start + idx.cigar_len]);

        uint32 counter_pass = 0;
        uint32 counter_bp_offset = 0;
        uint32 counter_inactive = 0;
        uint32 counter_clip = 0;

        for(uint32 cigar_event_index = cigar_start; cigar_event_index < cigar_end; cigar_event_index++)
        {
            uint16 read_bp_offset = ctx.cigar.cigar_event_read_coordinates[cigar_event_index];
            if (read_bp_offset == uint16(-1))
            {
                counter_bp_offset++;
                continue;
            }

            if (ctx.active_location_list[idx.read_start + read_bp_offset] == 0)
            {
                counter_inactive++;
                continue;
            }

            if (ctx.cigar.cigar_events[cigar_event_index] == cigar_event::S)
            {
                counter_clip++;
                continue;
            }

            counter_pass++;

            uint32 key = covariate_chain::encode(ctx, batch, read_index, read_bp_offset, cigar_event_index);
            uint32 insert_idx;

            //        printf("locating key %x...\n", key);
            insert_idx = table.find_insertion_point(key);

            //        if (table->exists(insert_idx) && table->table[insert_idx].key == key)
            //            printf("... found at index %d (%x)\n", insert_idx, table->table[insert_idx].key);
            //        else
            //            printf("... not found\n");

            //        {
            //            struct covariate_table_entry *loc2 = NULL;
            //            for(uint32 i = 0; i < table->num_entries; i++)
            //            {
            //                if (table->table[i].key == key)
            //                {
            //                    loc2 = &table->table[i];
            //                    break;
            //                }
            //            }
            //
            //            if (loc2 && &table->table[insert_idx] != loc2)
            //            {
            //                printf("bug!\n");
            //            }
            //        }

            switch(MODE)
            {
            case Gather:
                if (!table.exists(insert_idx) || table.keys[insert_idx] != key)
                {
                    //            printf("adding key %x (RG = %d)\n", key, covariate_chain::decode(key, 1));
                    bool r = table.insert(key, insert_idx);
                    if (!r)
                    {
                        // mark for rollback
//                        printf("overflow! rollback at read index %u event index %u\n", read_index, cigar_event_index);
                        last_cigar_event_index = cigar_event_index;
                        return false;
                    }
                } else {
                    //            printf("reusing key %x (RG = %d)\n", key, covariate_chain::decode(key, 1));
                    table.observations[insert_idx]++;
                }

                break;

            case Rollback:
                // note: rollback can leave entries with 0 observations; there's probably no need to cull them, however, as they will show up later
                assert(table.exists(insert_idx));
                assert(table.keys[insert_idx] == key);
                table.observations[insert_idx]--;

                break;
            }
        }

        return true;
    }

public:
    NVBIO_HOST_DEVICE bool process(bqsr_context::view ctx,
                                   const alignment_batch_device::const_view batch,
                                   D_VectorU32::plain_view_type& active_read_list,
                                   const uint32 read_id)
    {
        const uint32 read_index = active_read_list[read_id];
        bool ret;

        ret = process_read<Gather>(ctx, batch, read_index);
        if (ret == false)
        {
            // rollback this read in the table
            process_read<Rollback>(ctx, batch, read_index);
            return false;
        } else {
            return true;
        }
    }
};

template<typename covariate_chain>
__global__ void covariates_kernel(D_VectorU32::plain_view_type active_read_list,
                                  bqsr_context::view ctx,
                                  const alignment_batch_device::const_view batch)
{
    covariates_context::view& cv = ctx.covariates;

    covariate_gatherer<covariate_chain> gatherer(ctx, batch);
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // process reads until we overflow the local memory buffer
    while(tid < active_read_list.size())
    {
        uint32 read_index = ctx.active_read_list[tid];
        bool ret;

        ret = gatherer.process(ctx, batch, active_read_list, tid);
        if (ret == false)
        {
            break;
        }

        tid += blockDim.x * gridDim.x;
    }

    if (gatherer.table.size())
    {
        // allocate a chunk of memory to dump our local table into
        uint32 alloc;
        alloc = ctx.covariates.mempool.allocate(gatherer.table.size());
        if (alloc == uint32(-1))
        {
            // out of memory
            // do not flush and do not retire processed reads (they will be processed in the next pass)
            return;
        }

        memcpy(cv.mempool.keys.begin() + alloc, gatherer.table.keys, sizeof(gatherer.table.keys[0]) * gatherer.table.size());
        memcpy(cv.mempool.observations.begin() + alloc, gatherer.table.observations, sizeof(gatherer.table.observations[0]) * gatherer.table.size());
    }

    // retire all reads we processed
    uint32 tid_last = tid;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < active_read_list.size() && tid < tid_last)
    {
        active_read_list[tid] = uint32(-1);
        tid += blockDim.x * gridDim.x;
    }
}

#if 0
template<typename covariate_chain>
void covariates_cpu(D_CovariateTable::plain_view_type output,
                    bqsr_context::view ctx,
                    const alignment_batch_device::const_view batch)
{
    covariate_table table;

    for(uint32 i = 0; i < ctx.active_read_list.size(); i++)
    {
        uint32 read_index = ctx.active_read_list[i];
        gather_covariates_for_read<typename covariate_chain::chain>(ctx, batch, read_index, table);
    }

    for(int i = 0; i < LOCAL_TABLE_SIZE; i++)
    {
        output[i] = table[i];
    }

    ctx.covariates.table_size[0] = table.size();
}
#endif

struct pingpong_read_lists
{
    D_VectorU32& a;
    D_VectorU32& b;
    uint8 i;

    pingpong_read_lists(D_VectorU32& a, D_VectorU32& b)
        : a(a), b(b), i(0)
    { }

    D_VectorU32& source(void)
    {
        return (i ? a : b);
    }

    D_VectorU32& destination(void)
    {
        return (i ? b : a);
    }

    void swap(void)
    {
        i ^= 1;
    }
};

struct read_is_valid
{
    NVBIO_HOST_DEVICE bool operator() (const uint32 read_index)
    {
        return (read_index != uint32(-1));
    }
};

void gather_covariates(bqsr_context *context, const alignment_batch& batch)
{
    covariates_context& cv = context->covariates;

    // set up our ping-pong read lists
    pingpong_read_lists read_lists(context->temp_u32, context->temp_u32_2);

    // copy the current read list into the source list
    read_lists.source() = context->active_read_list;
    // resize the destination list to make sure we have enough space
    read_lists.destination().resize(context->active_read_list.size());

    // prepare our tables
    D_CovariateTable& table = cv.recal_table_1;

//    table.resize(0);
    cv.mempool.resize(LOCAL_TABLE_SIZE * 256);

    // set up our list of indices and temporary arrays for sorting
    D_VectorU32& indices = context->temp_u32_3;
    indices.resize(LOCAL_TABLE_SIZE * 256);

    D_VectorU32& temp_sorted = context->temp_u32_4;
    temp_sorted.resize(indices.size());

    uint32 active_reads = read_lists.source().size();

    // figure out our launch parameters and mempool size
    const uint32 threads_per_block = 128;
    // the number of reads per thread will affect how often we move local memory into the mempool
    // running with reads_per_thread = 1 means the mempool will fill up very quickly
    // on the other hand, a large reads_per_thread value will cause divergence due to occasional flushes of the local storage into the mempool
    // the ideal value would cause each thread to run for as long as possible without flushing until the very end
    const uint32 reads_per_thread = 32;
    const uint32 num_blocks = nvbio::util::divide_ri(read_lists.source().size(), threads_per_block * reads_per_thread);

    do {
        cv.mempool.clear();

        covariates_kernel<covariates_recaltable1::chain> <<<num_blocks, threads_per_block>>>(read_lists.source(), *context, batch.device);
        cudaDeviceSynchronize();

        // read back the number of items in the mempool
        const uint32 mempool_size = cv.mempool.items_allocated[0];

        // concat the mempool to the end of the current table
        table.concatenate(cv.mempool, mempool_size);
        // sort the concatenated table
        table.sort(indices, temp_sorted);
        // pack the table
        // (indices is reused as a temporary vector here)
        table.pack(indices, temp_sorted);

        // compact the active read list
        active_reads = nvbio::copy_if(read_lists.source().size(),
                                      read_lists.source().begin(),
                                      read_lists.destination().begin(),
                                      read_is_valid(),
                                      context->temp_storage);
        read_lists.destination().resize(active_reads);
        read_lists.swap();
    } while(active_reads);
}

void output_covariates(bqsr_context *context)
{
    covariates_recaltable1::dump_table(context, context->covariates.recal_table_1);
}
