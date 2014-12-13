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

#include "pipeline.h"

#include "alignment_data_device.h"
#include "sequence_data_device.h"
#include "variant_data_device.h"

#include "device/primitives/backends.h"

#include <thread>

#if ENABLE_TBB_BACKEND
#include <tbb/task_scheduler_init.h>
#endif

namespace firepony {

template <target_system system> void firepony_process_batch(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void firepony_postprocess(firepony_context<system>& context);

template <target_system system_dst, target_system system_src>
void firepony_gather_intermediates(firepony_context<system_dst>& context, firepony_context<system_src>& other)
{
    context.covariates.quality.concat(context.compute_device, other.compute_device, other.covariates.quality);
    context.covariates.cycle.concat(context.compute_device, other.compute_device, other.covariates.cycle);
    context.covariates.context.concat(context.compute_device, other.compute_device, other.covariates.context);
}

template <target_system system>
struct firepony_device_pipeline : public firepony_pipeline
{
    uint32 consumer_id;

    alignment_header<system> *header;
    sequence_data<system> *reference;
    variant_database<system> *dbsnp;

    firepony_context<system> *context;
    alignment_batch<system> *batch;

    io_thread *reader;

    std::thread thread;
    uint32 compute_device;

    firepony_device_pipeline(uint32 consumer_id, uint32 compute_device)
        : consumer_id(consumer_id), compute_device(compute_device)
    { }

    virtual std::string get_name(void) override;

    virtual target_system get_system(void) override
    {
        return system;
    }

    virtual int get_compute_device(void) override
    {
        return compute_device;
    }

    virtual pipeline_statistics& statistics(void) override
    {
        return context->stats;
    }

    virtual void setup(io_thread *reader,
                       const runtime_options *options,
                       alignment_header_host *h_header,
                       sequence_data_host *h_reference,
                       variant_database_host *h_dbsnp) override
    {
#if ENABLE_CUDA_BACKEND
        if (system == cuda)
        {
            cudaSetDevice(compute_device);
        }
#endif

        this->reader = reader;

        header = new alignment_header<system>(*h_header);
        reference = new sequence_data<system>(*h_reference);
        dbsnp = new variant_database<system>(*h_dbsnp);

        header->download();
        reference->download();
        dbsnp->download();

        context = new firepony_context<system>(compute_device, *options, *header, *reference, *dbsnp);
        batch = new alignment_batch<system>();
    }

    virtual void start(void) override
    {
        thread = std::thread(&firepony_device_pipeline<system>::run, this);
    }

    virtual void join(void) override
    {
        thread.join();
    }

    virtual void gather_intermediates(firepony_pipeline *other) override
    {
        switch(other->get_system())
        {
#if ENABLE_CUDA_BACKEND
        case firepony::cuda:
        {
            cudaSetDevice(compute_device);

            firepony_device_pipeline<firepony::cuda> *other_cuda = (firepony_device_pipeline<firepony::cuda> *) other;
            firepony_gather_intermediates(*context, *other_cuda->context);
            break;
        }
#endif

#if ENABLE_TBB_BACKEND
        case firepony::intel_tbb:
        {
            firepony_device_pipeline<firepony::intel_tbb> *other_tbb = (firepony_device_pipeline<firepony::intel_tbb> *) other;
            firepony_gather_intermediates(*context, *other_tbb->context);
            break;
        }
#endif

        default:
            assert(!"can't happen");
        }
    }

    virtual void postprocess(void) override
    {
#if ENABLE_CUDA_BACKEND
        if (system == cuda)
        {
            cudaSetDevice(compute_device);
        }
#endif

        firepony_postprocess(*context);
    }

private:
    void run(void)
    {
#if ENABLE_CUDA_BACKEND
        if (system == cuda)
        {
            cudaSetDevice(compute_device);
        }
#endif

        timer<host> io_timer;
        alignment_batch_host *h_batch;

        for(;;)
        {
            // try to get a batch to work on
            io_timer.start();
            h_batch = reader->get_batch();
            io_timer.stop();
            statistics().io.add(io_timer);

            if (h_batch == nullptr)
            {
                // no more data, we're done
                break;
            }

            // make sure our reference is up to date
            if (h_batch->reference_generation > reference->device.generation)
            {
                // reload the reference
                reader->reference->consumer_lock(consumer_id);
                reference->download();
                reader->reference->consumer_unlock(consumer_id);
            }

            // download to the device
            batch->download(h_batch);

            // process the batch
            firepony_process_batch(*context, *batch);

            // return it to the reader for reuse
            reader->retire_batch(h_batch);

            if (!context->options.debug)
            {
                fprintf(stderr, ".");
                fflush(stderr);
            }
        }
    }
};

#if ENABLE_CUDA_BACKEND
template<>
std::string firepony_device_pipeline<firepony::cuda>::get_name(void)
{
    cudaDeviceProp prop;

    cudaSetDevice(compute_device);
    cudaGetDeviceProperties(&prop, compute_device);

    char buf[1024];
    snprintf(buf, sizeof(buf),
             "%s (%lu MB, CUDA device %d)",
             prop.name, prop.totalGlobalMem / (1024 * 1024), compute_device);

    return std::string(buf);
}
#endif

#if ENABLE_TBB_BACKEND
tbb::task_scheduler_init tbb_scheduler_init(tbb::task_scheduler_init::deferred);
static int num_tbb_threads = -1;

template<>
std::string firepony_device_pipeline<firepony::intel_tbb>::get_name(void)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "CPU (Intel Threading Building Blocks, %d threads)", num_tbb_threads);
    return std::string(buf);
}

#endif

firepony_pipeline *firepony_pipeline::create(target_system system, uint32 device)
{
    static uint32 current_consumer_id = 0;
    uint32 consumer_id = current_consumer_id;
    current_consumer_id++;

    switch(system)
    {
#if ENABLE_CUDA_BACKEND
    case firepony::cuda:
        return new firepony_device_pipeline<firepony::cuda>(consumer_id, device);
#endif

#if ENABLE_TBB_BACKEND
    case firepony::intel_tbb:
        // reserve device threads for other devices and I/O
        num_tbb_threads = tbb::task_scheduler_init::default_num_threads() - device - 1;
        tbb_scheduler_init.initialize(num_tbb_threads);
        return new firepony_device_pipeline<firepony::intel_tbb>(consumer_id, num_tbb_threads);
#endif

    default:
        current_consumer_id--;  // we didn't actually create anything
        return nullptr;
    }
}

} // namespace firepony
