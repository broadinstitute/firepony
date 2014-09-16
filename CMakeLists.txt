nvbio_module(bqsr)

addsources(
bqsr.cu
bqsr_types.h
bqsr_context.h
bqsr_context.cu
bam_loader.cu
bam_loader.h
filters.h
filters.cu
reference.cu
reference.h
util.cu
util.h
variants.cu
variants.h
)

cuda_add_executable(bqsr ${bqsr_srcs})
target_link_libraries(bqsr nvbio zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})

