nvbio_module(bqsr)

addsources(
bam_loader.cu
bam_loader.h
baq.cu
baq.h
#baq-cpu.cu
#baq-cpu.h
bqsr.cu
bqsr_context.h
bqsr_context.cu
bqsr_types.h
cigar.h
cigar.cu
covariates.h
covariates_bit_packing.h
covariates_table.h
covariates_table.cu
covariates.cu
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
target_link_libraries(bqsr nvbio zlibstatic crcstatic ${gamgee_LIB} ${SYSTEM_LINK_LIBRARIES})
add_dependencies(bqsr gamgee)

