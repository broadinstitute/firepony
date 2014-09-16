nvbio_module(bqsr)

addsources(
bqsr.cu
bam_loader.cpp
bam_loader.h
reference.cu
reference.h
util.cu
util.h
)

cuda_add_executable(bqsr ${bqsr_srcs})
target_link_libraries(bqsr nvbio zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})

