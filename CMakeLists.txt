nvbio_module(bqsr)

addsources(
gamgee_loader.cu
gamgee_loader.h
baq.cu
baq.h
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

from_nvbio/alphabet_inl.h
from_nvbio/alphabet.h
from_nvbio/dna.h
from_nvbio/vcf.cu
from_nvbio/vcf.h
)

nvbio_add_module_directory(primitives)

cuda_add_executable(bqsr ${bqsr_srcs})
target_link_libraries(bqsr nvbio zlibstatic crcstatic ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})
add_dependencies(bqsr gamgee zlib_bqsr)

