Firepony
========

Firepony is a C++/CUDA reimplementation of the GATK base quality score
recalibration algorithm.

It is meant to act as a fast drop-in replacement for GATK BQSR in sequence analysis pipelines. The output of Firepony can be consumed by GATK for subsequent processing steps.

Firepony can run on both x86 CPUs and NVIDIA GPUs. Because it was designed from the ground-up in a data-parallel fashion, Firepony scales extremely well across both architectures.

