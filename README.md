This repository contains code and benchmark results from the paper "Arithmetic Circuit Implementations of S-boxes for SKINNY and PHOTON in MPC" by Aysajan Abidin, Erik Pohle and Bart Preneel published at ESORICS 2023.

## Content
- The two jupyter notebooks, `Skinny.ipynb` and `Photon.ipynb` contain example [SageMath](https://www.sagemath.org/) code how to find parameters for SKINNY and PHOTON. The code to find interpolation, polynomial decomposition and embedding parameters for binary fields with any modulus can be found in `crv.py`, `embedding.py` and `spnutils.py`.
- The folder `MP-SPDZ code` contains source code for the implementations for the [MP-SPDZ framework](https://github.com/data61/MP-SPDZ). Instructions on how to use the source code are detailed below.
- The folder `benchmark results` contains the raw and aggregated data (time and communication data) of the MPC benchmark that is reported in the paper.

## Source code for MPC benchmark
- Copy the contents of `MP-SPDZ code` into `Programs/Source/` of the MP-SPDZ framework.
- Compile the benchmark with `./compile.py skinny_benchmark <circuit> <SIMD>` (from the MP-SPDZ root directory) See `skinny_benchmark.mpc` for all available circuits and options. For example use `enc_skinny_64_128_mul_sq1` to compile the SQ1 implementation or `enc_skinny_64_128_crv` for the CRV implementation of SKINNY-64-128.
- Make sure that `USE_GF2N_LONG = 0` is set in `CONFIG.mine` in MP-SPDZ, otherwise the embeddings yield wrong results
- The benchmark in the paper was run with the MASCOT virtual machine, i.e., `mascot-party.x`
