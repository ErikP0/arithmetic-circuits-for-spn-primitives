This repository contains code and benchmark results from the paper "Arithmetic Circuit Implementations of S-boxes for SKINNY and PHOTON in MPC" by Aysajan Abidin, Erik Pohle and Bart Preneel published at ESORICS 2023 [[eprint]](https://eprint.iacr.org/2023/1426).

If content of this repository has been useful to you for academic work, please consider citing
```
@inproceedings{DBLP:conf/esorics/AbidinPP23,
  author       = {Aysajan Abidin and Erik Pohle and Bart Preneel},
  editor       = {Gene Tsudik and Mauro Conti and Kaitai Liang and Georgios Smaragdakis},
  title        = {Arithmetic Circuit Implementations of S-boxes for {SKINNY} and {PHOTON} in {MPC}},
  booktitle    = {Computer Security - {ESORICS} 2023 - 28th European Symposium on Research in Computer Security, Proceedings, Part {I}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14344},
  pages        = {86--105},
  publisher    = {Springer},
  year         = {2023},
  url          = {https://doi.org/10.1007/978-3-031-50594-2\_5},
  doi          = {10.1007/978-3-031-50594-2\_5},
}
```

## Content
- The two jupyter notebooks, `Skinny.ipynb` and `Photon.ipynb` contain example [SageMath](https://www.sagemath.org/) code how to find parameters for SKINNY and PHOTON. The code to find interpolation, polynomial decomposition and embedding parameters for binary fields with any modulus can be found in `crv.py`, `embedding.py` and `spnutils.py`.
- The folder `MP-SPDZ code` contains source code for the implementations for the [MP-SPDZ framework](https://github.com/data61/MP-SPDZ). Instructions on how to use the source code are detailed below.
- The folder `benchmark results` contains the raw and aggregated data (time and communication data) of the MPC benchmark that is reported in the paper.

## Source code for MPC benchmark
- Copy the contents of `MP-SPDZ code` into `Programs/Source/` of the MP-SPDZ framework.
- Compile the benchmark with `./compile.py skinny_benchmark <circuit> <SIMD>` (from the MP-SPDZ root directory) See `skinny_benchmark.mpc` for all available circuits and options. For example use `enc_skinny_64_128_mul_sq1` to compile the SQ1 implementation or `enc_skinny_64_128_crv` for the CRV implementation of SKINNY-64-128.
- Make sure that `USE_GF2N_LONG = 0` is set in `CONFIG.mine` in MP-SPDZ, otherwise the embeddings yield wrong results
- The benchmark in the paper was run with the MASCOT virtual machine, i.e., `mascot-party.x`
