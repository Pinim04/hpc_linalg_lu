<details open>
<summary><h1>Assignment #1</h1></summary>

## LU Decomposition — OpenMP Optimization

Profiling and OpenMP optimization of the PolyBench `lu` kernel (in-place LU
decomposition of an `N x N` matrix, no pivoting).

### Repository layout

```
Assignment#1/
└── OpenMP/
    ├── lu/
    │   ├── lu_baseline.c               # Reference sequential PolyBench kernel
    │   ├── lu_optimized.c              # Optimized OpenMP version
    │   ├── lu.c                        # Build target (script.sh copies one of the above here)
    │   ├── lu.h                        # Dataset sizes and DATA_TYPE
    │   ├── Makefile                    # Uses ../utilities/common.mk
    │   ├── script.sh                   # Full profiling pipeline (perf, gprof, cachegrind)
    │   ├── results/                    # Output of script.sh
    │   │   ├── baseline/{perf,gprof,valgrind}/
    │   │   └── optimized/{perf,valgrind}/
    │   ├── env/                        # Machine/environment snapshot taken by script.sh
    │   └── deprecated_versions_results/  # Measurements from earlier versions
    └── utilities/                      # PolyBench runtime (polybench.c/h) and build rules
```

### Optimizations

`lu_optimized.c` takes a different path depending on the problem size:

- **`N <= 128` (MINI, SMALL):** stays sequential. `A[k][k]` and `A[i][k]` are
  hoisted into scalars and the inner `j` loops use `#pragma omp simd`. At these
  sizes, thread overhead would cost more than it saves.
- **`N > 128` (STANDARD, LARGE, EXTRALARGE):** one `omp parallel` region wraps
  the entire `k` loop, so the threads are created only once. The row
  normalization uses `omp for simd schedule(static)`. The trailing-submatrix
  update uses `omp for schedule(static)` over `i`, with an inner `simd` loop.

  **Why `static` and not `guided`:** the work shrinks only across `k`. For a
  given `k`, every row `i > k` has the same length (`n-k-1`), so each `i` loop
  is already evenly balanced. `guided` balances nothing here and adds
  chunk-handout overhead at every `k` step. `static` also tends to give each
  thread the same rows in consecutive `k` steps, which helps cache reuse.
  `guided` only helps when cores run at different speeds, like the P/E cores
  on hybrid Intel CPUs, not on the Jetson Nano's four identical A57 cores. The
  guided-schedule results are kept in `results/*_guided/` for comparison.

### Dataset sizes

| Dataset    | N              |
|------------|----------------|
| MINI       | 32             |
| SMALL      | 128            |
| STANDARD   | 1024 (default) |
| LARGE      | 2000           |
| EXTRALARGE | 4000           |

### Building and running manually

```sh
cd OpenMP/lu
cp lu_optimized.c lu.c
make OPT="-O2 -g -fopenmp" EXT_CFLAGS="-DLARGE_DATASET -DPOLYBENCH_TIME" clean all
OMP_NUM_THREADS=4 ./lu_acc
```

Useful PolyBench flags (pass them via `EXT_CFLAGS`):

- `-D<SIZE>_DATASET` selects the problem size
- `-DPOLYBENCH_TIME` prints the kernel execution time
- `-DPOLYBENCH_DUMP_ARRAYS` dumps the result matrix to stderr (for correctness checks)

### Profiling pipeline

```sh
cd OpenMP/lu
./script.sh                 # all datasets
./script.sh SMALL LARGE     # only the selected ones
```

The script runs the baseline and then the optimized version. For each one it:

1. Saves the environment (kernel, gcc version, `OMP_NUM_THREADS`, CPU
   frequency, thermals, cache hierarchy) to `env/environment_conditions.txt`.
2. Does a warm-up run, then runs `perf stat -r 10` (cycles, instructions,
   branches, cache and L1-dcache events) on every dataset, writing to
   `results/<version>/perf/<DATASET>.txt`.
3. On **SMALL** checks correctness by diffing the baseline and optimized array dumps and runs Valgrind cachegrind + `cg_annotate`.
4. On **STANDARD** checks correctness by diffing the baseline and optimized array dumps and, for the baseline version only, produces a gprof flat profile.

Requirements: `gcc` with OpenMP support, `perf`, `valgrind`, `gprof`. If
`perf` isn't found, the script tries `module load perf/1.0`, for HPC clusters.

### Credits

The `lu` kernel and the `utilities/` runtime come from
[PolyBench/GPU-OpenMP](https://github.com/cavazos-lab/PolyBench-ACC) 1.0
(University of Delaware, Ohio State University, Louis-Noël Pouchet).

</details>

### Authors

- Eleonora Muzzi
- Matteo Pini