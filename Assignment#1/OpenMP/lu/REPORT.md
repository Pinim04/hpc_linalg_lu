# LU Decomposition on the Jetson Nano: Performance Report

Evaluation of `lu_optimized.c` (sequential SIMD path plus OpenMP parallel path)
against the PolyBench reference kernel `lu_baseline.c`.

## 1. Summary

| Dataset | N | Baseline | Optimized | Speedup |
|---|---|---|---|---|
| MINI | 32 | 31.8 µs | 24.5 µs | **1.30×** |
| SMALL | 128 | 1.754 ms | 1.336 ms | **1.31×** |
| STANDARD | 1024 | 1.086 s | 0.590 s | **1.84×** |
| LARGE | 2000 | 7.496 s | 4.636 s | **1.62×** |
| EXTRALARGE | 4000 | 60.60 s | 38.36 s | **1.58×** |

- **Small sizes (sequential path):** SIMD plus scalar hoisting removes 45% of
  the instructions and gives about 1.3×.
- **Large sizes (parallel path):** 4 threads give 1.6–1.8×. The limit is DRAM
  bandwidth, not computation. The memory bus saturates at about 9 GB/s, so any
  LU without cache blocking is capped at about 1.6× on this board. The L2
  counters confirm this.

## 2. Setup

| | |
|---|---|
| Board | NVIDIA Jetson Nano, L4T R32.7.6 (JetPack 4.6.6), kernel 4.9.337-tegra |
| CPU | 4× ARM Cortex-A57 at 1.479 GHz, 32 KB L1D per core, 2 MB shared L2 |
| Power / DVFS | `NV Power Mode: MAXN`, governor `schedutil`, all 4 CPUs online |
| Compiler | gcc 7.5.0 (Ubuntu/Linaro), `-O2 -g` (baseline), `-O2 -g -fopenmp` (optimized) |
| Threads | `OMP_NUM_THREADS=4`, `OMP_WAIT_POLICY` default |
| Data type | `double`, matrix sizes 8 KB (MINI) to 128 MB (EXTRALARGE) |

**Method.** `script.sh` builds each version per dataset with `-DPOLYBENCH_TIME`.
It does one warm-up run and then `perf stat -r 10`. The reported kernel time is
the mean of the 10 PolyBench timer values, which cover the kernel only, not
process startup. In addition:
- **SMALL:** cachegrind on both versions and a check that both versions
  produce identical output.
- **STANDARD:** a gprof profile of the baseline and a check that both versions
  produce identical output.

**Conditions.** Before each dataset, the script sampled the CPU usage of other
processes for 1 s. It stayed at 0–1% for the whole evaluation, so the node was
otherwise idle. CPU temperature ranged from 35 to 52 °C, with no thermal
throttling.

## 3. Kernel time

Mean of 10 runs per configuration. The spread is half the min–max range,
relative to the mean.

| Dataset | Baseline | Spread | Optimized | Spread | Speedup |
|---|---|---|---|---|---|
| MINI | 31.8 µs | 31–33 µs | 24.5 µs | 24–25 µs | 1.30× |
| SMALL | 1.754 ms | ±1.5% | 1.336 ms | ±2.1% | 1.31× |
| STANDARD | 1.086 s | ±1.4% | 0.590 s | ±6.6% | 1.84× |
| LARGE | 7.496 s | ±0.5% | 4.636 s | ±0.5% | 1.62× |
| EXTRALARGE | 60.60 s | ±0.2% | 38.36 s | ±1.6% | 1.58× |

The timings are very repeatable. Optimized STANDARD varies the most because each
`k` step does little work compared with the cost of synchronizing threads at its
two barriers.

## 4. Analysis

### 4.1 Where the time goes

The gprof flat profile (STANDARD, baseline) attributes **98.7%** of the run time
to `kernel_lu`; `init_array` takes 0.18%. Optimizing the kernel was the right
target. The profile was built with `-O0 -pg`, so its absolute time (5.65 s) is
not comparable to the `-O2` timings.

### 4.2 Sequential path (MINI, SMALL): SIMD and hoisting

For `N <= 128` the optimized kernel stays single-threaded:
- It hoists `A[k][k]` and `A[i][k]` into scalars.
- It vectorizes the inner `j` loops with `#pragma omp simd`. NEON registers are
  128 bits wide, so each vector instruction processes 2 doubles.

gcc 7 doesn't auto-vectorize at `-O2`, so the baseline stays scalar.

Cachegrind, SMALL (the whole program):

| | Baseline | Optimized | Change |
|---|---|---|---|
| Instructions | 5.89 M | 3.24 M | −45% |
| Data accesses | 2.86 M | 1.16 M | −59% |
| D1 misses | 97.1 k | 98.5 k | ≈ 0 |
| LL misses | 5.4 k | 6.2 k | ≈ 0 |

The work drops by almost half, but the time improves only 1.3×. The cache
misses don't change. The D1 miss *rate* rises from 3.4% to 8.5% only because
there are fewer accesses in total. At SMALL the 128 KB matrix doesn't fit in the
32 KB L1D, so the remaining misses limit the gain.

The perf counters for MINI and SMALL are not used for analysis. The process runs
for 30–60 ms while the kernel takes 0.02–1.8 ms, so those counters mostly
measure program startup and the OpenMP runtime.

### 4.3 Parallel path (STANDARD and larger)

For `N > 128`, one `omp parallel` region wraps the whole `k` loop, so threads
are created only once. Inside it:
- The row normalization is shared out with `omp for simd`.
- The trailing-submatrix update is split over rows `i` with `omp for`, and its
  inner `j` loop is vectorized.

Both loops use `schedule(static)`. For a given `k`, every row `i > k` has the
same length (`n-k-1`), so the iterations of each loop are perfectly balanced. A
dynamic schedule such as `guided` would balance nothing and only add the cost of
handing out chunks at every `k` step. `static` also tends to give each thread
the same rows in consecutive steps.

**Thread scaling** (STANDARD, mean of 4 runs per thread count):

| Threads | Time | vs 1 thread | vs baseline | Estimated DRAM bandwidth ¹ |
|---|---|---|---|---|
| 1 | 0.963 s | 1.00× | 1.13× | 5.4 GB/s |
| 2 | 0.689 s | 1.40× | 1.58× | 7.5 GB/s |
| 3 | 0.627 s | 1.54× | 1.73× | 8.2 GB/s |
| 4 | 0.602 s | 1.60× | 1.80× | 8.6 GB/s |
| 4, `OMP_WAIT_POLICY=active` | 0.606 s | 1.59× | 1.79× | — |

¹ Based on the 5.2 GB of DRAM traffic measured for STANDARD (§4.4).

- **SIMD alone, on one thread, is worth 1.13×.**
- **The gain flattens quickly.** The second thread gives +40%; the third and
  fourth together give +14%. This is the typical curve of a memory-bound loop:
  after about 2 cores the memory bus is saturated.
- **Busy-waiting (`active`) changes nothing.** Barrier wake-up latency is not a
  factor.

### 4.4 Memory-bandwidth ceiling

**The traffic model.**
- At step `k`, the kernel reads and writes every element of the remaining
  `m × m` submatrix, where `m = n-k-1`.
- Summed over all steps, that's `Σ m² ≈ n³/3` elements.
- The kernel has no blocking. Whenever the submatrix is larger than the 2 MB L2
  (`m > 512`), it has been evicted by the next step and streams from DRAM again.
- Total DRAM reads are therefore about `n³/3 × 8 B`, plus as much again written
  back.

**Spatial locality is already optimal.** With `j` innermost, accesses to
`A[i][j]` and `A[k][j]` are contiguous (C stores matrices row by row), so every
64-byte cache line fetched is fully used. What's missing is temporal locality:
data isn't reused across `k` steps.

The raw ARMv8 events `r16` (L2D_CACHE, L2 accesses) and `r17`
(L2D_CACHE_REFILL, L2 misses) confirm the model:

| Dataset | L2 refills, baseline | L2 refills, optimized | DRAM reads (refills × 64 B) | Model `n³/3 × 8 B` | Bandwidth, baseline ² | Bandwidth, optimized ² |
|---|---|---|---|---|---|---|
| STANDARD | 42.2 M | 40.4 M | 2.6–2.7 GB | 2.9 GB | 5.0 GB/s | 8.8 GB/s |
| LARGE | 345 M | 334 M | 21.4–22.1 GB | 21.3 GB | 5.9 GB/s | 9.2 GB/s |
| EXTRALARGE | 2.77 G | 2.71 G | 174–177 GB | 171 GB | 5.8 GB/s | 9.1 GB/s |

² Reads plus an equal volume of write-backs, divided by the kernel time.
Write-backs are not counted directly. They are assumed equal to the reads,
because every element read in a `k` step is also written.

What the table shows:

- **Both versions move the same data from DRAM.** The L2 refills match the
  analytical model within 4% for LARGE and EXTRALARGE. STANDARD comes in
  slightly below the model because its last ~512 steps fit in L2. Parallelism
  and SIMD change how fast the data is requested, not how much.
- **One core uses about 5.9 GB/s, and four cores saturate at about 9.1 GB/s.**
  The maximum speedup is therefore about 1.6×, which is what LARGE (1.62×) and
  EXTRALARGE (1.58×) achieve. STANDARD reaches 1.84× because its single-core
  baseline uses only 5.0 GB/s.
- **The kernel runs at the bandwidth limit.** LU performs `2n³/3`
  floating-point operations for `2 × n³/3 × 8 B` of traffic, which is 1/8 flop
  per byte. At 9.1 GB/s that allows at most about 1.13 GFLOP/s. The optimized
  EXTRALARGE run achieves `(2/3 · 4000³) / 38.36 s` = **1.11 GFLOP/s**.
- **The perf counters show cores waiting on memory.**
  - At EXTRALARGE the optimized version uses 217 G cycles in 38.4 s. That is
    96% of the cycles 4 cores at 1.479 GHz can deliver, so all cores are busy.
  - Yet IPC is 0.42, against 1.91 for the baseline.
  - The cores execute 47% fewer instructions (90.8 G vs 170.8 G, the effect of
    2-wide SIMD) and spend most of their cycles stalled on DRAM.

## 5. Conclusions

1. **The optimizations work as intended.**
   - The sequential path (SIMD plus hoisting) gives 1.3× on small inputs.
   - The parallel path gives 1.84× (STANDARD), 1.62× (LARGE) and 1.58×
     (EXTRALARGE) on 4 cores.
2. **The Nano's DRAM bandwidth limits the parallel speedup, not the code.**
   - The L2 refill counters match the analytical traffic model.
   - Bandwidth saturates at about 9.1 GB/s, against about 5.9 GB/s for one
     core. That bounds any unblocked LU at about 1.6×.
   - The kernel runs at the resulting limit of about 1.1 GFLOP/s.
3. **`schedule(static)` is the right choice.** Within each `k` step the
   trailing-update loop is perfectly balanced.

### Future work

- **Blocked (tiled) LU.** Splitting the update into `b × b` tiles turns it into
  matrix-multiply-like updates that reuse data from L2 across several `k` steps.
  That cuts DRAM traffic roughly by a factor of `b` and is the only way past the
  about 1.6× bandwidth ceiling. Changing the loop order can't do this, because
  spatial locality is already optimal.
- **Thread scan at LARGE and EXTRALARGE**, to confirm that the bandwidth curve
  flattens at the same point.

## Appendix: data files

| Data | Location |
|---|---|
| perf counters (cycles, instructions, L1D, L2 `r16`/`r17`), kernel times | `results/{baseline,optimized}/perf/<DATASET>.txt` |
| Cachegrind (SMALL) | `results/{baseline,optimized}/valgrind/` |
| gprof (STANDARD, baseline) | `results/baseline/gprof/` |
| Environment, temperature, background CPU | `env/environment_conditions.txt` |
