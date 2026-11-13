/*
 * peak_flops.c - FP64 compute roof for the roofline model (ARMv8 / NEON).
 *
 * Each thread runs NCHAINS independent vfmaq_f64 accumulator chains so the
 * FMA pipelines are never stalled on latency. One vfmaq_f64 = 2 lanes x
 * (mul + add) = 4 FLOP.
 *
 * Build: gcc -O2 -fopenmp peak_flops.c -o peak_flops
 * Run:   OMP_NUM_THREADS=<t> ./peak_flops
 */
#include <arm_neon.h>
#include <omp.h>
#include <stdio.h>

#define ITERS   100000000L
#define NCHAINS 16

int main(void)
{
  double sink = 0.0;
  int nthreads = 1;

  double t0 = omp_get_wtime();

  #pragma omp parallel reduction(+:sink)
  {
    #pragma omp single
    nthreads = omp_get_num_threads();

    float64x2_t acc[NCHAINS];
    const float64x2_t b = vdupq_n_f64(1.0000001);
    const float64x2_t c = vdupq_n_f64(0.9999999);
    for (int r = 0; r < NCHAINS; r++)
      acc[r] = vdupq_n_f64((double)r);

    for (long i = 0; i < ITERS; i++) {
      acc[0]  = vfmaq_f64(acc[0],  b, c);
      acc[1]  = vfmaq_f64(acc[1],  b, c);
      acc[2]  = vfmaq_f64(acc[2],  b, c);
      acc[3]  = vfmaq_f64(acc[3],  b, c);
      acc[4]  = vfmaq_f64(acc[4],  b, c);
      acc[5]  = vfmaq_f64(acc[5],  b, c);
      acc[6]  = vfmaq_f64(acc[6],  b, c);
      acc[7]  = vfmaq_f64(acc[7],  b, c);
      acc[8]  = vfmaq_f64(acc[8],  b, c);
      acc[9]  = vfmaq_f64(acc[9],  b, c);
      acc[10] = vfmaq_f64(acc[10], b, c);
      acc[11] = vfmaq_f64(acc[11], b, c);
      acc[12] = vfmaq_f64(acc[12], b, c);
      acc[13] = vfmaq_f64(acc[13], b, c);
      acc[14] = vfmaq_f64(acc[14], b, c);
      acc[15] = vfmaq_f64(acc[15], b, c);
    }

    float64x2_t s = acc[0];
    for (int r = 1; r < NCHAINS; r++)
      s = vaddq_f64(s, acc[r]);
    sink += vgetq_lane_f64(s, 0) + vgetq_lane_f64(s, 1);
  }

  double t = omp_get_wtime() - t0;
  double flop = (double)nthreads * ITERS * NCHAINS * 4.0;

  printf("peak_flops threads=%d time=%.3f s  %.2f GFLOP/s  (sink=%g)\n",
         nthreads, t, flop / t / 1e9, sink);
  return 0;
}
