#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Include polybench common header. */

#include <polybench.h>

extern "C" {
  #include "utils.h"
}

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"

/* Array initialization. */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){

  int i, j;

  for (i = 0; i < n; i++)

    for (j = 0; j < n; j++)

      A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
  
}

/* DCE code. Must scan the entire live-out data.
  Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){

  int i, j;

  for (i = 0; i < n; i++)

    for (j = 0; j < n; j++){

      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);

      if ((i * n + j) % 20 == 0)

        fprintf(stderr, "\n");

    }

  fprintf(stderr, "\n");

}

#define CUDA_CHECK(call)                                                                                 \
  do {                                                                                                   \
    cudaError_t err = (call);                                                                            \
    if (err != cudaSuccess) {                                                                            \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));         \
      exit(EXIT_FAILURE);                                                                                \
    }                                                                                                    \
  } while (0)

__global__ void lu_div_row(DATA_TYPE *A, int k, int n){
    
  int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);

  if(j < n){

    A[k * n + j] = A[k * n + j] / A[k * n + k];
  
  }

}

__global__ void lu_trailing(DATA_TYPE *A, int k, int n){
    
  int i = blockIdx.y * blockDim.y + threadIdx.y + (k + 1); // row index
  int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1); // column index

  if(i < n && j < n){

    A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
  
  }

}

/* Main computational kernel. The whole function will be timed,
  including the call and return. */
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){

  DATA_TYPE *flat_A = &A[0][0];

  int block1D = 256;
  dim3 block2D(16, 16);

  for(int k = 0; k < n - 1; k++){

    int rem = n - (k + 1);

    int grid1D = (rem + block1D - 1) / block1D;
    lu_div_row<<<grid1D, block1D>>>(flat_A, k, n);

    int grid2D_side = (rem + block2D.x - 1) / block2D.x;
    dim3 grid2D(grid2D_side, grid2D_side);
    lu_trailing<<<grid2D, block2D>>>(flat_A, k, n);

  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE (*A)[N];

  CUDA_CHECK(cudaMallocManaged((void**)&A, sizeof(DATA_TYPE) * (size_t)N * (size_t)N));

  /* Initialize array(s). */
  init_array(n, A);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_lu(n, A);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed by the function call in argument. */
  polybench_prevent_dce(print_array(n, A));

  /* Be clean. */
  CUDA_CHECK(cudaFree(A));

  return 0;
}