#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"
#define BLOCK_SIZE 64
#define min(a,b) ((a) < (b) ? (a) : (b))
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

/* Main computational kernel. The whole function will be timed,
  including the call and return. */
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){

  int i, j, k;

  if(n <= 128){

    for (k = 0; k < _PB_N; k++){

      DATA_TYPE A_kk = A[k][k];

      #pragma omp simd
      for (j = k + 1; j < _PB_N; j++)
      
        A[k][j] = A[k][j] / A_kk;

      for (i = k + 1; i < _PB_N; i++){

        DATA_TYPE A_ik = A[i][k];

        #pragma omp simd
        for (j = k + 1; j < _PB_N; j++)

          A[i][j] = A[i][j] - A_ik * A[k][j];
        
      }

    }

  }
  else{
    #pragma omp target data map(tofrom: A[0:_PB_N][0:_PB_N])
    {
      for (int kk = 0; kk < _PB_N; kk += BLOCK_SIZE) {
      
        int end_k = min(kk + BLOCK_SIZE, _PB_N);
        
        for (int k = kk; k < end_k; k++) {
        
          #pragma omp target teams distribute parallel for
          for (int j = k + 1; j < _PB_N; j++) {
          
            A[k][j] /= A[k][k];
            
          }
  
          #pragma omp target teams distribute parallel for collapse(2)
          for (int i = k + 1; i < end_k; i++) {
          
            for (int j = k + 1; j < _PB_N; j++) {
            
              A[i][j] -= A[i][k] * A[k][j];
              
            }
          }
  
          #pragma omp target teams distribute parallel for collapse(2)
          for (int i = end_k; i < _PB_N; i++) {
          
            for (int j = k + 1; j < end_k; j++) {
            
              A[i][j] -= A[i][k] * A[k][j];
              
            }
          }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = end_k; i < _PB_N; i++) {
        
          for (int j = end_k; j < _PB_N; j++) {
            
            for (int k = kk; k < end_k; k++) {
            
              A[i][j] -= A[i][k] * A[k][j];
              
            }
          }
        }
      }
    }
  }
  
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_lu(n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}