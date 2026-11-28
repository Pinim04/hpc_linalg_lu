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

#ifndef BX
#define BX 64
#endif

#ifndef BY
#define BY 4
#endif

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
    do {                                                                                                 \
        cudaError_t err = (call);                                                                        \
        if (err != cudaSuccess) {                                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

// launch bounds specifies the maximum number of threads per block and the minimum number of blocks per multiprocessor, prevents occupancy dropping in any case (can be omitted in this case because #registers used is <=13)
__global__ void __launch_bounds__(256, 8) lu_div_row(DATA_TYPE *A, int k, int n){
    
    int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);

    if(j < n){

        A[k * n + j] = A[k * n + j] / A[k * n + k];
    
    }

}

__global__ void __launch_bounds__(256, 8) lu_trailing(DATA_TYPE *A, int k, int n){
    
    int i = blockIdx.y * blockDim.y + threadIdx.y + (k + 1); // row index
    int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1); // column index

    if(i < n && j < n){

        A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
    
    }

}

/* Main computational kernel. The whole function will be timed, including the call and return. */
static void kernel_lu(int n, DATA_TYPE *h_A, DATA_TYPE *d_A, size_t nbytes){

    CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaFuncSetCacheConfig(lu_div_row, cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(lu_trailing, cudaFuncCachePreferL1));

    int block1D = 256;
    dim3 block2D(BX, BY); // put all 32 threads of a warp in the x dimension to promote coalescing

    for(int k = 0; k < n - 1; k++){

        int rem = n - (k + 1);

        int grid1D = (rem + block1D - 1) / block1D;
        lu_div_row<<<grid1D, block1D>>>(d_A, k, n);

        dim3 grid2D((rem + block2D.x - 1) / block2D.x, (rem + block2D.y - 1) / block2D.y);
        lu_trailing<<<grid2D, block2D>>>(d_A, k, n);

    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost));

}

int main(int argc, char **argv)
{
    /* Retrieve problem size. */
    int n = N;
    size_t nbytes = sizeof(DATA_TYPE) * (size_t)n * (size_t)n;

    /* Variable declaration/allocation. */
    DATA_TYPE *h_A = NULL;
    DATA_TYPE *d_A = NULL;

    CUDA_CHECK(cudaMallocHost((void**)&h_A, nbytes));
    CUDA_CHECK(cudaMalloc((void**)&d_A, nbytes));

    /* Initialize array(s). */
    init_array(n, (DATA_TYPE(*)[N])h_A);

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_lu(n, h_A, d_A, nbytes);

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed by the function call in argument. */
    polybench_prevent_dce(print_array(n, (DATA_TYPE(*)[N])h_A));

    /* Be clean. */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFreeHost(h_A));

    return 0;
}