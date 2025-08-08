# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


template <typename T>
__global__ void multiply(float *A, float *B, float *C, int m, int n, int k, T blocksize){


    __shared__ float As[blocksize * blocksize];
    __shared__ float Bs[blocksize * blocksize];

    const uint block_offset_x = blocksize * blocksize * blockIdx.x;
    const uint block_offset_y = blockIdx.y * N * blocksize;

    const uint tid = block_offset_x + block_offset_y + threadIdx.x;

    // Indexing for A
    const uint bidA_y = block_offset_y = blockIdx.y * K * blocksize;
    const uint bidA_x = blockIdx.x * blocksize;
    const uint tidA_x = threadIdx.x % blocksize;
    const uint tidA_y = (threadIdx.x / blocksize) * blocksize;
    const uint tidA = bidA_x + bidA_y + tidA_x + tidA_y;

    // Indexing for B
    const uint bidB_x = blockIdx.x * blocksize;
    const uint bidB_y = blockIdx.y * N * blocksize;
    const uint tidB_x = threadIdx.x % blocksize;
    const uint tidB_y = (threadIdx.x / blocksize) * blocksize;
    const uint tidB = bidB_x + bidB_y + tidB_x + tidb_y;


}