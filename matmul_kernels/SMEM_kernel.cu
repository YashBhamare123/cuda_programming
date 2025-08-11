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
__global__ void multiply(float *A, float *B, float *C, int M, int N, int K, T blocksize){


    __shared__ float As[blocksize * blocksize];
    __shared__ float Bs[blocksize * blocksize];

    const uint block_offset_C = blockIdx.y * (N / blocksize) * blocksize * blocksize + blocksize * blockIdx.x;
    const uint thread_offset_x = threadIdx.x % blocksize ;
    const uint thread_offset_A_y = K * threadIdx.x / blocksize;
    const uint thread_offset_B_y = N * threadIdx.x / blocksize;

    const uint block_offset_A = blockIdx.y * (K / blocksize) * blocksize * blocksize;
    const uint block_offset_B = blockIdx.x * blocksize;

    const uint C_index = block_offset_C + thread_offset_x;

    for (int i = 0; i < K / blocksize ; i += blocksize){
        A_index = block_offset_A + i + thread_offset_A_y + thread_offset_x;
        B_index = block_offset_B + i * N + thread_offset_B_y + thread_offset_x;

        As[threadIdx.x] = A[A_index];
        Bs[threadIdx.x] = B[B_index];

        __syncthreads();
        for (int j = 0; j < blocksize; j++){
            C[C_index] = As[j + blocksize * (threadIdx.x / blocksize)]* Bs[j * blocksize + thread_offset_x];
        }
    }
}

int main(){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int m = 4, n = 4 , k = 4;
    size_t size_A = sizeof(float) * m * k, size_B = sizeof(float) * k * n, size_C = sizeof(float) * m * n;
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    for (int i = 0; i< m * k; i ++)
        h_A[i] = 1;
    for (int i = 0; i< n * k; i ++)
        h_B[i] = 1;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    int blocksize = 2;
    dim3 dim = (n / blocksize, m /blocksize);
    multiply<<<dim, blocksize*blocksize>>>(d_A, d_B, d_C, m, n, k, blocksize);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < m * n; i ++)
        printf("%d\n", h_C[i]);
    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}