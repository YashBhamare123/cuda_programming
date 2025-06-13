#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILUREcuda-programming/cuda-course); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudnnGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void TanhKernel(float *A, float *C, int size){
    int id = blockIdx * blockDim + threadIdx;
    if (id < size)
        C[i] = tanhf([A[i]]);
}

void TanhCPU(float *A, float *C, int size){
    for (int i = 0; i <size; i++){
        C[i] = tanhf(A[i]);
    }
}

void initialize(float *A, int size){
    srand(62);
    for (int i = 0; i < size; i++){
        A[i] = rand();
    }
}

int main(){
    
    // initializing the tensors and copying data from host to device
    int batch_size = 32;
    int M = 128, N = 128;
    int channels = 64;

    int size = batch_size * M * N * channels;


    float *h_A, *h_C_cpu, *h_C_gpu, *h_C_cudnn;
    float *d_A, *d_C_gpu, d_C_cudnn;

    h_A = (float*)malloc(size * sizeof(float));
    h_C_cpu = (float*)malloc(size * sizeof(float));
    h_C_gpu = (float*)malloc(size * sizeof(float));

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    initialize(h_A, size);
    cudaMemcpy(&d_A, &h_A, size * sizeof(float), cudaMemcpyHostToDevice);

    // benchmarks
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int num_benchmark = 50;
    int num_warmup = 10;
    float *cpu_time[num_benchmark];
    float *gpu_time[num_benchmark];
    float *cudnn_time[num_benchmark];

    // gpu benchmark
    dim3 num_blocks = (batch_size, channels);
    dim3 num_threads = ()




}