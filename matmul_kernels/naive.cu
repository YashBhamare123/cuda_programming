#include <stdio.h>
#include <cuda_runtime.h>
# define M 100000
# define N 100000
# define K 100000

__global__ void multiply(float *A, float *B, float *C, int m, int n, int k){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset_A = index / k;
    int offset_B = index % k;
    if (index < m * n){
        C[index] = 0;
        for (int i = 0; i < k; i++){
            C[index] += A[offset_A * k + i]*B[offset_B + i * k];
        }
    }
}

int main(){
    int num = 9;
    size_t size = num * sizeof(float);
    float h_A[num] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
    float h_B[num] ={1., 1., 1., 1., 1., 1., 1., 1., 1.};
    float *h_C = (float*)malloc(size);
    float *d_A, *d_B, *d_C;
    

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    multiply<<<1 ,num>>>(d_A, d_B, d_C, 3, 3, 3);

    cudaDeviceSynchronize(); // redundant cause memcpy does implicit device sync
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < num; i++ ){
        printf("%f\n", h_C[i]);
    }


    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


