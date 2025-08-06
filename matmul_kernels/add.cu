#include <stdio.h>
#include <cuda_runtime.h>
# define M 100000
# define N 100000
# define K 100000

__global__ void add(float *A, float *B, float *C){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    C[index] = A[index] + B[index];
}


int main(){
    int size = 9;
    float h_A[size] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
    float h_B[size] ={1., 1., 1., 1., 1., 1., 1., 1., 1.};
    float *A = &h_A[0];
    float *B = &h_B[0];
    float *h_C = (float*)malloc(sizeof(float)* size);
    float *d_A, *d_B, *d_C;
    
    cudaMalloc((void **)&d_C, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    add<<<1 , size>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size ; i++ ){
        printf("%f", h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}