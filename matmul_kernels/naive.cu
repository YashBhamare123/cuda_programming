#include <stdio.h>
#include <cuda_runtime.h>
# define M 100000
# define N 100000
# define K 100000

__global__ void multiply(float *A, float *B, float *C, int m, int n, int k){
    int index = blockId.x * blockDim + threadId.x;
    int offset_A = index / k;
    int offset_B = index;
    if (index < m * n){
        C[index] = 0;
        for (int i = 0; i < k, i++){
            C[index] += A[offset_A + i]*B[offset_B + i * k];
        }
    }
}

int main(){
    int size = 9;
    float h_A[size] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
    float h_B[size] ={1., 1., 1., 1., 1., 1., 1., 1., 1.};
    float *h_C = (float*)malloc(sizeof(float)* size);
    float *d_A, *d_B, *d_C;
    
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, h_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, cudaMemcpyHostToDevice);

    multiply<<<1 , size>>(d_A, d_B, d_C, 3, 3, 3);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size, i++ ){
        cout<<h[i]<<endl;
    }


    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


