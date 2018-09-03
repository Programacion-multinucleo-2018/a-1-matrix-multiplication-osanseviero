// Matrix Multiplication in gpu with 2D grid of blocks with 1D block shape
// Compile with: nvcc -o test matrix_multiplication_2D_2D.cu -std=c++11

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
 
// Multiplies matrices using GPU with 2D grid
__global__ void multiply_matrix_gpu(long *matA, long *matB, long *matC, const int n) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;

    if (ix < n && iy < n) {
        for(int k=0; k<n; k++) {
            matC[iy*n+ix] += matA[iy*n+k] * matB[k*n+ix];
        }
    }
}

// Multiplies matrices in host
void multiply_matrix_host(long *matA, long *matB, long *matC, int n) {
    for(int i = 0; i<n; i++) {
        for(int j=0; j<n; j++) {
            for(int k=0; k<n; k++) {
                matC[i*n+j] += matA[i*n+k] * matB[j+k*n];
            }
        }
    }
}

// Compares two matrices
void checkResult(long *hostRef, long *gpuRef, const int n) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < n*n; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("host %ld gpu %ld\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match) printf("Matrix match.\n\n");
    else printf("Matrix does not not match.\n\n");
}
 
int main(int argc, char* argv[]) {
    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // Size of matrix
    int n = 1000;
    int bytes = n * n * sizeof(long*);

    // Host matrix memory
    long *h_a = (long *)malloc(bytes);
    long *h_b = (long *)malloc(bytes);

    // Results
    long *hostRef = (long *)malloc(bytes);
    long *gpuRef = (long *)malloc(bytes);

    // Initialize matrix on host
    for(int i = 0; i < n*n; i++ ) {
        h_a[i] = i+1;
        h_b[i] = i+1;
    }

    // Initialize matrix with 0s
    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    // Multiply matrix on host
    auto start_cpu = std::chrono::high_resolution_clock::now();
    multiply_matrix_host(h_a, h_b, hostRef, n);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time in host
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("multiply_matrix_host elapsed %f ms\n", duration_ms.count());

    // Device matrix global memory
    long *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    // Transfer data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, bytes);  // Initialize matrix with 0s

    // Kernel execution configuration
    dim3 block(128);
    dim3 grid((n + block.x - 1) / block.x, n);
    printf("grid.x %d grid.y %d block.x %d \n", grid.x, grid.y, block.x);

    // Execute kernel
    start_cpu = std::chrono::high_resolution_clock::now();
    multiply_matrix_gpu<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    duration_ms = end_cpu - start_cpu;
    printf("multiply_matrix_gpu elapsed %f ms\n", duration_ms.count());

    // Copy result from device to host
    cudaMemcpy(gpuRef, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check results
    checkResult(hostRef, gpuRef, n);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(hostRef);
    free(gpuRef);
    
    cudaDeviceReset();

    return 0;
}
