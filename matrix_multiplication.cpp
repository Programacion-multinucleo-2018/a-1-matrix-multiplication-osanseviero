// Implements matrix multiplication using CPU.
// Compile with g++ matrix_multiplication.cpp  -std=c++11

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// Multiplies two matrices and store result in an output matrix
void multiply_matrix(long *matA, long *matB, long *matC, int n) {
    for(int i = 0; i<n; i++) {
        for(int j=0; j<n; j++) {
            for(int k=0; k<n; k++) {
                matC[i*n+j] += matA[i*n+k] * matB[j+k*n];
            }
        }
    }
}
 
int main(int argc, char* argv[]) {
    // Size of matrix
    int n = 1000;
    int bytes = n * n * sizeof(long*);

    // Input matrix pointers
    long *a = (long *)malloc(bytes);
    long *b = (long *)malloc(bytes);

    // Output matrix pointer
    long *c = (long *)malloc(bytes);

    // Initialize matrix
    for(int i = 0; i < n*n; i++ ) {
        a[i] = i+1;
        b[i] = i+1;
    }

    auto start_cpu =  std::chrono::high_resolution_clock::now();
    multiply_matrix(a, b, c, n);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("multiply_matrix_gpu elapsed %f ms\n", duration_ms.count());

    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
