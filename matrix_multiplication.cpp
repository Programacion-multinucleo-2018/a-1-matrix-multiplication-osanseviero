#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
 
void multiply_matrix(long* input_matrix_a, long* input_matrix_b, long* output_matrix, int n) {
    for(int i = 0; i<n; i++) {
        for(int j=0; j<n; j++) {
            for(int k=0; k<n; k++) {
                output_matrix[i*n+j] += input_matrix_a[i*n+k] * input_matrix_b[j+k*n];
            }
        }
    }
}
 
int main(int argc, char* argv[]) {
    // Size of matrix
    int n = 1000;
    int bytes = n * n * sizeof(long*);

    // Input matrix pointers
    long* a = (long *)malloc(bytes);
    long* b = (long *)malloc(bytes);

    // Output matrix pointer
    long* c = (long *)malloc(bytes);

    // Initialize matrix
    for(int i = 0; i < n*n; i++ ) {
        a[i] = i+1;
        b[i] = i+1;
    }

    struct timespec start, finish;
    double elapsed;

    auto start_cpu =  std::chrono::high_resolution_clock::now();
    multiply_matrix(a, b, c, n);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    std::cout << "time: " << duration_ms.count() << std::endl;

    free(a);
    free(b);
    free(c);
    
    return 0;
}
