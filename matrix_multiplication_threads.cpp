#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>
 
void multiply_matrix_omp(long* input_matrix_a, long* input_matrix_b, long* output_matrix, int n) {
    int i = 0;
    #pragma omp parallel for private(i) shared(input_matrix_a, input_matrix_b, output_matrix)

    for(i = 0; i<n; i++) {
        for(int j=0; j<n; j++) {
            for(int k=0; k<n; k++) {
                output_matrix[i*n+j] += input_matrix_a[i*n+k] * input_matrix_b[j+k*n];
            }
        }
    }
}

void multiply_matrix(long* input_matrix_a, long* input_matrix_b, long* output_matrix, int n) {
    for(int i = 0; i<n; i++) {
        for(int j=0; j<n; j++) {
            for(int k=0; k<n; k++) {
                output_matrix[i*n+j] += input_matrix_a[i*n+k] * input_matrix_b[j+k*n];
            }
        }
    }
}

void checkResult(long *matrix_a, long *matrix_b, const int n) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < n*n; i++) {
        if (abs(matrix_a[i] - matrix_b[i]) > epsilon) {
            match = 0;
            break;
        }
    }

    if (match) printf("Matrix match.\n\n");
    else printf("Matrix does not not match.\n\n");
}
 
int main(int argc, char* argv[]) {
    // Size of matrix
    int n = 1000;
    int bytes = n * n * sizeof(long*);

    // Input matrix pointers
    long* a1 = (long *)malloc(bytes);
    long* b1 = (long *)malloc(bytes);
    long* a2 = (long *)malloc(bytes);
    long* b2 = (long *)malloc(bytes);

    // Output matrix pointer
    long* c1 = (long *)malloc(bytes);
    long* c2 = (long *)malloc(bytes);

    // Initialize matrix
    for(int i = 0; i < n*n; i++ ) {
        a1[i] = i+1;
        a2[i] = i+1;
        b1[i] = i+1;
        b2[i] = i+1;
    }

    struct timespec start, finish;
    double elapsed;

    auto start_cpu =  std::chrono::high_resolution_clock::now();
    multiply_matrix(a1, b1, c1, n);
    auto end_cpu =  std::chrono::high_resolution_clock::now();
    // Measure total time
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    std::cout << "time without threads: " << duration_ms.count() << std::endl;


    start_cpu =  std::chrono::high_resolution_clock::now();
    multiply_matrix_omp(a2, b2, c2, n);
    end_cpu =  std::chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    std::cout << "time with threads: " << duration_ms.count() << std::endl;

    // Check results
    checkResult(c1, c2, n);

    
    free(a1);
    free(b1);
    free(c1);

    free(a2);
    free(b2);
    free(c2);
    
    return 0;
}
