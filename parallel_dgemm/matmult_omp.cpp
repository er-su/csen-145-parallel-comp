// er-su
// CSEN 145 Fa 2024
// The purpose of this program is to time and compare the speed of dense matrix multiplication
// approaches. The first approach is parallelized block multiplication while the second approach
// is parallelized tile multiplication.

#include <iostream>
#include <chrono>  
#include <cstdlib> 
#include <ctime>   
#include <omp.h>
#include <algorithm>
using namespace std;
using namespace std::chrono;

const int TILESIZE = 64;

// Allocate a 1D array for a matrix
int* allocateMatrix(int rows, int cols) {
    int* matrix = new int[rows * cols];
    return matrix;
}

// Fill a given matrix with random values from 1 - 10
void populateMatrixWithRand(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
            matrix[i] = rand() % 10 + 1; 
    }
}

// The parallelized block algorithm. Takes two matrices and populates the result in the result matrix parameter
void matmult_block(const int* mat1, const int* mat2, int* result, const int rows1, const int cols1, const int cols2) {

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < rows1; i++) {
        for(int j = 0; j < cols2; j++) {
            int v = 0;
            for(int k = 0; k < cols1; k++) {
                v += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
            result[i * cols2 + j] = v;
        }
    }
}

// The parallelized tile version of the program. This program first initalizes all values inside the array to 0,
// then proceeds to generate tasks of completing a tile in the result matrix.
void matmult_tile(const int* mat1, const int* mat2, int* result, const int rows1, const int cols1, const int cols2) {
    #pragma omp parallel for
    for(int z = 0; z < rows1 * cols2; z++) {
        result[z] = 0;
    }  

    #pragma omp parallel shared(mat1, mat2) 
    {    
        #pragma omp single
        {
            // Iterate through the tiles
            for(int ii = 0; ii < rows1; ii += TILESIZE) {
                for(int jj = 0; jj < cols2; jj += TILESIZE) {
                    #pragma omp task firstprivate(ii, jj)
                    {
                        for(int kk = 0; kk < cols1; kk += TILESIZE) {

                            
                            // Evaulate a single tile
                            for(int i = ii; i < min(ii + TILESIZE, rows1); ++i) {
                                for(int j = jj; j < min(jj + TILESIZE, cols2); ++j) {
                                    int sum = 0;
                                    for(int k = kk; k < min(kk + TILESIZE, cols1); ++k) {                            
                                        sum += mat2[k * cols2 + j] * mat1[i * cols1 + k];
                                    }
                                    result[i * cols2 + j] += sum;
                                }
                            } 
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int n_cols1;
    int n_rows1;
    int n_cols2;
    int* mat1;
    int* mat2;
    int* result1;
    int* result2;

    if(argc != 4) {
        cout << "Execute in following format: ./matmult <n_rows1> <n_cols1> <n_cols2>" << endl;
        return 1;
    }

    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        #pragma omp single
        cout << "Running with threads: " << nt << endl; 
    }

    n_rows1 = atoi(argv[1]);
    n_cols1 = atoi(argv[2]);
    n_cols2 = atoi(argv[3]);

    mat1 = allocateMatrix(n_rows1, n_cols1);
    mat2 = allocateMatrix(n_cols1, n_cols2);
    result1 = allocateMatrix(n_rows1, n_cols2);
    result2 = allocateMatrix(n_rows1, n_cols2);

    populateMatrixWithRand(mat1, n_rows1, n_cols1);
    populateMatrixWithRand(mat2, n_cols1, n_cols2);

    // Time the block version
    auto start = omp_get_wtime();
    matmult_block(mat1, mat2, result1, n_rows1, n_cols1, n_cols2);
    auto stop = omp_get_wtime();
    auto duration =  (stop - start) * 1000000;

    cout << "Block: (Dimensions: " << n_rows1 << "x" << n_cols1 << " * " << n_cols1 << "x" << n_cols2 << ") - Time: " << duration << " microseconds\n";

    // Time the tile version
    start = omp_get_wtime();
    matmult_tile(mat1, mat2, result2, n_rows1, n_cols1, n_cols2);
    stop = omp_get_wtime();
    duration =  (stop - start) * 1000000;

    cout << "Tile: (Dimensions: " << n_rows1 << "x" << n_cols1 << " * " << n_cols1 << "x" << n_cols2 << ") - Time:  " << duration << " microseconds\n";

    delete[] result1;
    delete[] result2;
    delete[] mat1;
    delete[] mat2;



}
