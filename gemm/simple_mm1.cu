#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "../include/exercise.cuh"

#define BLOCK_SIZE 16

void cpu_matrix_mult(int *h_a, int *h_b, int *h_c_cpu, int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int tmp = 0;
            for (int h = 0; h < k; h++) {
                tmp += h_a[i*k+h]*h_b[h*n+j];
            }
            h_c_cpu[i*n+j] = tmp;
        }
    }
}

__global__ void gpu_matrix_mult(int *A,int *B, int *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    //printf("row: %d, col: %d\n", row, col);
    if(row < M && col < N)
    {
        for(int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char const *argv[])
{
    int m=1024;
    int n=1024;
    int k=1024;

    int *h_a, *h_b, *h_c;
    int *h_c_cpu;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_b, sizeof(int)*k*n));
    CHECK(cudaMallocHost((void **) &h_c, sizeof(int)*m*n));
    CHECK(cudaMallocHost((void **) &h_c_cpu, sizeof(int)*m*n));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            h_a[i * k + j] = rand() % 10;
        }
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            h_b[i * n + j] = rand() % 10;
        }
    }

    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **) &d_a, sizeof(int)*m*k));
    CHECK(cudaMalloc((void **) &d_b, sizeof(int)*k*n));
    CHECK(cudaMalloc((void **) &d_c, sizeof(int)*m*n));

    // copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*k*n, cudaMemcpyHostToDevice));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    double iStart, iElaps;

    iStart = cpuSecond();
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    CHECK(cudaMemcpy(h_c, d_c, sizeof(int)*m*n, cudaMemcpyDeviceToHost));
    printf("gpu simple matrix multiply elapsed %lf ms <<<grid <%d, %d>, block <%d, %d> >>>\n", iElaps, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    cpu_matrix_mult(h_a, h_b, h_c_cpu, m, n, k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(h_c[i*n+j]-h_c_cpu[i*n+j]) > (1.0e-10)) {
                printf("i: %d, j:%d, hc: %d, h_c_cpu: %d\n", i, j, h_c[i*n+j], h_c_cpu[i*n+j]);
            }
        }
    }

    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    return 0;
}