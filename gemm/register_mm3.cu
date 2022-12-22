#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>                                                                                          //#include <cublas_v2.h>

#include "../include/exercise.cuh"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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

/*
 * NOTE: every thread has different local var/register
 *
 * NOTICE: FOR WRITE
 * ADD register A, resigter B for: load tileA, tileB to compute accum
 */
template<
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_N,
    const int BLOCK_SIZE_K,
    const int THREAD_SIZE_Y,
    const int THREAD_SIZE_X
    >
__global__ void gpu_matrix_mult_register(
                int* __restrict__ A,
                int* __restrict__ B,
                int* __restrict__ C,
                int M, int N, int K)
{
    // shared memory for cached partial data of A and B
    __shared__ int tile_A[BLOCK_SIZE_M][BLOCK_SIZE_K]; //[128,8]
    __shared__ int tile_B[BLOCK_SIZE_K][BLOCK_SIZE_N]; //[8,128]
    // register for C, replace before sum
    int accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    // register for A, B
    int frag_a[THREAD_SIZE_Y];
    int frag_b[THREAD_SIZE_X];

    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszx * bszy;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // tid in block
    const int tid = ty * bszx + tx; // 0~256

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START=tid/BLOCK_SIZE_K;
    const int B_TILE_ROW_START=tid/BLOCK_SIZE_N;

    const int A_TILE_COL=tid%BLOCK_SIZE_K;
    const int B_TILE_COL=tid%BLOCK_SIZE_N;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE=THREAD_NUM_PER_BLOCK/BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE=THREAD_NUM_PER_BLOCK/BLOCK_SIZE_N;

    A = &A[(BLOCK_SIZE_M * by)* K]; // curr row first block
    B = &B[BLOCK_SIZE_N * bx]; // cur col first block

    // loop tiles
    for(int tile_idx = 0; tile_idx < K; tile_idx+=BLOCK_SIZE_K)
    {
        // load A from global memory to shared memory
        #pragma unroll
        for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
            tile_A[i+A_TILE_ROW_START][A_TILE_COL] = A[OFFSET(
                            A_TILE_ROW_START+i, //row
                            A_TILE_COL+tile_idx, //col
                            K)];
        }
        // load B from global memory to shared memory
        #pragma unroll
        for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE){
            tile_B[i+B_TILE_ROW_START][B_TILE_COL]=B[OFFSET(
                            B_TILE_ROW_START+tile_idx+i, // row
                            B_TILE_COL, // col
                            N)];
        }
        __syncthreads(); // sync threads per block

        // accum = tileA @ tileB, per thread
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE_K; k_idx++) {
            // NOTICE: reg is different in different thread
            // load A from shared memory to register: per col
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
                frag_a[thread_y] = tile_A[THREAD_SIZE_Y*ty + thread_y][k_idx];
            }
            // load B from shared memory to register: per row
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
                frag_b[thread_x] = tile_B[k_idx][THREAD_SIZE_X*tx + thread_x];
            }

            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
                #pragma unroll
                for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
        }
        __syncthreads();

    }
    // write accum to C
    #pragma unroll
    for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
        #pragma unroll
        for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
            const int row = BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty + thread_y;
            const int col = BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx + thread_x;
            if(row < M && col < N) {
                C[OFFSET(row, col, N)] = accum[thread_y][thread_x];
            }
        }
    }
}

/*
 * NOTE: every thread has different local var/register
 *
 * NOTICE: FOR READ
 * ADD register ldg_a, ldg_b: load global memory-> ldg_a/b -> share_memory
 */
template<
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_N,
    const int BLOCK_SIZE_K,
    const int THREAD_SIZE_Y,
    const int THREAD_SIZE_X
    >
__global__ void gpu_matrix_mult_register2(
                int* __restrict__ A,
                int* __restrict__ B,
                int* __restrict__ C,
                int M, int N, int K)
{
    // shared memory for cached partial data of A and B
    __shared__ int tile_A[BLOCK_SIZE_M][BLOCK_SIZE_K]; //[128,8]
    __shared__ int tile_B[BLOCK_SIZE_K][BLOCK_SIZE_N]; //[8,128]
    // register for C, replace before sum
    int accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    // register for A, B
    int frag_a[THREAD_SIZE_Y];
    int frag_b[THREAD_SIZE_X];

    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszx * bszy;

    // registers load global memory: means global->register->shared memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;
    int ldg_a_reg[ldg_num_a];
    int ldg_b_reg[ldg_num_b];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // tid in block
    const int tid = ty * bszx + tx; // 0~256

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START=tid/BLOCK_SIZE_K;
    const int B_TILE_ROW_START=tid/BLOCK_SIZE_N;

    const int A_TILE_COL=tid%BLOCK_SIZE_K;
    const int B_TILE_COL=tid%BLOCK_SIZE_N;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE=THREAD_NUM_PER_BLOCK/BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE=THREAD_NUM_PER_BLOCK/BLOCK_SIZE_N;

    A = &A[(BLOCK_SIZE_M * by)* K]; // curr row first block
    B = &B[BLOCK_SIZE_N * bx]; // cur col first block

    // loop tiles
    for(int tile_idx = 0; tile_idx < K; tile_idx+=BLOCK_SIZE_K)
    {
        // load A from global memory to shared memory
        #pragma unroll
        for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
            int ldg_index = i / A_TILE_ROW_STRIDE;
            ldg_a_reg[ldg_index] = A[OFFSET(
                            A_TILE_ROW_START+i, //row
                            A_TILE_COL+tile_idx, //col
                            K)];
            tile_A[i+A_TILE_ROW_START][A_TILE_COL] = ldg_a_reg[ldg_index];
        }
        // load B from global memory to shared memory
        #pragma unroll
        for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE){
            int ldg_index = i / B_TILE_ROW_STRIDE;
            ldg_b_reg[ldg_index] = B[OFFSET(
                            B_TILE_ROW_START+tile_idx+i, // row
                            B_TILE_COL, // col
                            N)];
            tile_B[i+B_TILE_ROW_START][B_TILE_COL]=ldg_b_reg[ldg_index];
        }
        __syncthreads(); // sync threads per block

        // accum = tileA @ tileB, per thread
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE_K; k_idx++) {
            // NOTICE: reg is different in different thread
            // load A from shared memory to register: per col
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
                frag_a[thread_y] = tile_A[THREAD_SIZE_Y*ty + thread_y][k_idx];
            }
            // load B from shared memory to register: per row
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
                frag_b[thread_x] = tile_B[k_idx][THREAD_SIZE_X*tx + thread_x];
            }

            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
                #pragma unroll
                for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
        }
        __syncthreads();

    }
    // write accum to C
    #pragma unroll
    for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
        #pragma unroll
        for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
            const int row = BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty + thread_y;
            const int col = BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx + thread_x;
            if(row < M && col < N) {
                C[OFFSET(row, col, N)] = accum[thread_y][thread_x];
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 8;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;

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

    unsigned int grid_rows = (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    unsigned int grid_cols = (n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE_M/THREAD_SIZE_Y, BLOCK_SIZE_N/THREAD_SIZE_X);
    double iStart, iElaps;

    iStart = cpuSecond();

    gpu_matrix_mult_register2<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X>
            <<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    CHECK(cudaMemcpy(h_c, d_c, sizeof(int)*m*n, cudaMemcpyDeviceToHost));
    printf("gpu register matrix multiply elapsed %lf ms <<<grid <%d, %d>, block <%d, %d> >>>\n", iElaps, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

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