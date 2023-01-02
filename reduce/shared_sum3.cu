#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/exercise.cuh"

#define BLOCK_SIZE 512
#define BLOCK_WARP 1

void cpu_matrix_sum(int *A, int *out, int N, int block_num)
{
    for (int i=0; i < block_num; i++) {
        int sum = 0;
        // TODO: CHECK
        for (int j=0; j < BLOCK_SIZE*BLOCK_WARP; j++) {
            sum = sum + A[i*BLOCK_SIZE*BLOCK_WARP+j];
        }
        out[i] = sum;
    }
}

// unroll all wrap
// global memory to shared memory
__global__ void gpu_matrix_sum_shared_interleaved_complete(int *g_idata, int *g_odata, unsigned int N)
{
	__shared__ int smem[BLOCK_SIZE];

    int bx = blockIdx.x; int tx = threadIdx.x;

    int tid = bx * blockDim.x + tx;

    if (tid >= N)
        return ;

	smem[tx] = g_idata[tid];

    // inplace reduction in global memory
    if (blockDim.x >= 1024 && tx < 512)
        smem[tx] += smem[tx+512];
    __syncthreads();
    if (blockDim.x >= 512 && tx < 256)
        smem[tx] += smem[tx+256];
    __syncthreads();
    if (blockDim.x >= 256 && tx < 128)
        smem[tx] += smem[tx+128];
    __syncthreads();
    if (blockDim.x >= 128 && tx < 64)
        smem[tx] += smem[tx+64];
    __syncthreads();

    if (tx < 32) {
        // write variable result to global memory
        volatile int *vsmem = smem;
        vsmem[tx] += vsmem[tx+32];
        vsmem[tx] += vsmem[tx+16];
        vsmem[tx] += vsmem[tx+8];
        vsmem[tx] += vsmem[tx+4];
        vsmem[tx] += vsmem[tx+2];
        vsmem[tx] += vsmem[tx+1];
    }

    if (tx == 0)
        g_odata[bx] = smem[0];
}

// every thread handle data of 8 block
// unroll all wrap
// global memory to shared memory
__global__ void gpu_matrix_sum_shared_interleaved_unroll8_complete(int *g_idata, int *g_odata, unsigned int N)
{
	__shared__ int smem[BLOCK_SIZE];

    int bx = blockIdx.x; int tx = threadIdx.x;

    int idx = bx * blockDim.x * 8 + tx;
    if (tx >= N)
        return ;

    if (idx + 7 * blockDim.x < N) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        smem[tx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // inplace reduction in global memory
    if (blockDim.x >= 1024 && tx < 512)
        smem[tx] += smem[tx+512];
    __syncthreads();
    if (blockDim.x >= 512 && tx < 256)
        smem[tx] += smem[tx+256];
    __syncthreads();
    if (blockDim.x >= 256 && tx < 128)
        smem[tx] += smem[tx+128];
    __syncthreads();
    if (blockDim.x >= 128 && tx < 64)
        smem[tx] += smem[tx+64];
    __syncthreads();

    if (tx < 32) {
        // write variable result to global memory
        volatile int *vsmem = smem;
        vsmem[tx] += vsmem[tx+32];
        vsmem[tx] += vsmem[tx+16];
        vsmem[tx] += vsmem[tx+8];
        vsmem[tx] += vsmem[tx+4];
        vsmem[tx] += vsmem[tx+2];
        vsmem[tx] += vsmem[tx+1];
    }

    if (tx == 0)
        g_odata[bx] = smem[0];
}

int main(int argc,char** argv)
{
    initDevice(0);

    int N=1024*1024;
    int block_num=(N+BLOCK_SIZE-1)/BLOCK_SIZE/BLOCK_WARP;

    int *h_a, *h_out, *h_cpu;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*N));
    CHECK(cudaMallocHost((void **) &h_out, sizeof(int)*block_num));
    CHECK(cudaMallocHost((void **) &h_cpu, sizeof(int)*block_num));

    for (int i = 0; i < N; ++i) {
        h_a[i] = rand() % 10;
    }

    int *d_a, *d_out;
    CHECK(cudaMalloc((void **) &d_a, sizeof(int)*N));
    CHECK(cudaMalloc((void **) &d_out, sizeof(int)*block_num));

    // copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*N, cudaMemcpyHostToDevice));

    dim3 dimGrid((unsigned int)(block_num), 1);
    dim3 dimBlock(BLOCK_SIZE, 1);
    double iStart, iElaps;

    iStart = cpuSecond();
    gpu_matrix_sum_shared_interleaved_complete<<<dimGrid, dimBlock>>>(d_a, d_out, N);
    //gpu_matrix_sum_shared_interleaved_unroll8_complete<<<dimGrid, dimBlock>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    CHECK(cudaMemcpy(h_out, d_out, sizeof(int)*block_num, cudaMemcpyDeviceToHost));
    printf("gpu shared array sum elapsed %lf ms <<<grid %d, block %d >>>\n", iElaps, dimGrid.x, dimBlock.x);

    cpu_matrix_sum(h_a, h_cpu, N, block_num);
    for (int i = 0; i < block_num; i++) {
        if (abs(h_out[i]-h_cpu[i]) != 0) {
            printf("i: %d, h_out: %d, h_cpu: %d\n", i, h_out[i], h_cpu[i]);
        }
    }

    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_out));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_out));
    CHECK(cudaFreeHost(h_cpu));
    return 0;
}
