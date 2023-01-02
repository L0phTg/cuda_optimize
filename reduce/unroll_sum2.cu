#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/exercise.cuh"

#define BLOCK_SIZE 512

void cpu_matrix_sum(int *A, int *out, int N, int block_num)
{
    for (int i=0; i < block_num; i++) {
        int sum = 0;
        // TODO: CHECK
        for (int j=0; j < BLOCK_SIZE*8; j++) {
            sum = sum + A[i*BLOCK_SIZE*8+j];
        }
        out[i] = sum;
    }
}

// every thread handle data of 2 block
__global__ void gpu_matrix_sum_interleaved_unroll2(int *g_idata, int *g_odata, unsigned int N)
{
    int bx = blockIdx.x; int tx = threadIdx.x;

    int idx = bx * blockDim.x * 2 + tx;
    if (tx >= N)
        return ;

    int *idata = g_idata + bx * blockDim.x * 2;
    if (idx + blockDim.x < N) {
        g_idata[idx] += g_idata[idx+blockDim.x];
    }
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>=1) {
        if (tx < stride) {
            idata[tx] += idata[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
        g_odata[bx] = idata[0];
}

// every thread handle data of 8 block
// unroll last warp
__global__ void gpu_matrix_sum_interleaved_unroll8(int *g_idata, int *g_odata, unsigned int N)
{
    int bx = blockIdx.x; int tx = threadIdx.x;

    int idx = bx * blockDim.x * 8 + tx;
    if (tx >= N)
        return ;

    int *idata = g_idata + bx * blockDim.x * 8;
    if (idx + 7 * blockDim.x < N) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 32; stride >>=1) {
        if (tx < stride) {
            idata[tx] += idata[tx + stride];
        }
        __syncthreads();
    }
    if (tx < 32) {
        // write variable result to global memory
        volatile int *vmem = idata;
        vmem[tx] += vmem[tx+32];
        vmem[tx] += vmem[tx+16];
        vmem[tx] += vmem[tx+8];
        vmem[tx] += vmem[tx+4];
        vmem[tx] += vmem[tx+2];
        vmem[tx] += vmem[tx+1];
    }

    if (tx == 0)
        g_odata[bx] = idata[0];
}

// every thread handle data of 8 block
// unroll all wrap
__global__ void gpu_matrix_sum_interleaved_unroll8_complete(int *g_idata, int *g_odata, unsigned int N)
{
    int bx = blockIdx.x; int tx = threadIdx.x;

    int idx = bx * blockDim.x * 8 + tx;
    if (tx >= N)
        return ;

    int *idata = g_idata + bx * blockDim.x * 8;
    if (idx + 7 * blockDim.x < N) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // inplace reduction in global memory
    if (blockDim.x >= 1024 && tx < 512)
        idata[tx] += idata[tx+512];
    __syncthreads();
    if (blockDim.x >= 512 && tx < 256)
        idata[tx] += idata[tx+256];
    __syncthreads();
    if (blockDim.x >= 256 && tx < 128)
        idata[tx] += idata[tx+128];
    __syncthreads();
    if (blockDim.x >= 128 && tx < 64)
        idata[tx] += idata[tx+64];
    __syncthreads();

    if (tx < 32) {
        // write variable result to global memory
        volatile int *vmem = idata;
        vmem[tx] += vmem[tx+32];
        vmem[tx] += vmem[tx+16];
        vmem[tx] += vmem[tx+8];
        vmem[tx] += vmem[tx+4];
        vmem[tx] += vmem[tx+2];
        vmem[tx] += vmem[tx+1];
    }

    if (tx == 0)
        g_odata[bx] = idata[0];
}

int main(int argc,char** argv)
{
    initDevice(0);

    int N=1024*1024;
    int block_num=(N+BLOCK_SIZE-1)/BLOCK_SIZE/8;

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
    gpu_matrix_sum_interleaved_unroll8_complete<<<dimGrid, dimBlock>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    CHECK(cudaMemcpy(h_out, d_out, sizeof(int)*block_num, cudaMemcpyDeviceToHost));
    printf("gpu unroll array sum elapsed %lf ms <<<grid %d, block %d >>>\n", iElaps, dimGrid.x, dimBlock.x);

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
