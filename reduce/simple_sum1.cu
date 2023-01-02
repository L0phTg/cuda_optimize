#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/exercise.cuh"

#define BLOCK_SIZE 512

void cpu_matrix_sum(int *A, int *out, int N, int block_num)
{
    for (int i=0; i < block_num; i++) {
        int sum = 0;
        // TODO: CHECK
        for (int j=0; j < BLOCK_SIZE; j++) {
            sum = sum + A[i*BLOCK_SIZE+j];
        }
        out[i] = sum;
    }
}

__global__ void gpu_matrix_sum_neighbored(int *g_idata, int *g_odata, unsigned int N)
{
    int tx = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (tx >= N)
        return ;

    for (int stride = 1; stride < blockDim.x; stride*=2) {
        if ((tx % (2 * stride)) == 0) {
            idata[tx] += idata[tx+ stride];   //
        }
        __syncthreads(); // sync within block
    }
    if (tx == 0)
        g_odata[blockIdx.x] = idata[0];
}

//
__global__ void gpu_matrix_sum_neighbored2(int *g_idata, int *g_odata, unsigned int N)
{
    int tx = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (tx >= N)
        return ;

    for (int stride = 1; stride < blockDim.x; stride*=2) {
        int idx = 2 * stride * tx;
        if (idx < blockDim.x) {
            idata[idx] += idata[idx + stride];
        }
        __syncthreads(); // sync within block
    }
    if (tx == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void gpu_matrix_sum_interleaved(int *g_idata, int *g_odata, unsigned int N)
{
    int bx = blockIdx.x; int tx = threadIdx.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return ;

    int *idata = g_idata + bx * blockDim.x;

    for (int stride = blockDim.x/2; stride > 0; stride >>=1) {
        if (tx < stride) {
            idata[tx] += idata[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
        g_odata[bx] = idata[0];
}


int main(int argc,char** argv)
{
    initDevice(0);

    int N=1024*1024;
    int block_num=(N+BLOCK_SIZE-1)/BLOCK_SIZE;

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

    dim3 dimGrid((unsigned int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, 1);
    double iStart, iElaps;

    iStart = cpuSecond();
    gpu_matrix_sum_neighbored<<<dimGrid, dimBlock>>>(d_a, d_out, N);
    //gpu_matrix_sum_neighbored2<<<dimGrid, dimBlock>>>(d_a, d_out, N);
    //gpu_matrix_sum_interleaved<<<dimGrid, dimBlock>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    CHECK(cudaMemcpy(h_out, d_out, sizeof(int)*block_num, cudaMemcpyDeviceToHost));
    printf("gpu simple array sum elapsed %lf ms <<<grid %d, block %d >>>\n", iElaps, dimGrid.x, dimBlock.x);

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
