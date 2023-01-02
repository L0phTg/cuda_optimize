#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include "../include/exercise.cuh"

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

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

__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor_sync(0xffffffff, localSum, 16);
    localSum += __shfl_xor_sync(0xffffffff, localSum, 8);
    localSum += __shfl_xor_sync(0xffffffff, localSum, 4);
    localSum += __shfl_xor_sync(0xffffffff, localSum, 2);
	localSum += __shfl_xor_sync(0xffffffff, localSum, 1);
    return localSum;
}

__device__ __forceinline__ int warpReduce2(int localSum)
{
    localSum += __shfl_down_sync(0xffffffff, localSum, 16);
    localSum += __shfl_down_sync(0xffffffff, localSum, 8);
    localSum += __shfl_down_sync(0xffffffff, localSum, 4);
    localSum += __shfl_down_sync(0xffffffff, localSum, 2);
    localSum += __shfl_down_sync(0xffffffff, localSum, 1);
	return localSum;
}

// shuffle reduce
// 1024 = 32 * 32 => need twice shuffle
__global__ void gpu_matrix_sum_shfl(int *g_idata, int *g_odata, unsigned int N)
{
	__shared__ int smem[BLOCK_SIZE/WARP_SIZE];

    int bszx = blockDim.x;
    int bx = blockIdx.x; int tx = threadIdx.x;

    unsigned int tid = bx * bszx + tx;

    if (tid >= N)
        return ;

	int sum=g_idata[tid];
	int laneIdx=threadIdx.x%WARP_SIZE;
	int warpIdx=threadIdx.x/WARP_SIZE;

	sum=warpReduce2(sum); // reduce every warp, 1024 reduce to 32 

	if(laneIdx==0)
		smem[warpIdx]=sum; // 32 warp reduce result saved to first 32 element in smem
	__syncthreads();

	sum=(tx < BLOCK_SIZE/WARP_SIZE)? smem[laneIdx] : 0;

	if(warpIdx==0)
		sum=warpReduce2(sum); // 32 reduce to 1

	if(tx == 0)
		g_odata[bx]=sum;
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

    dim3 dimGrid((unsigned int)(block_num), 1);
    dim3 dimBlock(BLOCK_SIZE, 1);
    double iStart, iElaps;

    iStart = cpuSecond();
	CHECK(cudaProfilerStart());
    gpu_matrix_sum_shfl<<<dimGrid, dimBlock>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
	CHECK(cudaProfilerStop());
    iElaps = cpuSecond() - iStart;

    CHECK(cudaMemcpy(h_out, d_out, sizeof(int)*block_num, cudaMemcpyDeviceToHost));
    printf("gpu shuffle array sum elapsed %lf ms <<<grid %d, block %d >>>\n", iElaps, dimGrid.x, dimBlock.x);

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
