CUDA算子开发笔记：
- 【*****】共享内存是BLOCK内部所有线程共享；
- 【*****】寄存器/局部变量是线程独有的；
- 【****】BLOCK的大小代表了计算网格的大小，即BLOCK的大小代表了每个BLOCK里面包含多少执行线程；
- 内存访问模式：
- 共享内存：
    - 连续的32-bit访存被分配到连续的banks, 每个bank每个周期可以响应一个地址
    - `bank冲突`：如果同一个warp中的线程访问同一个bank中的不同地址时将会发生bank冲突；
- shuffle指令：允许线程束内两个线程相互访问对方的寄存器，使得通信效率相比利用共享内存通信高很多。

# Reduce
函数签名：
- `g_idata`：要规约的一维数组；
- `N`：g_idata的数组长度；
- `g_odata`：保存规约的结果；由于cuda按照thread block来处理数据，所以g_odata为g_idata分块后，由**每块的reduce结果组成的数组**；
```c
__global__ void gpu_matrix_sum_XXX(
    int *g_idata,
    int *g_odata,
    unsigned int N
)
```
1. simple_sum1.cu：常规的规约算法：包含相邻配对和交错配对。这里涉及到的优化技巧：`改善线程束分化`、
    - 改善线程束分化：
        - `gpu_matrix_sum_neighbored`函数中：`(tx % (2 * stride)) == 0`导致在每次循环中, 每个warp利用到的线程数目warpSize/(2*stride), 但是用到的warp总数不变, 依然是block_size/warpSize（默认warpSize=32）, 所以线程利用率很低.
        -  `gpu_matrix_sum_neighbored2`函数中：将判断条件改为：`int idx = 2 * stride * tx; if (idx < blockDim.x) ...`可以发现, 在每次循环中, 使用到的warp总数变为了block_size/(2*stride)/(warpSize), 且每个warp的利用率为100%, 即相比与上面的方法，通过提高每个warp的线程利用率来减少了使用的warp总数, 减小了硬件的压力.

2. unroll_sum2.cu：在交错配对的函数上做优化，涉及到的优化技巧：`规约展开`、`循环展开`
    - 规约展开：
        - `gpu_matrix_sum_interleaved_unroll2`, 为了让每个线程处理两个block的数据, 所以添加了`g_idata[idx] += g_idata[idx+blockDim.x];`，即将相邻两个block的数据按元素相加;
        - `gpu_matrix_sum_interleaved_unroll8`，更进一步，每个线程处理八个block的数据
    - 循环展开，方式比较简单直接：
        - 针对threadIdx.x > 32的展开：
            ```c
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
            ```
        - 针对threadIdx.x < 32的展开，利用对`volatile`的读写来替代了对`__syncthreads()`的调用，即目前只有一个warp在干活，由于一个warp中的32个线程其实是在一个SIMD单元上，这32个线程每次都是执行同一条指令，这天然地保持了同步状态：
            ```c
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
            ```

3. shared_sum3.cu：在2的基础上做优化，涉及到的优化技巧：`引入共享内存`、`解决Bank冲突`
    - 引入共享内存，方式很直接，将每个block的数据放到共享内存上，然后再执行规约：
        ```c
            __shared__ int smem[BLOCK_SIZE];
            smem[tx] = g_idata[tid];
        ```
    - 解决Bank冲突（*如果同一个warp中的线程访问同一个bank中的不同地址*），这是在引入共享内存后随之需要考虑的问题：
        - 在`simple_sum1.cu`的相邻配对的函数中, 循环条件为`int stride = 1; stride < blockDim.x; stride*=2)`, 如果引入共享缓存的话，在第一个warp中, 0号线程读取内存索引为0,1位置的元素的值, 16号线程读取内存索引为32,33位置的元素的值，即通过一个warp中的两个线程(0, 16)读取了同一个bank(0, 32)中的元素, 导致了bank冲突
        - 在改为交错配对后，循环条件成为了`int stride = blockDim.x/2; stride > 0; stride >>=1`如果blockDim.x=1024, 第一个warp中，0号线程读取内存索引为(0, 512)的元素的值，1号线程读取内存索引为(1, 513)的元素的值. 不同的线程读取的share_memory中的元素(0, 1)不在同一个bank中, 所以避免了bank冲突.

4. shfl_sum4.cu，利用shuffle指令来优化束内规约，涉及到的优化技巧：`引入shuffle指令`：
    - shuffle指令可以允许线程束内两个线程相互访问对方的寄存器，下面是利用shuffle指令对一个warp中的数据进行求和的算法：
        ```c
        __device__ __forceinline__ int warpReduce2(int localSum)
        {
            localSum += __shfl_down_sync(0xffffffff, localSum, 16);
            localSum += __shfl_down_sync(0xffffffff, localSum, 8);
            localSum += __shfl_down_sync(0xffffffff, localSum, 4);
            localSum += __shfl_down_sync(0xffffffff, localSum, 2);
            localSum += __shfl_down_sync(0xffffffff, localSum, 1);
            return localSum;
        }
        ```

# GEMM
NOTE：
- A、B为输入矩阵，C为结果矩阵；
- A矩阵形状：M\*K，B矩阵形状：K\*N，C矩阵形状：M*N。
1. simple_mm1.cu：基本的GPU矩阵乘法算法。
    ```cpp
    __global__ void gpu_matrix_mult(int *A,int *B, int *C, int M, int N, int K)
    ```
2. shared_mm2.cu：包含三个版本：
    - 引入共享内存，这里假定BLOCK为方阵，同时共享内存块的大小和BLOCK大小相同。
    ```cpp
    template<const int BLOCK_SIZE>
    __global__ void gpu_matrix_mult_shared(int *A,int *B, int *C, int M, int N, int K)
    ```
    -  引入共享内存，假定BLOCK为矩形，且共享内存块的大小与BLOCK大小不同（BLOCK_SIZE_K > BLOCK_SIZE_M && BLOCK_SIZE_K > BLOCK_SIZE_N）。即每个BLOCK中可执行的线程数目小于了共享内存块的大小，需要分批次读取数据到共享内存中；
    ```cpp
    template<
        const int BLOCK_SIZE_M,
        const int BLOCK_SIZE_N,
        const int BLOCK_SIZE_K
        >
    __global__ void gpu_matrix_mult_shared2(int *A,int *B, int *C, int M, int N, int K)
    ```
    -  引入共享内存，上面的基础上引入了线程计算块`THREAD_SIZE_Y`和`THREAD_SIZE_X`, 即每个gpu执行线程负责计算`THREAD_SIZE_Y*THREAD_SIZE_X`大小的单元（之前的版本可以理解为每个gpu执行线程计算`1*1`的单元）; 同时每个线程对C写入时，也应该一次写入`THREAD_SIZE_Y*THREAD_SIZE_X`数量的元素, 所以引入了`int accum[THREAD_SIZE_Y][THREAD_SIZE_X]`作为写缓存(可以理解为register for C) 
    ```cpp
    template<
        const int BLOCK_SIZE_M,
        const int BLOCK_SIZE_N,
        const int BLOCK_SIZE_K,
        const int THREAD_SIZE_Y,
        const int THREAD_SIZE_X
        >
    __global__ void gpu_matrix_mult_shared3(int *A,int *B, int *C, int M, int N, int K)
    ```
3. register_mm3.cu：包含两个版本，函数签名同上：
    -  【加快写入】引入对tile_A, tile_B读取的寄存器缓存`frag_a[THREAS_SIZE_Y]`和`frag_b[THREAD_SIZE_Y]`,每个gpu执行线程循环多次从tile_A(按列读)和tile_B(按行读)中加载数据到寄存器frag_a和frag_b中，再由frag_a和frag_b计算得到accum, 最终将该执行线程负责的`THREAD_SIZE_Y*THREAD_SIZE_X`大小单元计算完成并写入到C中；
    -  【加快读取】引入对A, B读取的寄存器，即再将A中的元素加载到tile_A中时，先将其加载到ldg_a_reg寄存器上，然后再将其加载到tile_A中。
4. vectorized_mm4.cu：包含两个版本，函数签名同上：
    - 核函数从全局内存中读取数据时的最小粒度是32字节，所以在**数据对齐合并访问时**，可以一次读取4个float32或者int32；这里可以实现对A, B的向量化读取，tile_B的向量化，注意tile_A不行, 因为tile_A此时的形状是`[BLOCK_SIZE_M][BLOCK_SIZE_K]`, tile_B此时的形状是`[BLOCK_SIZE_K][BLOCK_SIZE_N]`，在计算得到accum时，我们加载tile_A中的一列`THREAD_SIZE_Y`大小加载到`frag_a`中，加载tile_B中的一行`THREAD_SIZE_X`大小加载到`frag_b`中，注意，因为这里tile_A按列读取的, 所以不能向量化访存.
    -  为了实现对tile_A的向量化读取，方法也很直接, 我们将tile_A转置即可, 这时tile_A的大小为`[BLOCK_SIZE_K][BLOCK_SIZE_M]`. 注意, 这时候还需要修改tile_A的写入逻辑，具体可以参看代码.

## 性能对比


# 参考资料

- <<CUDA C编程权威指南>>
- [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)
- [深入浅出GPU优化系列：GEMM优化（一）](https://zhuanlan.zhihu.com/p/435908830)
- [深入浅出GPU优化系列：GEMM优化（二）](https://zhuanlan.zhihu.com/p/442930482)
- https://github.com/Tony-Tan/CUDA_Freshman
- https://github.com/flame/how-to-optimize-gemm
- https://github.com/Liu-xiandong/How_to_optimize_in_GPU
