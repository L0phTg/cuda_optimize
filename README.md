CUDA算子开发笔记：
- 【*****】共享内存是BLOCK内部所有线程共享；
- 【*****】寄存器/局部变量是线程独有的；
- 【****】BLOCK的大小代表了计算网格的大小，即BLOCK的大小代表了每个BLOCK里面包含多少执行线程；
- 内存访问模式：

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

- [深入浅出GPU优化系列：GEMM优化（一）](https://zhuanlan.zhihu.com/p/435908830)
- [深入浅出GPU优化系列：GEMM优化（二）](https://zhuanlan.zhihu.com/p/442930482)
- https://github.com/flame/how-to-optimize-gemm
- https://github.com/Liu-xiandong/How_to_optimize_in_GPU
