/**
 * @file cc_gpu.cuh
 * @brief Single-GPU pull-based label-propagation baseline for CC.
 *
 * Each active vertex pulls the maximum label from incoming neighbors and
 * activates outgoing neighbors when its label increases.
 */
#include <cuda_runtime.h>
#include "include/cuda_event_timer.cuh"
#include <stdio.h>
#include <stdlib.h>

#define INF 100000
#define BLOCK_SIZE 256

/**
 * @brief Propagate maximum component labels for one sparse iteration.
 * @param d_values In-place vertex labels.
 * @param d_active Current work-set bitmap; zero means inactive.
 * @param d_update Next-iteration work-set bitmap.
 */
__global__ void ccPullKernel(
    int* d_values,
    const int* d_row_offsets,
    const int* d_column_indices,
    const int* d_column_offsets,
    const int* d_row_indices,
    const int* d_active,
    int* d_update,
    int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes || d_active[tid] == 0) return;

    int oldVal = d_values[tid];
    int newVal = oldVal;

    // 遍历入边,取最大值
    for (int i = d_column_offsets[tid]; i < d_column_offsets[tid + 1]; i++) {
        int neighbor = d_row_indices[i];
        int candidate = d_values[neighbor];
        if (candidate > newVal) newVal = candidate;
    }

    d_values[tid] = newVal;

    if (newVal > oldVal) {
        for (int i = d_row_offsets[tid]; i < d_row_offsets[tid + 1]; i++) {
            int dst = d_column_indices[i];
            d_update[dst] = 1;
        }
    }


}

/** @brief Count nonzero work-set entries. */
__global__ void countActiveKernel(
    const int* d_active,
    int* d_num_active,
    int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nodes && d_active[tid] != 0) {
        atomicAdd(d_num_active, 1);
    }
}

/**
 * @brief Execute single-GPU maximum-label propagation to convergence.
 * @param h_value Host output labels, one per vertex.
 * @param h_row_offsets Host outgoing CSR row offsets.
 * @param h_column_indices Host outgoing CSR destinations.
 * @param h_column_offsets Host incoming CSR column offsets.
 * @param h_row_indices Host incoming CSR sources.
 * @param num_nodes Number of vertices.
 * @param num_edges Number of directed edges.
 */
void ccGPU(
    int* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int num_edges
) {
    
    // ------------------- Host 临时数组 -------------------
    int* h_active = (int*)malloc(num_nodes*sizeof(int));
    for(int i=0;i<num_nodes;i++){
        h_active[i] = 1;      // 第一轮所有顶点活跃
        h_value[i] = i;
    }
    // ------------------- Device 内存 -------------------
    int *d_values, *d_row_offsets, *d_column_indices;
    int *d_column_offsets, *d_row_indices, *d_active, *d_update;
    int *d_num_active;

    cudaMalloc(&d_values, num_nodes * sizeof(int));
    cudaMalloc(&d_row_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_column_indices, num_edges * sizeof(int));
    cudaMalloc(&d_column_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_row_indices, num_edges * sizeof(int));
    cudaMalloc(&d_active, num_nodes * sizeof(int));
    cudaMalloc(&d_update, num_nodes * sizeof(int));
    cudaMalloc(&d_num_active, sizeof(int));

    cudaMemcpy(d_values, h_value, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_column_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_offsets, h_column_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, h_row_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active, h_active, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_update, 0, num_nodes*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    GraphCudaEventAccumulator graph_kernel_timer;
    graph_cuda_timer_create(&graph_kernel_timer);

    cudaEventRecord(start);
    // ------------------- BFS 迭代 -------------------
    int iter = 0;
    int active_nodes = num_nodes;   // 第一轮所有节点活跃
    int num_blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    while(iter < 1000 && active_nodes > 0) {

        iter++;

        // 执行拉模式核函数
        graph_cuda_timer_start(&graph_kernel_timer);
        ccPullKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_values, d_row_offsets, d_column_indices,
            d_column_offsets, d_row_indices,
            d_active, d_update, num_nodes
        );
        graph_cuda_timer_stop(&graph_kernel_timer);
        cudaDeviceSynchronize();
        graph_cuda_timer_accumulate(&graph_kernel_timer);

        // 更新下一轮活跃数组
        cudaMemcpy(d_active, d_update, num_nodes*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_update, 0, num_nodes*sizeof(int));

        // GPU 统计下一轮活跃顶点数，CPU 只回传一个 int
        cudaMemset(d_num_active, 0, sizeof(int));
        countActiveKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_active, d_num_active, num_nodes
        );
        cudaMemcpy(&active_nodes, d_num_active, sizeof(int), cudaMemcpyDeviceToHost);

    }

    // ------------------- 拷贝结果回 Host -------------------
    cudaMemcpy(h_value, d_values, num_nodes*sizeof(int), cudaMemcpyDeviceToHost);

    // 记录结束事件
    cudaEventRecord(stop);
    // 等待 stop 完成
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU time: %.4f ms\n", ms);
    printf("BENCHMARK_TIMING gpu_main_ms=%.4f graph_kernel_ms=%.4f "
           "cpu_check_tail_ms=0.0000 "
           "nccl_exchange_ms=0.0000 mpi_sync_ms=0.0000 communication_ms=0.0000\n",
           ms, graph_kernel_timer.total_ms);
    printf("BENCHMARK_ITERATIONS iterations=%d\n", iter);
    graph_cuda_timer_destroy(&graph_kernel_timer);
    
    // ------------------- 释放内存 -------------------
    cudaFree(d_values); cudaFree(d_row_offsets); cudaFree(d_column_indices);
    cudaFree(d_column_offsets); cudaFree(d_row_indices);
    cudaFree(d_active); cudaFree(d_update);
    cudaFree(d_num_active);
    free(h_active); 

    //printf("GPU CC finished in %d iterations.\n", iter);
}
