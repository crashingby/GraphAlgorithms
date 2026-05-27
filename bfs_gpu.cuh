#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define INF 100000  

// ==================== GPU BFS 拉模式核函数 ====================
__global__ void bfsPullKernel(
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

    // 遍历入边
    for (int i = d_column_offsets[tid]; i < d_column_offsets[tid + 1]; i++) {
        int neighbor = d_row_indices[i];
        int candidate = d_values[neighbor] + 1;
        if (candidate < newVal) newVal = candidate;
    }


    d_values[tid] = newVal;

    if (newVal < oldVal) {
        for (int i = d_row_offsets[tid]; i < d_row_offsets[tid + 1]; i++) {
            int dst = d_column_indices[i];
            d_update[dst] = 1;
        }
    }

}

// ==================== 统计并输出活跃顶点 ====================
int count_active_nodes(int* h_active, int num_nodes) {
    int count = 0;
    for(int i=0;i<num_nodes;i++) if(h_active[i] != 0) count++;
    return count;
}

// ==================== GPU BFS 主函数 ====================
void bfsGPU(
    int* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int num_edges,
    int src
) {
    // ------------------- Host 临时数组 -------------------
    int* h_active = (int*)malloc(num_nodes*sizeof(int));
    for(int i=0;i<num_nodes;i++){
        h_active[i] = 1;      // 第一轮所有顶点活跃
        h_value[i] = INF;
    }
    h_value[src] = 0;

    // ------------------- Device 内存 -------------------
    int *d_values, *d_row_offsets, *d_column_indices;
    int *d_column_offsets, *d_row_indices, *d_active, *d_update;

    cudaMalloc(&d_values, num_nodes * sizeof(int));
    cudaMalloc(&d_row_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_column_indices, num_edges * sizeof(int));
    cudaMalloc(&d_column_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_row_indices, num_edges * sizeof(int));
    cudaMalloc(&d_active, num_nodes * sizeof(int));
    cudaMalloc(&d_update, num_nodes * sizeof(int));

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

    cudaEventRecord(start);

    // ------------------- BFS 迭代 -------------------
    int iter = 0;
    int active_nodes = num_nodes;   // 第一轮所有节点活跃
    int num_blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    while(iter < 1000) {

        iter++;
         // 拷贝到 Host 统计并输出活跃顶点
        cudaMemcpy(h_active, d_active, num_nodes*sizeof(int), cudaMemcpyDeviceToHost);
        active_nodes = count_active_nodes(h_active, num_nodes);
        if(active_nodes == 0) break;

        // 执行拉模式核函数
        bfsPullKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_values, d_row_offsets, d_column_indices,
            d_column_offsets, d_row_indices,
            d_active, d_update, num_nodes
        );
        cudaDeviceSynchronize();

        // 更新下一轮活跃数组
        cudaMemcpy(d_active, d_update, num_nodes*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_update, 0, num_nodes*sizeof(int));
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

    // ------------------- 释放内存 -------------------
    cudaFree(d_values); cudaFree(d_row_offsets); cudaFree(d_column_indices);
    cudaFree(d_column_offsets); cudaFree(d_row_indices);
    cudaFree(d_active); cudaFree(d_update);
    free(h_active); 

    printf("GPU BFS finished in %d iterations.\n", iter);
}
