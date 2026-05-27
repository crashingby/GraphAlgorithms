#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <cub/cub.cuh> // 引入 CUB 高性能排序库


#define BLOCK_SIZE 256
#define INF 100000  
#define GPU_DEVICE 0

// ==================== 结构体 ====================
enum ValueTrend {
    TREND_NONE = 0,  // 不检测
    TREND_INC  = 1,  // 单调递增
    TREND_DEC  = 2   // 单调递减
};

struct MonotonicInfo {
    int dmr_error_flag;       // 冗余计算不一致标志（设备端写入）
    int monotonic_error_flag; // 单调性违反标志（设备端写入）
};

// ==================== Kernel: 计算活跃顶点得分 ====================
__global__ void collectAndScoreKernel(
    int* d_active, 
    const int* d_row_offsets, 
    int num_nodes, 
    int max_outdegree,
    int* d_active_nodes, 
    float* d_scores, 
    int* d_num_active) 
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        int val = d_active[v];
        if (val != -1) {
            // 原子加获取当前顶点在紧凑数组中的索引
            int idx = atomicAdd(d_num_active, 1);
            d_active_nodes[idx] = v;
            
            int outdeg = d_row_offsets[v + 1] - d_row_offsets[v];
            float dist_score = 1.0f / (1.0f + (float)val);
            // 计算得分
            d_scores[idx] = 0.5f * (float)outdeg / (float)max_outdegree + 0.5f * dist_score;
        }
    }
}

// ==================== Kernel: 标记关键顶点 ====================
__global__ void markCriticalKernel(
    int* d_active, 
    const int* d_active_nodes, 
    int num_active, 
    int topN) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_active) {
        int v = d_active_nodes[tid];
        // 排序后，前 topN 个是关键顶点(2)，其余是普通活跃顶点(1)
        if (tid < topN) {
            d_active[v] = 2;
        } else {
            d_active[v] = 1;
        }
    }
}

// ==================== Kernel：BFS 拉模式 + 双模冗余 ====================
__global__ void bfsPullDualKernel(
    int* d_values,
    const int* d_row_offsets,
    const int* d_column_indices,
    const int* d_column_offsets,
    const int* d_row_indices,
    const int* d_active, 
    int* d_update,        // 写下一轮活跃顶点的值  
    int num_nodes,
    int* d_delta,        // 每个顶点本轮变化幅度（设备数组）
    MonotonicInfo* d_info, // 设备端的检测标志结构体
    ValueTrend trend
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // ==================== 共享内存 ====================
    __shared__ int critical_list[256];
    __shared__ int idle_count;
    __shared__ int critical_count;
    __shared__ int redundant_results[256];

    if (local_tid == 0) {
        idle_count = 0;
        critical_count = 0;
    }
    __syncthreads();

    bool valid_tid = (tid < num_nodes);
    bool is_active = valid_tid && (d_active[tid] != -1);
    bool is_critical = is_active && (d_active[tid] == 2);
    bool is_idle = (!is_active || !valid_tid);

    // 收集关键线程
    int cid = -1;
    if(is_critical){
        cid = atomicAdd(&critical_count, 1);
        critical_list[cid] = tid; 
    }

    // 收集空闲线程
    int iid = -1;
    if(is_idle){
        iid = atomicAdd(&idle_count, 1);
    }
    __syncthreads();

    // ==================== 主线程计算（活跃线程） ====================
    int oldVal = INF;
    if (valid_tid) oldVal = d_values[tid];
    int main_newVal = INF;

    if(is_active){
        main_newVal = d_values[tid];
        for (int i = d_column_offsets[tid]; i < d_column_offsets[tid + 1]; i++) {
            int dst = d_row_indices[i];
            int candidate = d_values[dst] + 1;
            if(candidate < main_newVal) main_newVal = candidate;
        }
    }
    __syncthreads();

    // ==================== 冗余计算（空闲线程执行） ====================
    if(is_idle) {
        int my_idle_id = iid;   // 本线程在 idle_list 的索引
        if(my_idle_id < critical_count){
            int target_tid = critical_list[my_idle_id];
            int redundant_newVal = d_values[target_tid];
            for (int i = d_column_offsets[target_tid]; i < d_column_offsets[target_tid + 1]; i++) {
                int dst = d_row_indices[i];
                int candidate = d_values[dst] + 1;
                if(candidate < redundant_newVal) redundant_newVal = candidate;
            }
            redundant_results[my_idle_id] = redundant_newVal;  // 将冗余结果写入共享内存
        }
    }
    __syncthreads();

    // ==================== 冗余结果对比（关键线程执行） ====================
    if(is_critical){
        int my_index = cid; 
        if(idle_count > my_index){
            int redundant_newVal = redundant_results[my_index];  // 冗余结果
            if(redundant_newVal != main_newVal){
                atomicExch(&(d_info->dmr_error_flag), 1);  //将 DMR 错误写入设备端结构（单个写操作即可）
            }
        }
    }
    __syncthreads();

    // ==================== 主线程写回（活跃线程） ====================
    if (is_active) {
        d_values[tid] = main_newVal;
        if (main_newVal < oldVal) {
            for (int i = d_row_offsets[tid]; i < d_row_offsets[tid + 1]; i++) {
                int dst = d_column_indices[i];
                // 写下一轮活跃顶点值（可能有写冲突，但仅用于激活判断）
                d_update[dst] = d_values[dst];
            }
            int delta = main_newVal - oldVal; 
            d_delta[tid] = delta; 

            // 记录单调性违反
            if ((trend == TREND_INC && delta < 0) || (trend == TREND_DEC && delta > 0)) {
                atomicExch(&(d_info->monotonic_error_flag), 1);
            }
        } else {
            d_delta[tid] = 0;
        }
    } else if (valid_tid) {
        d_delta[tid] = 0;
    }
}

// ==================== 计算整个图的最大出度 ====================
int compute_max_outdegree(const int* h_row_offsets, int num_nodes) {
    int max_outdegree = 0;
    for (int v = 0; v < num_nodes; v++) {
        int outdegree = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdegree > max_outdegree) max_outdegree = outdegree;
    }
    return max_outdegree;
}

// ==================== 主函数：全GPU容错版 (CUB极限优化) ====================
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
    printf("开始 GPU BFS（全GPU排序容错实现 - CUB优化版）...\n"); fflush(stdout);

    int max_outdegree = compute_max_outdegree(h_row_offsets, num_nodes);
    
    // ------------------- 分配 Host 内存 -------------------
    int* h_active; 
    int* h_delta[2]; 
    MonotonicInfo* h_info[2]; 

    cudaMallocHost(&h_active, num_nodes * sizeof(int));
    cudaMallocHost(&h_delta[0], num_nodes * sizeof(int));
    cudaMallocHost(&h_delta[1], num_nodes * sizeof(int));
    cudaMallocHost(&h_info[0], sizeof(MonotonicInfo));
    cudaMallocHost(&h_info[1], sizeof(MonotonicInfo));
    
    for(int i = 0; i < num_nodes; i++){
        h_active[i] = 1;      
        h_value[i] = INF;
    }
    h_value[src] = 0;

    // ------------------- 分配 Device 内存 -------------------
    int *d_values, *d_row_offsets, *d_column_indices;
    int *d_column_offsets, *d_row_indices, *d_active, *d_update;
    int* d_delta[2];  
    MonotonicInfo* d_info[2]; 

    cudaMalloc(&d_values, num_nodes * sizeof(int));
    cudaMalloc(&d_row_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_column_indices, num_edges * sizeof(int));
    cudaMalloc(&d_column_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_row_indices, num_edges * sizeof(int));
    cudaMalloc(&d_active, num_nodes * sizeof(int));
    cudaMalloc(&d_update, num_nodes * sizeof(int));
    
    for(int i=0;i<2;i++){
        cudaMalloc(&d_delta[i], num_nodes*sizeof(int));
        cudaMallocManaged(&d_info[i], sizeof(MonotonicInfo));
    }

    // ========== 新增：供 CUB 双缓冲排序使用的显存 ==========
    int *d_nodes_in, *d_nodes_out, *d_num_active;
    float *d_scores_in, *d_scores_out;
    cudaMalloc(&d_nodes_in, num_nodes * sizeof(int));
    cudaMalloc(&d_nodes_out, num_nodes * sizeof(int));
    cudaMalloc(&d_scores_in, num_nodes * sizeof(float));
    cudaMalloc(&d_scores_out, num_nodes * sizeof(float));
    cudaMalloc(&d_num_active, sizeof(int));

    // ========== 提前分配 CUB 所需的临时排序空间（核心优化，消除循环内 cudaMalloc） ==========
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    // 第一次调用传 NULL，目的是获取排序 num_nodes 长度所需的最大内存大小
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        d_scores_in, d_scores_out, d_nodes_in, d_nodes_out, num_nodes);
    // 一次性分配好内存，循环内直接复用
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaMemcpy(d_values, h_value, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_column_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_offsets, h_column_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, h_row_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active, h_active, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_update, -1, num_nodes*sizeof(int));

    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cudaEvent_t copy_delta_event[2], start, stop;
    cudaEventCreateWithFlags(&copy_delta_event[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&copy_delta_event[1], cudaEventDisableTiming);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int iter = 0;
    int pingpong = 0;
    float pre_avg_delta = INF;
    int active_nodes = num_nodes;
    int num_blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ================= 初始化的第一次排序 =================
    cudaMemset(d_num_active, 0, sizeof(int));
    collectAndScoreKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_active, d_row_offsets, num_nodes, max_outdegree, 
        d_nodes_in, d_scores_in, d_num_active
    );
    // 使用 CUB 排序，输入在 _in，输出在 _out
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        d_scores_in, d_scores_out, d_nodes_in, d_nodes_out, num_nodes);
        
    int topN = std::max(1, (int)(num_nodes * 0.1f));
    // 注意：传给 markCriticalKernel 的是排序后的 d_nodes_out
    markCriticalKernel<<<num_blocks, BLOCK_SIZE>>>(d_active, d_nodes_out, num_nodes, topN);

    // ------------------- 主循环 -------------------
    while(iter < 1000) {
        iter++;
        int cur = pingpong;
        int prev = 1 - pingpong;

        // 1. 启动计算
        bfsPullDualKernel<<<num_blocks, BLOCK_SIZE, 0, stream[cur]>>>(
            d_values, d_row_offsets, d_column_indices,
            d_column_offsets, d_row_indices,
            d_active, d_update, num_nodes, 
            d_delta[cur], d_info[cur], TREND_DEC
        );

        // 2. 状态切换
        cudaMemcpyAsync(d_active, d_update, num_nodes * sizeof(int), cudaMemcpyDeviceToDevice, stream[cur]);
        cudaMemsetAsync(d_update, -1, num_nodes * sizeof(int), stream[cur]);
        
        // 3. 异步回传检测数据
        cudaMemcpyAsync(h_delta[cur], d_delta[cur], num_nodes * sizeof(int), cudaMemcpyDeviceToHost, stream[cur]);
        cudaMemcpyAsync(h_info[cur], d_info[cur], sizeof(MonotonicInfo), cudaMemcpyDeviceToHost, stream[cur]);
        cudaEventRecord(copy_delta_event[cur], stream[cur]);

        // CPU 错误检测 (利用上一轮数据)
        if(iter > 1){
            if (cudaEventQuery(copy_delta_event[prev]) == cudaSuccess) {
                long long sum_delta = 0;
                int count_update = 0;
                for (int i = 0; i < num_nodes; i++) {
                    int dv = h_delta[prev][i];
                    if (dv != 0) {
                        sum_delta += (long long)dv;
                        count_update++;
                    }
                }
                float avg_delta = 0.0f;
                if (count_update > 0) {
                    avg_delta = fabsf((float)sum_delta) / (float)count_update;
                    pre_avg_delta = avg_delta;
                }
            }
       }
    
        // ================= GPU 内部计算活跃数与排序 =================
        cudaStreamSynchronize(stream[cur]); 

        cudaMemsetAsync(d_num_active, 0, sizeof(int), stream[cur]);
        collectAndScoreKernel<<<num_blocks, BLOCK_SIZE, 0, stream[cur]>>>(
            d_active, d_row_offsets, num_nodes, max_outdegree, 
            d_nodes_in, d_scores_in, d_num_active
        );

        // 获取活跃节点数量，如果为 0 直接退出
        cudaMemcpyAsync(&active_nodes, d_num_active, sizeof(int), cudaMemcpyDeviceToHost, stream[cur]);
        cudaStreamSynchronize(stream[cur]); 
        if(active_nodes == 0) break;

        // CUB 高性能异步排序：完全没有内存分配开销，使用我们循环前预先申请的 d_temp_storage
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_scores_in, d_scores_out, 
            d_nodes_in, d_nodes_out, 
            active_nodes, 
            0, sizeof(float)*8, stream[cur]
        );

        topN = std::max(1, (int)(active_nodes * 0.1f));
        int active_blocks = (active_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        markCriticalKernel<<<active_blocks, BLOCK_SIZE, 0, stream[cur]>>>(d_active, d_nodes_out, active_nodes, topN);

        pingpong = prev;
    }

    cudaMemcpy(h_value, d_values, num_nodes*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU time: %.4f ms\n", ms);

    // ================= 释放资源 =================
    cudaFree(d_values); cudaFree(d_row_offsets); cudaFree(d_column_indices);
    cudaFree(d_column_offsets); cudaFree(d_row_indices);
    cudaFree(d_active); cudaFree(d_update); 
    cudaFree(d_nodes_in); cudaFree(d_nodes_out); 
    cudaFree(d_scores_in); cudaFree(d_scores_out); 
    cudaFree(d_num_active); cudaFree(d_temp_storage);
    
    for (int i = 0; i < 2; i++) {
        cudaFree(d_delta[i]);
        cudaFree(d_info[i]);
    }
    cudaFreeHost(h_active); 
    cudaFreeHost(h_delta[0]); cudaFreeHost(h_delta[1]);
    cudaFreeHost(h_info[0]); cudaFreeHost(h_info[1]);

    cudaEventDestroy(copy_delta_event[0]);
    cudaEventDestroy(copy_delta_event[1]);
    cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);

    printf("GPU BFS finished in %d iterations.\n", iter);
}