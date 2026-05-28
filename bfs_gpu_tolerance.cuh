#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <unistd.h>

#define BLOCK_SIZE 256
#define INF 100000  

// ==================== 结构体 ====================
enum ValueTrend {
    TREND_NONE = 0,  // 不检测
    TREND_INC  = 1,  // 单调递增
    TREND_DEC  = 2   // 单调递减
};

struct MonotonicInfo {
int dmr_error_flag; // 冗余计算不一致标志（设备端写入）
int monotonic_error_flag; // 单调性违反标志（设备端写入）
};

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
        if (cid < BLOCK_SIZE) critical_list[cid] = tid;
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
        if(my_idle_id < critical_count && my_idle_id < BLOCK_SIZE){
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
        if(my_index >= 0 && my_index < idle_count && my_index < BLOCK_SIZE){
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

// ==================== 辅助：统计活跃顶点数（CPU） ====================
int count_active_nodes(int* h_active, int num_nodes){
    int cnt = 0;
    for(int i=0;i<num_nodes;i++) if(h_active[i]!=-1) cnt++;
    return cnt;
}


// ==================== 挑选关键顶点（CPU）====================
void select_critical_vertices(int* h_active, const int* h_row_offsets, int num_nodes, int max_outdegree) {
    std::vector<std::pair<int,float>> scores; // <vertex_id, score>
    for(int v=0; v<num_nodes; v++){
        if(h_active[v] != -1){
            int value = h_active[v];
            h_active[v] = 1;
            int outdeg = h_row_offsets[v+1] - h_row_offsets[v];
            float dist_score = 1.0f / (1.0f + value);
            float score = 0.5f * (float)outdeg / max_outdegree + 0.5f * dist_score;
            scores.emplace_back(v, score);
        }
    }
    // 挑选评分制最高的前10%活跃顶点作为关键顶点
    std::sort(scores.begin(), scores.end(), [](auto& a, auto& b){ return a.second > b.second; });  
    int topN = std::max(1, (int)(scores.size() * 0.1f));
    for (int i = 0; i < topN; i++) h_active[scores[i].first] = 2;
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



// ==================== 主函数：实现真正的 CPU–GPU 异步 ====================
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
    printf("开始 GPU BFS（容错实现）...\n"); fflush(stdout);

    int max_outdegree = compute_max_outdegree(h_row_offsets, num_nodes);
    
    // ------------------- 分配 host（page-locked，便于异步拷贝） -------------------
    int* h_active; 
    int* h_delta[2]; 
    MonotonicInfo* h_info[2]; 

    cudaMallocHost(&h_active, num_nodes * sizeof(int));
    cudaMallocHost(&h_delta[0], num_nodes * sizeof(int));
    cudaMallocHost(&h_delta[1], num_nodes * sizeof(int));
    cudaMallocHost(&h_info[0], sizeof(MonotonicInfo));
    cudaMallocHost(&h_info[1], sizeof(MonotonicInfo));
    
    // 初始化 host 数据
    for(int i = 0; i < num_nodes; i++){
        h_active[i] = 1;      // 第一轮所有顶点活跃
        h_value[i] = INF;
    }
    h_value[src] = 0;

    // ------------------- 分配 device 内存 -------------------
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
        cudaMalloc(&d_delta[i],num_nodes*sizeof(int));
        cudaMallocManaged(&d_info[i],sizeof(MonotonicInfo));
    }
    // 复制图结构到 device（一次性）
    cudaMemcpy(d_values, h_value, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_column_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_offsets, h_column_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, h_row_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    
    // 初始化 device active（第一轮全部活跃）
    cudaMemcpy(d_active, h_active, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_update, -1, num_nodes*sizeof(int));

    // ------------------- 创建两个 stream 与事件 -------------------
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    // kernel 完成事件；active 更新完成事件；host 拷贝完成事件
    cudaEvent_t copy_delta_event[2], copy_active_event;
    cudaEventCreateWithFlags(&copy_delta_event[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&copy_delta_event[1], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&copy_active_event, cudaEventDisableTiming);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ------------------- BFS 迭代 -------------------
    int iter = 0;
    int pingpong = 0;
    float pre_avg_delta = INF;
    int active_nodes = num_nodes;
    // ------------------- 初始：根据初始 h_active 选出 critical(0)，并拷贝到 device -------------------
    select_critical_vertices(h_active, h_row_offsets, num_nodes, max_outdegree);
    // 将 critical 传到 device，为迭代0的 kernel 准备
    cudaMemcpyAsync(d_active, h_active, num_nodes * sizeof(int), cudaMemcpyHostToDevice, stream[0]);

    // ------------------- 主循环 -------------------
    while(iter < 1000) {
        iter++;
        //printf("[DEBUG] iter=%d, active_nodes=%d\n", iter, active_nodes);
        
        int cur = pingpong;
        int prev = 1 - pingpong;

        // 启动第 cur 轮 kernel（依赖 d_critical[cur]，已经在上一步准备）
        cudaMemsetAsync(d_info[cur], 0, sizeof(MonotonicInfo), stream[cur]);
        int num_blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfsPullDualKernel<<<num_blocks, BLOCK_SIZE, 0, stream[cur]>>>(
            d_values,d_row_offsets,d_column_indices,
            d_column_offsets,d_row_indices,
            d_active,d_update,num_nodes, 
            d_delta[cur], d_info[cur], TREND_DEC
        );
        // kernel 完成后用 d_update 更新 d_active（device->device），并清空 d_update，用于下一轮启动前进行关键顶点计算
        cudaMemcpyAsync(d_active, d_update, num_nodes * sizeof(int), cudaMemcpyDeviceToDevice, stream[cur]);
        cudaMemsetAsync(d_update, -1, num_nodes * sizeof(int), stream[cur]);
        cudaEventRecord(copy_active_event, stream[cur]);   

        // kernel 完成后立即异步拷贝 delta/info，供下一轮GPU计算时过程中CPU检测
        cudaMemcpyAsync(h_delta[cur], d_delta[cur], num_nodes * sizeof(int), cudaMemcpyDeviceToHost, stream[cur]);
        cudaMemcpyAsync(h_info[cur], d_info[cur], sizeof(MonotonicInfo), cudaMemcpyDeviceToHost, stream[cur]);
        cudaEventRecord(copy_delta_event[cur], stream[cur]);   // 记录拷贝完成事件

        // -------------------  CPU：异步检查上一轮 DMR/单调性 -------------------
        if(iter > 1){
            // 上一轮的 kernel 肯定已完成, 则我们可以直接拿到 delta/info
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
                // 检查 avg_delta 与上一轮比较
                float avg_delta = 0.0f;
                if (count_update > 0) {
                    avg_delta = fabsf((float)sum_delta) / (float)count_update;
                    if (avg_delta > pre_avg_delta) {
                        //printf("平均变化幅度上升：prev=%.6f now=%.6f（iter %d），需要重启。", pre_avg_delta, avg_delta, iter - 1);
                    }
                    pre_avg_delta = avg_delta;
                }
                // 检查单调性/DMR，如果触发选择重启
                if (h_info[prev]->dmr_error_flag || h_info[prev]->monotonic_error_flag) {
                    //printf("检测到 DMR/单调性错误（iter %d），标记需要重启。", iter - 1);
                }
            }
       }
    
        // ------------------- CPU：同步获取下一轮活跃顶点 -------------------
        cudaEventSynchronize(copy_active_event); // 只等待 active 更新完成
        cudaMemcpy(h_active, d_active, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        active_nodes = count_active_nodes(h_active, num_nodes);
        if(active_nodes == 0) break;
        
        select_critical_vertices(h_active, h_row_offsets, num_nodes, max_outdegree);
        cudaMemcpy(d_active, h_active, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
        
        // 切换 pingpong
        pingpong = prev;
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
    for (int i = 0; i < 2; i++) {
        cudaFree(d_delta[i]);
        cudaFree(d_info[i]);
    }

    cudaFreeHost(h_active); 
    cudaFreeHost(h_delta[0]); cudaFreeHost(h_delta[1]);
    cudaFreeHost(h_info[0]); cudaFreeHost(h_info[1]);


    cudaEventDestroy(copy_delta_event[0]);
    cudaEventDestroy(copy_delta_event[1]);
    cudaEventDestroy(copy_active_event);

    cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);

    printf("GPU BFS finished in %d iterations.\n", iter);
}
