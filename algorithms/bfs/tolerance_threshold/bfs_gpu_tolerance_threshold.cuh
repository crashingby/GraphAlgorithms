#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <unistd.h>
#include <nvToolsExt.h>

#define BLOCK_SIZE 256
#define INF 100000  
#define GPU_DEVICE 0
#define CHECK_BUFFER_COUNT 64

// ==================== 结构体 ====================
enum ValueTrend {
    TREND_NONE = 0,  // 不检测
    TREND_INC  = 1,  // 单调递增
    TREND_DEC  = 2   // 单调递减
};

struct MonotonicInfo {
    int dmr_error_flag;       // 冗余计算不一致标志（设备端写入）
    int monotonic_error_flag; // 单调性违反标志（设备端写入）
    unsigned long long sum_abs_delta;
    int count_update;
};

struct AsyncCheckResult {
    bool done = false;
    unsigned long long sum_abs_delta = 0;
    int count_update = 0;
    int dmr_error = 0;
    int monotonic_error = 0;
};

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    explicit NvtxRange(const std::string& name) : name_(name) { nvtxRangePushA(name_.c_str()); }
    ~NvtxRange() { nvtxRangePop(); }

    std::string name_;
};

inline std::string nvtx_name(const char* label, int iter) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s iter=%d gpu=%d", label, iter, GPU_DEVICE);
    return std::string(buf);
}

// ==================== Kernel: 计算得分并根据阈值标记关键顶点 ====================
// 替换了原来的 collectAndScoreKernel 和 markCriticalKernel
__global__ void scoreAndMarkKernel(
    int* d_active, 
    const int* d_row_offsets, 
    int num_nodes, 
    int max_outdegree,
    float alpha,
    float beta,
    float threshold,
    int* d_num_active) 
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        int val = d_active[v];
        if (val != -1) {
            // 统计当前活跃节点数
            atomicAdd(d_num_active, 1);
            
            // 计算当前活跃节点得分
            int outdeg = d_row_offsets[v + 1] - d_row_offsets[v];
            float dist_score = 1.0f / (1.0f + (float)val);
            float score = alpha * ((float)outdeg / (float)max_outdegree) + beta * dist_score;
            
            // 根据阈值判定是否为关键节点 (2 为关键，1 为普通活跃)
            if (score >= threshold) {
                d_active[v] = 2;
            } else {
                d_active[v] = 1;
            }
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
            unsigned long long abs_delta = (delta < 0)
                ? (unsigned long long)(-(long long)delta)
                : (unsigned long long)delta;
            atomicAdd(&(d_info->sum_abs_delta), abs_delta);
            atomicAdd(&(d_info->count_update), 1);

            // 记录单调性违反
            if ((trend == TREND_INC && delta < 0) || (trend == TREND_DEC && delta > 0)) {
                atomicExch(&(d_info->monotonic_error_flag), 1);
            }
        }
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

// ==================== 主函数：全GPU容错版 (阈值判定版) ====================
void bfsGPU(
    int* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int num_edges,
    int src,
    float alpha,      // 新增：出度得分权重
    float beta,       // 新增：距离得分权重
    float threshold   // 新增：关键节点得分阈值
) {
    printf("开始 GPU BFS（阈值判定容错实现）...\n"); fflush(stdout);
    printf("配置参数 -> Alpha: %.2f | Beta: %.2f | Threshold: %.2f\n", alpha, beta, threshold);

    int max_outdegree = compute_max_outdegree(h_row_offsets, num_nodes);
    
    // ------------------- 分配 Host 内存 -------------------
    int* h_active;
    MonotonicInfo* h_info[CHECK_BUFFER_COUNT];

    cudaMallocHost(&h_active, num_nodes * sizeof(int));
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        cudaMallocHost(&h_info[i], sizeof(MonotonicInfo));
    }
    
    for(int i = 0; i < num_nodes; i++){
        h_active[i] = 1;      
        h_value[i] = INF;
    }
    h_value[src] = 0;

    // ------------------- 分配 Device 内存 -------------------
    int *d_values, *d_row_offsets, *d_column_indices;
    int *d_column_offsets, *d_row_indices, *d_active, *d_update;
    int *d_num_active;
    MonotonicInfo* d_info[CHECK_BUFFER_COUNT];
    MonotonicInfo* d_info_scratch;

    cudaMalloc(&d_values, num_nodes * sizeof(int));
    cudaMalloc(&d_row_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_column_indices, num_edges * sizeof(int));
    cudaMalloc(&d_column_offsets, (num_nodes+1) * sizeof(int));
    cudaMalloc(&d_row_indices, num_edges * sizeof(int));
    cudaMalloc(&d_active, num_nodes * sizeof(int));
    cudaMalloc(&d_update, num_nodes * sizeof(int));
    cudaMalloc(&d_num_active, sizeof(int));
    
    for(int i = 0; i < CHECK_BUFFER_COUNT; i++){
        cudaMalloc(&d_info[i], sizeof(MonotonicInfo));
    }
    cudaMalloc(&d_info_scratch, sizeof(MonotonicInfo));

    cudaMemcpy(d_values, h_value, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_column_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_offsets, h_column_offsets, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, h_row_indices, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active, h_active, num_nodes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_update, -1, num_nodes*sizeof(int));

    cudaStream_t compute_stream, check_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreateWithFlags(&check_stream, cudaStreamNonBlocking);

    cudaEvent_t compute_done_event[CHECK_BUFFER_COUNT], check_done_event[CHECK_BUFFER_COUNT], start, stop;
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        cudaEventCreateWithFlags(&compute_done_event[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&check_done_event[i], cudaEventDisableTiming);
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int iter = 0;
    std::array<std::atomic<int>, CHECK_BUFFER_COUNT> pending_check;
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        pending_check[i].store(0);
    }
    std::atomic<bool> check_stop(false);
    std::vector<AsyncCheckResult> check_results(1001);
    bool detected_dmr = false;
    bool detected_monotonic = false;
    bool detected_avg_increase = false;
    int first_dmr_iter = -1;
    int first_monotonic_iter = -1;
    int first_avg_increase_iter = -1;
    float pre_avg_delta = (float)INF;

    std::thread check_worker([&]() {
        cudaSetDevice(GPU_DEVICE);
        while (!check_stop.load(std::memory_order_relaxed)) {
            bool progressed = false;
            for (int buf = 0; buf < CHECK_BUFFER_COUNT; ++buf) {
                int iter_id = pending_check[buf].load(std::memory_order_acquire);
                if (iter_id <= 0) continue;

                if (cudaEventQuery(check_done_event[buf]) == cudaSuccess) {
                    NvtxRange scan_range(nvtx_name("worker CPU scan", iter_id));

                    AsyncCheckResult result;
                    result.done = true;
                    result.dmr_error = h_info[buf]->dmr_error_flag;
                    result.monotonic_error = h_info[buf]->monotonic_error_flag;
                    result.sum_abs_delta = h_info[buf]->sum_abs_delta;
                    result.count_update = h_info[buf]->count_update;

                    if (result.count_update > 0) {
                        float avg_delta = (float)result.sum_abs_delta / (float)result.count_update;
                        if (pre_avg_delta < (float)INF && avg_delta > pre_avg_delta && first_avg_increase_iter < 0) {
                            detected_avg_increase = true;
                            first_avg_increase_iter = iter_id;
                        }
                        pre_avg_delta = avg_delta;
                    }
                    if (result.dmr_error && first_dmr_iter < 0) {
                        detected_dmr = true;
                        first_dmr_iter = iter_id;
                    }
                    if (result.monotonic_error && first_monotonic_iter < 0) {
                        detected_monotonic = true;
                        first_monotonic_iter = iter_id;
                    }

                    if (iter_id >= 0 && iter_id < (int)check_results.size()) {
                        check_results[iter_id] = result;
                    }
                    pending_check[buf].store(0, std::memory_order_release);
                    progressed = true;
                }
            }

            if (!progressed) {
                std::this_thread::yield();
            }
        }
    });

    int active_nodes = num_nodes;
    int num_blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ================= 初始化的第一次评分与标记 =================
    cudaMemset(d_num_active, 0, sizeof(int));
    scoreAndMarkKernel<<<num_blocks, BLOCK_SIZE, 0, compute_stream>>>(
        d_active, d_row_offsets, num_nodes, max_outdegree,
        alpha, beta, threshold, d_num_active
    );

    // ------------------- 主循环 -------------------
    while(iter < 1000) {
        iter++;
        {
            NvtxRange enqueue_range(nvtx_name("main enqueue bfs/check", iter));

            int check_buf = iter % CHECK_BUFFER_COUNT;
            int expected = 0;
            bool do_async_check = pending_check[check_buf].compare_exchange_strong(
                expected, -1, std::memory_order_acq_rel);
            MonotonicInfo* iter_info = do_async_check ? d_info[check_buf] : d_info_scratch;

            cudaMemsetAsync(iter_info, 0, sizeof(MonotonicInfo), compute_stream);

            // 1. 启动计算
            bfsPullDualKernel<<<num_blocks, BLOCK_SIZE, 0, compute_stream>>>(
                d_values, d_row_offsets, d_column_indices,
                d_column_offsets, d_row_indices,
                d_active, d_update, num_nodes,
                iter_info, TREND_DEC
            );

            if (do_async_check) {
                NvtxRange copy_enqueue_range(nvtx_name("main enqueue check_stream D2H", iter));
                cudaEventRecord(compute_done_event[check_buf], compute_stream);
                cudaStreamWaitEvent(check_stream, compute_done_event[check_buf], 0);
                cudaMemcpyAsync(h_info[check_buf], iter_info, sizeof(MonotonicInfo), cudaMemcpyDeviceToHost, check_stream);
                cudaEventRecord(check_done_event[check_buf], check_stream);
                pending_check[check_buf].store(iter, std::memory_order_release);
            }

            // 2. 状态切换
            cudaMemcpyAsync(d_active, d_update, num_nodes * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream);
            cudaMemsetAsync(d_update, -1, num_nodes * sizeof(int), compute_stream);
        }

        // ================= GPU 内部计算活跃数与基于阈值的关键节点筛选 =================
        {
            NvtxRange sync_range(nvtx_name("main compute sync", iter));
            cudaStreamSynchronize(compute_stream);
        }

        {
            NvtxRange score_range(nvtx_name("main score active", iter));
            cudaMemsetAsync(d_num_active, 0, sizeof(int), compute_stream);
            scoreAndMarkKernel<<<num_blocks, BLOCK_SIZE, 0, compute_stream>>>(
                d_active, d_row_offsets, num_nodes, max_outdegree,
                alpha, beta, threshold, d_num_active
            );
        }

        // 获取活跃节点数量，如果为 0 直接退出
        {
            NvtxRange count_range(nvtx_name("main active count sync", iter));
            cudaMemcpyAsync(&active_nodes, d_num_active, sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
            cudaStreamSynchronize(compute_stream);
        }
        if(active_nodes == 0) break;
    }

    bool waiting_checks = true;
    while (waiting_checks) {
        waiting_checks = false;
        for (int buf = 0; buf < CHECK_BUFFER_COUNT; ++buf) {
            if (pending_check[buf].load(std::memory_order_acquire) != 0) waiting_checks = true;
        }
        if (waiting_checks) std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    check_stop.store(true, std::memory_order_release);
    check_worker.join();

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
    cudaFree(d_num_active); 
    
    for (int i = 0; i < CHECK_BUFFER_COUNT; i++) {
        cudaFree(d_info[i]);
    }
    cudaFree(d_info_scratch);
    cudaFreeHost(h_active);
    for (int i = 0; i < CHECK_BUFFER_COUNT; i++) {
        cudaFreeHost(h_info[i]);
    }

    for (int i = 0; i < CHECK_BUFFER_COUNT; i++) {
        cudaEventDestroy(compute_done_event[i]);
        cudaEventDestroy(check_done_event[i]);
    }
    cudaStreamDestroy(check_stream);
    cudaStreamDestroy(compute_stream);

    printf("GPU BFS finished in %d iterations.\n", iter);
    printf("异步检测标记: DMR=%d", detected_dmr ? 1 : 0);
    if (first_dmr_iter >= 0) printf("(first_iter=%d)", first_dmr_iter);
    printf(", monotonic=%d", detected_monotonic ? 1 : 0);
    if (first_monotonic_iter >= 0) printf("(first_iter=%d)", first_monotonic_iter);
    printf(", avg_delta_increase=%d", detected_avg_increase ? 1 : 0);
    if (first_avg_increase_iter >= 0) printf("(first_iter=%d)", first_avg_increase_iter);
    printf("\n");
}