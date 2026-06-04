#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include "include/spsc_queue.h"

#define BLOCK_SIZE 256
#define INF 100000
#define GPU_DEVICE 0
#define CHECK_BUFFER_COUNT 16

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#endif

enum ValueTrend {
    TREND_NONE = 0,
    TREND_INC = 1,
    TREND_DEC = 2
};

struct CheckTask {
    int iter = 0;
    int buf = -1;
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

inline int compute_max_outdegree(const int* h_row_offsets, int num_nodes) {
    int max_outdegree = 1;
    for (int v = 0; v < num_nodes; ++v) {
        int outdeg = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdeg > max_outdegree) max_outdegree = outdeg;
    }
    return max_outdegree;
}

__global__ void scoreAndMarkIntKernel(
    int* d_active,
    const int* d_values,
    const int* d_row_offsets,
    int num_nodes,
    int max_outdegree,
    float alpha,
    float beta,
    float threshold,
    int* d_num_active
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes && d_active[v] != -1) {
        atomicAdd(d_num_active, 1);
        int outdeg = d_row_offsets[v + 1] - d_row_offsets[v];
        int safe_max = max_outdegree > 0 ? max_outdegree : 1;
        float value_score = 1.0f / (1.0f + fabsf((float)d_values[v]));
        float score = alpha * ((float)outdeg / (float)safe_max) + beta * value_score;
        d_active[v] = (score >= threshold) ? 2 : 1;
    }
}

struct MonotonicInfo {
    int dmr_error_flag;
    int monotonic_error_flag;
    int count_update;
};

struct AsyncCheckResult {
    bool done = false;
    long long sum_delta = 0;
    int count_update = 0;
    int dmr_error = 0;
    int monotonic_error = 0;
};


__global__ void ccPullDualKernel(
    int* d_values,
    const int* d_row_offsets,
    const int* d_column_indices,
    const int* d_column_offsets,
    const int* d_row_indices,
    const int* d_active,
    int* d_update,
    int num_nodes,
    int* d_compact_delta,
    MonotonicInfo* d_info,
    ValueTrend trend
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int critical_list[BLOCK_SIZE];
    __shared__ int idle_count;
    __shared__ int critical_count;
    __shared__ int redundant_results[BLOCK_SIZE];

    if (local_tid == 0) { idle_count = 0; critical_count = 0; }
    __syncthreads();
    bool valid = tid < num_nodes;
    bool active = valid && d_active[tid] != -1;
    bool critical = active && d_active[tid] == 2;
    bool idle = !active || !valid;
    int cid = -1;
    int iid = -1;
    if (critical) { cid = atomicAdd(&critical_count, 1); if (cid < BLOCK_SIZE) critical_list[cid] = tid; }
    if (idle) iid = atomicAdd(&idle_count, 1);
    __syncthreads();

    int oldVal = valid ? d_values[tid] : 0;
    int mainVal = oldVal;
    if (active) {
        for (int i = d_column_offsets[tid]; i < d_column_offsets[tid + 1]; ++i) {
            int candidate = d_values[d_row_indices[i]];
            if (candidate > mainVal) mainVal = candidate;
        }
    }
    __syncthreads();

    if (idle && iid < critical_count && iid < BLOCK_SIZE) {
        int target = critical_list[iid];
        int r = d_values[target];
        for (int i = d_column_offsets[target]; i < d_column_offsets[target + 1]; ++i) {
            int candidate = d_values[d_row_indices[i]];
            if (candidate > r) r = candidate;
        }
        redundant_results[iid] = r;
    }
    __syncthreads();
    if (critical && cid >= 0 && cid < idle_count && cid < BLOCK_SIZE && redundant_results[cid] != mainVal) {
        atomicExch(&(d_info->dmr_error_flag), 1);
    }
    __syncthreads();

    if (active) {
        d_values[tid] = mainVal;
        if (mainVal > oldVal) {
            for (int i = d_row_offsets[tid]; i < d_row_offsets[tid + 1]; ++i) {
                d_update[d_column_indices[i]] = 1;
            }
            int delta = mainVal - oldVal;
            int delta_idx = atomicAdd(&(d_info->count_update), 1);
            d_compact_delta[delta_idx] = delta;
            if ((trend == TREND_INC && delta < 0) || (trend == TREND_DEC && delta > 0)) {
                atomicExch(&(d_info->monotonic_error_flag), 1);
            }
        }
    }
}

void ccGPU(
    int* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int num_edges,
    float alpha,
    float beta,
    float threshold
) {
    printf("开始 GPU CC（紧凑delta队列检测容错实现）...\n");
    printf("配置参数 -> Alpha: %.2f | Beta: %.2f | Threshold: %.2f\n", alpha, beta, threshold);

    int max_outdegree = compute_max_outdegree(h_row_offsets, num_nodes);
    int* h_active;
    int* h_compact_delta[CHECK_BUFFER_COUNT];
    MonotonicInfo* h_info[CHECK_BUFFER_COUNT];
    CUDA_CHECK(cudaMallocHost(&h_active, num_nodes * sizeof(int)));
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        CUDA_CHECK(cudaMallocHost(&h_compact_delta[i], num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_info[i], sizeof(MonotonicInfo)));
    }
    for (int i = 0; i < num_nodes; ++i) { h_active[i] = 1; h_value[i] = i; }

    int *d_values, *d_ro, *d_ci, *d_co, *d_ri, *d_active, *d_update, *d_num_active;
    int* d_compact_delta[CHECK_BUFFER_COUNT];
    int* d_compact_delta_scratch;
    MonotonicInfo* d_info[CHECK_BUFFER_COUNT];
    MonotonicInfo* d_info_scratch;
    CUDA_CHECK(cudaMalloc(&d_values, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ro, (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_co, (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ri, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_active, sizeof(int)));
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        CUDA_CHECK(cudaMalloc(&d_compact_delta[i], num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info[i], sizeof(MonotonicInfo)));
    }
    CUDA_CHECK(cudaMalloc(&d_compact_delta_scratch, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info_scratch, sizeof(MonotonicInfo)));

    CUDA_CHECK(cudaMemcpy(d_values, h_value, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ro, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, h_column_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_co, h_column_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ri, h_row_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, h_active, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_update, 0xff, num_nodes * sizeof(int)));

    cudaStream_t compute_stream, check_stream, delta_copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(&check_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&delta_copy_stream, cudaStreamNonBlocking));
    cudaEvent_t compute_done_event[CHECK_BUFFER_COUNT], check_done_event[CHECK_BUFFER_COUNT], start, stop;
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_event[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&check_done_event[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    SpscQueue<int, CHECK_BUFFER_COUNT> free_buffers;
    SpscQueue<CheckTask, CHECK_BUFFER_COUNT> pending_tasks;
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) free_buffers.try_push(i);

    std::atomic<bool> check_stop(false);
    std::vector<AsyncCheckResult> check_results(1001);
    bool detected_dmr = false, detected_monotonic = false, detected_avg_increase = false;
    int first_dmr_iter = -1, first_monotonic_iter = -1, first_avg_increase_iter = -1;
    float pre_avg_delta = (float)INF;

    std::thread check_worker([&]() {
        cudaSetDevice(GPU_DEVICE);
        while (!check_stop.load(std::memory_order_relaxed) || !pending_tasks.empty()) {
            CheckTask task;
            bool progressed = false;
            if (pending_tasks.peek(task) && cudaEventQuery(check_done_event[task.buf]) == cudaSuccess) {
                pending_tasks.try_pop(task);
                NvtxRange scan_range(nvtx_name("worker CPU scan", task.iter));
                AsyncCheckResult result;
                result.done = true;
                result.dmr_error = h_info[task.buf]->dmr_error_flag;
                result.monotonic_error = h_info[task.buf]->monotonic_error_flag;
                result.count_update = h_info[task.buf]->count_update;
                if (result.count_update > 0) {
                    {
                        NvtxRange copy_range(nvtx_name("worker D2H compact delta", task.iter));
                        CUDA_CHECK(cudaMemcpyAsync(
                            h_compact_delta[task.buf], d_compact_delta[task.buf],
                            result.count_update * sizeof(int), cudaMemcpyDeviceToHost, delta_copy_stream
                        ));
                        CUDA_CHECK(cudaStreamSynchronize(delta_copy_stream));
                    }
                    {
                        NvtxRange scan_compact_range(nvtx_name("worker CPU scan compact delta", task.iter));
                        for (int i = 0; i < result.count_update; ++i) {
                            result.sum_delta += (long long)h_compact_delta[task.buf][i];
                        }
                    }
                    float avg_delta = fabsf((float)result.sum_delta) / (float)result.count_update;
                    if (pre_avg_delta < (float)INF && avg_delta > pre_avg_delta && first_avg_increase_iter < 0) {
                        detected_avg_increase = true;
                        first_avg_increase_iter = task.iter;
                    }
                    pre_avg_delta = avg_delta;
                }
                if (result.dmr_error && first_dmr_iter < 0) { detected_dmr = true; first_dmr_iter = task.iter; }
                if (result.monotonic_error && first_monotonic_iter < 0) { detected_monotonic = true; first_monotonic_iter = task.iter; }
                if (task.iter >= 0 && task.iter < (int)check_results.size()) check_results[task.iter] = result;
                free_buffers.try_push(task.buf);
                progressed = true;
            }
            if (!progressed) std::this_thread::yield();
        }
    });

    int blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iter = 0;
    int active_nodes = num_nodes;
    CUDA_CHECK(cudaMemsetAsync(d_num_active, 0, sizeof(int), compute_stream));
    scoreAndMarkIntKernel<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(
        d_active, d_values, d_ro, num_nodes, max_outdegree, alpha, beta, threshold, d_num_active
    );

    while (iter < 1000) {
        ++iter;
        {
            NvtxRange enqueue_range(nvtx_name("main enqueue cc/check", iter));
            int check_buf = -1;
            bool do_async_check = free_buffers.try_pop(check_buf);
            int* iter_compact_delta = do_async_check ? d_compact_delta[check_buf] : d_compact_delta_scratch;
            MonotonicInfo* iter_info = do_async_check ? d_info[check_buf] : d_info_scratch;
            CUDA_CHECK(cudaMemsetAsync(iter_info, 0, sizeof(MonotonicInfo), compute_stream));
            ccPullDualKernel<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(
                d_values, d_ro, d_ci, d_co, d_ri, d_active, d_update, num_nodes,
                iter_compact_delta, iter_info, TREND_INC
            );
            if (do_async_check) {
                NvtxRange copy_range(nvtx_name("main enqueue check_stream D2H", iter));
                CUDA_CHECK(cudaEventRecord(compute_done_event[check_buf], compute_stream));
                CUDA_CHECK(cudaStreamWaitEvent(check_stream, compute_done_event[check_buf], 0));
                CUDA_CHECK(cudaMemcpyAsync(h_info[check_buf], iter_info, sizeof(MonotonicInfo), cudaMemcpyDeviceToHost, check_stream));
                CUDA_CHECK(cudaEventRecord(check_done_event[check_buf], check_stream));
                if (!pending_tasks.try_push({iter, check_buf})) {
                    fprintf(stderr, "检测任务队列已满，跳过 iter=%d 的 CPU 检测。\n", iter);
                    CUDA_CHECK(cudaEventSynchronize(check_done_event[check_buf]));
                    free_buffers.try_push(check_buf);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(d_active, d_update, num_nodes * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream));
            CUDA_CHECK(cudaMemsetAsync(d_update, 0xff, num_nodes * sizeof(int), compute_stream));
        }
        { NvtxRange sync_range(nvtx_name("main compute sync", iter)); CUDA_CHECK(cudaStreamSynchronize(compute_stream)); }
        {
            NvtxRange score_range(nvtx_name("main score active", iter));
            CUDA_CHECK(cudaMemsetAsync(d_num_active, 0, sizeof(int), compute_stream));
            scoreAndMarkIntKernel<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(
                d_active, d_values, d_ro, num_nodes, max_outdegree, alpha, beta, threshold, d_num_active
            );
        }
        {
            NvtxRange count_range(nvtx_name("main active count sync", iter));
            CUDA_CHECK(cudaMemcpyAsync(&active_nodes, d_num_active, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        }
        if (active_nodes == 0) break;
    }

    while (!pending_tasks.empty()) std::this_thread::yield();
    check_stop.store(true, std::memory_order_release);
    check_worker.join();
    CUDA_CHECK(cudaMemcpy(h_value, d_values, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU time: %.4f ms\n", ms);

    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_ro)); CUDA_CHECK(cudaFree(d_ci));
    CUDA_CHECK(cudaFree(d_co)); CUDA_CHECK(cudaFree(d_ri)); CUDA_CHECK(cudaFree(d_active));
    CUDA_CHECK(cudaFree(d_update)); CUDA_CHECK(cudaFree(d_num_active));
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        CUDA_CHECK(cudaFree(d_compact_delta[i])); CUDA_CHECK(cudaFree(d_info[i]));
        CUDA_CHECK(cudaFreeHost(h_compact_delta[i])); CUDA_CHECK(cudaFreeHost(h_info[i]));
        CUDA_CHECK(cudaEventDestroy(compute_done_event[i])); CUDA_CHECK(cudaEventDestroy(check_done_event[i]));
    }
    CUDA_CHECK(cudaFree(d_compact_delta_scratch)); CUDA_CHECK(cudaFree(d_info_scratch));
    CUDA_CHECK(cudaFreeHost(h_active));
    CUDA_CHECK(cudaStreamDestroy(delta_copy_stream)); CUDA_CHECK(cudaStreamDestroy(check_stream)); CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    printf("GPU CC finished in %d iterations.\n", iter);
    printf("异步检测标记: DMR=%d", detected_dmr ? 1 : 0);
    if (first_dmr_iter >= 0) printf("(first_iter=%d)", first_dmr_iter);
    printf(", monotonic=%d", detected_monotonic ? 1 : 0);
    if (first_monotonic_iter >= 0) printf("(first_iter=%d)", first_monotonic_iter);
    printf(", avg_delta_increase=%d", detected_avg_increase ? 1 : 0);
    if (first_avg_increase_iter >= 0) printf("(first_iter=%d)", first_avg_increase_iter);
    printf("\n");
}

