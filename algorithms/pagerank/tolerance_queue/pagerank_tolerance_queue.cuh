/**
 * @file pagerank_tolerance_queue.cuh
 * @brief Single-GPU PageRank with selective DMR and asynchronous CPU checks.
 */
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include "include/spsc_queue.h"
#include "include/cuda_event_timer.cuh"

#define BLOCK_SIZE 256
#define INF 100000
#define GPU_DEVICE 0
#define CHECK_BUFFER_COUNT 64

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#endif

/** @brief Value-trend selector retained for the common detector vocabulary. */
enum ValueTrend {
    TREND_NONE = 0,
    TREND_INC = 1,
    TREND_DEC = 2
};

/** @brief Queue item binding an iteration to a reusable check buffer. */
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

/** @brief Return the maximum outgoing degree used to normalize criticality. */
inline int compute_max_outdegree(const int* h_row_offsets, int num_nodes) {
    int max_outdegree = 1;
    for (int v = 0; v < num_nodes; ++v) {
        int outdeg = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdeg > max_outdegree) max_outdegree = outdeg;
    }
    return max_outdegree;
}


/** @brief Count active vertices and classify ordinary versus critical work. */
__global__ void scoreAndMarkFloatKernel(
    int* d_active,
    const float* d_values,
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
        float value_score = fabsf(d_values[v]);
        float score = alpha * ((float)outdeg / (float)safe_max) + beta * value_score;
        d_active[v] = (score >= threshold) ? 2 : 1;
    }
}

#define PR_ALPHA 0.85f
#define PR_TOL 1e-3f

/** @brief Per-iteration DMR and residual summary produced on the device. */
struct MonotonicInfo {
    int dmr_error_flag;
    int monotonic_error_flag;
    float sum_abs_delta;
    int count_update;
};

/** @brief Host-side interpretation of one completed residual summary. */
struct AsyncCheckResult {
    bool done = false;
    double sum_abs_delta = 0.0;
    int count_update = 0;
    int dmr_error = 0;
    int monotonic_error = 0;
};

/**
 * @brief Update ranks and selectively duplicate critical pull calculations.
 * @note DMR mismatches are reported only; the primary result is retained.
 * @details Active and assigned redundant lanes select a compute vertex before
 * entering one common incoming-neighbor accumulation. Role divergence is
 * limited to target selection and result placement; only the primary lane
 * performs authoritative writeback.
 */
__global__ void pagerankPullDualKernel(
    float* d_values,
    const int* d_row_offsets,
    const int* d_column_indices,
    const int* d_column_offsets,
    const int* d_row_indices,
    const int* d_active,
    int* d_update,
    int num_nodes,
    MonotonicInfo* d_info
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int critical_list[BLOCK_SIZE];
    __shared__ int idle_count;
    __shared__ int critical_count;
    __shared__ float redundant_results[BLOCK_SIZE];

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

    float oldVal = valid ? d_values[tid] : 0.0f;
    const bool does_redundant_work =
        idle && iid >= 0 && iid < critical_count && iid < BLOCK_SIZE;
    int compute_vertex = -1;
    if (active) compute_vertex = tid;
    else if (does_redundant_work) compute_vertex = critical_list[iid];

    float computed_value = 0.0f;
    if (compute_vertex >= 0) {
        float sum = 0.0f;
        for (int i = d_column_offsets[compute_vertex];
             i < d_column_offsets[compute_vertex + 1]; ++i) {
            int neighbor = d_row_indices[i];
            int outDegree = d_row_offsets[neighbor + 1] - d_row_offsets[neighbor];
            if (outDegree > 0) sum += d_values[neighbor] / (float)outDegree;
        }
        computed_value = (1.0f - PR_ALPHA) + PR_ALPHA * sum;
    }

    float mainVal = oldVal;
    if (active) {
        mainVal = computed_value;
    } else if (does_redundant_work) {
        redundant_results[iid] = computed_value;
    }
    __syncthreads();
    if (critical && cid >= 0 && cid < idle_count && cid < BLOCK_SIZE && fabsf(redundant_results[cid] - mainVal) > 1e-5f) {
        atomicExch(&(d_info->dmr_error_flag), 1);
    }
    __syncthreads();

    if (active) {
        d_values[tid] = mainVal;
        float delta = fabsf(mainVal - oldVal);
        if (delta > 0.0f) {
            atomicAdd(&(d_info->sum_abs_delta), delta);
            atomicAdd(&(d_info->count_update), 1);
        }
        if (delta >= PR_TOL) {
            for (int i = d_row_offsets[tid]; i < d_row_offsets[tid + 1]; ++i) {
                d_update[d_column_indices[i]] = 1;
            }
        }
    }
}

/**
 * @brief Execute PageRank with asynchronous queue-based DMR/residual checks.
 *
 * A background CPU consumer reads completed pinned summaries and compares the
 * mean residual across iterations while the main thread continues GPU work.
 */
void pagerankGPU(
    float* h_value,
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
    printf("开始 GPU PageRank（队列异步检测容错实现）...\n");
    printf("配置参数 -> Alpha: %.2f | Beta: %.2f | Threshold: %.2f\n", alpha, beta, threshold);

    int max_outdegree = compute_max_outdegree(h_row_offsets, num_nodes);
    int* h_active;
    MonotonicInfo* h_info[CHECK_BUFFER_COUNT];
    CUDA_CHECK(cudaMallocHost(&h_active, num_nodes * sizeof(int)));
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) CUDA_CHECK(cudaMallocHost(&h_info[i], sizeof(MonotonicInfo)));
    for (int i = 0; i < num_nodes; ++i) { h_active[i] = 1; h_value[i] = 1.0f / (float)num_nodes; }

    float* d_values;
    int *d_ro, *d_ci, *d_co, *d_ri, *d_active, *d_update, *d_num_active;
    MonotonicInfo* d_info[CHECK_BUFFER_COUNT];
    MonotonicInfo* d_info_scratch;
    CUDA_CHECK(cudaMalloc(&d_values, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ro, (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_co, (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ri, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_active, sizeof(int)));
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) CUDA_CHECK(cudaMalloc(&d_info[i], sizeof(MonotonicInfo)));
    CUDA_CHECK(cudaMalloc(&d_info_scratch, sizeof(MonotonicInfo)));

    CUDA_CHECK(cudaMemcpy(d_values, h_value, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ro, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, h_column_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_co, h_column_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ri, h_row_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, h_active, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_update, 0xff, num_nodes * sizeof(int)));

    cudaStream_t compute_stream, check_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(&check_stream, cudaStreamNonBlocking));
    cudaEvent_t compute_done_event[CHECK_BUFFER_COUNT], check_done_event[CHECK_BUFFER_COUNT], start, stop;
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_event[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&check_done_event[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    GraphCudaEventAccumulator graph_kernel_timer;
    CUDA_CHECK(graph_cuda_timer_create(&graph_kernel_timer));

    SpscQueue<int, CHECK_BUFFER_COUNT> free_buffers;
    SpscQueue<CheckTask, CHECK_BUFFER_COUNT> pending_tasks;
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) free_buffers.try_push(i);
    std::atomic<bool> check_stop(false);
    std::vector<AsyncCheckResult> check_results(1001);
    bool detected_dmr = false, detected_monotonic = false, detected_avg_increase = false;
    int first_dmr_iter = -1, first_monotonic_iter = -1, first_avg_increase_iter = -1;
    double pre_avg_delta = (double)INF;

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
                result.sum_abs_delta = h_info[task.buf]->sum_abs_delta;
                result.count_update = h_info[task.buf]->count_update;
                if (result.count_update > 0) {
                    double avg_delta = result.sum_abs_delta / (double)result.count_update;
                    if (pre_avg_delta < (double)INF && avg_delta > pre_avg_delta && first_avg_increase_iter < 0) {
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

    CUDA_CHECK(cudaEventRecord(start));
    int blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iter = 0;
    int active_nodes = num_nodes;
    CUDA_CHECK(cudaMemsetAsync(d_num_active, 0, sizeof(int), compute_stream));
    scoreAndMarkFloatKernel<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(
        d_active, d_values, d_ro, num_nodes, max_outdegree, alpha, beta, threshold, d_num_active
    );

    while (iter < 1000) {
        ++iter;
        {
            NvtxRange enqueue_range(nvtx_name("main enqueue pagerank/check", iter));
            int check_buf = -1;
            bool do_async_check = free_buffers.try_pop(check_buf);
            MonotonicInfo* iter_info = do_async_check ? d_info[check_buf] : d_info_scratch;
            CUDA_CHECK(cudaMemsetAsync(iter_info, 0, sizeof(MonotonicInfo), compute_stream));
            CUDA_CHECK(graph_cuda_timer_start(&graph_kernel_timer, compute_stream));
            pagerankPullDualKernel<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(
                d_values, d_ro, d_ci, d_co, d_ri, d_active, d_update, num_nodes, iter_info
            );
            CUDA_CHECK(graph_cuda_timer_stop(&graph_kernel_timer, compute_stream));
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
        { NvtxRange sync_range(nvtx_name("main compute sync", iter)); CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            CUDA_CHECK(graph_cuda_timer_accumulate(&graph_kernel_timer)); }
        {
            NvtxRange score_range(nvtx_name("main score active", iter));
            CUDA_CHECK(cudaMemsetAsync(d_num_active, 0, sizeof(int), compute_stream));
            scoreAndMarkFloatKernel<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(
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

    CUDA_CHECK(cudaMemcpy(h_value, d_values, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    const auto tail_start = std::chrono::steady_clock::now();
    while (!pending_tasks.empty()) std::this_thread::yield();
    check_stop.store(true, std::memory_order_release);
    check_worker.join();
    const double cpu_check_tail_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tail_start).count();

    printf("GPU time: %.4f ms\n", ms);
    printf("BENCHMARK_TIMING gpu_main_ms=%.4f graph_kernel_ms=%.4f "
           "cpu_check_tail_ms=%.4f "
           "nccl_exchange_ms=0.0000 mpi_sync_ms=0.0000 communication_ms=0.0000\n",
           ms, graph_kernel_timer.total_ms, cpu_check_tail_ms);
    printf("BENCHMARK_ITERATIONS iterations=%d\n", iter);

    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_ro)); CUDA_CHECK(cudaFree(d_ci));
    CUDA_CHECK(cudaFree(d_co)); CUDA_CHECK(cudaFree(d_ri)); CUDA_CHECK(cudaFree(d_active));
    CUDA_CHECK(cudaFree(d_update)); CUDA_CHECK(cudaFree(d_num_active));
    for (int i = 0; i < CHECK_BUFFER_COUNT; ++i) {
        CUDA_CHECK(cudaFree(d_info[i])); CUDA_CHECK(cudaFreeHost(h_info[i]));
        CUDA_CHECK(cudaEventDestroy(compute_done_event[i])); CUDA_CHECK(cudaEventDestroy(check_done_event[i]));
    }
    CUDA_CHECK(cudaFree(d_info_scratch)); CUDA_CHECK(cudaFreeHost(h_active));
    CUDA_CHECK(graph_cuda_timer_destroy(&graph_kernel_timer));
    CUDA_CHECK(cudaStreamDestroy(check_stream)); CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    printf("GPU PageRank finished in %d iterations.\n", iter);
    printf("异步检测标记: DMR=%d", detected_dmr ? 1 : 0);
    if (first_dmr_iter >= 0) printf("(first_iter=%d)", first_dmr_iter);
    printf(", monotonic=%d", detected_monotonic ? 1 : 0);
    if (first_monotonic_iter >= 0) printf("(first_iter=%d)", first_monotonic_iter);
    printf(", avg_delta_increase=%d", detected_avg_increase ? 1 : 0);
    if (first_avg_increase_iter >= 0) printf("(first_iter=%d)", first_avg_increase_iter);
    printf("\n");
}
