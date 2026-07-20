/**
 * @file pagerank_tolerance_multiGPU.cuh
 * @brief Distributed PageRank with selective DMR and asynchronous CPU checks.
 */
#ifndef PAGERANK_TOLERANCE_MULTIGPU_CUH
#define PAGERANK_TOLERANCE_MULTIGPU_CUH

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "include/distributed_benchmark.cuh"
#include "include/distributed_partition.cuh"
#include "include/spsc_queue.h"

#define PR_DIST_ALPHA 0.85f
#define PR_DIST_TOL 1e-3f

static constexpr int PR_DIST_CHECK_BUFFER_COUNT = 64;
static constexpr int PR_DIST_MAX_ITERATIONS = 1000;

/**
 * @brief GPU 端为一次 PageRank 迭代生成的容错检查摘要。
 *
 * 该结构只包含固定大小的聚合信息，因此可以由独立 check stream
 * 异步复制到页锁定主机内存。PageRank 的值不具有单调性约束，故这里
 * 不包含 monotonic error 标志。
 */
struct PagerankDistributedCheckInfo {
    int dmr_error_flag;
    float sum_abs_delta;
    int count_update;
};

/** @brief 主线程提交给 CPU 检查线程的异步复制任务。 */
struct PagerankDistributedCheckTask {
    int iter = 0;
    int buffer = -1;
};

/** @brief CPU 检查线程按迭代保存的本 rank 检查结果。 */
struct PagerankDistributedAsyncCheckResult {
    bool done = false;
    float sum_abs_delta = 0.0f;
    int count_update = 0;
    int dmr_error = 0;
};

/**
 * @brief 对当前 active 集打分并标出需要 DMR 检查的关键顶点。
 *
 * active 值 2 表示关键顶点，1 表示普通活跃顶点，-1 表示非活跃顶点。
 * count 统计进入本轮打分的 owned 活跃顶点数。
 *
 * @param active owned 顶点的活跃状态，kernel 会原地写入关键等级。
 * @param values owned 顶点当前的 PageRank 值。
 * @param row_offsets owned 顶点的出边 CSR 偏移。
 * @param owned_count 当前 rank 拥有的顶点数。
 * @param max_outdegree 全图最大出度，用于归一化关键顶点评分。
 * @param alpha 关键顶点评分中的出度权重。
 * @param beta 关键顶点评分中的当前值权重。
 * @param threshold 关键顶点判定阈值。
 * @param count 本轮参与打分的活跃 owned 顶点数。
 */
__global__ void pagerankDistributedScoreAndMarkKernel(
    int* active,
    const float* values,
    const int* row_offsets,
    int owned_count,
    int max_outdegree,
    float alpha,
    float beta,
    float threshold,
    int* count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= owned_count || active[v] == -1) return;

    atomicAdd(count, 1);
    int outdegree = row_offsets[v + 1] - row_offsets[v];
    int safe_max = max_outdegree > 0 ? max_outdegree : 1;
    float value_score = fabsf(values[v]);
    float score = alpha * ((float)outdegree / (float)safe_max) + beta * value_score;
    active[v] = (score >= threshold) ? 2 : 1;
}

/**
 * @brief 更新 owned 顶点的 PageRank，并为关键顶点执行块内选择性 DMR。
 *
 * 每个活跃线程完成一次 pull 更新；同一 block 中的空闲线程为关键顶点
 * 重算结果，差值超过 1e-6 时设置 DMR 标志。该 kernel 还累计本轮发生
 * 变化的绝对 delta 及其数量，供 CPU 在所有 rank 间计算全局平均 delta。
 * PageRank 不执行单调性检查。
 *
 * 活跃 lane 与被分配冗余任务的 idle lane 先选择 owned 计算顶点，再共同
 * 执行同一段 owned-plus-ghost 入邻居累加。只有目标选择、结果存放、比较
 * 和主结果写回保留角色相关分支。
 *
 * @warning `__syncthreads` 只同步当前 block，其他 block 仍可原地更新
 * values；主、副计算可能观察到不同阶段的值，因此 DMR 不一致不一定是
 * 硬件故障。NaN 也不会被当前大于 epsilon 的比较可靠捕获。
 *
 * @param values owned 顶点在前、ghost 顶点在后的本地值数组。
 * @param local_outdegree 本地 owned/ghost 顶点对应的全局出度。
 * @param row_offsets owned 顶点的出边 CSR 偏移。
 * @param column_indices 出邻居的本地编号。
 * @param column_offsets owned 顶点的入边 CSC 偏移。
 * @param row_indices 入邻居的本地编号。
 * @param active owned 顶点活跃状态及关键标记。
 * @param update 本地 owned/ghost 顶点的下一轮激活标记。
 * @param owned_count 当前 rank 拥有的顶点数。
 * @param local_node_count owned 与 ghost 顶点总数。
 * @param check_info 本轮固定大小的 GPU 检查摘要。
 */
__global__ void pagerankDistributedToleranceKernel(
    float* values,
    const int* local_outdegree,
    const int* row_offsets,
    const int* column_indices,
    const int* column_offsets,
    const int* row_indices,
    const int* active,
    int* update,
    int owned_count,
    int local_node_count,
    PagerankDistributedCheckInfo* check_info
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    __shared__ int critical_list[DG_BLOCK_SIZE];
    __shared__ float redundant_results[DG_BLOCK_SIZE];
    __shared__ int idle_count;
    __shared__ int critical_count;

    if (local_tid == 0) {
        idle_count = 0;
        critical_count = 0;
    }
    __syncthreads();

    bool valid = tid < owned_count;
    bool is_active = valid && active[tid] != -1;
    bool critical = is_active && active[tid] == 2;
    bool idle = !valid || !is_active;

    int critical_id = -1;
    int idle_id = -1;
    if (critical) {
        critical_id = atomicAdd(&critical_count, 1);
        if (critical_id < DG_BLOCK_SIZE) critical_list[critical_id] = tid;
    }
    if (idle) idle_id = atomicAdd(&idle_count, 1);
    __syncthreads();

    float old_value = valid ? values[tid] : 0.0f;
    const bool does_redundant_work =
        idle && idle_id >= 0 && idle_id < critical_count &&
        idle_id < DG_BLOCK_SIZE;
    int compute_vertex = -1;
    if (is_active) {
        compute_vertex = tid;
    } else if (does_redundant_work) {
        compute_vertex = critical_list[idle_id];
    }

    float computed_value = 0.0f;
    if (compute_vertex >= 0) {
        float sum = 0.0f;
        for (int e = column_offsets[compute_vertex];
             e < column_offsets[compute_vertex + 1]; ++e) {
            int src = row_indices[e];
            if (src >= 0 && src < local_node_count) {
                int outdegree = local_outdegree[src];
                if (outdegree > 0) sum += values[src] / (float)outdegree;
            }
        }
        computed_value = (1.0f - PR_DIST_ALPHA) + PR_DIST_ALPHA * sum;
    }

    float new_value = old_value;
    if (is_active) {
        new_value = computed_value;
    } else if (does_redundant_work) {
        redundant_results[idle_id] = computed_value;
    }
    __syncthreads();

    if (critical && critical_id >= 0 && critical_id < idle_count &&
        critical_id < DG_BLOCK_SIZE && fabsf(redundant_results[critical_id] - new_value) > 1e-6f) {
        atomicExch(&check_info->dmr_error_flag, 1);
    }
    __syncthreads();

    if (!valid) return;
    if (!is_active) return;

    values[tid] = new_value;
    float diff = fabsf(new_value - old_value);
    if (diff != 0.0f) {
        atomicAdd(&check_info->sum_abs_delta, diff);
        atomicAdd(&check_info->count_update, 1);
    }
    if (diff >= PR_DIST_TOL) {
        for (int e = row_offsets[tid]; e < row_offsets[tid + 1]; ++e) {
            int dst = column_indices[e];
            if (dst >= 0 && dst < local_node_count) update[dst] = 1;
        }
    }
}

/**
 * @brief 运行基于 MPI rank 分区和 NCCL ghost 同步的容错 PageRank。
 *
 * 算法更新、激活传播和 ghost value 同步均位于 compute stream。每轮
 * 容错 kernel 完成后，函数从固定 buffer 池取一个检查缓冲，通过 event
 * 令独立 check stream 异步执行 D2H；后台 CPU 线程轮询完成事件，并把
 * DMR 与 delta 摘要按迭代保存。迭代结束后，各 rank 使用 MPI 汇总完整
 * 的检查结果，报告全局 DMR 和平均 delta 增大；检测只报告、不恢复。
 * 若 buffer 池暂时耗尽，该轮仍使用 scratch 完成算法，但不向最终检测
 * 报告贡献 DMR 或 residual 证据。当前差值比较也不单独捕获 NaN。
 *
 * @param h_value 输出全图 PageRank 值；每个 rank 最终均通过 Allgatherv 获得完整结果。
 * @param h_row_offsets 全图出边 CSR 行偏移。
 * @param h_column_indices 全图出边 CSR 列索引。
 * @param h_column_offsets 全图入边 CSC 列偏移。
 * @param h_row_indices 全图入边 CSC 行索引。
 * @param num_nodes 全图顶点数。
 * @param num_edges 全图边数（当前入口不直接使用）。
 * @param alpha 关键顶点评分中的出度权重。
 * @param beta 关键顶点评分中的当前值权重。
 * @param threshold 关键顶点评分阈值。
 *
 * @pre MPI 已以不低于 MPI_THREAD_FUNNELED 的线程级别初始化。
 * @pre num_nodes 不小于 MPI rank 数，且每个 rank 均可访问 CUDA 设备。
 */
inline void pagerankMultiGPU(
    float* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int num_edges,
    float alpha = 0.5f,
    float beta = 0.5f,
    float threshold = 0.3f
) {
    (void)num_edges;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        fprintf(stderr, "pagerankMultiGPU 需要先调用 MPI_Init。\n");
        exit(EXIT_FAILURE);
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int device = dg_select_rank_device();
    ncclComm_t comm = dg_create_nccl_comm(rank, world_size);

    std::vector<DgSubgraphHost> parts;
    dg_build_all_subgraphs(world_size, num_nodes, h_row_offsets, h_column_indices,
                           h_column_offsets, h_row_indices, parts);
    const DgSubgraphHost& part = parts[rank];

    if (rank == 0) printf("发现 %d 个 MPI rank，开启 NCCL 分布式 PageRank 容错版本。\n", world_size);
    printf("Rank %d GPU %d: owned [%d, %d), owned_count=%d, local_count=%d, ghosts=%zu\n",
           rank, device, part.start_node, part.end_node, part.owned_count,
           part.local_node_count, part.ghost_global_ids.size());

    std::vector<std::vector<DgPeerPlanHost>> host_plans;
    dg_build_peer_plans(parts, world_size, num_nodes, host_plans);
    std::vector<DgPeerPlanDevice> plans;
    dg_upload_peer_plans(host_plans, rank, world_size, plans, true);

    int max_outdegree = 1;
    for (int v = 0; v < num_nodes; ++v) {
        int outdegree = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdegree > max_outdegree) max_outdegree = outdegree;
    }

    float* d_values = nullptr;
    int *d_local_outdegree = nullptr, *d_row_offsets = nullptr, *d_column_indices = nullptr;
    int *d_column_offsets = nullptr, *d_row_indices = nullptr, *d_active = nullptr;
    int *d_update = nullptr, *d_active_count = nullptr;
    PagerankDistributedCheckInfo* d_check_info_scratch = nullptr;
    std::vector<PagerankDistributedCheckInfo*> d_check_info(
        PR_DIST_CHECK_BUFFER_COUNT, nullptr);
    std::vector<PagerankDistributedCheckInfo*> h_check_info(
        PR_DIST_CHECK_BUFFER_COUNT, nullptr);
    std::vector<cudaEvent_t> compute_done_events(PR_DIST_CHECK_BUFFER_COUNT);
    std::vector<cudaEvent_t> check_done_events(PR_DIST_CHECK_BUFFER_COUNT);

    CUDA_CHECK(cudaMalloc(&d_values, part.local_node_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_local_outdegree, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, part.column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_indices, part.row_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, part.owned_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_check_info_scratch, sizeof(PagerankDistributedCheckInfo)));
    for (int buffer = 0; buffer < PR_DIST_CHECK_BUFFER_COUNT; ++buffer) {
        CUDA_CHECK(cudaMalloc(
            &d_check_info[buffer], sizeof(PagerankDistributedCheckInfo)));
        CUDA_CHECK(cudaMallocHost(
            &h_check_info[buffer], sizeof(PagerankDistributedCheckInfo)));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &compute_done_events[buffer], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &check_done_events[buffer], cudaEventDisableTiming));
    }

    std::vector<float> h_local_value(part.local_node_count, 1.0f / (float)num_nodes);
    std::vector<int> h_local_outdegree(part.local_node_count, 0);
    for (int lv = 0; lv < part.local_node_count; ++lv) {
        int gv = part.local_to_global[lv];
        h_local_outdegree[lv] = h_row_offsets[gv + 1] - h_row_offsets[gv];
    }
    std::vector<int> h_active(part.owned_count, 1);

    CUDA_CHECK(cudaMemcpy(d_values, h_local_value.data(), part.local_node_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_outdegree, h_local_outdegree.data(), part.local_node_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, part.row_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_indices, part.column_indices.data(), part.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_offsets, part.column_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_indices, part.row_indices.data(), part.row_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, h_active.data(), part.owned_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_update, 0xff, part.local_node_count * sizeof(int)));

    cudaStream_t compute_stream, check_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(&check_stream, cudaStreamNonBlocking));
    dg_nccl_exchange_float_values(
        d_values, plans, world_size, rank, comm, compute_stream);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    SpscQueue<int, PR_DIST_CHECK_BUFFER_COUNT> free_check_buffers;
    SpscQueue<PagerankDistributedCheckTask, PR_DIST_CHECK_BUFFER_COUNT>
        pending_check_tasks;
    for (int buffer = 0; buffer < PR_DIST_CHECK_BUFFER_COUNT; ++buffer) {
        free_check_buffers.try_push(buffer);
    }

    std::atomic<bool> stop_check_worker(false);
    std::mutex pending_mutex;
    std::condition_variable pending_cv;
    std::vector<PagerankDistributedAsyncCheckResult> check_results(
        PR_DIST_MAX_ITERATIONS + 1);

    // The worker sleeps between tasks and blocks on the oldest completed-copy
    // event. It never calls MPI, preserving MPI_THREAD_FUNNELED semantics.
    std::thread check_worker([&]() {
        CUDA_CHECK(cudaSetDevice(device));
        while (true) {
            PagerankDistributedCheckTask task;
            {
                std::unique_lock<std::mutex> lock(pending_mutex);
                pending_cv.wait(lock, [&]() {
                    return stop_check_worker.load(std::memory_order_acquire) ||
                           !pending_check_tasks.empty();
                });
                if (stop_check_worker.load(std::memory_order_acquire) &&
                    pending_check_tasks.empty()) {
                    break;
                }
                if (!pending_check_tasks.try_pop(task)) continue;
            }

            CUDA_CHECK(cudaEventSynchronize(check_done_events[task.buffer]));
            PagerankDistributedAsyncCheckResult result;
            result.done = true;
            result.dmr_error = h_check_info[task.buffer]->dmr_error_flag;
            result.sum_abs_delta = h_check_info[task.buffer]->sum_abs_delta;
            result.count_update = h_check_info[task.buffer]->count_update;
            if (task.iter >= 0 &&
                task.iter < static_cast<int>(check_results.size())) {
                check_results[task.iter] = result;
            }
            free_check_buffers.try_push(task.buffer);
        }
    });

    DgIterationPhaseTimer phase_timer;
    GraphCudaEventAccumulator graph_kernel_timer;
    CUDA_CHECK(graph_cuda_timer_create(&graph_kernel_timer));
    DgBenchmarkTiming timing;
    int iter = 0;
    int total_active = num_nodes;
    const auto main_loop_start = std::chrono::steady_clock::now();

    while (iter < PR_DIST_MAX_ITERATIONS && total_active > 0) {
        ++iter;
        phase_timer.start_pre(compute_stream);
        CUDA_CHECK(cudaMemsetAsync(
            d_active_count, 0, sizeof(int), compute_stream));
        pagerankDistributedScoreAndMarkKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, compute_stream>>>(
            d_active, d_values, d_row_offsets, part.owned_count, max_outdegree,
            alpha, beta, threshold, d_active_count);

        int check_buffer = -1;
        bool do_async_check = free_check_buffers.try_pop(check_buffer);
        PagerankDistributedCheckInfo* iteration_check_info =
            do_async_check ? d_check_info[check_buffer] : d_check_info_scratch;
        CUDA_CHECK(cudaMemsetAsync(
            iteration_check_info, 0,
            sizeof(PagerankDistributedCheckInfo), compute_stream));
        CUDA_CHECK(graph_cuda_timer_start(&graph_kernel_timer, compute_stream));
        pagerankDistributedToleranceKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, compute_stream>>>(
            d_values, d_local_outdegree, d_row_offsets, d_column_indices, d_column_offsets, d_row_indices,
            d_active, d_update, part.owned_count, part.local_node_count,
            iteration_check_info);
        CUDA_CHECK(graph_cuda_timer_stop(&graph_kernel_timer, compute_stream));

        if (do_async_check) {
            CUDA_CHECK(cudaEventRecord(
                compute_done_events[check_buffer], compute_stream));
            CUDA_CHECK(cudaStreamWaitEvent(
                check_stream, compute_done_events[check_buffer], 0));
            CUDA_CHECK(cudaMemcpyAsync(
                h_check_info[check_buffer],
                d_check_info[check_buffer],
                sizeof(PagerankDistributedCheckInfo),
                cudaMemcpyDeviceToHost,
                check_stream));
            CUDA_CHECK(cudaEventRecord(
                check_done_events[check_buffer], check_stream));
            {
                std::lock_guard<std::mutex> lock(pending_mutex);
                if (!pending_check_tasks.try_push({iter, check_buffer})) {
                    fprintf(stderr, "PageRank async check queue invariant failed.\n");
                    exit(EXIT_FAILURE);
                }
            }
            pending_cv.notify_one();
        }

        CUDA_CHECK(cudaMemsetAsync(
            d_active, 0xff, part.owned_count * sizeof(int), compute_stream));
        dgCopyOwnedUpdateKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, compute_stream>>>(
            d_update, d_active, part.owned_count);
        phase_timer.stop_pre(compute_stream);

        phase_timer.start_comm(compute_stream);
        dg_nccl_exchange_activation(
            d_update, d_active, plans, world_size, rank, comm, compute_stream);
        dg_nccl_exchange_float_values(
            d_values, plans, world_size, rank, comm, compute_stream);
        phase_timer.stop_comm(compute_stream);

        phase_timer.start_post(compute_stream);
        CUDA_CHECK(cudaMemsetAsync(
            d_update, 0xff, part.local_node_count * sizeof(int), compute_stream));
        CUDA_CHECK(cudaMemsetAsync(
            d_active_count, 0, sizeof(int), compute_stream));
        dgCountActiveKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, compute_stream>>>(
            d_active, d_active_count, part.owned_count);

        int local_active = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &local_active, d_active_count, sizeof(int),
            cudaMemcpyDeviceToHost, compute_stream));
        phase_timer.stop_post(compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        phase_timer.accumulate(timing.gpu_compute_ms, timing.nccl_exchange_ms);
        CUDA_CHECK(graph_cuda_timer_accumulate(&graph_kernel_timer));

        dg_timed_allreduce(
            &local_active, &total_active, 1, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD, timing.mpi_sync_ms);
    }

    timing.main_loop_ms = dg_elapsed_ms(
        main_loop_start, std::chrono::steady_clock::now());

    // Drain measures only work still owned by the checker; result collectives
    // belong to a separate post-check aggregation phase.
    const auto drain_start = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        stop_check_worker.store(true, std::memory_order_release);
    }
    pending_cv.notify_one();
    if (check_worker.joinable()) check_worker.join();
    timing.cpu_check_drain_ms = dg_elapsed_ms(
        drain_start, std::chrono::steady_clock::now());

    const auto postcheck_start = std::chrono::steady_clock::now();

    int global_dmr = 0;
    int global_residual_increase = 0;
    int first_dmr_iter = -1;
    int first_residual_increase_iter = -1;
    int checked_iteration_count = 0;
    double previous_avg_delta = std::numeric_limits<double>::infinity();

    for (int check_iter = 1; check_iter <= iter; ++check_iter) {
        int local_done = check_results[check_iter].done ? 1 : 0;
        int all_done = 0;
        dg_timed_allreduce(
            &local_done, &all_done, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        if (!all_done) continue;
        ++checked_iteration_count;

        float local_sum_delta = check_results[check_iter].sum_abs_delta;
        float global_sum_delta = 0.0f;
        int local_count_delta = check_results[check_iter].count_update;
        int global_count_delta = 0;
        int local_dmr = check_results[check_iter].dmr_error ? 1 : 0;
        int iteration_global_dmr = 0;

        dg_timed_allreduce(
            &local_sum_delta, &global_sum_delta, 1, MPI_FLOAT, MPI_SUM,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        dg_timed_allreduce(
            &local_count_delta, &global_count_delta, 1, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        dg_timed_allreduce(
            &local_dmr, &iteration_global_dmr, 1, MPI_INT, MPI_LOR,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);

        if (iteration_global_dmr) {
            global_dmr = 1;
            if (first_dmr_iter < 0) first_dmr_iter = check_iter;
        }
        if (global_count_delta > 0) {
            double avg_delta =
                static_cast<double>(global_sum_delta) /
                static_cast<double>(global_count_delta);
            if (previous_avg_delta <
                    std::numeric_limits<double>::infinity() &&
                avg_delta > previous_avg_delta) {
                global_residual_increase = 1;
                if (first_residual_increase_iter < 0) {
                    first_residual_increase_iter = check_iter;
                }
            }
            previous_avg_delta = avg_delta;
        }
    }

    timing.postcheck_total_ms = dg_elapsed_ms(
        postcheck_start, std::chrono::steady_clock::now());
    timing.graph_kernel_ms = graph_kernel_timer.total_ms;
    dg_report_benchmark_timing(timing, rank, world_size);

    std::vector<float> h_owned(part.owned_count);
    CUDA_CHECK(cudaMemcpy(h_owned.data(), d_values, part.owned_count * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<int> counts = dg_owned_counts(world_size, num_nodes);
    std::vector<int> displs = dg_displacements(counts);
    MPI_Allgatherv(h_owned.data(), part.owned_count, MPI_FLOAT,
                   h_value, counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("异步CPU检测(PageRank): DMR=%d", global_dmr);
        if (first_dmr_iter >= 0) printf("(first_iter=%d)", first_dmr_iter);
        printf(", residual_increase=%d", global_residual_increase);
        if (first_residual_increase_iter >= 0) {
            printf("(first_iter=%d)", first_residual_increase_iter);
        }
        printf(", monotonic=N/A\n");
        printf("异步检测覆盖(PageRank): checked=%d/%d, skipped=%d\n",
               checked_iteration_count, iter, iter - checked_iteration_count);
        printf("NCCL 分布式 PageRank 容错版本迭代 %d 次结束。\n", iter);
        printf("BENCHMARK_ITERATIONS iterations=%d\n", iter);
    }

    phase_timer.destroy();
    CUDA_CHECK(graph_cuda_timer_destroy(&graph_kernel_timer));
    CUDA_CHECK(cudaStreamDestroy(check_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    dg_free_peer_plans(plans);
    cudaFree(d_values);
    cudaFree(d_local_outdegree);
    cudaFree(d_row_offsets);
    cudaFree(d_column_indices);
    cudaFree(d_column_offsets);
    cudaFree(d_row_indices);
    cudaFree(d_active);
    cudaFree(d_update);
    cudaFree(d_active_count);
    cudaFree(d_check_info_scratch);
    for (int buffer = 0; buffer < PR_DIST_CHECK_BUFFER_COUNT; ++buffer) {
        cudaFree(d_check_info[buffer]);
        cudaFreeHost(h_check_info[buffer]);
        cudaEventDestroy(compute_done_events[buffer]);
        cudaEventDestroy(check_done_events[buffer]);
    }
    NCCL_CHECK(ncclCommDestroy(comm));
}

#endif
