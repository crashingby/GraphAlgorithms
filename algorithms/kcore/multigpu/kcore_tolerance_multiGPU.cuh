/**
 * @file kcore_tolerance_multiGPU.cuh
 * @brief Distributed k-core with selective DMR and asynchronous CPU checks.
 */
#ifndef KCORE_TOLERANCE_MULTIGPU_CUH
#define KCORE_TOLERANCE_MULTIGPU_CUH

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "include/distributed_benchmark.cuh"
#include "include/distributed_partition.cuh"
#include "include/spsc_queue.h"

/** Number of reusable device/host slots in the asynchronous checker. */
constexpr int KCORE_CHECK_BUFFER_COUNT = 64;

/** Maximum number of KCore iterations and therefore result slots. */
constexpr int KCORE_MAX_ITERATIONS = 1000;

/**
 * @brief Per-iteration error evidence accumulated by the GPU.
 *
 * A buffer is written by exactly one KCore iteration. The compute stream
 * records an event after the kernel, and the check stream then copies this
 * fixed-size structure to pinned host memory without blocking the main loop.
 */
struct KcoreDistributedCheckInfo {
    /** Set when the primary and redundant computations disagree. */
    int dmr_error_flag;
    /** Set when a degree increases, which violates KCore monotonicity. */
    int monotonic_error_flag;
    /** Sum of absolute degree changes produced by this rank. */
    unsigned long long sum_abs_delta;
    /** Number of owned vertices whose degree changed. */
    int count_update;
};

/** @brief Host-side snapshot produced by the asynchronous check worker. */
struct KcoreAsyncCheckResult {
    bool done = false;
    unsigned long long sum_abs_delta = 0;
    int count_update = 0;
    int dmr_error = 0;
    int monotonic_error = 0;
};

/** @brief Queue item associating an iteration with a reusable check buffer. */
struct KcoreCheckTask {
    int iter = 0;
    int buffer = -1;
};

/**
 * @brief Classify active owned vertices as ordinary or critical.
 *
 * The score combines normalized out-degree and the current KCore degree.
 * A value of `2` marks a critical vertex whose update should be recomputed by
 * an idle thread in kcoreDistributedToleranceKernel(); `1` is ordinary and
 * `-1` remains inactive.
 *
 * @param active Active-state array for owned vertices.
 * @param values Current local degree values; owned entries come first.
 * @param row_offsets Outgoing CSR offsets for owned vertices.
 * @param owned_count Number of vertices owned by this rank.
 * @param max_outdegree Global maximum out-degree used for normalization.
 * @param alpha Weight of normalized out-degree.
 * @param beta Weight of the current-value score.
 * @param threshold Critical-vertex score threshold.
 * @param count Number of active owned vertices encountered.
 */
__global__ void kcoreDistributedScoreAndMarkKernel(
    int* active,
    const int* values,
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
    float value_score = 1.0f / (1.0f + fabsf((float)values[v]));
    float score = alpha * ((float)outdegree / (float)safe_max) + beta * value_score;
    active[v] = (score >= threshold) ? 2 : 1;
}

/**
 * @brief Recompute local degrees, peel vertices, and collect fault evidence.
 *
 * Active owned vertices perform the normal KCore update. Within each block,
 * idle lanes recompute as many critical updates as possible; disagreement is
 * reported through @p check_info. The kernel also checks that local degrees
 * never increase and accumulates degree-change statistics. Peeling and
 * propagation semantics are unchanged: a vertex whose new degree is below
 * @p k is marked dead, and its outgoing neighbours are activated through
 * @p update, including ghost entries that will be sent to their owners.
 *
 * Active and assigned redundant lanes select an owned compute vertex before
 * entering one common owned-plus-ghost degree recount. Role divergence is
 * limited to target selection, result placement, comparison, and writeback.
 *
 * @warning Block barriers do not freeze the grid-wide alive array. Other
 * blocks may peel vertices between primary and redundant recounts, so DMR is
 * an anomaly signal rather than definitive hardware-fault evidence.
 *
 * @param values Degree values for owned vertices followed by ghost cache.
 * @param alive Alive flags for owned vertices followed by ghost cache.
 * @param row_offsets Outgoing CSR offsets for owned vertices.
 * @param column_indices Outgoing destination local IDs.
 * @param column_offsets Incoming CSC offsets for owned vertices.
 * @param row_indices Incoming source local IDs.
 * @param active Current active states for owned vertices.
 * @param update Next-iteration activation flags in local ID space.
 * @param owned_count Number of vertices owned by this rank.
 * @param local_node_count Number of owned plus ghost vertices on this rank.
 * @param k Requested KCore threshold.
 * @param check_info Per-iteration GPU check accumulator.
 */
__global__ void kcoreDistributedToleranceKernel(
    int* values,
    int* alive,
    const int* row_offsets,
    const int* column_indices,
    const int* column_offsets,
    const int* row_indices,
    const int* active,
    int* update,
    int owned_count,
    int local_node_count,
    int k,
    KcoreDistributedCheckInfo* check_info
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    __shared__ int critical_list[DG_BLOCK_SIZE];
    __shared__ int redundant_results[DG_BLOCK_SIZE];
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

    int old_value = valid ? values[tid] : 0;
    const bool does_redundant_work =
        idle && idle_id >= 0 && idle_id < critical_count &&
        idle_id < DG_BLOCK_SIZE;
    int compute_vertex = -1;
    if (is_active) {
        compute_vertex = tid;
    } else if (does_redundant_work) {
        compute_vertex = critical_list[idle_id];
    }

    int computed_value = 0;
    if (compute_vertex >= 0) {
        for (int e = column_offsets[compute_vertex];
             e < column_offsets[compute_vertex + 1]; ++e) {
            int src = row_indices[e];
            if (src >= 0 && src < local_node_count && alive[src] != 0) {
                ++computed_value;
            }
        }
    }

    int new_value = old_value;
    if (is_active) {
        new_value = computed_value;
    } else if (does_redundant_work) {
        redundant_results[idle_id] = computed_value;
    }
    __syncthreads();

    if (critical && critical_id >= 0 && critical_id < idle_count &&
        critical_id < DG_BLOCK_SIZE && redundant_results[critical_id] != new_value) {
        atomicExch(&check_info->dmr_error_flag, 1);
    }
    __syncthreads();

    if (!valid) return;
    if (!is_active) return;

    values[tid] = new_value;
    int diff = new_value - old_value;
    if (diff != 0) {
        atomicAdd(&check_info->sum_abs_delta, (unsigned long long)llabs((long long)diff));
        atomicAdd(&check_info->count_update, 1);
    }
    if (diff > 0) atomicExch(&check_info->monotonic_error_flag, 1);

    if (alive[tid] != 0 && new_value < k) {
        alive[tid] = 0;
        for (int e = row_offsets[tid]; e < row_offsets[tid + 1]; ++e) {
            int dst = column_indices[e];
            if (dst >= 0 && dst < local_node_count && alive[dst] != 0) update[dst] = 1;
        }
    }
}

/**
 * @brief Run fault-detecting distributed KCore with asynchronous CPU checks.
 *
 * Each MPI rank owns a contiguous vertex partition and stores remote
 * neighbours as ghosts. NCCL exchanges activation flags and alive-state
 * caches after every peeling step. Fault evidence is transferred through a
 * fixed pool of pinned buffers on a non-blocking check stream; a background
 * CPU worker polls completion events and stores results by iteration. Once
 * computation finishes, MPI reductions combine DMR, monotonicity, and average
 * delta-increase evidence across ranks. Detection is observational only: it
 * does not roll back or repair KCore state. If no check buffer is free, the
 * iteration uses scratch storage and contributes no final detection evidence.
 *
 * @param[out] h_value Global degree/result array, assembled on every rank.
 * @param h_row_offsets Global outgoing CSR offsets.
 * @param h_column_indices Global outgoing CSR destinations.
 * @param h_column_offsets Global incoming CSC offsets.
 * @param h_row_indices Global incoming CSC sources.
 * @param num_nodes Number of graph vertices.
 * @param num_edges Number of graph edges (unused after graph construction).
 * @param k Requested KCore threshold.
 * @param alpha Criticality weight for normalized out-degree.
 * @param beta Criticality weight for current degree.
 * @param threshold Score at or above which an active vertex is critical.
 *
 * @pre MPI has been initialized with at least MPI_THREAD_FUNNELED.
 * @pre Every rank can access a CUDA device and num_nodes is at least the rank count.
 */
inline void kcoreMultiGPU(
    int* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int num_edges,
    int k,
    float alpha = 0.5f,
    float beta = 0.5f,
    float threshold = 0.3f
) {
    (void)num_edges;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        fprintf(stderr, "kcoreMultiGPU 需要先调用 MPI_Init。\n");
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

    if (rank == 0) printf("发现 %d 个 MPI rank，开启 NCCL 分布式 KCore 容错版本。\n", world_size);
    printf("Rank %d GPU %d: owned [%d, %d), owned_count=%d, local_count=%d, ghosts=%zu\n",
           rank, device, part.start_node, part.end_node, part.owned_count,
           part.local_node_count, part.ghost_global_ids.size());

    std::vector<std::vector<DgPeerPlanHost>> host_plans;
    dg_build_peer_plans(parts, world_size, num_nodes, host_plans);
    std::vector<DgPeerPlanDevice> plans;
    dg_upload_peer_plans(host_plans, rank, world_size, plans, false);

    int max_outdegree = 1;
    for (int v = 0; v < num_nodes; ++v) {
        int outdegree = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdegree > max_outdegree) max_outdegree = outdegree;
    }

    int *d_values = nullptr, *d_alive = nullptr, *d_row_offsets = nullptr;
    int *d_column_indices = nullptr, *d_column_offsets = nullptr, *d_row_indices = nullptr;
    int *d_active = nullptr, *d_update = nullptr, *d_active_count = nullptr;
    KcoreDistributedCheckInfo* d_check_info_scratch = nullptr;
    std::vector<KcoreDistributedCheckInfo*> d_check_info(KCORE_CHECK_BUFFER_COUNT, nullptr);
    std::vector<KcoreDistributedCheckInfo*> h_check_info(KCORE_CHECK_BUFFER_COUNT, nullptr);
    std::vector<cudaEvent_t> compute_done_events(KCORE_CHECK_BUFFER_COUNT);
    std::vector<cudaEvent_t> check_done_events(KCORE_CHECK_BUFFER_COUNT);

    CUDA_CHECK(cudaMalloc(&d_values, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_alive, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, part.column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_indices, part.row_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, part.owned_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_check_info_scratch, sizeof(KcoreDistributedCheckInfo)));
    for (int buffer = 0; buffer < KCORE_CHECK_BUFFER_COUNT; ++buffer) {
        CUDA_CHECK(cudaMalloc(&d_check_info[buffer], sizeof(KcoreDistributedCheckInfo)));
        CUDA_CHECK(cudaMallocHost(&h_check_info[buffer], sizeof(KcoreDistributedCheckInfo)));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &compute_done_events[buffer], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &check_done_events[buffer], cudaEventDisableTiming));
    }

    std::vector<int> h_local_value(part.local_node_count, 0);
    for (int lv = 0; lv < part.owned_count; ++lv) {
        h_local_value[lv] = part.row_offsets[lv + 1] - part.row_offsets[lv];
    }
    std::vector<int> h_alive(part.local_node_count, 1);
    std::vector<int> h_active(part.owned_count, 1);

    CUDA_CHECK(cudaMemcpy(d_values, h_local_value.data(), part.local_node_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alive, h_alive.data(), part.local_node_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, part.row_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_indices, part.column_indices.data(), part.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_offsets, part.column_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_indices, part.row_indices.data(), part.row_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, h_active.data(), part.owned_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_update, 0xff, part.local_node_count * sizeof(int)));

    cudaStream_t stream, check_stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(&check_stream, cudaStreamNonBlocking));
    dg_nccl_exchange_int_values(d_alive, plans, world_size, rank, comm, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The main thread consumes free slots and produces pending tasks. The
    // worker does the inverse, so both queues remain single-producer/single-consumer.
    SpscQueue<int, KCORE_CHECK_BUFFER_COUNT> free_buffers;
    SpscQueue<KcoreCheckTask, KCORE_CHECK_BUFFER_COUNT> pending_tasks;
    for (int buffer = 0; buffer < KCORE_CHECK_BUFFER_COUNT; ++buffer) {
        free_buffers.try_push(buffer);
    }
    std::atomic<bool> check_stop(false);
    std::mutex pending_mutex;
    std::condition_variable pending_cv;
    std::vector<KcoreAsyncCheckResult> check_results(KCORE_MAX_ITERATIONS + 1);

    // Wait without polling, then synchronize only the oldest pending copy.
    // This keeps the checker from competing with the main MPI thread for CPU.
    std::thread check_worker([&]() {
        CUDA_CHECK(cudaSetDevice(device));
        while (true) {
            KcoreCheckTask task;
            {
                std::unique_lock<std::mutex> lock(pending_mutex);
                pending_cv.wait(lock, [&]() {
                    return check_stop.load(std::memory_order_acquire) ||
                           !pending_tasks.empty();
                });
                if (check_stop.load(std::memory_order_acquire) &&
                    pending_tasks.empty()) {
                    break;
                }
                if (!pending_tasks.try_pop(task)) continue;
            }

            CUDA_CHECK(cudaEventSynchronize(check_done_events[task.buffer]));
            const KcoreDistributedCheckInfo& info = *h_check_info[task.buffer];
            KcoreAsyncCheckResult result;
            result.done = true;
            result.sum_abs_delta = info.sum_abs_delta;
            result.count_update = info.count_update;
            result.dmr_error = info.dmr_error_flag;
            result.monotonic_error = info.monotonic_error_flag;
            if (task.iter >= 0 && task.iter < (int)check_results.size()) {
                check_results[task.iter] = result;
            }
            free_buffers.try_push(task.buffer);
        }
    });

    DgIterationPhaseTimer phase_timer;
    GraphCudaEventAccumulator graph_kernel_timer;
    CUDA_CHECK(graph_cuda_timer_create(&graph_kernel_timer));
    DgBenchmarkTiming timing;
    int iter = 0;
    int total_active = num_nodes;
    const auto main_loop_start = std::chrono::steady_clock::now();

    while (iter < KCORE_MAX_ITERATIONS && total_active > 0) {
        ++iter;
        phase_timer.start_pre(stream);
        CUDA_CHECK(cudaMemsetAsync(d_active_count, 0, sizeof(int), stream));
        kcoreDistributedScoreAndMarkKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_active, d_values, d_row_offsets, part.owned_count, max_outdegree,
            alpha, beta, threshold, d_active_count);

        int check_buffer = -1;
        bool enqueue_check = free_buffers.try_pop(check_buffer);
        KcoreDistributedCheckInfo* iteration_info = enqueue_check
            ? d_check_info[check_buffer]
            : d_check_info_scratch;

        CUDA_CHECK(cudaMemsetAsync(
            iteration_info, 0, sizeof(KcoreDistributedCheckInfo), stream));
        CUDA_CHECK(graph_cuda_timer_start(&graph_kernel_timer, stream));
        kcoreDistributedToleranceKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_values, d_alive, d_row_offsets, d_column_indices, d_column_offsets, d_row_indices,
            d_active, d_update, part.owned_count, part.local_node_count, k, iteration_info);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(graph_cuda_timer_stop(&graph_kernel_timer, stream));

        if (enqueue_check) {
            // The check stream starts the tiny D2H copy as soon as this
            // iteration kernel finishes; NCCL and next-active construction
            // continue independently on the compute stream.
            CUDA_CHECK(cudaEventRecord(compute_done_events[check_buffer], stream));
            CUDA_CHECK(cudaStreamWaitEvent(
                check_stream, compute_done_events[check_buffer], 0));
            CUDA_CHECK(cudaMemcpyAsync(
                h_check_info[check_buffer],
                d_check_info[check_buffer],
                sizeof(KcoreDistributedCheckInfo),
                cudaMemcpyDeviceToHost,
                check_stream));
            CUDA_CHECK(cudaEventRecord(check_done_events[check_buffer], check_stream));
            {
                std::lock_guard<std::mutex> lock(pending_mutex);
                if (!pending_tasks.try_push({iter, check_buffer})) {
                    fprintf(stderr, "KCore async check queue invariant failed.\n");
                    exit(EXIT_FAILURE);
                }
            }
            pending_cv.notify_one();
        }

        CUDA_CHECK(cudaMemsetAsync(d_active, 0xff, part.owned_count * sizeof(int), stream));
        dgCopyOwnedUpdateKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_update, d_active, part.owned_count);
        phase_timer.stop_pre(stream);

        phase_timer.start_comm(stream);
        dg_nccl_exchange_activation(d_update, d_active, plans, world_size, rank, comm, stream);
        dg_nccl_exchange_int_values(d_alive, plans, world_size, rank, comm, stream);
        phase_timer.stop_comm(stream);

        phase_timer.start_post(stream);
        CUDA_CHECK(cudaMemsetAsync(d_update, 0xff, part.local_node_count * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_active_count, 0, sizeof(int), stream));
        dgCountActiveKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_active, d_active_count, part.owned_count);

        int local_active = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &local_active, d_active_count, sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        phase_timer.stop_post(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        phase_timer.accumulate(timing.gpu_compute_ms, timing.nccl_exchange_ms);
        CUDA_CHECK(graph_cuda_timer_accumulate(&graph_kernel_timer));

        dg_timed_allreduce(
            &local_active, &total_active, 1, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD, timing.mpi_sync_ms);
    }

    timing.main_loop_ms = dg_elapsed_ms(
        main_loop_start, std::chrono::steady_clock::now());

    // Only unfinished checker work belongs to drain. MPI result aggregation is
    // deliberately excluded and measured in the following phase.
    const auto drain_start = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        check_stop.store(true, std::memory_order_release);
    }
    pending_cv.notify_one();
    if (check_worker.joinable()) check_worker.join();
    timing.cpu_check_drain_ms = dg_elapsed_ms(
        drain_start, std::chrono::steady_clock::now());

    const auto postcheck_start = std::chrono::steady_clock::now();

    bool detected_dmr = false;
    bool detected_monotonic = false;
    bool detected_avg_increase = false;
    int first_dmr_iter = -1;
    int first_monotonic_iter = -1;
    int first_avg_increase_iter = -1;
    int checked_iteration_count = 0;
    bool have_previous_avg_delta = false;
    double previous_avg_delta = 0.0;

    for (int check_iter = 1; check_iter <= iter; ++check_iter) {
        int local_done = check_results[check_iter].done ? 1 : 0;
        int all_done = 0;
        dg_timed_allreduce(
            &local_done, &all_done, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        if (!all_done) continue;
        ++checked_iteration_count;

        unsigned long long local_sum = check_results[check_iter].sum_abs_delta;
        unsigned long long global_sum = 0;
        int local_count = check_results[check_iter].count_update;
        int global_count = 0;
        int local_dmr = check_results[check_iter].dmr_error ? 1 : 0;
        int global_dmr = 0;
        int local_monotonic = check_results[check_iter].monotonic_error ? 1 : 0;
        int global_monotonic = 0;

        dg_timed_allreduce(
            &local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        dg_timed_allreduce(
            &local_count, &global_count, 1, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        dg_timed_allreduce(
            &local_dmr, &global_dmr, 1, MPI_INT, MPI_LOR,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        dg_timed_allreduce(
            &local_monotonic, &global_monotonic, 1, MPI_INT, MPI_LOR,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);

        if (global_dmr) {
            detected_dmr = true;
            if (first_dmr_iter < 0) first_dmr_iter = check_iter;
        }
        if (global_monotonic) {
            detected_monotonic = true;
            if (first_monotonic_iter < 0) first_monotonic_iter = check_iter;
        }
        if (global_count > 0) {
            double average_delta = (double)global_sum / (double)global_count;
            if (have_previous_avg_delta && average_delta > previous_avg_delta) {
                detected_avg_increase = true;
                if (first_avg_increase_iter < 0) first_avg_increase_iter = check_iter;
            }
            previous_avg_delta = average_delta;
            have_previous_avg_delta = true;
        }
    }

    timing.postcheck_total_ms = dg_elapsed_ms(
        postcheck_start, std::chrono::steady_clock::now());
    timing.graph_kernel_ms = graph_kernel_timer.total_ms;
    dg_report_benchmark_timing(timing, rank, world_size);

    std::vector<int> h_owned(part.owned_count);
    CUDA_CHECK(cudaMemcpy(h_owned.data(), d_values, part.owned_count * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> counts = dg_owned_counts(world_size, num_nodes);
    std::vector<int> displs = dg_displacements(counts);
    MPI_Allgatherv(h_owned.data(), part.owned_count, MPI_INT,
                   h_value, counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("异步CPU检测(KCore): DMR=%d", detected_dmr ? 1 : 0);
        if (first_dmr_iter >= 0) printf("(first_iter=%d)", first_dmr_iter);
        printf(", monotonic=%d", detected_monotonic ? 1 : 0);
        if (first_monotonic_iter >= 0) {
            printf("(first_iter=%d)", first_monotonic_iter);
        }
        printf(", avg_delta_increase=%d", detected_avg_increase ? 1 : 0);
        if (first_avg_increase_iter >= 0) {
            printf("(first_iter=%d)", first_avg_increase_iter);
        }
        printf("\n");
        printf("异步检测覆盖(KCore): checked=%d/%d, skipped=%d\n",
               checked_iteration_count, iter, iter - checked_iteration_count);
        printf("NCCL 分布式 KCore 容错版本迭代 %d 次结束。\n", iter);
        printf("BENCHMARK_ITERATIONS iterations=%d\n", iter);
    }

    phase_timer.destroy();
    CUDA_CHECK(graph_cuda_timer_destroy(&graph_kernel_timer));
    for (int buffer = 0; buffer < KCORE_CHECK_BUFFER_COUNT; ++buffer) {
        cudaFree(d_check_info[buffer]);
        cudaFreeHost(h_check_info[buffer]);
        cudaEventDestroy(compute_done_events[buffer]);
        cudaEventDestroy(check_done_events[buffer]);
    }
    CUDA_CHECK(cudaStreamDestroy(check_stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    dg_free_peer_plans(plans);
    cudaFree(d_values);
    cudaFree(d_alive);
    cudaFree(d_row_offsets);
    cudaFree(d_column_indices);
    cudaFree(d_column_offsets);
    cudaFree(d_row_indices);
    cudaFree(d_active);
    cudaFree(d_update);
    cudaFree(d_active_count);
    cudaFree(d_check_info_scratch);
    NCCL_CHECK(ncclCommDestroy(comm));
}

#endif  // KCORE_TOLERANCE_MULTIGPU_CUH
