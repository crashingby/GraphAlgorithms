/**
 * @file distributed_benchmark.cuh
 * @brief Common, non-overlapping timing utilities for distributed algorithms.
 *
 * The main-loop wall interval is the end-to-end rank-local iteration span. GPU
 * compute, NCCL exchange, and termination MPI are explanatory subcomponents of
 * that span. Core graph-kernel time is a narrower CUDA-event subset of GPU
 * compute. After the loop, checker drain and detection-result aggregation are
 * measured as two disjoint phases. MPI inside aggregation is also reported as
 * a subset and must not be added to the total a second time.
 */
#ifndef DISTRIBUTED_BENCHMARK_CUH
#define DISTRIBUTED_BENCHMARK_CUH

#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#include "include/cuda_event_timer.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(err__));                                     \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)
#endif

/** @brief Convert a steady-clock duration to milliseconds. */
inline double dg_elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& stop
) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

/**
 * @brief CUDA events for compute ranges before and after NCCL in one iteration.
 *
 * A caller records all six events on one stream, synchronizes that stream, and
 * calls accumulate(). gpu_compute_ms then contains only CUDA work outside the
 * communication range, while nccl_exchange_ms contains pack/NCCL/unpack only.
 */
class DgIterationPhaseTimer {
public:
    DgIterationPhaseTimer() {
        CUDA_CHECK(cudaEventCreate(&pre_start_));
        CUDA_CHECK(cudaEventCreate(&pre_stop_));
        CUDA_CHECK(cudaEventCreate(&comm_start_));
        CUDA_CHECK(cudaEventCreate(&comm_stop_));
        CUDA_CHECK(cudaEventCreate(&post_start_));
        CUDA_CHECK(cudaEventCreate(&post_stop_));
    }

    DgIterationPhaseTimer(const DgIterationPhaseTimer&) = delete;
    DgIterationPhaseTimer& operator=(const DgIterationPhaseTimer&) = delete;

    /** @brief Start rank-local GPU work preceding communication. */
    void start_pre(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(pre_start_, stream));
    }

    /** @brief Stop rank-local GPU work preceding communication. */
    void stop_pre(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(pre_stop_, stream));
    }

    /** @brief Start pack/NCCL/unpack timing. */
    void start_comm(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(comm_start_, stream));
    }

    /** @brief Stop pack/NCCL/unpack timing. */
    void stop_comm(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(comm_stop_, stream));
    }

    /** @brief Start rank-local GPU work following communication. */
    void start_post(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(post_start_, stream));
    }

    /** @brief Stop rank-local GPU work following communication. */
    void stop_post(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(post_stop_, stream));
    }

    /**
     * @brief Add one completed iteration's disjoint CUDA ranges.
     * @pre The stream containing all recorded events has completed.
     */
    void accumulate(double& gpu_compute_ms, double& nccl_exchange_ms) const {
        float pre_ms = 0.0f;
        float comm_ms = 0.0f;
        float post_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pre_ms, pre_start_, pre_stop_));
        CUDA_CHECK(cudaEventElapsedTime(&comm_ms, comm_start_, comm_stop_));
        CUDA_CHECK(cudaEventElapsedTime(&post_ms, post_start_, post_stop_));
        gpu_compute_ms += static_cast<double>(pre_ms + post_ms);
        nccl_exchange_ms += static_cast<double>(comm_ms);
    }

    /** @brief Destroy the six CUDA timing events. */
    void destroy() {
        CUDA_CHECK(cudaEventDestroy(pre_start_));
        CUDA_CHECK(cudaEventDestroy(pre_stop_));
        CUDA_CHECK(cudaEventDestroy(comm_start_));
        CUDA_CHECK(cudaEventDestroy(comm_stop_));
        CUDA_CHECK(cudaEventDestroy(post_start_));
        CUDA_CHECK(cudaEventDestroy(post_stop_));
    }

private:
    cudaEvent_t pre_start_{};
    cudaEvent_t pre_stop_{};
    cudaEvent_t comm_start_{};
    cudaEvent_t comm_stop_{};
    cudaEvent_t post_start_{};
    cudaEvent_t post_stop_{};
};

/**
 * @brief Execute one blocking all-reduce and accumulate its call duration.
 *
 * The duration includes both collective progress and time spent waiting for
 * slower ranks to enter the collective; callers must label it as sync/wait.
 */
inline void dg_timed_allreduce(
    const void* send_buffer,
    void* receive_buffer,
    int count,
    MPI_Datatype datatype,
    MPI_Op operation,
    MPI_Comm communicator,
    double& accumulated_ms
) {
    const double start = MPI_Wtime();
    MPI_Allreduce(
        send_buffer, receive_buffer, count, datatype, operation, communicator);
    accumulated_ms += (MPI_Wtime() - start) * 1000.0;
}

/** @brief Disjoint rank-local phases plus explanatory communication subsets. */
struct DgBenchmarkTiming {
    /** Complete iteration loop, including compute, NCCL, MPI, and host gaps. */
    double main_loop_ms = 0.0;
    /** Non-NCCL CUDA ranges inside the main loop. */
    double gpu_compute_ms = 0.0;
    /** Core graph-kernel time; a subset of gpu_compute_ms. */
    double graph_kernel_ms = 0.0;
    /** Time from checker shutdown request until its worker has joined. */
    double cpu_check_drain_ms = 0.0;
    /** Pack, NCCL transfer, and unpack CUDA ranges inside the main loop. */
    double nccl_exchange_ms = 0.0;
    /** Termination MPI_Allreduce call durations, including rank wait. */
    double mpi_sync_ms = 0.0;
    /** Entire result-aggregation phase after checker drain. */
    double postcheck_total_ms = 0.0;
    /** MPI calls inside postcheck_total_ms; a subset, not an extra phase. */
    double postcheck_mpi_ms = 0.0;
};

/**
 * @brief Reduce rank timings and emit the canonical BENCHMARK_TIMING record.
 *
 * gpu_main_ms and cpu_check_tail_ms are compatibility aliases for main_loop_ms
 * and cpu_check_drain_ms. graph_kernel_ms is reduced by rank maximum, while
 * rank_graph_kernel_avg_ms is the arithmetic rank mean. Communication contains
 * main-loop NCCL, termination MPI, and post-check MPI. Algorithm total contains
 * exactly three disjoint
 * phases: main loop + checker drain + post-check total.
 */
inline void dg_report_benchmark_timing(
    const DgBenchmarkTiming& timing,
    int rank,
    int world_size
) {
    enum Metric {
        MAIN_LOOP,
        GPU_COMPUTE,
        GRAPH_KERNEL,
        CPU_DRAIN,
        NCCL_EXCHANGE,
        MPI_SYNC,
        POSTCHECK_TOTAL,
        POSTCHECK_MPI,
        COMMUNICATION,
        ALGORITHM_TOTAL,
        METRIC_COUNT
    };

    const double communication_ms =
        timing.nccl_exchange_ms + timing.mpi_sync_ms + timing.postcheck_mpi_ms;
    const double algorithm_total_ms =
        timing.main_loop_ms + timing.cpu_check_drain_ms + timing.postcheck_total_ms;
    const double local[METRIC_COUNT] = {
        timing.main_loop_ms,
        timing.gpu_compute_ms,
        timing.graph_kernel_ms,
        timing.cpu_check_drain_ms,
        timing.nccl_exchange_ms,
        timing.mpi_sync_ms,
        timing.postcheck_total_ms,
        timing.postcheck_mpi_ms,
        communication_ms,
        algorithm_total_ms,
    };
    double sums[METRIC_COUNT] = {0.0};
    double maxima[METRIC_COUNT] = {0.0};
    MPI_Reduce(local, sums, METRIC_COUNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, maxima, METRIC_COUNT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank != 0) return;
    const double ranks = static_cast<double>(world_size);
    printf(
        "BENCHMARK_TIMING "
        "gpu_main_ms=%.4f main_loop_ms=%.4f gpu_compute_ms=%.4f "
        "graph_kernel_ms=%.4f "
        "cpu_check_tail_ms=%.4f cpu_check_drain_ms=%.4f "
        "nccl_exchange_ms=%.4f mpi_sync_ms=%.4f "
        "postcheck_total_ms=%.4f postcheck_mpi_ms=%.4f communication_ms=%.4f "
        "rank_gpu_main_avg_ms=%.4f rank_main_loop_avg_ms=%.4f "
        "rank_gpu_compute_avg_ms=%.4f rank_graph_kernel_avg_ms=%.4f "
        "rank_cpu_check_tail_avg_ms=%.4f rank_cpu_check_drain_avg_ms=%.4f "
        "rank_nccl_exchange_avg_ms=%.4f rank_mpi_sync_avg_ms=%.4f "
        "rank_postcheck_total_avg_ms=%.4f rank_postcheck_mpi_avg_ms=%.4f "
        "rank_communication_avg_ms=%.4f rank_algorithm_total_avg_ms=%.4f "
        "rank_algorithm_total_max_ms=%.4f\n",
        maxima[MAIN_LOOP], maxima[MAIN_LOOP], maxima[GPU_COMPUTE],
        maxima[GRAPH_KERNEL], maxima[CPU_DRAIN], maxima[CPU_DRAIN],
        maxima[NCCL_EXCHANGE],
        maxima[MPI_SYNC], maxima[POSTCHECK_TOTAL], maxima[POSTCHECK_MPI],
        maxima[COMMUNICATION], sums[MAIN_LOOP] / ranks,
        sums[MAIN_LOOP] / ranks, sums[GPU_COMPUTE] / ranks,
        sums[GRAPH_KERNEL] / ranks,
        sums[CPU_DRAIN] / ranks, sums[CPU_DRAIN] / ranks,
        sums[NCCL_EXCHANGE] / ranks, sums[MPI_SYNC] / ranks,
        sums[POSTCHECK_TOTAL] / ranks, sums[POSTCHECK_MPI] / ranks,
        sums[COMMUNICATION] / ranks, sums[ALGORITHM_TOTAL] / ranks,
        maxima[ALGORITHM_TOTAL]);
}

#endif
