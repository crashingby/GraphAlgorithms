/**
 * @file cuda_event_timer.cuh
 * @brief Lightweight CUDA-event accumulator for one repeated device range.
 *
 * The accumulator intentionally has no MPI or NCCL dependency, so the same
 * timing primitive can be used by single-GPU and distributed executables.
 * CUDA events measure elapsed device-stream time; NVTX ranges remain useful
 * for interactive profiling but do not provide machine-readable benchmark
 * values to the experiment runner.
 */
#ifndef GRAPH_CUDA_EVENT_TIMER_CUH
#define GRAPH_CUDA_EVENT_TIMER_CUH

#include <cuda_runtime.h>

/** @brief Pair of reusable CUDA events plus their accumulated elapsed time. */
struct GraphCudaEventAccumulator {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    double total_ms = 0.0;
};

/** @brief Allocate the two timing-enabled events. */
inline cudaError_t graph_cuda_timer_create(
    GraphCudaEventAccumulator* timer
) {
    cudaError_t status = cudaEventCreate(&timer->start);
    if (status != cudaSuccess) return status;
    status = cudaEventCreate(&timer->stop);
    if (status != cudaSuccess) {
        cudaEventDestroy(timer->start);
        timer->start = nullptr;
        return status;
    }
    timer->total_ms = 0.0;
    return cudaSuccess;
}

/** @brief Record the beginning of the measured range on @p stream. */
inline cudaError_t graph_cuda_timer_start(
    GraphCudaEventAccumulator* timer,
    cudaStream_t stream = nullptr
) {
    return cudaEventRecord(timer->start, stream);
}

/** @brief Record the end of the measured range on @p stream. */
inline cudaError_t graph_cuda_timer_stop(
    GraphCudaEventAccumulator* timer,
    cudaStream_t stream = nullptr
) {
    return cudaEventRecord(timer->stop, stream);
}

/**
 * @brief Add the most recently recorded range to total_ms.
 * @pre The stream containing @c stop has completed.
 *
 * Callers must accumulate before re-recording either event for the next
 * iteration. No synchronization is performed here, keeping the measured
 * region and the caller's existing synchronization policy independent.
 */
inline cudaError_t graph_cuda_timer_accumulate(
    GraphCudaEventAccumulator* timer
) {
    float elapsed_ms = 0.0f;
    cudaError_t status = cudaEventElapsedTime(
        &elapsed_ms, timer->start, timer->stop);
    if (status == cudaSuccess) {
        timer->total_ms += static_cast<double>(elapsed_ms);
    }
    return status;
}

/** @brief Destroy both CUDA events owned by @p timer. */
inline cudaError_t graph_cuda_timer_destroy(
    GraphCudaEventAccumulator* timer
) {
    cudaError_t start_status = cudaSuccess;
    cudaError_t stop_status = cudaSuccess;
    if (timer->start != nullptr) start_status = cudaEventDestroy(timer->start);
    if (timer->stop != nullptr) stop_status = cudaEventDestroy(timer->stop);
    timer->start = nullptr;
    timer->stop = nullptr;
    if (start_status != cudaSuccess) return start_status;
    return stop_status;
}

#endif
