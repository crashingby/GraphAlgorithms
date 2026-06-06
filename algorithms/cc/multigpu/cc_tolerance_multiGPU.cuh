#ifndef CC_TOLERANCE_MULTIGPU_CUH
#define CC_TOLERANCE_MULTIGPU_CUH

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "include/distributed_partition.cuh"

struct CcDistributedCheckInfo {
    int dmr_error_flag;
    int monotonic_error_flag;
    unsigned long long sum_abs_delta;
    int count_update;
};

__global__ void ccDistributedScoreAndMarkKernel(
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

__global__ void ccDistributedToleranceKernel(
    int* values,
    const int* row_offsets,
    const int* column_indices,
    const int* column_offsets,
    const int* row_indices,
    const int* active,
    int* update,
    int owned_count,
    int local_node_count,
    CcDistributedCheckInfo* check_info
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
    int new_value = old_value;
    if (is_active) {
        for (int e = column_offsets[tid]; e < column_offsets[tid + 1]; ++e) {
            int src = row_indices[e];
            if (src >= 0 && src < local_node_count && values[src] > new_value) {
                new_value = values[src];
            }
        }
    }
    __syncthreads();

    if (idle && idle_id < critical_count && idle_id < DG_BLOCK_SIZE) {
        int target = critical_list[idle_id];
        int redundant = values[target];
        for (int e = column_offsets[target]; e < column_offsets[target + 1]; ++e) {
            int src = row_indices[e];
            if (src >= 0 && src < local_node_count && values[src] > redundant) {
                redundant = values[src];
            }
        }
        redundant_results[idle_id] = redundant;
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
    if (new_value > old_value) {
        for (int e = row_offsets[tid]; e < row_offsets[tid + 1]; ++e) {
            int dst = column_indices[e];
            if (dst >= 0 && dst < local_node_count) update[dst] = 1;
        }
    } else if (new_value < old_value) {
        atomicExch(&check_info->monotonic_error_flag, 1);
    }
}

inline void ccMultiGPU(
    int* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int /*num_edges*/,
    float alpha = 0.5f,
    float beta = 0.5f,
    float threshold = 0.3f
) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        fprintf(stderr, "ccMultiGPU 需要先调用 MPI_Init。\n");
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

    if (rank == 0) {
        printf("发现 %d 个 MPI rank，开启 NCCL 分布式 CC 容错版本。\n", world_size);
    }
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

    int *d_values = nullptr, *d_row_offsets = nullptr, *d_column_indices = nullptr;
    int *d_column_offsets = nullptr, *d_row_indices = nullptr, *d_active = nullptr;
    int *d_update = nullptr, *d_active_count = nullptr;
    CcDistributedCheckInfo* d_check_info = nullptr;

    CUDA_CHECK(cudaMalloc(&d_values, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, part.column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_indices, part.row_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, part.owned_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_check_info, sizeof(CcDistributedCheckInfo)));

    std::vector<int> h_local_value(part.local_node_count);
    for (int lv = 0; lv < part.local_node_count; ++lv) h_local_value[lv] = part.local_to_global[lv];
    std::vector<int> h_active(part.owned_count, 1);

    CUDA_CHECK(cudaMemcpy(d_values, h_local_value.data(), part.local_node_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, part.row_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_indices, part.column_indices.data(), part.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_offsets, part.column_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_indices, part.row_indices.data(), part.row_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, h_active.data(), part.owned_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_update, 0xff, part.local_node_count * sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    dg_nccl_exchange_int_values(d_values, plans, world_size, rank, comm, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    int iter = 0;
    int total_active = num_nodes;
    double pre_avg_delta = 0.0;
    int local_dmr = 0, local_monotonic = 0, local_avg_increase = 0;

    while (iter < 1000 && total_active > 0) {
        ++iter;
        CUDA_CHECK(cudaMemsetAsync(d_active_count, 0, sizeof(int), stream));
        ccDistributedScoreAndMarkKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_active, d_values, d_row_offsets, part.owned_count, max_outdegree,
            alpha, beta, threshold, d_active_count);

        CUDA_CHECK(cudaMemsetAsync(d_check_info, 0, sizeof(CcDistributedCheckInfo), stream));
        ccDistributedToleranceKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_values, d_row_offsets, d_column_indices, d_column_offsets, d_row_indices,
            d_active, d_update, part.owned_count, part.local_node_count, d_check_info);

        CUDA_CHECK(cudaMemsetAsync(d_active, 0xff, part.owned_count * sizeof(int), stream));
        dgCopyOwnedUpdateKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_update, d_active, part.owned_count);
        dg_nccl_exchange_activation(d_update, d_active, plans, world_size, rank, comm, stream);
        dg_nccl_exchange_int_values(d_values, plans, world_size, rank, comm, stream);

        CUDA_CHECK(cudaMemsetAsync(d_update, 0xff, part.local_node_count * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_active_count, 0, sizeof(int), stream));
        dgCountActiveKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_active, d_active_count, part.owned_count);

        int local_active = 0;
        CcDistributedCheckInfo h_info{0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpyAsync(&local_active, d_active_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&h_info, d_check_info, sizeof(CcDistributedCheckInfo), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        unsigned long long local_sum_delta = h_info.sum_abs_delta;
        int local_count_delta = h_info.count_update;
        unsigned long long global_sum_delta = 0;
        int global_count_delta = 0;
        MPI_Allreduce(&local_active, &total_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sum_delta, &global_sum_delta, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_count_delta, &global_count_delta, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (global_count_delta > 0) {
            double avg_delta = (double)global_sum_delta / (double)global_count_delta;
            if (avg_delta > pre_avg_delta && iter > 1) local_avg_increase = 1;
            pre_avg_delta = avg_delta;
        }
        if (h_info.dmr_error_flag) local_dmr = 1;
        if (h_info.monotonic_error_flag) local_monotonic = 1;
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int global_dmr = 0, global_monotonic = 0, global_avg_increase = 0;
    MPI_Allreduce(&local_dmr, &global_dmr, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&local_monotonic, &global_monotonic, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&local_avg_increase, &global_avg_increase, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    std::vector<int> h_owned(part.owned_count);
    CUDA_CHECK(cudaMemcpy(h_owned.data(), d_values, part.owned_count * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> counts = dg_owned_counts(world_size, num_nodes);
    std::vector<int> displs = dg_displacements(counts);
    MPI_Allgatherv(h_owned.data(), part.owned_count, MPI_INT,
                   h_value, counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("异步CPU检测(CC): DMR=%d, monotonic=%d, avg_delta_increase=%d\n",
               global_dmr, global_monotonic, global_avg_increase);
        printf("GPU time: %.4f ms\n", ms);
        printf("NCCL 分布式 CC 容错版本迭代 %d 次结束。\n", iter);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    dg_free_peer_plans(plans);
    cudaFree(d_values);
    cudaFree(d_row_offsets);
    cudaFree(d_column_indices);
    cudaFree(d_column_offsets);
    cudaFree(d_row_indices);
    cudaFree(d_active);
    cudaFree(d_update);
    cudaFree(d_active_count);
    cudaFree(d_check_info);
    NCCL_CHECK(ncclCommDestroy(comm));
}

#endif
