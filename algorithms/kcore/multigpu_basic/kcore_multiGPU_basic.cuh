/**
 * @file kcore_multiGPU_basic.cuh
 * @brief MPI/NCCL rank-per-GPU k-core baseline without fault detection.
 */
#ifndef KCORE_MULTIGPU_BASIC_CUH
#define KCORE_MULTIGPU_BASIC_CUH

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "include/distributed_benchmark.cuh"
#include "include/distributed_partition.cuh"

/**
 * @brief Recount live neighbors and peel owned vertices below @p k.
 * @param values Owned-plus-ghost remaining-degree estimates.
 * @param alive Owned-plus-ghost live/dead cache.
 * @param active Work set for owned vertices only; -1 means inactive.
 * @param update Next work set over owned and ghost vertices.
 */
__global__ void kcoreDistributedPullKernel(
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
    int k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= owned_count || active[tid] == -1) return;

    int new_value = 0;
    for (int e = column_offsets[tid]; e < column_offsets[tid + 1]; ++e) {
        int src = row_indices[e];
        if (src >= 0 && src < local_node_count && alive[src] != 0) ++new_value;
    }
    values[tid] = new_value;

    if (alive[tid] != 0 && new_value < k) {
        alive[tid] = 0;
        for (int e = row_offsets[tid]; e < row_offsets[tid + 1]; ++e) {
            int dst = column_indices[e];
            if (dst >= 0 && dst < local_node_count && alive[dst] != 0) update[dst] = 1;
        }
    }
}

/**
 * @brief Execute distributed k-core peeling without resilience checks.
 *
 * Remote deletion effects travel as activation flags while ghost @c alive
 * values are refreshed through NCCL after each iteration.
 */
inline void kcoreMultiGPUBasic(
    int* h_value,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    int num_nodes,
    int /*num_edges*/,
    int k
) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int device = dg_select_rank_device();
    ncclComm_t comm = dg_create_nccl_comm(rank, world_size);

    std::vector<DgSubgraphHost> parts;
    dg_build_all_subgraphs(
        world_size,
        num_nodes,
        h_row_offsets,
        h_column_indices,
        h_column_offsets,
        h_row_indices,
        parts
    );
    const DgSubgraphHost& part = parts[rank];

    if (rank == 0) {
        printf("发现 %d 个 MPI rank，开启 KCore 多节点无容错版本。\n", world_size);
    }
    printf(
        "Rank %d GPU %d: owned [%d, %d), owned_count=%d, local_count=%d, ghosts=%zu\n",
        rank,
        device,
        part.start_node,
        part.end_node,
        part.owned_count,
        part.local_node_count,
        part.ghost_global_ids.size()
    );

    std::vector<std::vector<DgPeerPlanHost>> host_plans;
    dg_build_peer_plans(parts, world_size, num_nodes, host_plans);
    std::vector<DgPeerPlanDevice> plans;
    dg_upload_peer_plans(host_plans, rank, world_size, plans, false);

    int *d_values = nullptr;
    int *d_alive = nullptr;
    int *d_row_offsets = nullptr;
    int *d_column_indices = nullptr;
    int *d_column_offsets = nullptr;
    int *d_row_indices = nullptr;
    int *d_active = nullptr;
    int *d_update = nullptr;
    int *d_active_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_values, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_alive, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, part.column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_indices, part.row_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, part.owned_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, part.local_node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));

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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Match the tolerance multiGPU timing boundary: warm up/synchronize ghost alive flags before timing.
    dg_nccl_exchange_int_values(d_alive, plans, world_size, rank, comm, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DgIterationPhaseTimer phase_timer;
    GraphCudaEventAccumulator graph_kernel_timer;
    CUDA_CHECK(graph_cuda_timer_create(&graph_kernel_timer));
    DgBenchmarkTiming timing;
    int iter = 0;
    int total_active = num_nodes;
    const auto main_loop_start = std::chrono::steady_clock::now();

    while (iter < 1000 && total_active > 0) {
        ++iter;
        phase_timer.start_pre(stream);
        CUDA_CHECK(graph_cuda_timer_start(&graph_kernel_timer, stream));
        kcoreDistributedPullKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_values,
            d_alive,
            d_row_offsets,
            d_column_indices,
            d_column_offsets,
            d_row_indices,
            d_active,
            d_update,
            part.owned_count,
            part.local_node_count,
            k
        );
        CUDA_CHECK(graph_cuda_timer_stop(&graph_kernel_timer, stream));

        CUDA_CHECK(cudaMemsetAsync(d_active, 0xff, part.owned_count * sizeof(int), stream));
        dgCopyOwnedUpdateKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_update,
            d_active,
            part.owned_count
        );
        phase_timer.stop_pre(stream);
        phase_timer.start_comm(stream);
        dg_nccl_exchange_activation(d_update, d_active, plans, world_size, rank, comm, stream);
        dg_nccl_exchange_int_values(d_alive, plans, world_size, rank, comm, stream);
        phase_timer.stop_comm(stream);

        phase_timer.start_post(stream);
        CUDA_CHECK(cudaMemsetAsync(d_update, 0xff, part.local_node_count * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_active_count, 0, sizeof(int), stream));
        dgCountActiveKernel<<<dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_active,
            d_active_count,
            part.owned_count
        );

        int local_active = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &local_active, d_active_count, sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        phase_timer.stop_post(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        phase_timer.accumulate(timing.gpu_compute_ms, timing.nccl_exchange_ms);
        CUDA_CHECK(graph_cuda_timer_accumulate(&graph_kernel_timer));

        const double mpi_sync_start = MPI_Wtime();
        MPI_Allreduce(
            &local_active, &total_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        timing.mpi_sync_ms += (MPI_Wtime() - mpi_sync_start) * 1000.0;
    }

    timing.main_loop_ms = dg_elapsed_ms(
        main_loop_start, std::chrono::steady_clock::now());
    timing.graph_kernel_ms = graph_kernel_timer.total_ms;
    dg_report_benchmark_timing(timing, rank, world_size);

    std::vector<int> h_owned(part.owned_count);
    CUDA_CHECK(cudaMemcpy(h_owned.data(), d_values, part.owned_count * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> counts = dg_owned_counts(world_size, num_nodes);
    std::vector<int> displs = dg_displacements(counts);
    MPI_Allgatherv(h_owned.data(), part.owned_count, MPI_INT,
                   h_value, counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("KCore 多节点无容错版本迭代 %d 次结束。\n", iter);
        printf("BENCHMARK_ITERATIONS iterations=%d\n", iter);
    }

    phase_timer.destroy();
    CUDA_CHECK(graph_cuda_timer_destroy(&graph_kernel_timer));
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
    NCCL_CHECK(ncclCommDestroy(comm));
}

#endif
