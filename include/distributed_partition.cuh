/**
 * @file distributed_partition.cuh
 * @brief Shared contiguous partitioning and NCCL exchange utilities.
 *
 * Every MPI rank owns a contiguous global vertex range. Remote predecessors or
 * successors referenced by the owned subgraph are appended as ghost vertices.
 * Precomputed peer plans map sparse activation flags and owned values between
 * local indices on communicating ranks.
 */
#ifndef GRAPH_DISTRIBUTED_PARTITION_CUH
#define GRAPH_DISTRIBUTED_PARTITION_CUH

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DG_BLOCK_SIZE 256

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
                cudaGetErrorString(err__));                                      \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)
#endif

#ifndef NCCL_CHECK
#define NCCL_CHECK(call) do {                                                    \
    ncclResult_t res__ = (call);                                                  \
    if (res__ != ncclSuccess) {                                                   \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,          \
                ncclGetErrorString(res__));                                      \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
} while (0)
#endif

/** @brief Host description of one owned-plus-ghost graph partition. */
struct DgSubgraphHost {
    int rank = 0;
    int start_node = 0;
    int end_node = 0;
    int owned_count = 0;
    int local_node_count = 0;
    std::vector<int> local_to_global;
    std::unordered_map<int, int> global_to_local;
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<int> column_offsets;
    std::vector<int> row_indices;
    std::vector<int> ghost_local_ids;
    std::vector<int> ghost_global_ids;
};

/**
 * @brief Host-side sparse index plan for communication with one peer.
 *
 * Activation indices propagate remote work to the owner. Value indices refresh
 * ghost caches from the owner after each graph iteration.
 */
struct DgPeerPlanHost {
    std::vector<int> act_send_local;
    std::vector<int> act_recv_owned;
    std::vector<int> value_send_owned;
    std::vector<int> value_recv_ghost;
};

/** @brief Device indices and staging buffers corresponding to a peer plan. */
struct DgPeerPlanDevice {
    int act_send_count = 0;
    int act_recv_count = 0;
    int value_send_count = 0;
    int value_recv_count = 0;

    int* d_act_send_local = nullptr;
    int* d_act_recv_owned = nullptr;
    int* d_value_send_owned = nullptr;
    int* d_value_recv_ghost = nullptr;

    int* d_act_send_buf = nullptr;
    int* d_act_recv_buf = nullptr;
    int* d_value_send_buf_int = nullptr;
    int* d_value_recv_buf_int = nullptr;
    float* d_value_send_buf_float = nullptr;
    float* d_value_recv_buf_float = nullptr;
};

inline int dg_partition_start(int rank, int world_size, int num_nodes) {
    int base = num_nodes / world_size;
    int rem = num_nodes % world_size;
    return rank * base + std::min(rank, rem);
}

inline int dg_partition_end(int rank, int world_size, int num_nodes) {
    return dg_partition_start(rank + 1, world_size, num_nodes);
}

inline int dg_owner_of_vertex(int v, int world_size, int num_nodes) {
    int base = num_nodes / world_size;
    int rem = num_nodes % world_size;
    int big = (base + 1) * rem;
    if (v < big) return v / (base + 1);
    if (base == 0) return rem - 1;
    return rem + (v - big) / base;
}

inline int dg_get_or_add_local(DgSubgraphHost& p, int global_v, bool owned) {
    auto it = p.global_to_local.find(global_v);
    if (it != p.global_to_local.end()) return it->second;

    int local = (int)p.local_to_global.size();
    p.global_to_local[global_v] = local;
    p.local_to_global.push_back(global_v);
    if (!owned) {
        p.ghost_local_ids.push_back(local);
        p.ghost_global_ids.push_back(global_v);
    }
    return local;
}

/**
 * @brief Build one rank partition from the complete outgoing and incoming CSR.
 * @param rank Partition rank to build.
 * @param world_size Number of MPI ranks.
 * @param num_nodes Global vertex count.
 * @param p Output partition; owned vertices precede all ghosts.
 */
inline void dg_build_subgraph(
    int rank,
    int world_size,
    int num_nodes,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    DgSubgraphHost& p
) {
    p.rank = rank;
    p.start_node = dg_partition_start(rank, world_size, num_nodes);
    p.end_node = dg_partition_end(rank, world_size, num_nodes);
    p.owned_count = std::max(0, p.end_node - p.start_node);
    p.local_node_count = 0;
    p.local_to_global.clear();
    p.global_to_local.clear();
    p.ghost_local_ids.clear();
    p.ghost_global_ids.clear();

    for (int v = p.start_node; v < p.end_node; ++v) {
        int local = (int)p.local_to_global.size();
        p.global_to_local[v] = local;
        p.local_to_global.push_back(v);
    }

    p.row_offsets.assign(p.owned_count + 1, 0);
    p.column_offsets.assign(p.owned_count + 1, 0);
    p.column_indices.clear();
    p.row_indices.clear();

    int out = 0;
    for (int lv = 0; lv < p.owned_count; ++lv) {
        int gv = p.start_node + lv;
        p.row_offsets[lv] = out;
        for (int e = h_row_offsets[gv]; e < h_row_offsets[gv + 1]; ++e) {
            int dst = h_column_indices[e];
            if (dst < 0 || dst >= num_nodes) continue;
            bool owned = dst >= p.start_node && dst < p.end_node;
            p.column_indices.push_back(dg_get_or_add_local(p, dst, owned));
            ++out;
        }
        p.row_offsets[lv + 1] = out;
    }

    int in = 0;
    for (int lv = 0; lv < p.owned_count; ++lv) {
        int gv = p.start_node + lv;
        p.column_offsets[lv] = in;
        for (int e = h_column_offsets[gv]; e < h_column_offsets[gv + 1]; ++e) {
            int src = h_row_indices[e];
            if (src < 0 || src >= num_nodes) continue;
            bool owned = src >= p.start_node && src < p.end_node;
            p.row_indices.push_back(dg_get_or_add_local(p, src, owned));
            ++in;
        }
        p.column_offsets[lv + 1] = in;
    }

    p.local_node_count = (int)p.local_to_global.size();
    if (p.column_indices.empty()) p.column_indices.push_back(0);
    if (p.row_indices.empty()) p.row_indices.push_back(0);
}

inline void dg_build_all_subgraphs(
    int world_size,
    int num_nodes,
    const int* h_row_offsets,
    const int* h_column_indices,
    const int* h_column_offsets,
    const int* h_row_indices,
    std::vector<DgSubgraphHost>& parts
) {
    parts.resize(world_size);
    for (int r = 0; r < world_size; ++r) {
        dg_build_subgraph(
            r,
            world_size,
            num_nodes,
            h_row_offsets,
            h_column_indices,
            h_column_offsets,
            h_row_indices,
            parts[r]
        );
    }
}

/**
 * @brief Derive pairwise activation and ghost-value exchange plans.
 * @param parts Partition descriptions for every rank.
 * @param plans Output matrix indexed as sender rank then peer rank.
 */
inline void dg_build_peer_plans(
    const std::vector<DgSubgraphHost>& parts,
    int world_size,
    int num_nodes,
    std::vector<std::vector<DgPeerPlanHost>>& plans
) {
    plans.assign(world_size, std::vector<DgPeerPlanHost>(world_size));

    for (int r = 0; r < world_size; ++r) {
        const DgSubgraphHost& p = parts[r];
        // Activation is a vertex flag, not an edge payload. A remote owned
        // vertex therefore needs at most one entry from this sender even when
        // several cross-partition edges target it.
        std::vector<std::unordered_set<int>> seen_remote_targets(world_size);
        for (int lv = 0; lv < p.owned_count; ++lv) {
            for (int e = p.row_offsets[lv]; e < p.row_offsets[lv + 1]; ++e) {
                int dst_local = p.column_indices[e];
                int dst_global = p.local_to_global[dst_local];
                if (dst_global < 0 || dst_global >= num_nodes) continue;
                int peer = dg_owner_of_vertex(dst_global, world_size, num_nodes);
                if (peer < 0 || peer >= world_size || peer == r) continue;
                if (!seen_remote_targets[peer].insert(dst_global).second) continue;
                plans[r][peer].act_send_local.push_back(dst_local);
                plans[peer][r].act_recv_owned.push_back(dst_global - parts[peer].start_node);
            }
        }
    }

    for (int r = 0; r < world_size; ++r) {
        const DgSubgraphHost& recv_part = parts[r];
        for (size_t i = 0; i < recv_part.ghost_global_ids.size(); ++i) {
            int ghost_global = recv_part.ghost_global_ids[i];
            if (ghost_global < 0 || ghost_global >= num_nodes) continue;
            int owner = dg_owner_of_vertex(ghost_global, world_size, num_nodes);
            if (owner < 0 || owner >= world_size || owner == r) continue;
            plans[owner][r].value_send_owned.push_back(ghost_global - parts[owner].start_node);
            plans[r][owner].value_recv_ghost.push_back(recv_part.ghost_local_ids[i]);
        }
    }
}

inline void dg_upload_index(const std::vector<int>& h, int** d) {
    if (h.empty()) {
        *d = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(d, h.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
}

inline void dg_alloc_int(int** d, int n) {
    if (n <= 0) {
        *d = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(d, n * sizeof(int)));
}

inline void dg_alloc_float(float** d, int n) {
    if (n <= 0) {
        *d = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(d, n * sizeof(float)));
}

inline void dg_upload_peer_plans(
    const std::vector<std::vector<DgPeerPlanHost>>& host_plans,
    int rank,
    int world_size,
    std::vector<DgPeerPlanDevice>& dev_plans,
    bool need_float_values
) {
    dev_plans.resize(world_size);
    for (int peer = 0; peer < world_size; ++peer) {
        const DgPeerPlanHost& hp = host_plans[rank][peer];
        DgPeerPlanDevice& dp = dev_plans[peer];

        dp.act_send_count = (int)hp.act_send_local.size();
        dp.act_recv_count = (int)hp.act_recv_owned.size();
        dp.value_send_count = (int)hp.value_send_owned.size();
        dp.value_recv_count = (int)hp.value_recv_ghost.size();

        dg_upload_index(hp.act_send_local, &dp.d_act_send_local);
        dg_upload_index(hp.act_recv_owned, &dp.d_act_recv_owned);
        dg_upload_index(hp.value_send_owned, &dp.d_value_send_owned);
        dg_upload_index(hp.value_recv_ghost, &dp.d_value_recv_ghost);

        dg_alloc_int(&dp.d_act_send_buf, dp.act_send_count);
        dg_alloc_int(&dp.d_act_recv_buf, dp.act_recv_count);

        if (need_float_values) {
            dg_alloc_float(&dp.d_value_send_buf_float, dp.value_send_count);
            dg_alloc_float(&dp.d_value_recv_buf_float, dp.value_recv_count);
        } else {
            dg_alloc_int(&dp.d_value_send_buf_int, dp.value_send_count);
            dg_alloc_int(&dp.d_value_recv_buf_int, dp.value_recv_count);
        }
    }
}

inline void dg_free_peer_plans(std::vector<DgPeerPlanDevice>& dev_plans) {
    for (DgPeerPlanDevice& p : dev_plans) {
        cudaFree(p.d_act_send_local);
        cudaFree(p.d_act_recv_owned);
        cudaFree(p.d_value_send_owned);
        cudaFree(p.d_value_recv_ghost);
        cudaFree(p.d_act_send_buf);
        cudaFree(p.d_act_recv_buf);
        cudaFree(p.d_value_send_buf_int);
        cudaFree(p.d_value_recv_buf_int);
        cudaFree(p.d_value_send_buf_float);
        cudaFree(p.d_value_recv_buf_float);
    }
}

__global__ void dgPackIntKernel(const int* src, const int* idx, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[idx[i]];
}

__global__ void dgUnpackIntKernel(const int* in, const int* idx, int* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[idx[i]] = in[i];
}

__global__ void dgPackFloatKernel(const float* src, const int* idx, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[idx[i]];
}

__global__ void dgUnpackFloatKernel(const float* in, const int* idx, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[idx[i]] = in[i];
}

__global__ void dgApplyActivationKernel(
    const int* flags,
    const int* owned_idx,
    int* next_active,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i] != -1) next_active[owned_idx[i]] = 1;
}

__global__ void dgCopyOwnedUpdateKernel(
    const int* update,
    int* next_active,
    int owned_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < owned_count) next_active[i] = update[i];
}

__global__ void dgCountActiveKernel(const int* active, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && active[i] != -1) atomicAdd(count, 1);
}

inline int dg_blocks(int n) {
    return (n + DG_BLOCK_SIZE - 1) / DG_BLOCK_SIZE;
}

/**
 * @brief Exchange sparse remote activation flags and apply them at each owner.
 * @param d_update Local update array spanning owned and ghost indices.
 * @param d_next_active Owned-only work set receiving local and remote updates.
 */
inline void dg_nccl_exchange_activation(
    int* d_update,
    int* d_next_active,
    std::vector<DgPeerPlanDevice>& plans,
    int world_size,
    int rank,
    ncclComm_t comm,
    cudaStream_t stream
) {
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.act_send_count > 0) {
            dgPackIntKernel<<<dg_blocks(p.act_send_count), DG_BLOCK_SIZE, 0, stream>>>(
                d_update,
                p.d_act_send_local,
                p.d_act_send_buf,
                p.act_send_count
            );
        }
    }

    NCCL_CHECK(ncclGroupStart());
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.act_recv_count > 0) {
            NCCL_CHECK(ncclRecv(
                p.d_act_recv_buf,
                p.act_recv_count,
                ncclInt,
                peer,
                comm,
                stream
            ));
        }
        if (p.act_send_count > 0) {
            NCCL_CHECK(ncclSend(
                p.d_act_send_buf,
                p.act_send_count,
                ncclInt,
                peer,
                comm,
                stream
            ));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.act_recv_count > 0) {
            dgApplyActivationKernel<<<dg_blocks(p.act_recv_count), DG_BLOCK_SIZE, 0, stream>>>(
                p.d_act_recv_buf,
                p.d_act_recv_owned,
                d_next_active,
                p.act_recv_count
            );
        }
    }
}

/** @brief Refresh integer-valued ghost caches using precomputed peer plans. */
inline void dg_nccl_exchange_int_values(
    int* d_values,
    std::vector<DgPeerPlanDevice>& plans,
    int world_size,
    int rank,
    ncclComm_t comm,
    cudaStream_t stream
) {
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.value_send_count > 0) {
            dgPackIntKernel<<<dg_blocks(p.value_send_count), DG_BLOCK_SIZE, 0, stream>>>(
                d_values,
                p.d_value_send_owned,
                p.d_value_send_buf_int,
                p.value_send_count
            );
        }
    }

    NCCL_CHECK(ncclGroupStart());
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.value_recv_count > 0) {
            NCCL_CHECK(ncclRecv(
                p.d_value_recv_buf_int,
                p.value_recv_count,
                ncclInt,
                peer,
                comm,
                stream
            ));
        }
        if (p.value_send_count > 0) {
            NCCL_CHECK(ncclSend(
                p.d_value_send_buf_int,
                p.value_send_count,
                ncclInt,
                peer,
                comm,
                stream
            ));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.value_recv_count > 0) {
            dgUnpackIntKernel<<<dg_blocks(p.value_recv_count), DG_BLOCK_SIZE, 0, stream>>>(
                p.d_value_recv_buf_int,
                p.d_value_recv_ghost,
                d_values,
                p.value_recv_count
            );
        }
    }
}

/** @brief Refresh floating-point ghost caches using precomputed peer plans. */
inline void dg_nccl_exchange_float_values(
    float* d_values,
    std::vector<DgPeerPlanDevice>& plans,
    int world_size,
    int rank,
    ncclComm_t comm,
    cudaStream_t stream
) {
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.value_send_count > 0) {
            dgPackFloatKernel<<<dg_blocks(p.value_send_count), DG_BLOCK_SIZE, 0, stream>>>(
                d_values,
                p.d_value_send_owned,
                p.d_value_send_buf_float,
                p.value_send_count
            );
        }
    }

    NCCL_CHECK(ncclGroupStart());
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.value_recv_count > 0) {
            NCCL_CHECK(ncclRecv(
                p.d_value_recv_buf_float,
                p.value_recv_count,
                ncclFloat,
                peer,
                comm,
                stream
            ));
        }
        if (p.value_send_count > 0) {
            NCCL_CHECK(ncclSend(
                p.d_value_send_buf_float,
                p.value_send_count,
                ncclFloat,
                peer,
                comm,
                stream
            ));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        DgPeerPlanDevice& p = plans[peer];
        if (p.value_recv_count > 0) {
            dgUnpackFloatKernel<<<dg_blocks(p.value_recv_count), DG_BLOCK_SIZE, 0, stream>>>(
                p.d_value_recv_buf_float,
                p.d_value_recv_ghost,
                d_values,
                p.value_recv_count
            );
        }
    }
}

/** @brief Bind each node-local MPI rank to one visible CUDA device. */
inline int dg_select_rank_device() {
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(
        MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        0,
        MPI_INFO_NULL,
        &local_comm
    );
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "No CUDA device visible for this MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

/** @brief Create one NCCL communicator spanning all MPI ranks. */
inline ncclComm_t dg_create_nccl_comm(int rank, int world_size) {
    ncclUniqueId id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
    return comm;
}

inline std::vector<int> dg_owned_counts(int world_size, int num_nodes) {
    std::vector<int> counts(world_size);
    for (int r = 0; r < world_size; ++r) {
        counts[r] = dg_partition_end(r, world_size, num_nodes)
            - dg_partition_start(r, world_size, num_nodes);
    }
    return counts;
}

inline std::vector<int> dg_displacements(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    for (size_t i = 1; i < counts.size(); ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    return displs;
}

#endif
