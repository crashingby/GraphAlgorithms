/**
 * @file bfs_tolerance_multiGPU.cuh
 * @brief Distributed BFS with selective DMR and asynchronous CPU checks.
 *
 * This implementation keeps the historical BFS-specific owned/ghost partition
 * and peer plans. Each rank owns a compute stream, a nonblocking check stream,
 * a reusable pinned summary pool, and one CPU consumer thread.
 */
#ifndef FAULT_TOLERANT_BFS_MULTIGPU_NCCL_PARTITIONED_CUH
#define FAULT_TOLERANT_BFS_MULTIGPU_NCCL_PARTITIONED_CUH

#include <cuda_runtime.h>
#include <nccl.h>
#include <nvToolsExt.h>
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>

#include "include/spsc_queue.h"

#define BLOCK_SIZE 256
#define INF 100000
#define MAX_GPUS 8
#define CHECK_BUFFER_COUNT 64

// ==================== 数据结构 ====================
/** @brief Expected per-vertex value direction used by the detector. */
enum ValueTrend { TREND_NONE = 0, TREND_INC = 1, TREND_DEC = 2 };

/** @brief Per-rank, per-iteration device summary for asynchronous checking. */
struct MonotonicInfo {
    int dmr_error_flag;
    int monotonic_error_flag;
    unsigned long long sum_abs_delta;
    int count_update;
};

/** @brief Completed host summary retained until cross-rank aggregation. */
struct AsyncCheckResult {
    bool done = false;
    unsigned long long sum_abs_delta = 0;
    int count_update = 0;
    int dmr_error = 0;
    int monotonic_error = 0;
};

/** @brief SPSC task that associates an iteration with a check buffer. */
struct CheckTask {
    int iter = 0;
    int buf = -1;
};

/** @brief RAII guard for one NVTX profiling range. */
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    explicit NvtxRange(const std::string& name) : name_(name) { nvtxRangePushA(name_.c_str()); }
    ~NvtxRange() { nvtxRangePop(); }

    std::string name_;
};

/** @brief Format a stable NVTX range name for an iteration and rank-local GPU. */
inline std::string nvtx_name(const char* label, int iter, int gpu) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s iter=%d gpu=%d", label, iter, gpu);
    return std::string(buf);
}

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

// ==================== GPU Kernels ====================
// 说明：保留原 kernel 的 DMR 冗余计算语义：
// - active 线程执行主计算；
// - idle 线程帮 critical 顶点重复计算；
// - critical 主/副结果不一致则设置 dmr_error_flag。
// 但索引空间改成“局部子图”：owned 顶点排在 [0, owned_count)，ghost 顶点排在后面。
/**
 * @brief Relax owned BFS vertices and duplicate selected critical pulls.
 *
 * Idle lanes recompute critical owned vertices from the same owned-plus-ghost
 * arrays. There is no grid-wide snapshot: another block may update a value
 * between the primary and redundant reads, so a mismatch is anomaly evidence
 * rather than proof of a hardware fault. The primary result remains
 * authoritative and no recovery is attempted.
 */
__global__ void bfsPullDualMultiGPUKernel(
    int* d_values,
    const int* d_row_offsets, const int* d_column_indices,
    const int* d_column_offsets, const int* d_row_indices,
    const int* d_active, int* d_update,
    int owned_count, int local_node_count,
    MonotonicInfo* d_info, ValueTrend trend
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // local owned id
    int local_tid = threadIdx.x;

    __shared__ int critical_list[BLOCK_SIZE];
    __shared__ int idle_count;
    __shared__ int critical_count;
    __shared__ int redundant_results[BLOCK_SIZE];

    if (local_tid == 0) { idle_count = 0; critical_count = 0; }
    __syncthreads();

    bool valid_tid = (tid < owned_count);
    bool is_active = valid_tid && (d_active[tid] != -1);
    bool is_critical = is_active && (d_active[tid] == 2);
    bool is_idle = (!is_active || !valid_tid);

    int cid = -1;
    if (is_critical) {
        cid = atomicAdd(&critical_count, 1);
        if (cid < BLOCK_SIZE) critical_list[cid] = tid;
    }

    int iid = -1;
    if (is_idle) iid = atomicAdd(&idle_count, 1);
    __syncthreads();

    int oldVal = INF;
    if (valid_tid) oldVal = d_values[tid];

    int main_newVal = INF;
    if (is_active) {
        main_newVal = d_values[tid];
        for (int i = d_column_offsets[tid]; i < d_column_offsets[tid + 1]; i++) {
            int src_local = d_row_indices[i];              // owned or ghost local id
            if (src_local >= 0 && src_local < local_node_count) {
                int candidate = d_values[src_local] + 1;   // ghost value is local cache
                if (candidate < main_newVal) main_newVal = candidate;
            }
        }
    }
    __syncthreads();

    if (is_idle) {
        int my_idle_id = iid;
        if (my_idle_id < critical_count && my_idle_id < BLOCK_SIZE) {
            int target_tid = critical_list[my_idle_id];
            int redundant_newVal = d_values[target_tid];
            for (int i = d_column_offsets[target_tid]; i < d_column_offsets[target_tid + 1]; i++) {
                int src_local = d_row_indices[i];
                if (src_local >= 0 && src_local < local_node_count) {
                    int candidate = d_values[src_local] + 1;
                    if (candidate < redundant_newVal) redundant_newVal = candidate;
                }
            }
            redundant_results[my_idle_id] = redundant_newVal;
        }
    }
    __syncthreads();

    if (is_critical) {
        int my_index = cid;
        if (my_index >= 0 && my_index < idle_count && my_index < BLOCK_SIZE) {
            int redundant_newVal = redundant_results[my_index];
            if (redundant_newVal != main_newVal) {
                atomicExch(&(d_info->dmr_error_flag), 1);
            }
        }
    }
    __syncthreads();

    if (is_active) {
        d_values[tid] = main_newVal;
        if (main_newVal < oldVal) {
            // scatter 激活：出邻居可能是 owned，也可能是 ghost。
            // ghost 激活不会在本 GPU 生效，会由 host 侧通信计划打包发给 owner GPU。
            for (int i = d_row_offsets[tid]; i < d_row_offsets[tid + 1]; i++) {
                int dst_local = d_column_indices[i];
                if (dst_local >= 0 && dst_local < local_node_count) {
                    d_update[dst_local] = 1;
                }
            }
            int delta = main_newVal - oldVal;
            unsigned long long abs_delta = (delta < 0)
                ? (unsigned long long)(-(long long)delta)
                : (unsigned long long)delta;
            atomicAdd(&(d_info->sum_abs_delta), abs_delta);
            atomicAdd(&(d_info->count_update), 1);
            if ((trend == TREND_INC && delta < 0) || (trend == TREND_DEC && delta > 0)) {
                atomicExch(&(d_info->monotonic_error_flag), 1);
            }
        }
    }
}

/** @brief Copy locally owned activation flags into the next work set. */
__global__ void copyOwnedUpdateToNextActiveKernel(
    const int* d_update, int* d_next_active, int owned_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < owned_count) d_next_active[i] = d_update[i];
}

/** @brief Gather sparse integer entries into an NCCL send buffer. */
__global__ void packByIndexKernel(
    const int* d_src, const int* d_indices, int* d_out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_out[i] = d_src[d_indices[i]];
}

/** @brief Scatter an NCCL receive buffer into sparse local indices. */
__global__ void unpackByIndexKernel(
    const int* d_in, const int* d_indices, int* d_dst, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_dst[d_indices[i]] = d_in[i];
}

/** @brief Merge received remote activation flags into owned work. */
__global__ void applyActivationRecvKernel(
    const int* d_flags, const int* d_owned_indices, int* d_next_active, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_flags[i] != -1) d_next_active[d_owned_indices[i]] = 1;
}

/**
 * @brief Classify the owned work set as inactive, ordinary, or critical.
 * @note Encodings are -1, 1, and 2 respectively.
 */
__global__ void scoreAndMarkKernel(
    int* d_active,
    const int* d_row_offsets,
    int num_nodes,
    int max_outdegree,
    float alpha,
    float beta,
    float threshold,
    int* d_num_active
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        int val = d_active[v];
        if (val != -1) {
            atomicAdd(d_num_active, 1);

            int outdeg = d_row_offsets[v + 1] - d_row_offsets[v];
            int safe_max_outdegree = (max_outdegree > 0) ? max_outdegree : 1;

            float dist_score = 1.0f / (1.0f + (float)val);
            float degree_score = (float)outdeg / (float)safe_max_outdegree;
            float score = alpha * degree_score + beta * dist_score;

            // 2 表示关键活跃顶点，1 表示普通活跃顶点，-1 表示非活跃。
            d_active[v] = (score >= threshold) ? 2 : 1;
        }
    }
}

// ==================== CPU 辅助函数 ====================
/** @brief Count host work-set entries whose inactive sentinel is not set. */
inline int count_active_nodes(const int* h_active, int num_nodes) {
    int cnt = 0;
    for (int i = 0; i < num_nodes; i++) if (h_active[i] != -1) cnt++;
    return cnt;
}

/** @brief Compute a nonzero maximum outdegree for score normalization. */
inline int compute_max_outdegree(const int* h_row_offsets, int num_nodes) {
    int max_outdegree = 0;
    for (int v = 0; v < num_nodes; v++) {
        int outdegree = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdegree > max_outdegree) max_outdegree = outdegree;
    }
    return std::max(max_outdegree, 1);
}



/** @brief Map a global vertex to its owner in the historical ceil partition. */
inline int owner_of_vertex(int v, int nodes_per_gpu, int num_gpus) {
    int owner = v / nodes_per_gpu;
    if (owner >= num_gpus) owner = num_gpus - 1;
    return owner;
}

/** @brief Historical BFS-specific host representation of one partition. */
struct GpuSubgraphHost {
    int start_node = 0;
    int end_node = 0;
    int owned_count = 0;
    int local_node_count = 0;

    std::vector<int> local_to_global;  // owned first, then ghosts
    std::unordered_map<int, int> global_to_local;

    std::vector<int> row_offsets;       // length owned_count + 1
    std::vector<int> column_indices;    // outgoing dst local ids, owned or ghost
    std::vector<int> column_offsets;    // length owned_count + 1
    std::vector<int> row_indices;       // incoming src local ids, owned or ghost

    std::vector<int> ghost_local_ids;
    std::vector<int> ghost_global_ids;
    std::vector<int> ghost_owner;
};

/** @brief Resolve a global vertex to an existing or newly appended local index. */
inline int get_or_add_local_vertex(GpuSubgraphHost& part, int global_v, bool owned) {
    auto it = part.global_to_local.find(global_v);
    if (it != part.global_to_local.end()) return it->second;

    int local = (int)part.local_to_global.size();
    part.global_to_local[global_v] = local;
    part.local_to_global.push_back(global_v);

    if (!owned) {
        part.ghost_local_ids.push_back(local);
        part.ghost_global_ids.push_back(global_v);
    }
    return local;
}

/** @brief Build one BFS owned-plus-ghost partition from complete host CSR data. */
inline void build_vertex_partition_subgraph(
    int gpu_id, int num_gpus, int num_nodes, int nodes_per_gpu,
    const int* h_row_offsets, const int* h_column_indices,
    const int* h_column_offsets, const int* h_row_indices,
    GpuSubgraphHost& part
) {
    part.start_node = gpu_id * nodes_per_gpu;
    part.end_node = std::min(part.start_node + nodes_per_gpu, num_nodes);
    part.owned_count = std::max(0, part.end_node - part.start_node);

    part.local_to_global.clear();
    part.global_to_local.clear();
    part.ghost_local_ids.clear();
    part.ghost_global_ids.clear();
    part.ghost_owner.clear();

    // owned 顶点必须排在最前面，这样 local_id == global_id - start_node。
    for (int v = part.start_node; v < part.end_node; ++v) {
        int local = (int)part.local_to_global.size();
        part.global_to_local[v] = local;
        part.local_to_global.push_back(v);
    }

    part.row_offsets.assign(part.owned_count + 1, 0);
    part.column_offsets.assign(part.owned_count + 1, 0);
    part.column_indices.clear();
    part.row_indices.clear();

    int out_cursor = 0;
    for (int local_v = 0; local_v < part.owned_count; ++local_v) {
        int global_v = part.start_node + local_v;
        part.row_offsets[local_v] = out_cursor;
        for (int e = h_row_offsets[global_v]; e < h_row_offsets[global_v + 1]; ++e) {
            int global_dst = h_column_indices[e];
            bool owned_dst = (global_dst >= part.start_node && global_dst < part.end_node);
            int local_dst = get_or_add_local_vertex(part, global_dst, owned_dst);
            part.column_indices.push_back(local_dst);
            ++out_cursor;
        }
        part.row_offsets[local_v + 1] = out_cursor;
    }

    int in_cursor = 0;
    for (int local_v = 0; local_v < part.owned_count; ++local_v) {
        int global_v = part.start_node + local_v;
        part.column_offsets[local_v] = in_cursor;
        for (int e = h_column_offsets[global_v]; e < h_column_offsets[global_v + 1]; ++e) {
            int global_src = h_row_indices[e];
            bool owned_src = (global_src >= part.start_node && global_src < part.end_node);
            int local_src = get_or_add_local_vertex(part, global_src, owned_src);
            part.row_indices.push_back(local_src);
            ++in_cursor;
        }
        part.column_offsets[local_v + 1] = in_cursor;
    }

    part.local_node_count = (int)part.local_to_global.size();
    for (int gid : part.ghost_global_ids) {
        part.ghost_owner.push_back(owner_of_vertex(gid, nodes_per_gpu, num_gpus));
    }

    if (part.column_indices.empty()) part.column_indices.push_back(0);
    if (part.row_indices.empty()) part.row_indices.push_back(0);
}

/** @brief Sparse activation and value-exchange indices for one BFS peer. */
struct PeerPlanHost {
    std::vector<int> act_send_local;       // sender ghost local ids, read d_update[ghost]
    std::vector<int> act_recv_owned;       // receiver owned local ids, write d_next_active[owned]
    std::vector<int> value_send_owned;     // owner local ids, read d_values[owned]
    std::vector<int> value_recv_ghost;     // receiver ghost local ids, write d_values[ghost]
};

/** @brief Device-side indices and NCCL staging buffers for one BFS peer. */
struct PeerPlanDevice {
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
    int* d_value_send_buf = nullptr;
    int* d_value_recv_buf = nullptr;
};

/** @brief Upload a host index vector, preserving nullptr for an empty plan. */
inline void upload_index_vector(const std::vector<int>& h, int** d_ptr) {
    if (h.empty()) { *d_ptr = nullptr; return; }
    CUDA_CHECK(cudaMalloc(d_ptr, h.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
}

/** @brief Allocate an integer staging buffer, preserving nullptr for zero size. */
inline void alloc_int_buffer(int** d_ptr, int n) {
    if (n <= 0) { *d_ptr = nullptr; return; }
    CUDA_CHECK(cudaMalloc(d_ptr, n * sizeof(int)));
}

// ==================== 主函数：MPI rank + Ghost 节点 + NCCL 点对点通信 ====================
/**
 * @brief Execute distributed BFS with asynchronous queue-based checks.
 *
 * The main thread produces per-iteration summaries into a bounded reusable
 * buffer pool. A CPU worker consumes event-complete summaries concurrently.
 * After graph convergence, MPI collectives aggregate only iterations checked
 * by every rank, then report global DMR, monotonic, and residual anomalies.
 * A buffer-starved iteration still executes with scratch storage but contributes
 * no detection evidence to the final report.
 *
 * @param h_value Output distances for all global vertices.
 * @param h_row_offsets Global outgoing CSR offsets.
 * @param h_column_indices Global outgoing CSR destinations.
 * @param h_column_offsets Global incoming CSR offsets.
 * @param h_row_indices Global incoming CSR sources.
 * @param num_nodes Global vertex count.
 * @param num_edges Global edge count, retained for the common entry signature.
 * @param src BFS source vertex.
 * @param alpha Outdegree weight in criticality scoring.
 * @param beta Active-value weight in criticality scoring.
 * @param threshold Critical-vertex score threshold.
 *
 * @pre MPI has been initialized with at least MPI_THREAD_FUNNELED.
 * @pre Every rank supplies the same graph and can access a CUDA device.
 */
inline void bfsMultiGPU(
    int* h_value,
    const int* h_row_offsets, const int* h_column_indices,
    const int* h_column_offsets, const int* h_row_indices,
    int num_nodes, int num_edges, int src,
    float alpha = 0.5f,
    float beta = 0.5f,
    float threshold = 0.6f
) {
    (void)num_edges;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        fprintf(stderr, "bfsMultiGPU 需要先调用 MPI_Init。\n");
        exit(EXIT_FAILURE);
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 0) {
        fprintf(stderr, "MPI world_size 非法。\n");
        exit(EXIT_FAILURE);
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    int local_device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&local_device_count));
    if (local_device_count <= 0) {
        fprintf(stderr, "rank %d: 当前 host 没有可用 GPU。\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int device_id = local_rank % local_device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    ncclUniqueId nccl_id;
    if (world_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_id, world_rank));

    int nodes_per_rank = (num_nodes + world_size - 1) / world_size;
    int max_outdegree = compute_max_outdegree(h_row_offsets, num_nodes);

    if (world_rank == 0) {
        printf("发现 %d 个 MPI rank，开启 NCCL 分布式 BFS：每 rank 绑定一张 GPU，owned + ghost + send/recv activation/value。\n", world_size);
    }

    // 每个 rank 都构造全局 host 侧划分，用于生成对等通信计划；device 侧只上传本 rank 的子图。
    std::vector<GpuSubgraphHost> parts(world_size);
    for (int r = 0; r < world_size; ++r) {
        build_vertex_partition_subgraph(
            r, world_size, num_nodes, nodes_per_rank,
            h_row_offsets, h_column_indices, h_column_offsets, h_row_indices,
            parts[r]
        );
        if (world_rank == 0) {
            printf("Rank %d: owned [%d, %d), owned_count=%d, local_count=%d, ghosts=%zu\n",
                   r, parts[r].start_node, parts[r].end_node,
                   parts[r].owned_count, parts[r].local_node_count,
                   parts[r].ghost_local_ids.size());
        }
    }
    const auto& part = parts[world_rank];

    std::vector<std::vector<PeerPlanHost>> h_plan(world_size, std::vector<PeerPlanHost>(world_size));
    for (int sender = 0; sender < world_size; ++sender) {
        const auto& sp = parts[sender];
        for (size_t k = 0; k < sp.ghost_local_ids.size(); ++k) {
            int ghost_local = sp.ghost_local_ids[k];
            int ghost_global = sp.ghost_global_ids[k];
            int owner = owner_of_vertex(ghost_global, nodes_per_rank, world_size);
            if (owner == sender) continue;

            auto it = parts[owner].global_to_local.find(ghost_global);
            if (it == parts[owner].global_to_local.end()) continue;
            int owner_local = it->second;

            h_plan[sender][owner].act_send_local.push_back(ghost_local);
            h_plan[owner][sender].act_recv_owned.push_back(owner_local);
            h_plan[owner][sender].value_send_owned.push_back(owner_local);
            h_plan[sender][owner].value_recv_ghost.push_back(ghost_local);
        }
    }

    for (int i = 0; i < num_nodes; ++i) h_value[i] = INF;
    h_value[src] = 0;

    int* h_active_global = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_active_global, num_nodes * sizeof(int)));
    for (int i = 0; i < num_nodes; ++i) h_active_global[i] = -1;
    h_active_global[src] = 1;
    for (int e = h_row_offsets[src]; e < h_row_offsets[src + 1]; ++e) {
        int dst = h_column_indices[e];
        if (dst >= 0 && dst < num_nodes) h_active_global[dst] = 1;
    }

    int local_alloc_count = std::max(1, part.local_node_count);
    int owned_alloc_count = std::max(1, part.owned_count);

    int *d_values = nullptr, *d_active = nullptr, *d_next_active = nullptr, *d_update = nullptr;
    int *d_row_offsets = nullptr, *d_column_indices = nullptr;
    int *d_column_offsets = nullptr, *d_row_indices = nullptr;
    int *d_num_active = nullptr;
    MonotonicInfo* d_info_scratch = nullptr;
    std::vector<MonotonicInfo*> d_info(CHECK_BUFFER_COUNT, nullptr);
    std::vector<MonotonicInfo*> h_info(CHECK_BUFFER_COUNT, nullptr);
    std::vector<cudaEvent_t> compute_done_events(CHECK_BUFFER_COUNT);
    std::vector<cudaEvent_t> check_events(CHECK_BUFFER_COUNT);

    CUDA_CHECK(cudaMalloc(&d_values, local_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, owned_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_active, owned_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, local_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, part.column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_indices, part.row_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_active, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info_scratch, sizeof(MonotonicInfo)));
    for (int b = 0; b < CHECK_BUFFER_COUNT; ++b) {
        CUDA_CHECK(cudaMalloc(&d_info[b], sizeof(MonotonicInfo)));
        CUDA_CHECK(cudaMallocHost(&h_info[b], sizeof(MonotonicInfo)));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_events[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&check_events[b], cudaEventDisableTiming));
    }

    std::vector<int> h_local_values(local_alloc_count, INF);
    std::vector<int> h_local_active(owned_alloc_count, -1);
    for (int local = 0; local < part.local_node_count; ++local) {
        int global = part.local_to_global[local];
        h_local_values[local] = h_value[global];
    }
    for (int local = 0; local < part.owned_count; ++local) {
        int global = part.start_node + local;
        h_local_active[local] = h_active_global[global];
    }

    CUDA_CHECK(cudaMemcpy(d_values, h_local_values.data(), local_alloc_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, h_local_active.data(), owned_alloc_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_update, 0xff, local_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_next_active, 0xff, owned_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, part.row_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_offsets, part.column_offsets.data(), (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_indices, part.column_indices.data(), part.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_indices, part.row_indices.data(), part.row_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t stream, check_stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(&check_stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMemsetAsync(d_num_active, 0, sizeof(int), stream));
    if (part.owned_count > 0) {
        int blocks = (part.owned_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scoreAndMarkKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_active, d_row_offsets, part.owned_count, max_outdegree,
            alpha, beta, threshold, d_num_active
        );
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<PeerPlanDevice> d_plan(world_size);
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == world_rank) continue;
        auto& hp = h_plan[world_rank][peer];
        auto& dp = d_plan[peer];
        dp.act_send_count = (int)hp.act_send_local.size();
        dp.act_recv_count = (int)hp.act_recv_owned.size();
        dp.value_send_count = (int)hp.value_send_owned.size();
        dp.value_recv_count = (int)hp.value_recv_ghost.size();

        upload_index_vector(hp.act_send_local, &dp.d_act_send_local);
        upload_index_vector(hp.act_recv_owned, &dp.d_act_recv_owned);
        upload_index_vector(hp.value_send_owned, &dp.d_value_send_owned);
        upload_index_vector(hp.value_recv_ghost, &dp.d_value_recv_ghost);

        alloc_int_buffer(&dp.d_act_send_buf, dp.act_send_count);
        alloc_int_buffer(&dp.d_act_recv_buf, dp.act_recv_count);
        alloc_int_buffer(&dp.d_value_send_buf, dp.value_send_count);
        alloc_int_buffer(&dp.d_value_recv_buf, dp.value_recv_count);
    }

    SpscQueue<int, CHECK_BUFFER_COUNT> free_buffers;
    SpscQueue<CheckTask, CHECK_BUFFER_COUNT> pending_tasks;
    for (int b = 0; b < CHECK_BUFFER_COUNT; ++b) free_buffers.try_push(b);
    std::atomic<bool> check_stop(false);
    std::vector<AsyncCheckResult> check_results(1001);

    std::thread check_worker([&]() {
        CUDA_CHECK(cudaSetDevice(device_id));
        while (!check_stop.load(std::memory_order_relaxed) || !pending_tasks.empty()) {
            bool progressed = false;
            CheckTask task;
            if (pending_tasks.peek(task)) {
                cudaError_t event_status = cudaEventQuery(check_events[task.buf]);
                if (event_status == cudaSuccess) {
                    pending_tasks.try_pop(task);
                    NvtxRange scan_range(nvtx_name("worker CPU scan", task.iter, world_rank));

                    AsyncCheckResult result;
                    result.done = true;
                    result.dmr_error = h_info[task.buf]->dmr_error_flag;
                    result.monotonic_error = h_info[task.buf]->monotonic_error_flag;
                    result.sum_abs_delta = h_info[task.buf]->sum_abs_delta;
                    result.count_update = h_info[task.buf]->count_update;
                    if (task.iter >= 0 && task.iter < (int)check_results.size()) {
                        check_results[task.iter] = result;
                    }
                    free_buffers.try_push(task.buf);
                    progressed = true;
                } else if (event_status != cudaErrorNotReady) {
                    CUDA_CHECK(event_status);
                }
            }
            if (!progressed) std::this_thread::yield();
        }
    });

    auto launch_pack_activation = [&]() {
        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == world_rank) continue;
            auto& dp = d_plan[peer];
            if (dp.act_send_count > 0) {
                int blocks = (dp.act_send_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                packByIndexKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    d_update, dp.d_act_send_local, dp.d_act_send_buf, dp.act_send_count
                );
            }
        }
    };

    auto launch_apply_activation = [&]() {
        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == world_rank) continue;
            auto& dp = d_plan[peer];
            if (dp.act_recv_count > 0) {
                int blocks = (dp.act_recv_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                applyActivationRecvKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    dp.d_act_recv_buf, dp.d_act_recv_owned, d_next_active, dp.act_recv_count
                );
            }
        }
    };

    auto launch_pack_values = [&]() {
        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == world_rank) continue;
            auto& dp = d_plan[peer];
            if (dp.value_send_count > 0) {
                int blocks = (dp.value_send_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                packByIndexKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    d_values, dp.d_value_send_owned, dp.d_value_send_buf, dp.value_send_count
                );
            }
        }
    };

    auto launch_unpack_values = [&]() {
        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == world_rank) continue;
            auto& dp = d_plan[peer];
            if (dp.value_recv_count > 0) {
                int blocks = (dp.value_recv_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                unpackByIndexKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    dp.d_value_recv_buf, dp.d_value_recv_ghost, d_values, dp.value_recv_count
                );
            }
        }
    };

    auto nccl_exchange_activation = [&]() {
        NCCL_CHECK(ncclGroupStart());
        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == world_rank) continue;
            auto& dp = d_plan[peer];
            if (dp.act_recv_count > 0) {
                NCCL_CHECK(ncclRecv(dp.d_act_recv_buf, dp.act_recv_count, ncclInt, peer, comm, stream));
            }
            if (dp.act_send_count > 0) {
                NCCL_CHECK(ncclSend(dp.d_act_send_buf, dp.act_send_count, ncclInt, peer, comm, stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
    };

    auto nccl_exchange_values = [&]() {
        NCCL_CHECK(ncclGroupStart());
        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == world_rank) continue;
            auto& dp = d_plan[peer];
            if (dp.value_recv_count > 0) {
                NCCL_CHECK(ncclRecv(dp.d_value_recv_buf, dp.value_recv_count, ncclInt, peer, comm, stream));
            }
            if (dp.value_send_count > 0) {
                NCCL_CHECK(ncclSend(dp.d_value_send_buf, dp.value_send_count, ncclInt, peer, comm, stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
    };

    launch_pack_values();
    nccl_exchange_values();
    launch_unpack_values();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt, stream));

    int iter = 0;
    while (iter < 1000) {
        ++iter;

        {
            NvtxRange enqueue_range(nvtx_name("main enqueue bfs/check", iter, world_rank));
            int check_buf = -1;
            bool do_async_check = free_buffers.try_pop(check_buf);
            MonotonicInfo* iter_info = do_async_check ? d_info[check_buf] : d_info_scratch;

            CUDA_CHECK(cudaMemsetAsync(iter_info, 0, sizeof(MonotonicInfo), stream));
            CUDA_CHECK(cudaMemsetAsync(d_update, 0xff, local_alloc_count * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(d_next_active, 0xff, owned_alloc_count * sizeof(int), stream));

            if (part.owned_count > 0) {
                int blocks = (part.owned_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                bfsPullDualMultiGPUKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    d_values, d_row_offsets, d_column_indices,
                    d_column_offsets, d_row_indices,
                    d_active, d_update,
                    part.owned_count, part.local_node_count,
                    iter_info, TREND_DEC
                );
                CUDA_CHECK(cudaGetLastError());

                copyOwnedUpdateToNextActiveKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    d_update, d_next_active, part.owned_count
                );
            }

            if (do_async_check) {
                NvtxRange copy_enqueue_range(nvtx_name("main enqueue check_stream D2H", iter, world_rank));
                CUDA_CHECK(cudaEventRecord(compute_done_events[check_buf], stream));
                CUDA_CHECK(cudaStreamWaitEvent(check_stream, compute_done_events[check_buf], 0));
                CUDA_CHECK(cudaMemcpyAsync(
                    h_info[check_buf],
                    d_info[check_buf],
                    sizeof(MonotonicInfo),
                    cudaMemcpyDeviceToHost,
                    check_stream
                ));
                CUDA_CHECK(cudaEventRecord(check_events[check_buf], check_stream));
                // Keep queue ownership strictly SPSC: only the worker returns
                // slots to free_buffers. With equal pool and queue capacities,
                // this loop normally succeeds on its first attempt.
                while (!pending_tasks.try_push({iter, check_buf})) {
                    std::this_thread::yield();
                }
            }
        }

        {
            NvtxRange range(nvtx_name("main NCCL activation", iter, -1));
            launch_pack_activation();
            nccl_exchange_activation();
            launch_apply_activation();
        }

        {
            NvtxRange range(nvtx_name("main NCCL values", iter, -1));
            launch_pack_values();
            nccl_exchange_values();
            launch_unpack_values();
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));

        int local_active_nodes = 0;
        {
            NvtxRange range(nvtx_name("main score active", iter, world_rank));
            CUDA_CHECK(cudaMemsetAsync(d_num_active, 0, sizeof(int), stream));
            if (part.owned_count > 0) {
                int blocks = (part.owned_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                scoreAndMarkKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    d_next_active, d_row_offsets, part.owned_count, max_outdegree,
                    alpha, beta, threshold, d_num_active
                );
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaMemcpyAsync(&local_active_nodes, d_num_active, sizeof(int), cudaMemcpyDeviceToHost, stream));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        int total_active_nodes = 0;
        MPI_Allreduce(&local_active_nodes, &total_active_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (total_active_nodes == 0) break;

        std::swap(d_active, d_next_active);
    }

    // Stop before draining CPU checks so all four checked algorithms report
    // the same GPU-main-pipeline interval. Rank zero prints the slowest rank.
    CUDA_CHECK(cudaEventRecord(stop_evt, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
    float max_ms = 0.0f;
    MPI_Reduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    while (!pending_tasks.empty()) std::this_thread::yield();
    check_stop.store(true, std::memory_order_release);
    if (check_worker.joinable()) check_worker.join();

    bool detected_dmr = false;
    bool detected_monotonic = false;
    bool detected_avg_increase = false;
    int first_dmr_iter = -1;
    int first_monotonic_iter = -1;
    int first_avg_increase_iter = -1;
    int checked_iteration_count = 0;
    float previous_avg_delta = (float)INF;

    for (int check_iter = 1; check_iter <= iter; ++check_iter) {
        int local_done = check_results[check_iter].done ? 1 : 0;
        int all_done = 0;
        MPI_Allreduce(&local_done, &all_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
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

        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_dmr, &global_dmr, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(&local_monotonic, &global_monotonic, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        if (global_dmr) {
            detected_dmr = true;
            if (first_dmr_iter < 0) first_dmr_iter = check_iter;
        }
        if (global_monotonic) {
            detected_monotonic = true;
            if (first_monotonic_iter < 0) first_monotonic_iter = check_iter;
        }
        if (global_count > 0) {
            float avg_delta = (float)global_sum / (float)global_count;
            if (previous_avg_delta < (float)INF && avg_delta > previous_avg_delta) {
                detected_avg_increase = true;
                if (first_avg_increase_iter < 0) first_avg_increase_iter = check_iter;
            }
            previous_avg_delta = avg_delta;
        }
    }

    std::vector<int> h_owned_values(part.owned_count, INF);
    if (part.owned_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_owned_values.data(), d_values, part.owned_count * sizeof(int), cudaMemcpyDeviceToHost));
    }

    std::vector<int> recv_counts(world_size), recv_displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        recv_counts[r] = parts[r].owned_count;
        recv_displs[r] = parts[r].start_node;
    }
    MPI_Allgatherv(
        h_owned_values.data(), part.owned_count, MPI_INT,
        h_value, recv_counts.data(), recv_displs.data(), MPI_INT,
        MPI_COMM_WORLD
    );

    if (world_rank == 0) {
        printf("GPU time: %.4f ms\n", max_ms);
        printf("NCCL 分布式 BFS 迭代 %d 次正常结束。\n", iter);
        printf("异步检测标记: DMR=%d", detected_dmr ? 1 : 0);
        if (first_dmr_iter >= 0) printf("(first_iter=%d)", first_dmr_iter);
        printf(", monotonic=%d", detected_monotonic ? 1 : 0);
        if (first_monotonic_iter >= 0) printf("(first_iter=%d)", first_monotonic_iter);
        printf(", avg_delta_increase=%d", detected_avg_increase ? 1 : 0);
        if (first_avg_increase_iter >= 0) printf("(first_iter=%d)", first_avg_increase_iter);
        printf("\n");
        printf("异步检测覆盖: checked=%d/%d, skipped=%d\n",
               checked_iteration_count, iter, iter - checked_iteration_count);
    }

    cudaFree(d_values);
    cudaFree(d_active);
    cudaFree(d_next_active);
    cudaFree(d_update);
    cudaFree(d_row_offsets);
    cudaFree(d_column_indices);
    cudaFree(d_column_offsets);
    cudaFree(d_row_indices);
    cudaFree(d_num_active);
    cudaFree(d_info_scratch);
    for (int b = 0; b < CHECK_BUFFER_COUNT; ++b) {
        cudaFree(d_info[b]);
        cudaFreeHost(h_info[b]);
        cudaEventDestroy(compute_done_events[b]);
        cudaEventDestroy(check_events[b]);
    }
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == world_rank) continue;
        auto& dp = d_plan[peer];
        cudaFree(dp.d_act_send_local);
        cudaFree(dp.d_act_recv_owned);
        cudaFree(dp.d_value_send_owned);
        cudaFree(dp.d_value_recv_ghost);
        cudaFree(dp.d_act_send_buf);
        cudaFree(dp.d_act_recv_buf);
        cudaFree(dp.d_value_send_buf);
        cudaFree(dp.d_value_recv_buf);
    }
    cudaStreamDestroy(check_stream);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    CUDA_CHECK(cudaFreeHost(h_active_global));
    ncclCommDestroy(comm);
    MPI_Comm_free(&local_comm);
}

#endif // FAULT_TOLERANT_BFS_MULTIGPU_NCCL_PARTITIONED_CUH
