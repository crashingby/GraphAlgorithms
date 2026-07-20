/**
 * @file bfs_tolerance_multiGPU.cuh
 * @brief Distributed BFS with selective DMR and asynchronous CPU checks.
 *
 * Checked and unchecked BFS share the same balanced owned-plus-ghost
 * partition, deduplicated peer plans, initialization, and collective order.
 * The tolerance-only work is criticality scoring, selective DMR, and publishing
 * fixed-size summaries to one asynchronous CPU consumer.
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
#include <condition_variable>
#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>

#include "include/distributed_benchmark.cuh"
#include "include/distributed_partition.cuh"
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
 *
 * Active and assigned redundant lanes select an owned compute vertex before
 * entering one common owned-plus-ghost pull loop. Role divergence is limited
 * to target selection, result placement, comparison, and primary writeback.
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

    const bool does_redundant_work =
        is_idle && iid >= 0 && iid < critical_count && iid < BLOCK_SIZE;
    int compute_vertex = -1;
    if (is_active) {
        compute_vertex = tid;
    } else if (does_redundant_work) {
        compute_vertex = critical_list[iid];
    }

    int computed_value = INF;
    if (compute_vertex >= 0) {
        computed_value = d_values[compute_vertex];
        for (int i = d_column_offsets[compute_vertex];
             i < d_column_offsets[compute_vertex + 1]; ++i) {
            int src_local = d_row_indices[i];              // owned or ghost local id
            if (src_local >= 0 && src_local < local_node_count) {
                int candidate = d_values[src_local] + 1;   // ghost value is local cache
                if (candidate < computed_value) computed_value = candidate;
            }
        }
    }

    int main_newVal = INF;
    if (is_active) {
        main_newVal = computed_value;
    } else if (does_redundant_work) {
        redundant_results[iid] = computed_value;
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
    if (v < num_nodes) {
        int active_flag = d_active[v];
        if (active_flag != -1) {
            atomicAdd(d_num_active, 1);

            int outdeg = d_row_offsets[v + 1] - d_row_offsets[v];
            int safe_max_outdegree = (max_outdegree > 0) ? max_outdegree : 1;

            // Criticality depends on the BFS distance, never on the work-set
            // encoding (-1/1/2).
            float dist_score = 1.0f / (1.0f + (float)d_values[v]);
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

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int device = dg_select_rank_device();
    ncclComm_t comm = dg_create_nccl_comm(rank, world_size);

    std::vector<DgSubgraphHost> parts;
    dg_build_all_subgraphs(
        world_size, num_nodes, h_row_offsets, h_column_indices,
        h_column_offsets, h_row_indices, parts);
    const DgSubgraphHost& part = parts[rank];

    if (rank == 0) {
        printf(
            "发现 %d 个 MPI rank，开启 BFS 多节点容错版本；"
            "分区和通信计划与无容错版本完全共享。\n",
            world_size);
    }
    printf(
        "Rank %d GPU %d: owned [%d, %d), owned_count=%d, "
        "local_count=%d, ghosts=%zu\n",
        rank, device, part.start_node, part.end_node, part.owned_count,
        part.local_node_count, part.ghost_global_ids.size());

    std::vector<std::vector<DgPeerPlanHost>> host_plans;
    dg_build_peer_plans(parts, world_size, num_nodes, host_plans);
    std::vector<DgPeerPlanDevice> plans;
    dg_upload_peer_plans(host_plans, rank, world_size, plans, false);

    const int local_alloc_count = std::max(1, part.local_node_count);
    const int owned_alloc_count = std::max(1, part.owned_count);

    int* d_values = nullptr;
    int* d_row_offsets = nullptr;
    int* d_column_indices = nullptr;
    int* d_column_offsets = nullptr;
    int* d_row_indices = nullptr;
    int* d_active = nullptr;
    int* d_update = nullptr;
    int* d_active_count = nullptr;
    MonotonicInfo* d_info_scratch = nullptr;
    std::vector<MonotonicInfo*> d_info(CHECK_BUFFER_COUNT, nullptr);
    std::vector<MonotonicInfo*> h_info(CHECK_BUFFER_COUNT, nullptr);
    std::vector<cudaEvent_t> compute_done_events(CHECK_BUFFER_COUNT);
    std::vector<cudaEvent_t> check_done_events(CHECK_BUFFER_COUNT);

    CUDA_CHECK(cudaMalloc(&d_values, local_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &d_row_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &d_column_indices, part.column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &d_column_offsets, (part.owned_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &d_row_indices, part.row_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, owned_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update, local_alloc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info_scratch, sizeof(MonotonicInfo)));
    for (int buffer = 0; buffer < CHECK_BUFFER_COUNT; ++buffer) {
        CUDA_CHECK(cudaMalloc(&d_info[buffer], sizeof(MonotonicInfo)));
        CUDA_CHECK(cudaMallocHost(&h_info[buffer], sizeof(MonotonicInfo)));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &compute_done_events[buffer], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &check_done_events[buffer], cudaEventDisableTiming));
    }

    std::vector<int> h_local_value(local_alloc_count, INF);
    for (int local = 0; local < part.local_node_count; ++local) {
        if (part.local_to_global[local] == src) h_local_value[local] = 0;
    }

    // Pull BFS starts with exactly the vertices that can observe the source.
    // This is intentionally identical to bfsMultiGPUBasic().
    std::vector<int> h_active(owned_alloc_count, -1);
    auto activate_owned = [&](int global_vertex) {
        if (global_vertex >= part.start_node &&
            global_vertex < part.end_node) {
            h_active[global_vertex - part.start_node] = 1;
        }
    };
    activate_owned(src);
    for (int edge = h_row_offsets[src];
         edge < h_row_offsets[src + 1]; ++edge) {
        activate_owned(h_column_indices[edge]);
    }

    CUDA_CHECK(cudaMemcpy(
        d_values, h_local_value.data(),
        part.local_node_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_row_offsets, part.row_offsets.data(),
        (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_column_indices, part.column_indices.data(),
        part.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_column_offsets, part.column_offsets.data(),
        (part.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_row_indices, part.row_indices.data(),
        part.row_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_active, h_active.data(),
        part.owned_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(
        d_update, 0xff, local_alloc_count * sizeof(int)));

    cudaStream_t stream;
    cudaStream_t check_stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &check_stream, cudaStreamNonBlocking));

    // Initialization is outside every benchmark phase in both BFS variants.
    dg_nccl_exchange_int_values(
        d_values, plans, world_size, rank, comm, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SpscQueue<int, CHECK_BUFFER_COUNT> free_buffers;
    SpscQueue<CheckTask, CHECK_BUFFER_COUNT> pending_tasks;
    for (int buffer = 0; buffer < CHECK_BUFFER_COUNT; ++buffer) {
        free_buffers.try_push(buffer);
    }

    std::atomic<bool> check_stop(false);
    std::mutex pending_mutex;
    std::condition_variable pending_cv;
    std::vector<AsyncCheckResult> check_results(1001);

    /**
     * Consume fixed-size summaries in FIFO order.
     *
     * The condition variable removes busy polling. cudaEventSynchronize blocks
     * this worker only; it never invokes MPI, which preserves FUNNELED access.
     */
    std::thread check_worker([&]() {
        CUDA_CHECK(cudaSetDevice(device));
        while (true) {
            CheckTask task;
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

            CUDA_CHECK(cudaEventSynchronize(
                check_done_events[task.buf]));
            AsyncCheckResult result;
            result.done = true;
            result.dmr_error = h_info[task.buf]->dmr_error_flag;
            result.monotonic_error =
                h_info[task.buf]->monotonic_error_flag;
            result.sum_abs_delta = h_info[task.buf]->sum_abs_delta;
            result.count_update = h_info[task.buf]->count_update;
            if (task.iter >= 0 &&
                task.iter < static_cast<int>(check_results.size())) {
                check_results[task.iter] = result;
            }
            free_buffers.try_push(task.buf);
        }
    });

    DgIterationPhaseTimer phase_timer;
    GraphCudaEventAccumulator graph_kernel_timer;
    CUDA_CHECK(graph_cuda_timer_create(&graph_kernel_timer));
    DgBenchmarkTiming timing;
    int iter = 0;
    int total_active = num_nodes;
    const int max_outdegree =
        compute_max_outdegree(h_row_offsets, num_nodes);
    const auto main_loop_start = std::chrono::steady_clock::now();

    while (iter < 1000 && total_active > 0) {
        ++iter;
        phase_timer.start_pre(stream);

        // Classification and selective DMR are the only algorithmic GPU work
        // added relative to the shared baseline loop.
        CUDA_CHECK(cudaMemsetAsync(
            d_active_count, 0, sizeof(int), stream));
        scoreAndMarkKernel<<<
            dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_active, d_values, d_row_offsets, part.owned_count,
            max_outdegree, alpha, beta, threshold, d_active_count);
        CUDA_CHECK(cudaGetLastError());

        int check_buffer = -1;
        const bool do_async_check = free_buffers.try_pop(check_buffer);
        MonotonicInfo* iteration_info =
            do_async_check ? d_info[check_buffer] : d_info_scratch;
        CUDA_CHECK(cudaMemsetAsync(
            iteration_info, 0, sizeof(MonotonicInfo), stream));
        CUDA_CHECK(cudaMemsetAsync(
            d_update, 0xff, local_alloc_count * sizeof(int), stream));

        CUDA_CHECK(graph_cuda_timer_start(&graph_kernel_timer, stream));
        bfsPullDualMultiGPUKernel<<<
            dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_values, d_row_offsets, d_column_indices,
            d_column_offsets, d_row_indices, d_active, d_update,
            part.owned_count, part.local_node_count,
            iteration_info, TREND_DEC);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(graph_cuda_timer_stop(&graph_kernel_timer, stream));

        if (do_async_check) {
            CUDA_CHECK(cudaEventRecord(
                compute_done_events[check_buffer], stream));
            CUDA_CHECK(cudaStreamWaitEvent(
                check_stream, compute_done_events[check_buffer], 0));
            CUDA_CHECK(cudaMemcpyAsync(
                h_info[check_buffer], d_info[check_buffer],
                sizeof(MonotonicInfo), cudaMemcpyDeviceToHost,
                check_stream));
            CUDA_CHECK(cudaEventRecord(
                check_done_events[check_buffer], check_stream));
            {
                std::lock_guard<std::mutex> lock(pending_mutex);
                if (!pending_tasks.try_push({iter, check_buffer})) {
                    fprintf(
                        stderr,
                        "BFS async check queue invariant failed.\n");
                    exit(EXIT_FAILURE);
                }
            }
            pending_cv.notify_one();
        }

        CUDA_CHECK(cudaMemsetAsync(
            d_active, 0xff, owned_alloc_count * sizeof(int), stream));
        dgCopyOwnedUpdateKernel<<<
            dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_update, d_active, part.owned_count);
        CUDA_CHECK(cudaGetLastError());
        phase_timer.stop_pre(stream);

        phase_timer.start_comm(stream);
        dg_nccl_exchange_activation(
            d_update, d_active, plans, world_size, rank, comm, stream);
        dg_nccl_exchange_int_values(
            d_values, plans, world_size, rank, comm, stream);
        phase_timer.stop_comm(stream);

        phase_timer.start_post(stream);
        CUDA_CHECK(cudaMemsetAsync(
            d_update, 0xff, local_alloc_count * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(
            d_active_count, 0, sizeof(int), stream));
        dgCountActiveKernel<<<
            dg_blocks(part.owned_count), DG_BLOCK_SIZE, 0, stream>>>(
            d_active, d_active_count, part.owned_count);
        CUDA_CHECK(cudaGetLastError());

        int local_active = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &local_active, d_active_count, sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        phase_timer.stop_post(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        phase_timer.accumulate(
            timing.gpu_compute_ms, timing.nccl_exchange_ms);
        CUDA_CHECK(graph_cuda_timer_accumulate(&graph_kernel_timer));

        dg_timed_allreduce(
            &local_active, &total_active, 1, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD, timing.mpi_sync_ms);
    }

    timing.main_loop_ms = dg_elapsed_ms(
        main_loop_start, std::chrono::steady_clock::now());

    // Drain ends as soon as the last queued host summary has been consumed.
    // It intentionally excludes all cross-rank detection aggregation below.
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
    double previous_avg_delta = static_cast<double>(INF);

    for (int check_iter = 1; check_iter <= iter; ++check_iter) {
        const int local_done =
            check_results[check_iter].done ? 1 : 0;
        int all_done = 0;
        dg_timed_allreduce(
            &local_done, &all_done, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD, timing.postcheck_mpi_ms);
        if (!all_done) continue;
        ++checked_iteration_count;

        const unsigned long long local_sum =
            check_results[check_iter].sum_abs_delta;
        unsigned long long global_sum = 0;
        const int local_count =
            check_results[check_iter].count_update;
        int global_count = 0;
        const int local_dmr =
            check_results[check_iter].dmr_error ? 1 : 0;
        int global_dmr = 0;
        const int local_monotonic =
            check_results[check_iter].monotonic_error ? 1 : 0;
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
            if (first_monotonic_iter < 0) {
                first_monotonic_iter = check_iter;
            }
        }
        if (global_count > 0) {
            const double average_delta =
                static_cast<double>(global_sum) /
                static_cast<double>(global_count);
            if (previous_avg_delta < static_cast<double>(INF) &&
                average_delta > previous_avg_delta) {
                detected_avg_increase = true;
                if (first_avg_increase_iter < 0) {
                    first_avg_increase_iter = check_iter;
                }
            }
            previous_avg_delta = average_delta;
        }
    }

    timing.postcheck_total_ms = dg_elapsed_ms(
        postcheck_start, std::chrono::steady_clock::now());
    timing.graph_kernel_ms = graph_kernel_timer.total_ms;
    dg_report_benchmark_timing(timing, rank, world_size);

    std::vector<int> h_owned(part.owned_count, INF);
    CUDA_CHECK(cudaMemcpy(
        h_owned.data(), d_values, part.owned_count * sizeof(int),
        cudaMemcpyDeviceToHost));
    std::vector<int> counts = dg_owned_counts(world_size, num_nodes);
    std::vector<int> displs = dg_displacements(counts);
    MPI_Allgatherv(
        h_owned.data(), part.owned_count, MPI_INT,
        h_value, counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("NCCL 分布式 BFS 容错版本迭代 %d 次结束。\n", iter);
        printf("异步CPU检测(BFS): DMR=%d", detected_dmr ? 1 : 0);
        if (first_dmr_iter >= 0) {
            printf("(first_iter=%d)", first_dmr_iter);
        }
        printf(", monotonic=%d", detected_monotonic ? 1 : 0);
        if (first_monotonic_iter >= 0) {
            printf("(first_iter=%d)", first_monotonic_iter);
        }
        printf(
            ", avg_delta_increase=%d",
            detected_avg_increase ? 1 : 0);
        if (first_avg_increase_iter >= 0) {
            printf("(first_iter=%d)", first_avg_increase_iter);
        }
        printf("\n");
        printf(
            "异步检测覆盖(BFS): checked=%d/%d, skipped=%d\n",
            checked_iteration_count, iter,
            iter - checked_iteration_count);
        printf("BENCHMARK_ITERATIONS iterations=%d\n", iter);
    }

    phase_timer.destroy();
    CUDA_CHECK(graph_cuda_timer_destroy(&graph_kernel_timer));
    CUDA_CHECK(cudaStreamDestroy(check_stream));
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
    cudaFree(d_info_scratch);
    for (int buffer = 0; buffer < CHECK_BUFFER_COUNT; ++buffer) {
        cudaFree(d_info[buffer]);
        cudaFreeHost(h_info[buffer]);
        cudaEventDestroy(compute_done_events[buffer]);
        cudaEventDestroy(check_done_events[buffer]);
    }
    NCCL_CHECK(ncclCommDestroy(comm));
}

#endif // FAULT_TOLERANT_BFS_MULTIGPU_NCCL_PARTITIONED_CUH
