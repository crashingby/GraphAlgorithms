#ifndef FAULT_TOLERANT_BFS_MULTIGPU_NCCL_PARTITIONED_CUH
#define FAULT_TOLERANT_BFS_MULTIGPU_NCCL_PARTITIONED_CUH

#include <cuda_runtime.h>
#include <nccl.h>
#include <nvToolsExt.h>

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

#define BLOCK_SIZE 256
#define INF 100000
#define MAX_GPUS 8

// ==================== 数据结构 ====================
enum ValueTrend { TREND_NONE = 0, TREND_INC = 1, TREND_DEC = 2 };

struct MonotonicInfo {
    int dmr_error_flag;
    int monotonic_error_flag;
};

struct AsyncCheckWorkerState {
    std::atomic<bool> stop{false};
};

struct AsyncCheckResult {
    bool done = false;
    long long sum_delta = 0;
    int count_update = 0;
    int dmr_error = 0;
    int monotonic_error = 0;
};

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    explicit NvtxRange(const std::string& name) : name_(name) { nvtxRangePushA(name_.c_str()); }
    ~NvtxRange() { nvtxRangePop(); }

    std::string name_;
};

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
__global__ void bfsPullDualMultiGPUKernel(
    int* d_values,
    const int* d_row_offsets, const int* d_column_indices,
    const int* d_column_offsets, const int* d_row_indices,
    const int* d_active, int* d_update,
    int owned_count, int local_node_count,
    int* d_delta, MonotonicInfo* d_info, ValueTrend trend
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
            d_delta[tid] = delta;
            if ((trend == TREND_INC && delta < 0) || (trend == TREND_DEC && delta > 0)) {
                atomicExch(&(d_info->monotonic_error_flag), 1);
            }
        } else {
            d_delta[tid] = 0;
        }
    } else if (valid_tid) {
        d_delta[tid] = 0;
    }
}

__global__ void copyOwnedUpdateToNextActiveKernel(
    const int* d_update, int* d_next_active, int owned_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < owned_count) d_next_active[i] = d_update[i];
}

__global__ void packByIndexKernel(
    const int* d_src, const int* d_indices, int* d_out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_out[i] = d_src[d_indices[i]];
}

__global__ void unpackByIndexKernel(
    const int* d_in, const int* d_indices, int* d_dst, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_dst[d_indices[i]] = d_in[i];
}

__global__ void applyActivationRecvKernel(
    const int* d_flags, const int* d_owned_indices, int* d_next_active, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_flags[i] != -1) d_next_active[d_owned_indices[i]] = 1;
}

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
inline int count_active_nodes(const int* h_active, int num_nodes) {
    int cnt = 0;
    for (int i = 0; i < num_nodes; i++) if (h_active[i] != -1) cnt++;
    return cnt;
}

inline int compute_max_outdegree(const int* h_row_offsets, int num_nodes) {
    int max_outdegree = 0;
    for (int v = 0; v < num_nodes; v++) {
        int outdegree = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdegree > max_outdegree) max_outdegree = outdegree;
    }
    return std::max(max_outdegree, 1);
}



inline int owner_of_vertex(int v, int nodes_per_gpu, int num_gpus) {
    int owner = v / nodes_per_gpu;
    if (owner >= num_gpus) owner = num_gpus - 1;
    return owner;
}

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

struct PeerPlanHost {
    std::vector<int> act_send_local;       // sender ghost local ids, read d_update[ghost]
    std::vector<int> act_recv_owned;       // receiver owned local ids, write d_next_active[owned]
    std::vector<int> value_send_owned;     // owner local ids, read d_values[owned]
    std::vector<int> value_recv_ghost;     // receiver ghost local ids, write d_values[ghost]
};

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

inline void upload_index_vector(const std::vector<int>& h, int** d_ptr) {
    if (h.empty()) { *d_ptr = nullptr; return; }
    CUDA_CHECK(cudaMalloc(d_ptr, h.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
}

inline void alloc_int_buffer(int** d_ptr, int n) {
    if (n <= 0) { *d_ptr = nullptr; return; }
    CUDA_CHECK(cudaMalloc(d_ptr, n * sizeof(int)));
}

// ==================== 主函数：局部子图 + Ghost 节点 + NCCL 点对点通信 ====================
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

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) {
        fprintf(stderr, "没有可用 GPU。\n");
        exit(EXIT_FAILURE);
    }
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    int nodes_per_gpu = (num_nodes + num_gpus - 1) / num_gpus;
    int max_outdegree = compute_max_outdegree(h_row_offsets, num_nodes);

    printf("发现 %d 个 GPU，开启 NCCL 子图划分 BFS：owned + ghost + send/recv activation/value。\n", num_gpus);

    std::vector<int> devs(num_gpus);
    std::iota(devs.begin(), devs.end(), 0);
    std::vector<ncclComm_t> comms(num_gpus);
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));

    // 1) 构造每张 GPU 的 owned + ghost 子图。
    std::vector<GpuSubgraphHost> parts(num_gpus);
    for (int d = 0; d < num_gpus; ++d) {
        build_vertex_partition_subgraph(
            d, num_gpus, num_nodes, nodes_per_gpu,
            h_row_offsets, h_column_indices, h_column_offsets, h_row_indices,
            parts[d]
        );
        printf("GPU %d: owned [%d, %d), owned_count=%d, local_count=%d, ghosts=%zu\n",
               d, parts[d].start_node, parts[d].end_node,
               parts[d].owned_count, parts[d].local_node_count, parts[d].ghost_local_ids.size());
    }

    // 2) 生成通信计划。
    std::vector<std::vector<PeerPlanHost>> h_plan(num_gpus, std::vector<PeerPlanHost>(num_gpus));
    for (int sender = 0; sender < num_gpus; ++sender) {
        const auto& sp = parts[sender];
        for (size_t k = 0; k < sp.ghost_local_ids.size(); ++k) {
            int ghost_local = sp.ghost_local_ids[k];
            int ghost_global = sp.ghost_global_ids[k];
            int owner = owner_of_vertex(ghost_global, nodes_per_gpu, num_gpus);
            if (owner == sender) continue;

            auto it = parts[owner].global_to_local.find(ghost_global);
            if (it == parts[owner].global_to_local.end()) continue;
            int owner_local = it->second;  // owned local id on owner GPU

            // activation: sender d_update[ghost] -> owner d_next_active[owned]
            h_plan[sender][owner].act_send_local.push_back(ghost_local);
            h_plan[owner][sender].act_recv_owned.push_back(owner_local);

            // value sync: owner d_values[owned] -> sender d_values[ghost]
            h_plan[owner][sender].value_send_owned.push_back(owner_local);
            h_plan[sender][owner].value_recv_ghost.push_back(ghost_local);
        }
    }

    // 3) 初始化 host active / values。
    for (int i = 0; i < num_nodes; ++i) h_value[i] = INF;
    h_value[src] = 0;

    int* h_active_global = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_active_global, num_nodes * sizeof(int)));
    for (int i = 0; i < num_nodes; ++i) h_active_global[i] = -1;

    // pull 版本：source 自身不一定会产生变小，所以第一轮激活 source 的出邻居。
    h_active_global[src] = 1;
    for (int e = h_row_offsets[src]; e < h_row_offsets[src + 1]; ++e) {
        int dst = h_column_indices[e];
        if (dst >= 0 && dst < num_nodes) h_active_global[dst] = 1;
    }
    // 4) Device 资源。
    std::vector<int*> d_values(num_gpus, nullptr), d_active(num_gpus, nullptr);
    std::vector<int*> d_next_active(num_gpus, nullptr), d_update(num_gpus, nullptr);
    std::vector<int*> d_row_offsets(num_gpus, nullptr), d_column_indices(num_gpus, nullptr);
    std::vector<int*> d_column_offsets(num_gpus, nullptr), d_row_indices(num_gpus, nullptr);
    std::vector<std::vector<int*>> d_delta(num_gpus, std::vector<int*>(2, nullptr));
    std::vector<int*> d_delta_scratch(num_gpus, nullptr);
    std::vector<int*> d_num_active(num_gpus, nullptr);
    std::vector<int> h_num_active(num_gpus, 0);
    std::vector<std::vector<MonotonicInfo*>> d_info(num_gpus, std::vector<MonotonicInfo*>(2, nullptr));
    std::vector<MonotonicInfo*> d_info_scratch(num_gpus, nullptr);
    std::vector<cudaStream_t> streams(num_gpus);
    std::vector<cudaStream_t> check_streams(num_gpus);
    std::vector<std::vector<PeerPlanDevice>> d_plan(num_gpus, std::vector<PeerPlanDevice>(num_gpus));

    std::vector<std::vector<int*>> h_delta(num_gpus, std::vector<int*>(2, nullptr));
    std::vector<std::vector<MonotonicInfo*>> h_info(num_gpus, std::vector<MonotonicInfo*>(2, nullptr));
    std::vector<std::vector<cudaEvent_t>> compute_done_events(num_gpus, std::vector<cudaEvent_t>(2));
    std::vector<std::vector<cudaEvent_t>> check_events(num_gpus, std::vector<cudaEvent_t>(2));
    std::vector<std::unique_ptr<AsyncCheckWorkerState>> check_states(num_gpus);
    std::unique_ptr<std::atomic<int>[]> pending_check(new std::atomic<int>[num_gpus * 2]);
    for (int i = 0; i < num_gpus * 2; ++i) pending_check[i].store(0);
    auto pending_index = [](int d, int buf) { return d * 2 + buf; };

    std::vector<std::thread> check_workers;
    std::vector<std::vector<AsyncCheckResult>> check_results(1001, std::vector<AsyncCheckResult>(num_gpus));
    std::mutex check_results_mutex;
    std::atomic<int> max_dispatched_check_iter(0);
    std::atomic<bool> detected_dmr(false);
    std::atomic<bool> detected_monotonic(false);
    std::atomic<bool> detected_avg_increase(false);
    int first_dmr_iter = -1;
    int first_monotonic_iter = -1;
    int first_avg_increase_iter = -1;
    int next_check_iter_to_aggregate = 1;
    float previous_avg_delta = (float)INF;

    for (int d = 0; d < num_gpus; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        const auto& p = parts[d];
        CUDA_CHECK(cudaStreamCreate(&streams[d]));
        CUDA_CHECK(cudaStreamCreate(&check_streams[d]));

        int local_alloc_count = std::max(1, p.local_node_count);
        int owned_alloc_count = std::max(1, p.owned_count);

        CUDA_CHECK(cudaMalloc(&d_values[d], local_alloc_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_active[d], owned_alloc_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_next_active[d], owned_alloc_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_update[d], local_alloc_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_row_offsets[d], (p.owned_count + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_column_offsets[d], (p.owned_count + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_column_indices[d], p.column_indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_row_indices[d], p.row_indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_delta_scratch[d], owned_alloc_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_num_active[d], sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info_scratch[d], sizeof(MonotonicInfo)));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&d_delta[d][b], owned_alloc_count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_info[d][b], sizeof(MonotonicInfo)));
            CUDA_CHECK(cudaMallocHost(&h_delta[d][b], owned_alloc_count * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&h_info[d][b], sizeof(MonotonicInfo)));
            CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_events[d][b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&check_events[d][b], cudaEventDisableTiming));
        }
        check_states[d].reset(new AsyncCheckWorkerState());

        std::vector<int> h_local_values(local_alloc_count, INF);
        std::vector<int> h_local_active(owned_alloc_count, -1);
        for (int local = 0; local < p.local_node_count; ++local) {
            int global = p.local_to_global[local];
            h_local_values[local] = h_value[global];
        }
        for (int local = 0; local < p.owned_count; ++local) {
            int global = p.start_node + local;
            h_local_active[local] = h_active_global[global];
        }

        CUDA_CHECK(cudaMemcpy(d_values[d], h_local_values.data(), local_alloc_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_active[d], h_local_active.data(), owned_alloc_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_update[d], 0xff, local_alloc_count * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_next_active[d], 0xff, owned_alloc_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_offsets[d], p.row_offsets.data(), (p.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_column_offsets[d], p.column_offsets.data(), (p.owned_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_column_indices[d], p.column_indices.data(), p.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_row_indices[d], p.row_indices.data(), p.row_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

        // 初始 active 也直接在本 GPU 上按阈值标记 critical，不再回 CPU 做 topK。
        CUDA_CHECK(cudaMemsetAsync(d_num_active[d], 0, sizeof(int), streams[d]));
        if (p.owned_count > 0) {
            int blocks = (p.owned_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            scoreAndMarkKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                d_active[d], d_row_offsets[d], p.owned_count, max_outdegree,
                alpha, beta, threshold, d_num_active[d]
            );
            CUDA_CHECK(cudaGetLastError());
        }

        for (int peer = 0; peer < num_gpus; ++peer) {
            if (peer == d) continue;
            auto& hp = h_plan[d][peer];
            auto& dp = d_plan[d][peer];
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
    }

    auto try_aggregate_checks = [&]() {
        std::lock_guard<std::mutex> lock(check_results_mutex);
        int dispatched = max_dispatched_check_iter.load();
        while (next_check_iter_to_aggregate <= dispatched) {
            bool all_done = true;
            for (int d = 0; d < num_gpus; ++d) {
                if (!check_results[next_check_iter_to_aggregate][d].done) {
                    all_done = false;
                    break;
                }
            }
            if (!all_done) break;

            long long sum_delta = 0;
            int count_update = 0;
            bool iter_dmr = false;
            bool iter_monotonic = false;
            for (int d = 0; d < num_gpus; ++d) {
                const auto& r = check_results[next_check_iter_to_aggregate][d];
                sum_delta += r.sum_delta;
                count_update += r.count_update;
                iter_dmr = iter_dmr || (r.dmr_error != 0);
                iter_monotonic = iter_monotonic || (r.monotonic_error != 0);
            }

            if (iter_dmr) {
                detected_dmr.store(true);
                if (first_dmr_iter < 0) first_dmr_iter = next_check_iter_to_aggregate;
            }
            if (iter_monotonic) {
                detected_monotonic.store(true);
                if (first_monotonic_iter < 0) first_monotonic_iter = next_check_iter_to_aggregate;
            }
            if (count_update > 0) {
                float avg_delta = std::fabs((float)sum_delta) / (float)count_update;
                if (avg_delta > previous_avg_delta) {
                    detected_avg_increase.store(true);
                    if (first_avg_increase_iter < 0) first_avg_increase_iter = next_check_iter_to_aggregate;
                }
                previous_avg_delta = avg_delta;
            }
            ++next_check_iter_to_aggregate;
        }
    };

    auto try_reserve_check_slot = [&](int d, int buf, int iter_id) {
        int expected = 0;
        if (!pending_check[pending_index(d, buf)].compare_exchange_strong(expected, -1)) {
            check_results[iter_id][d].done = true;
            return false;
        }
        return true;
    };

    check_workers.reserve(num_gpus);
    for (int d = 0; d < num_gpus; ++d) {
        check_workers.emplace_back([&, d]() {
            CUDA_CHECK(cudaSetDevice(d));
            const int owned_count = parts[d].owned_count;
            while (!check_states[d]->stop.load()) {
                bool did_work = false;
                for (int buf = 0; buf < 2; ++buf) {
                    int iter_id = pending_check[pending_index(d, buf)].load();
                    if (iter_id <= 0) continue;

                    cudaError_t event_status = cudaEventQuery(check_events[d][buf]);
                    if (event_status == cudaErrorNotReady) continue;
                    CUDA_CHECK(event_status);
                    did_work = true;

                    NvtxRange scan_range(nvtx_name("worker CPU scan", iter_id, d));
                    long long sum_delta = 0;
                    int count_update = 0;
                    for (int i = 0; i < owned_count; ++i) {
                        int dv = h_delta[d][buf][i];
                        if (dv != 0) {
                            sum_delta += (long long)dv;
                            ++count_update;
                        }
                    }

                    auto& r = check_results[iter_id][d];
                    r.done = true;
                    r.sum_delta = sum_delta;
                    r.count_update = count_update;
                    r.dmr_error = h_info[d][buf]->dmr_error_flag;
                    r.monotonic_error = h_info[d][buf]->monotonic_error_flag;
                    pending_check[pending_index(d, buf)].store(0);
                }

                if (!did_work) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
            }
        });
    }

    auto launch_pack_activation = [&](int d) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int peer = 0; peer < num_gpus; ++peer) {
            if (peer == d) continue;
            auto& dp = d_plan[d][peer];
            if (dp.act_send_count > 0) {
                int blocks = (dp.act_send_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                packByIndexKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                    d_update[d], dp.d_act_send_local, dp.d_act_send_buf, dp.act_send_count
                );
            }
        }
    };

    auto launch_apply_activation = [&](int d) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int peer = 0; peer < num_gpus; ++peer) {
            if (peer == d) continue;
            auto& dp = d_plan[d][peer];
            if (dp.act_recv_count > 0) {
                int blocks = (dp.act_recv_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                applyActivationRecvKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                    dp.d_act_recv_buf, dp.d_act_recv_owned, d_next_active[d], dp.act_recv_count
                );
            }
        }
    };

    auto launch_pack_values = [&](int d) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int peer = 0; peer < num_gpus; ++peer) {
            if (peer == d) continue;
            auto& dp = d_plan[d][peer];
            if (dp.value_send_count > 0) {
                int blocks = (dp.value_send_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                packByIndexKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                    d_values[d], dp.d_value_send_owned, dp.d_value_send_buf, dp.value_send_count
                );
            }
        }
    };

    auto launch_unpack_values = [&](int d) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int peer = 0; peer < num_gpus; ++peer) {
            if (peer == d) continue;
            auto& dp = d_plan[d][peer];
            if (dp.value_recv_count > 0) {
                int blocks = (dp.value_recv_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                unpackByIndexKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                    dp.d_value_recv_buf, dp.d_value_recv_ghost, d_values[d], dp.value_recv_count
                );
            }
        }
    };

    auto nccl_exchange_activation = [&]() {
        NCCL_CHECK(ncclGroupStart());
        for (int d = 0; d < num_gpus; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            for (int peer = 0; peer < num_gpus; ++peer) {
                if (peer == d) continue;
                auto& dp = d_plan[d][peer];
                if (dp.act_recv_count > 0) {
                    NCCL_CHECK(ncclRecv(dp.d_act_recv_buf, dp.act_recv_count, ncclInt, peer, comms[d], streams[d]));
                }
                if (dp.act_send_count > 0) {
                    NCCL_CHECK(ncclSend(dp.d_act_send_buf, dp.act_send_count, ncclInt, peer, comms[d], streams[d]));
                }
            }
        }
        NCCL_CHECK(ncclGroupEnd());
    };

    auto nccl_exchange_values = [&]() {
        NCCL_CHECK(ncclGroupStart());
        for (int d = 0; d < num_gpus; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            for (int peer = 0; peer < num_gpus; ++peer) {
                if (peer == d) continue;
                auto& dp = d_plan[d][peer];
                if (dp.value_recv_count > 0) {
                    NCCL_CHECK(ncclRecv(dp.d_value_recv_buf, dp.value_recv_count, ncclInt, peer, comms[d], streams[d]));
                }
                if (dp.value_send_count > 0) {
                    NCCL_CHECK(ncclSend(dp.d_value_send_buf, dp.value_send_count, ncclInt, peer, comms[d], streams[d]));
                }
            }
        }
        NCCL_CHECK(ncclGroupEnd());
    };

    // 初始 ghost value 同步：让 source=0 等初值能出现在需要它的 ghost cache 中。
    for (int d = 0; d < num_gpus; ++d) launch_pack_values(d);
    nccl_exchange_values();
    for (int d = 0; d < num_gpus; ++d) launch_unpack_values(d);
    for (int d = 0; d < num_gpus; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(streams[d]));
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt));

    int iter = 0;
    while (iter < 1000) {
        ++iter;

        // 1) 本地 owned 顶点计算，ghost 只读。
        for (int d = 0; d < num_gpus; ++d) {
            NvtxRange enqueue_range(nvtx_name("main enqueue bfs/check", iter, d));
            const int check_buf = iter & 1;
            const bool do_async_check = try_reserve_check_slot(d, check_buf, iter);

            CUDA_CHECK(cudaSetDevice(d));
            const auto& p = parts[d];
            int* iter_delta = do_async_check ? d_delta[d][check_buf] : d_delta_scratch[d];
            MonotonicInfo* iter_info = do_async_check ? d_info[d][check_buf] : d_info_scratch[d];

            CUDA_CHECK(cudaMemsetAsync(iter_info, 0, sizeof(MonotonicInfo), streams[d]));
            CUDA_CHECK(cudaMemsetAsync(d_update[d], 0xff, std::max(1, p.local_node_count) * sizeof(int), streams[d]));
            CUDA_CHECK(cudaMemsetAsync(d_next_active[d], 0xff, std::max(1, p.owned_count) * sizeof(int), streams[d]));

            if (p.owned_count > 0) {
                int blocks = (p.owned_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                bfsPullDualMultiGPUKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                    d_values[d], d_row_offsets[d], d_column_indices[d],
                    d_column_offsets[d], d_row_indices[d],
                    d_active[d], d_update[d],
                    p.owned_count, p.local_node_count,
                    iter_delta, iter_info, TREND_DEC
                );
                CUDA_CHECK(cudaGetLastError());

                copyOwnedUpdateToNextActiveKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                    d_update[d], d_next_active[d], p.owned_count
                );
            }

            if (do_async_check) {
                NvtxRange copy_enqueue_range(nvtx_name("main enqueue check_stream D2H", iter, d));
                CUDA_CHECK(cudaEventRecord(compute_done_events[d][check_buf], streams[d]));
                CUDA_CHECK(cudaStreamWaitEvent(check_streams[d], compute_done_events[d][check_buf], 0));
                CUDA_CHECK(cudaMemcpyAsync(
                    h_delta[d][check_buf],
                    d_delta[d][check_buf],
                    std::max(1, p.owned_count) * sizeof(int),
                    cudaMemcpyDeviceToHost,
                    check_streams[d]
                ));
                CUDA_CHECK(cudaMemcpyAsync(
                    h_info[d][check_buf],
                    d_info[d][check_buf],
                    sizeof(MonotonicInfo),
                    cudaMemcpyDeviceToHost,
                    check_streams[d]
                ));
                CUDA_CHECK(cudaEventRecord(check_events[d][check_buf], check_streams[d]));
                pending_check[pending_index(d, check_buf)].store(iter);
            }
        }
        int old_max_check_iter = max_dispatched_check_iter.load();
        while (old_max_check_iter < iter &&
               !max_dispatched_check_iter.compare_exchange_weak(old_max_check_iter, iter)) {}

        // 2) 把 ghost 激活打包，通过 NCCL 发给对应 owner GPU。
        {
            NvtxRange range(nvtx_name("main NCCL activation", iter, -1));
            for (int d = 0; d < num_gpus; ++d) launch_pack_activation(d);
            nccl_exchange_activation();
            for (int d = 0; d < num_gpus; ++d) launch_apply_activation(d);
        }

        // 3) 把本轮更新后的 owned values 同步给需要它的 ghost 节点。
        {
            NvtxRange range(nvtx_name("main NCCL values", iter, -1));
            for (int d = 0; d < num_gpus; ++d) launch_pack_values(d);
            nccl_exchange_values();
            for (int d = 0; d < num_gpus; ++d) launch_unpack_values(d);
        }

        for (int d = 0; d < num_gpus; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaStreamSynchronize(streams[d]));
        }

        // 4) 直接在每张 GPU 上对下一轮 active 做阈值筛选，标记普通/关键顶点。
        {
            NvtxRange range(nvtx_name("main score active", iter, -1));
        for (int d = 0; d < num_gpus; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            const auto& p = parts[d];

            CUDA_CHECK(cudaMemsetAsync(d_num_active[d], 0, sizeof(int), streams[d]));

            if (p.owned_count > 0) {
                int blocks = (p.owned_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                scoreAndMarkKernel<<<blocks, BLOCK_SIZE, 0, streams[d]>>>(
                    d_next_active[d], d_row_offsets[d], p.owned_count, max_outdegree,
                    alpha, beta, threshold, d_num_active[d]
                );
                CUDA_CHECK(cudaGetLastError());
            }
        }
        }

        // 5) 只回收每张 GPU 的 active 计数，用于终止判断；不再回收 active 数组。
        int total_active_nodes = 0;
        for (int d = 0; d < num_gpus; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaMemcpyAsync(
                &h_num_active[d],
                d_num_active[d],
                sizeof(int),
                cudaMemcpyDeviceToHost,
                streams[d]
            ));
        }

        {
            NvtxRange range(nvtx_name("main active count sync", iter, -1));
        for (int d = 0; d < num_gpus; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaStreamSynchronize(streams[d]));
            total_active_nodes += h_num_active[d];
        }
        }

        if (total_active_nodes == 0) break;

        // 6) 下一轮直接使用已经标记好的 d_next_active，交换指针即可。
        for (int d = 0; d < num_gpus; ++d) {
            std::swap(d_active[d], d_next_active[d]);
        }

    }

    bool pending_checks_left = true;
    while (pending_checks_left) {
        pending_checks_left = false;
        for (int d = 0; d < num_gpus; ++d) {
            for (int buf = 0; buf < 2; ++buf) {
                if (pending_check[pending_index(d, buf)].load() > 0) {
                    pending_checks_left = true;
                }
            }
        }
        if (pending_checks_left) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    for (int d = 0; d < num_gpus; ++d) {
        check_states[d]->stop.store(true);
    }
    for (auto& worker : check_workers) {
        if (worker.joinable()) worker.join();
    }
    try_aggregate_checks();

    // 6) 收集 owned values 回 host。
    for (int d = 0; d < num_gpus; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        const auto& p = parts[d];
        if (p.owned_count <= 0) continue;
        std::vector<int> h_owned_values(p.owned_count, INF);
        CUDA_CHECK(cudaMemcpy(h_owned_values.data(), d_values[d], p.owned_count * sizeof(int), cudaMemcpyDeviceToHost));
        for (int local = 0; local < p.owned_count; ++local) {
            h_value[p.start_node + local] = h_owned_values[local];
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
    printf("GPU time: %.4f ms\n", ms);
    printf("NCCL 子图划分 BFS 迭代 %d 次正常结束。\n", iter);
    printf("异步检测标记: DMR=%d", detected_dmr.load() ? 1 : 0);
    if (first_dmr_iter >= 0) printf("(first_iter=%d)", first_dmr_iter);
    printf(", monotonic=%d", detected_monotonic.load() ? 1 : 0);
    if (first_monotonic_iter >= 0) printf("(first_iter=%d)", first_monotonic_iter);
    printf(", avg_delta_increase=%d", detected_avg_increase.load() ? 1 : 0);
    if (first_avg_increase_iter >= 0) printf("(first_iter=%d)", first_avg_increase_iter);
    printf("\n");

    // 7) 释放资源。
    for (int d = 0; d < num_gpus; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        cudaFree(d_values[d]);
        cudaFree(d_active[d]);
        cudaFree(d_next_active[d]);
        cudaFree(d_update[d]);
        cudaFree(d_row_offsets[d]);
        cudaFree(d_column_indices[d]);
        cudaFree(d_column_offsets[d]);
        cudaFree(d_row_indices[d]);
        cudaFree(d_delta_scratch[d]);
        cudaFree(d_num_active[d]);
        cudaFree(d_info_scratch[d]);
        for (int b = 0; b < 2; ++b) {
            cudaFree(d_delta[d][b]);
            cudaFree(d_info[d][b]);
            cudaFreeHost(h_delta[d][b]);
            cudaFreeHost(h_info[d][b]);
            cudaEventDestroy(compute_done_events[d][b]);
            cudaEventDestroy(check_events[d][b]);
        }

        for (int peer = 0; peer < num_gpus; ++peer) {
            if (peer == d) continue;
            auto& dp = d_plan[d][peer];
            cudaFree(dp.d_act_send_local);
            cudaFree(dp.d_act_recv_owned);
            cudaFree(dp.d_value_send_owned);
            cudaFree(dp.d_value_recv_ghost);
            cudaFree(dp.d_act_send_buf);
            cudaFree(dp.d_act_recv_buf);
            cudaFree(dp.d_value_send_buf);
            cudaFree(dp.d_value_recv_buf);
        }
        cudaStreamDestroy(check_streams[d]);
        cudaStreamDestroy(streams[d]);
        ncclCommDestroy(comms[d]);
    }

    CUDA_CHECK(cudaFreeHost(h_active_global));
    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
}

#endif // FAULT_TOLERANT_BFS_MULTIGPU_NCCL_PARTITIONED_CUH
