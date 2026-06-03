#ifndef PARTITIONED_MULTIGPU_COMMON_CUH
#define PARTITIONED_MULTIGPU_COMMON_CUH
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#define BLOCK_SIZE 256
#define MAX_GPUS 8
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { cudaError_t e=(call); if (e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE);} } while (0)
#endif
#ifndef NCCL_CHECK
#define NCCL_CHECK(call) do { ncclResult_t r=(call); if (r!=ncclSuccess){fprintf(stderr,"NCCL error %s:%d: %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); exit(EXIT_FAILURE);} } while (0)
#endif
inline int mg_owner_of_vertex(int v, int nodes_per_gpu, int num_gpus) {
int owner=v/nodes_per_gpu;
return owner>=num_gpus? num_gpus - 1 : owner;
}

struct MgSubgraphHost {
int start_node=0, end_node=0, owned_count=0, local_node_count=0;
std::vector<int> local_to_global;
std::unordered_map<int, int> global_to_local;
std::vector<int> row_offsets, column_indices, column_offsets, row_indices, ghost_local_ids, ghost_global_ids;
};
inline int mg_get_or_add_local(MgSubgraphHost& p, int gv, bool owned) {
auto it=p.global_to_local.find(gv);
if (it!=p.global_to_local.end()) return it->second;
int local=(int)p.local_to_global.size();
p.global_to_local[gv]=local;
p.local_to_global.push_back(gv);
if (!owned) {
p.ghost_local_ids.push_back(local);
p.ghost_global_ids.push_back(gv);
}
return local;
}

inline void mg_build_subgraph(int gpu_id, int num_gpus, int num_nodes, int nodes_per_gpu, const int* h_ro, const int* h_ci, const int* h_co, const int* h_ri, MgSubgraphHost& p) {
p.start_node=gpu_id*nodes_per_gpu;
p.end_node=std::min(p.start_node+nodes_per_gpu, num_nodes);
p.owned_count=std::max(0, p.end_node-p.start_node);
p.local_to_global.clear();
p.global_to_local.clear();
p.ghost_local_ids.clear();
p.ghost_global_ids.clear();
for (int v = p.start_node; v < p.end_node; ++v) {
int local=(int)p.local_to_global.size();
p.global_to_local[v]=local;
p.local_to_global.push_back(v);
}
p.row_offsets.assign(p.owned_count+1, 0);
p.column_offsets.assign(p.owned_count+1, 0);
p.column_indices.clear();
p.row_indices.clear();
int out=0;
for (int lv = 0; lv < p.owned_count; ++lv) {
int gv=p.start_node+lv;
p.row_offsets[lv]=out;
for (int e = h_ro[gv]; e < h_ro[gv + 1]; ++e) {
int gd=h_ci[e];
bool owned=gd>=p.start_node && gd<p.end_node;
p.column_indices.push_back(mg_get_or_add_local(p, gd, owned));
++out;
}
p.row_offsets[lv+1]=out;
}
int in=0;
for (int lv = 0; lv < p.owned_count; ++lv) {
int gv=p.start_node+lv;
p.column_offsets[lv]=in;
for (int e = h_co[gv]; e < h_co[gv + 1]; ++e) {
int gs=h_ri[e];
bool owned=gs>=p.start_node && gs<p.end_node;
p.row_indices.push_back(mg_get_or_add_local(p, gs, owned));
++in;
}
p.column_offsets[lv+1]=in;
}
p.local_node_count=(int)p.local_to_global.size();
if (p.column_indices.empty()) p.column_indices.push_back(0);
if (p.row_indices.empty()) p.row_indices.push_back(0);
}

struct MgPeerPlanHost {
std::vector<int> act_send_local, act_recv_owned, value_send_owned, value_recv_ghost;
};
struct MgPeerPlanDevice {
int act_send_count=0, act_recv_count=0, value_send_count=0, value_recv_count=0;
int *d_act_send_local=nullptr, *d_act_recv_owned=nullptr, *d_value_send_owned=nullptr, *d_value_recv_ghost=nullptr;
int *d_act_send_buf=nullptr, *d_act_recv_buf=nullptr, *d_value_send_buf_int=nullptr, *d_value_recv_buf_int=nullptr;
float *d_value_send_buf_float=nullptr, *d_value_recv_buf_float=nullptr;
};
inline void mg_upload_index(const std::vector<int>& h, int** d) {
if (h.empty()) {
*d=nullptr;
return;
}
CUDA_CHECK(cudaMalloc(d, h.size()*sizeof(int)));
CUDA_CHECK(cudaMemcpy(*d, h.data(), h.size()*sizeof(int), cudaMemcpyHostToDevice));
}

inline void mg_alloc_int(int** d, int n) {
if (n<=0) {
*d=nullptr;
return;
}
CUDA_CHECK(cudaMalloc(d, n*sizeof(int)));
}

inline void mg_alloc_float(float** d, int n) {
if (n<=0) {
*d=nullptr;
return;
}
CUDA_CHECK(cudaMalloc(d, n*sizeof(float)));
}

__global__ void mgPackIntKernel(const int* src, const int* idx, int* out, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if (i<n) out[i]=src[idx[i]];
}

__global__ void mgUnpackIntKernel(const int* in, const int* idx, int* dst, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if (i<n) dst[idx[i]]=in[i];
}

__global__ void mgPackFloatKernel(const float* src, const int* idx, float* out, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if (i<n) out[i]=src[idx[i]];
}

__global__ void mgUnpackFloatKernel(const float* in, const int* idx, float* dst, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if (i<n) dst[idx[i]]=in[i];
}

__global__ void mgApplyActivationKernel(const int* flags, const int* owned_idx, int* next_active, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if (i<n && flags[i]!=-1) next_active[owned_idx[i]]=1;
}

__global__ void mgCopyOwnedUpdateKernel(const int* update, int* next_active, int owned_count) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if (i<owned_count) next_active[i]=update[i];
}

__global__ void mgCountActiveKernel(const int* active, int* count, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if (i<n && active[i]!=-1) atomicAdd(count, 1);
}

__global__ void mgScoreAndMarkIntKernel(int* active, const int* values, const int* ro, int owned_count, int max_outdegree, float alpha, float beta, float threshold, int* count) {
int v=blockIdx.x*blockDim.x+threadIdx.x;
if (v<owned_count && active[v]!=-1) {
atomicAdd(count, 1);
int outdeg=ro[v+1]-ro[v];
int safe=max_outdegree>0? max_outdegree : 1;
float value_score=1.0f/(1.0f+fabsf((float)values[v]));
float score=alpha*((float)outdeg/(float)safe)+beta*value_score;
active[v]=(score>=threshold)? 2 : 1;
}
}

__global__ void mgScoreAndMarkFloatKernel(int* active, const float* values, const int* ro, int owned_count, int max_outdegree, float alpha, float beta, float threshold, int* count) {
int v=blockIdx.x*blockDim.x+threadIdx.x;
if (v<owned_count && active[v]!=-1) {
atomicAdd(count, 1);
int outdeg=ro[v+1]-ro[v];
int safe=max_outdegree>0? max_outdegree : 1;
float value_score=fabsf(values[v]);
float score=alpha*((float)outdeg/(float)safe)+beta*value_score;
active[v]=(score>=threshold)? 2 : 1;
}
}

struct MgCheckInfo {
int dmr_error_flag=0;
int monotonic_error_flag=0;
};
struct MgAsyncCheckResult {
bool done=false;
double sum_delta=0.0;
int count=0;
int dmr_error=0;
int monotonic_error=0;
};
inline int mg_pending_index(int gpu, int buf) {
return gpu*2+buf;
}
#endif

#ifndef CC_TOLERANCE_MULTIGPU_CUH
#define CC_TOLERANCE_MULTIGPU_CUH

__global__ void ccPartitionKernel(int* values, const int* ro, const int* ci, const int* co, const int* ri, const int* active, int* update, int owned_count, int local_count, int* delta, MgCheckInfo* check_info) {
int tid=blockIdx.x*blockDim.x+threadIdx.x;
int local_tid=threadIdx.x;
__shared__ int critical_list[BLOCK_SIZE];
__shared__ int idle_count, critical_count, redundant_results[BLOCK_SIZE];
if (local_tid==0) {
idle_count=0;
critical_count=0;
}
__syncthreads();
bool valid=tid<owned_count;
bool is_active=valid && active[tid]!=-1;
bool critical=is_active && active[tid]==2;
bool idle=!is_active || !valid;
int cid=-1, iid=-1;
if (critical) {
cid=atomicAdd(&critical_count, 1);
if (cid<BLOCK_SIZE) critical_list[cid]=tid;
}
if (idle) iid=atomicAdd(&idle_count, 1);
__syncthreads();
int old=valid?values[tid]: 0;
int nv=old;
if (is_active) {
for (int i=co[tid];
i<co[tid+1];
++i) {
int src=ri[i];
if (src>=0 && src<local_count) {
int cand=values[src];
if (cand>nv) nv=cand;
}
}
}
__syncthreads();
if (idle && iid<critical_count && iid<BLOCK_SIZE) {
int target=critical_list[iid];
int r=values[target];
for (int i=co[target];
i<co[target+1];
++i) {
int src=ri[i];
if (src>=0 && src<local_count) {
int cand=values[src];
if (cand>r) r=cand;
}
} redundant_results[iid]=r;
}
__syncthreads();
if (critical && cid>=0 && cid<idle_count && cid<BLOCK_SIZE && redundant_results[cid]!=nv) atomicExch(&check_info->dmr_error_flag, 1);
__syncthreads();
if (is_active) {
values[tid]=nv;
if (nv>old) {
for (int i=ro[tid];
i<ro[tid+1];
++i) {
int dst=ci[i];
if (dst>=0 && dst<local_count) update[dst]=1;
}
int diff=nv-old;
delta[tid]=diff;
if (diff<0) atomicExch(&check_info->monotonic_error_flag, 1);
} else delta[tid]=0;
} else if (valid) delta[tid]=0;
}

inline void ccMultiGPU(int* h_value, const int* h_ro, const int* h_ci, const int* h_co, const int* h_ri, int num_nodes, int num_edges, float alpha=0.5f, float beta=0.5f, float threshold=0.3f) {
(void)num_edges;
(void)alpha;
(void)beta;
int ng=0;
CUDA_CHECK(cudaGetDeviceCount(&ng));
if (ng<=0) {
fprintf(stderr, "没有可用 GPU。\n");
exit(EXIT_FAILURE);
}
if (ng>MAX_GPUS) ng=MAX_GPUS;
int nodes_per_gpu=(num_nodes+ng-1)/ng;
int max_outdegree=1;
for (int v=0;
v<num_nodes;
++v) {
int outdeg=h_ro[v+1]-h_ro[v];
if (outdeg>max_outdegree) max_outdegree=outdeg;
} printf("发现 %d 个 GPU，开启 NCCL 子图划分 CC：owned + ghost。\n", ng);
std::vector<int> devs(ng);
std::iota(devs.begin(), devs.end(), 0);
std::vector<ncclComm_t> comms(ng);
NCCL_CHECK(ncclCommInitAll(comms.data(), ng, devs.data()));
std::vector<MgSubgraphHost> parts(ng);
for (int d=0;
d<ng;
++d) {
mg_build_subgraph(d, ng, num_nodes, nodes_per_gpu, h_ro, h_ci, h_co, h_ri, parts[d]);
printf("GPU %d: owned [%d, %d), owned_count=%d, local_count=%d, ghosts=%zu\n", d, parts[d].start_node, parts[d].end_node, parts[d].owned_count, parts[d].local_node_count, parts[d].ghost_local_ids.size());
}
std::vector<std::vector<MgPeerPlanHost>> hp(ng, std::vector<MgPeerPlanHost>(ng));
for (int s=0;
s<ng;
++s) {
for (size_t k=0;
k<parts[s].ghost_local_ids.size();
++k) {
int gl=parts[s].ghost_local_ids[k], gg=parts[s].ghost_global_ids[k], owner=mg_owner_of_vertex(gg, nodes_per_gpu, ng);
if (owner==s) continue;
auto it=parts[owner].global_to_local.find(gg);
if (it==parts[owner].global_to_local.end()) continue;
int ol=it->second;
hp[s][owner].act_send_local.push_back(gl);
hp[owner][s].act_recv_owned.push_back(ol);
hp[owner][s].value_send_owned.push_back(ol);
hp[s][owner].value_recv_ghost.push_back(gl);
}
}
std::vector<int*> values(ng), active(ng), next_active(ng), update(ng), ro(ng), ci(ng), co(ng), ri(ng), num_active(ng), check_delta(ng), scratch_delta(ng);
std::vector<MgCheckInfo*> check_info(ng), scratch_info(ng);
std::vector<int*> h_check_delta(ng*2, nullptr);
std::vector<MgCheckInfo*> h_check_info(ng*2, nullptr);
std::vector<cudaEvent_t> compute_done(ng*2), check_done(ng*2);
std::vector<std::atomic<int>> pending(ng*2);
for (auto& v: pending) v.store(0);
std::vector<MgAsyncCheckResult> check_results(1001*ng);
std::atomic<bool> check_stop(false);
std::vector<int> h_num_active(ng);
std::vector<cudaStream_t> streams(ng), check_streams(ng);
std::vector<std::vector<MgPeerPlanDevice>> dp(ng, std::vector<MgPeerPlanDevice>(ng));
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
auto& p=parts[d];
CUDA_CHECK(cudaStreamCreate(&streams[d]));
CUDA_CHECK(cudaStreamCreateWithFlags(&check_streams[d], cudaStreamNonBlocking));
int la=std::max(1, p.local_node_count), oa=std::max(1, p.owned_count);
CUDA_CHECK(cudaMalloc(&values[d], la*sizeof(int)));
CUDA_CHECK(cudaMalloc(&active[d], oa*sizeof(int)));
CUDA_CHECK(cudaMalloc(&next_active[d], oa*sizeof(int)));
CUDA_CHECK(cudaMalloc(&update[d], la*sizeof(int)));
CUDA_CHECK(cudaMalloc(&ro[d], (p.owned_count+1)*sizeof(int)));
CUDA_CHECK(cudaMalloc(&co[d], (p.owned_count+1)*sizeof(int)));
CUDA_CHECK(cudaMalloc(&ci[d], p.column_indices.size()*sizeof(int)));
CUDA_CHECK(cudaMalloc(&ri[d], p.row_indices.size()*sizeof(int)));
CUDA_CHECK(cudaMalloc(&num_active[d], sizeof(int)));
CUDA_CHECK(cudaMalloc(&check_delta[d], oa*sizeof(int)));
CUDA_CHECK(cudaMalloc(&scratch_delta[d], oa*sizeof(int)));
CUDA_CHECK(cudaMalloc(&check_info[d], sizeof(MgCheckInfo)));
CUDA_CHECK(cudaMalloc(&scratch_info[d], sizeof(MgCheckInfo)));
for (int buf=0;
buf<2;
++buf) {
int slot=mg_pending_index(d, buf);
CUDA_CHECK(cudaHostAlloc(&h_check_delta[slot], oa*sizeof(int), cudaHostAllocDefault));
CUDA_CHECK(cudaHostAlloc(&h_check_info[slot], sizeof(MgCheckInfo), cudaHostAllocDefault));
CUDA_CHECK(cudaEventCreateWithFlags(&compute_done[slot], cudaEventDisableTiming));
CUDA_CHECK(cudaEventCreateWithFlags(&check_done[slot], cudaEventDisableTiming));
}
std::vector<int> hv(la), ha(oa, 1);
for (int l=0;
l<p.local_node_count;
++l) hv[l]=p.local_to_global[l];
CUDA_CHECK(cudaMemcpy(values[d], hv.data(), la*sizeof(int), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(active[d], ha.data(), oa*sizeof(int), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemset(update[d], 0xff, la*sizeof(int)));
CUDA_CHECK(cudaMemset(next_active[d], 0xff, oa*sizeof(int)));
CUDA_CHECK(cudaMemcpy(ro[d], p.row_offsets.data(), (p.owned_count+1)*sizeof(int), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(co[d], p.column_offsets.data(), (p.owned_count+1)*sizeof(int), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(ci[d], p.column_indices.data(), p.column_indices.size()*sizeof(int), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(ri[d], p.row_indices.data(), p.row_indices.size()*sizeof(int), cudaMemcpyHostToDevice));
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& h=hp[d][peer];
auto& q=dp[d][peer];
q.act_send_count=h.act_send_local.size();
q.act_recv_count=h.act_recv_owned.size();
q.value_send_count=h.value_send_owned.size();
q.value_recv_count=h.value_recv_ghost.size();
mg_upload_index(h.act_send_local, &q.d_act_send_local);
mg_upload_index(h.act_recv_owned, &q.d_act_recv_owned);
mg_upload_index(h.value_send_owned, &q.d_value_send_owned);
mg_upload_index(h.value_recv_ghost, &q.d_value_recv_ghost);
mg_alloc_int(&q.d_act_send_buf, q.act_send_count);
mg_alloc_int(&q.d_act_recv_buf, q.act_recv_count);
mg_alloc_int(&q.d_value_send_buf_int, q.value_send_count);
mg_alloc_int(&q.d_value_recv_buf_int, q.value_recv_count);
}
} auto pack_act=[&](int d) {
CUDA_CHECK(cudaSetDevice(d));
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& q=dp[d][peer];
if (q.act_send_count>0) {
int b=(q.act_send_count+BLOCK_SIZE-1)/BLOCK_SIZE;
mgPackIntKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(update[d], q.d_act_send_local, q.d_act_send_buf, q.act_send_count);
}
}
};
auto apply_act=[&](int d) {
CUDA_CHECK(cudaSetDevice(d));
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& q=dp[d][peer];
if (q.act_recv_count>0) {
int b=(q.act_recv_count+BLOCK_SIZE-1)/BLOCK_SIZE;
mgApplyActivationKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(q.d_act_recv_buf, q.d_act_recv_owned, next_active[d], q.act_recv_count);
}
}
};
auto pack_val=[&](int d) {
CUDA_CHECK(cudaSetDevice(d));
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& q=dp[d][peer];
if (q.value_send_count>0) {
int b=(q.value_send_count+BLOCK_SIZE-1)/BLOCK_SIZE;
mgPackIntKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(values[d], q.d_value_send_owned, q.d_value_send_buf_int, q.value_send_count);
}
}
};
auto unpack_val=[&](int d) {
CUDA_CHECK(cudaSetDevice(d));
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& q=dp[d][peer];
if (q.value_recv_count>0) {
int b=(q.value_recv_count+BLOCK_SIZE-1)/BLOCK_SIZE;
mgUnpackIntKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(q.d_value_recv_buf_int, q.d_value_recv_ghost, values[d], q.value_recv_count);
}
}
};
auto xact=[&]() {
NCCL_CHECK(ncclGroupStart());
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& q=dp[d][peer];
if (q.act_recv_count>0) NCCL_CHECK(ncclRecv(q.d_act_recv_buf, q.act_recv_count, ncclInt, peer, comms[d], streams[d]));
if (q.act_send_count>0) NCCL_CHECK(ncclSend(q.d_act_send_buf, q.act_send_count, ncclInt, peer, comms[d], streams[d]));
}
}
NCCL_CHECK(ncclGroupEnd());
};
auto xval=[&]() {
NCCL_CHECK(ncclGroupStart());
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& q=dp[d][peer];
if (q.value_recv_count>0) NCCL_CHECK(ncclRecv(q.d_value_recv_buf_int, q.value_recv_count, ncclInt, peer, comms[d], streams[d]));
if (q.value_send_count>0) NCCL_CHECK(ncclSend(q.d_value_send_buf_int, q.value_send_count, ncclInt, peer, comms[d], streams[d]));
}
}
NCCL_CHECK(ncclGroupEnd());
};
for (int d=0;
d<ng;
++d) pack_val(d);
xval();
for (int d=0;
d<ng;
++d) unpack_val(d);
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
CUDA_CHECK(cudaStreamSynchronize(streams[d]));
}
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
CUDA_CHECK(cudaMemsetAsync(num_active[d], 0, sizeof(int), streams[d]));
if (parts[d].owned_count>0) {
int b=(parts[d].owned_count+BLOCK_SIZE-1)/BLOCK_SIZE;
mgScoreAndMarkIntKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(active[d], values[d], ro[d], parts[d].owned_count, max_outdegree, alpha, beta, threshold, num_active[d]);
}
}
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
CUDA_CHECK(cudaStreamSynchronize(streams[d]));
}
std::vector<std::thread> check_workers;
for (int wd=0;
wd<ng;
++wd) {
check_workers.emplace_back([&, wd]() {
CUDA_CHECK(cudaSetDevice(wd));
while (!check_stop.load(std::memory_order_relaxed)) {
bool progressed=false;
for (int buf=0;
buf<2;
++buf) {
int slot=mg_pending_index(wd, buf);
int it=pending[slot].load(std::memory_order_acquire);
if (it>0 && cudaEventQuery(check_done[slot])==cudaSuccess) {
auto& p=parts[wd];
MgAsyncCheckResult r;
r.done=true;
r.count=p.owned_count;
for (int i=0;
i<p.owned_count;
++i) r.sum_delta+=std::fabs((double)h_check_delta[slot][i]);
r.dmr_error=h_check_info[slot]->dmr_error_flag;
r.monotonic_error=h_check_info[slot]->monotonic_error_flag;
if (it>=0 && it<=1000) check_results[it*ng+wd]=r;
pending[slot].store(0, std::memory_order_release);
progressed=true;
}
}
if (!progressed) std:: this_thread:: sleep_for (std:: chrono:: microseconds(50));
}
});
} cudaEvent_t start, stop;
CUDA_CHECK(cudaSetDevice(0));
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
int iter=0;
while (iter<1000) {
++iter;
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
auto& p=parts[d];
CUDA_CHECK(cudaMemsetAsync(update[d], 0xff, std::max(1, p.local_node_count)*sizeof(int), streams[d]));
CUDA_CHECK(cudaMemsetAsync(next_active[d], 0xff, std::max(1, p.owned_count)*sizeof(int), streams[d]));
if (p.owned_count>0) {
int b=(p.owned_count+BLOCK_SIZE-1)/BLOCK_SIZE;
int check_buf=iter&1;
int slot=mg_pending_index(d, check_buf);
int expected=0;
bool do_check=pending[slot].compare_exchange_strong(expected, -1, std::memory_order_acq_rel);
int* iter_delta=do_check?check_delta[d]: scratch_delta[d];
MgCheckInfo* iter_info=do_check?check_info[d]: scratch_info[d];
CUDA_CHECK(cudaMemsetAsync(iter_info, 0, sizeof(MgCheckInfo), streams[d]));
ccPartitionKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(values[d], ro[d], ci[d], co[d], ri[d], active[d], update[d], p.owned_count, p.local_node_count, iter_delta, iter_info);
mgCopyOwnedUpdateKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(update[d], next_active[d], p.owned_count);
if (do_check) {
CUDA_CHECK(cudaEventRecord(compute_done[slot], streams[d]));
CUDA_CHECK(cudaStreamWaitEvent(check_streams[d], compute_done[slot], 0));
CUDA_CHECK(cudaMemcpyAsync(h_check_delta[slot], iter_delta, p.owned_count*sizeof(int), cudaMemcpyDeviceToHost, check_streams[d]));
CUDA_CHECK(cudaMemcpyAsync(h_check_info[slot], iter_info, sizeof(MgCheckInfo), cudaMemcpyDeviceToHost, check_streams[d]));
CUDA_CHECK(cudaEventRecord(check_done[slot], check_streams[d]));
pending[slot].store(iter, std::memory_order_release);
}
}
}
for (int d=0;
d<ng;
++d) pack_act(d);
xact();
for (int d=0;
d<ng;
++d) apply_act(d);
for (int d=0;
d<ng;
++d) pack_val(d);
xval();
for (int d=0;
d<ng;
++d) unpack_val(d);
int total=0;
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
CUDA_CHECK(cudaMemsetAsync(num_active[d], 0, sizeof(int), streams[d]));
if (parts[d].owned_count>0) {
int b=(parts[d].owned_count+BLOCK_SIZE-1)/BLOCK_SIZE;
mgScoreAndMarkIntKernel<<<b, BLOCK_SIZE, 0, streams[d]>>>(next_active[d], values[d], ro[d], parts[d].owned_count, max_outdegree, alpha, beta, threshold, num_active[d]);
}
CUDA_CHECK(cudaMemcpyAsync(&h_num_active[d], num_active[d], sizeof(int), cudaMemcpyDeviceToHost, streams[d]));
}
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
CUDA_CHECK(cudaStreamSynchronize(streams[d]));
total+=h_num_active[d];
}
if (total==0) break;
for (int d=0;
d<ng;
++d) std:: swap(active[d], next_active[d]);
}
bool waiting_checks=true;
while (waiting_checks) {
waiting_checks=false;
for (int d=0;
d<ng;
++d) for (int buf=0;
buf<2;
++buf) if (pending[mg_pending_index(d, buf)].load(std::memory_order_acquire)!=0) waiting_checks=true;
if (waiting_checks) std:: this_thread:: sleep_for (std:: chrono:: microseconds(50));
} check_stop.store(true, std::memory_order_release);
for (auto& t: check_workers) t.join();
int dmr_flag=0, monotonic_flag=0, avg_delta_increase_flag=0, first_dmr=-1, first_mono=-1, first_avg=-1;
double prev_avg=1e100;
for (int it=1;
it<=iter;
++it) {
double sum=0.0;
int cnt=0;
for (int d=0;
d<ng;
++d) {
auto& r=check_results[it*ng+d];
if (!r.done) continue;
sum+=r.sum_delta;
cnt+=r.count;
if (r.dmr_error && first_dmr<0) {
dmr_flag=1;
first_dmr=it;
}
if (r.monotonic_error && first_mono<0) {
monotonic_flag=1;
first_mono=it;
}
}
if (cnt>0) {
double avg=sum/(double)cnt;
if (prev_avg<1e99 && avg>prev_avg*(1.0+(double)threshold) && first_avg<0) {
avg_delta_increase_flag=1;
first_avg=it;
} prev_avg=avg;
}
} printf("异步CPU检测(CC): DMR=%d(first_iter=%d), monotonic=%d(first_iter=%d), avg_delta_increase=%d(first_iter=%d)\n", dmr_flag, first_dmr, monotonic_flag, first_mono, avg_delta_increase_flag, first_avg);
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
auto& p=parts[d];
if (p.owned_count<=0) continue;
std::vector<int> out(p.owned_count);
CUDA_CHECK(cudaMemcpy(out.data(), values[d], p.owned_count*sizeof(int), cudaMemcpyDeviceToHost));
for (int l=0;
l<p.owned_count;
++l) h_value[p.start_node+l]=out[l];
}
CUDA_CHECK(cudaSetDevice(0));
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms=0;
cudaEventElapsedTime(&ms, start, stop);
printf("GPU time: %.4f ms\n", ms);
printf("NCCL 子图划分 CC 迭代 %d 次结束。\n", iter);
for (int d=0;
d<ng;
++d) {
CUDA_CHECK(cudaSetDevice(d));
cudaFree(values[d]);
cudaFree(active[d]);
cudaFree(next_active[d]);
cudaFree(update[d]);
cudaFree(ro[d]);
cudaFree(ci[d]);
cudaFree(co[d]);
cudaFree(ri[d]);
cudaFree(num_active[d]);
cudaFree(check_delta[d]);
cudaFree(scratch_delta[d]);
cudaFree(check_info[d]);
cudaFree(scratch_info[d]);
for (int buf=0;
buf<2;
++buf) {
int slot=mg_pending_index(d, buf);
cudaFreeHost(h_check_delta[slot]);
cudaFreeHost(h_check_info[slot]);
cudaEventDestroy(compute_done[slot]);
cudaEventDestroy(check_done[slot]);
}
for (int peer=0;
peer<ng;
++peer) {
if (peer==d)continue;
auto& q=dp[d][peer];
cudaFree(q.d_act_send_local);
cudaFree(q.d_act_recv_owned);
cudaFree(q.d_value_send_owned);
cudaFree(q.d_value_recv_ghost);
cudaFree(q.d_act_send_buf);
cudaFree(q.d_act_recv_buf);
cudaFree(q.d_value_send_buf_int);
cudaFree(q.d_value_recv_buf_int);
} cudaStreamDestroy(check_streams[d]);
cudaStreamDestroy(streams[d]);
ncclCommDestroy(comms[d]);
} cudaEventDestroy(start);
cudaEventDestroy(stop);
}
#endif
