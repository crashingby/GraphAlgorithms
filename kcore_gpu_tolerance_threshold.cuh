
#ifndef SIMPLE_TOLERANCE_UTILS_CUH
#define SIMPLE_TOLERANCE_UTILS_CUH
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BLOCK_SIZE 256
#define INF 100000

enum ValueTrend { TREND_NONE = 0, TREND_INC = 1, TREND_DEC = 2 };
struct MonotonicInfo { int dmr_error_flag; int monotonic_error_flag; };

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)
#endif

__global__ void scoreAndMarkIntKernel(int* d_active, const int* d_values, const int* d_row_offsets,
                                      int num_nodes, int max_outdegree, float alpha, float beta,
                                      float threshold, int* d_num_active) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes && d_active[v] != -1) {
        atomicAdd(d_num_active, 1);
        int outdeg = d_row_offsets[v + 1] - d_row_offsets[v];
        int safe_max = max_outdegree > 0 ? max_outdegree : 1;
        float value_score = 1.0f / (1.0f + fabsf((float)d_values[v]));
        float score = alpha * ((float)outdeg / (float)safe_max) + beta * value_score;
        d_active[v] = (score >= threshold) ? 2 : 1;
    }
}

__global__ void scoreAndMarkFloatKernel(int* d_active, const float* d_values, const int* d_row_offsets,
                                        int num_nodes, int max_outdegree, float alpha, float beta,
                                        float threshold, int* d_num_active) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes && d_active[v] != -1) {
        atomicAdd(d_num_active, 1);
        int outdeg = d_row_offsets[v + 1] - d_row_offsets[v];
        int safe_max = max_outdegree > 0 ? max_outdegree : 1;
        float value_score = fabsf(d_values[v]);
        float score = alpha * ((float)outdeg / (float)safe_max) + beta * value_score;
        d_active[v] = (score >= threshold) ? 2 : 1;
    }
}

inline int compute_max_outdegree_tol(const int* h_row_offsets, int num_nodes) {
    int max_outdegree = 1;
    for (int v = 0; v < num_nodes; ++v) {
        int outdeg = h_row_offsets[v + 1] - h_row_offsets[v];
        if (outdeg > max_outdegree) max_outdegree = outdeg;
    }
    return max_outdegree;
}
#endif


#ifndef KCORE_GPU_TOLERANCE_THRESHOLD_CUH
#define KCORE_GPU_TOLERANCE_THRESHOLD_CUH

__global__ void kcorePullDualKernel(int* d_values, int* d_alive, const int* d_row_offsets, const int* d_column_indices,
                                    const int* d_column_offsets, const int* d_row_indices, const int* d_active,
                                    int* d_update, int num_nodes, int k, int* d_delta, MonotonicInfo* d_info,
                                    ValueTrend trend) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int critical_list[BLOCK_SIZE];
    __shared__ int idle_count, critical_count, redundant_results[BLOCK_SIZE];
    if(local_tid==0){idle_count=0; critical_count=0;} __syncthreads();
    bool valid=tid<num_nodes, active=valid&&d_active[tid]!=-1, critical=active&&d_active[tid]==2, idle=!active||!valid;
    int cid=-1,iid=-1; if(critical){cid=atomicAdd(&critical_count,1); if(cid<BLOCK_SIZE) critical_list[cid]=tid;} if(idle) iid=atomicAdd(&idle_count,1); __syncthreads();
    int oldVal=valid?d_values[tid]:0; int mainVal=oldVal;
    if(active){ mainVal=0; for(int i=d_column_offsets[tid]; i<d_column_offsets[tid+1]; ++i){ int n=d_row_indices[i]; if(d_alive[n]!=0) mainVal++; } }
    __syncthreads();
    if(idle && iid<critical_count && iid<BLOCK_SIZE){ int target=critical_list[iid]; int r=0; for(int i=d_column_offsets[target]; i<d_column_offsets[target+1]; ++i){ if(d_alive[d_row_indices[i]]!=0) r++; } redundant_results[iid]=r; }
    __syncthreads();
    if(critical && cid>=0 && cid<idle_count && cid<BLOCK_SIZE && redundant_results[cid]!=mainVal) atomicExch(&d_info->dmr_error_flag,1);
    __syncthreads();
    if(active){ d_values[tid]=mainVal; int delta=mainVal-oldVal; if(mainVal<k && d_alive[tid]!=0){ d_alive[tid]=0; for(int i=d_row_offsets[tid]; i<d_row_offsets[tid+1]; ++i){ int dst=d_column_indices[i]; if(d_alive[dst]!=0) d_update[dst]=1; } d_delta[tid]=delta; if((trend==TREND_INC&&delta<0)||(trend==TREND_DEC&&delta>0)) atomicExch(&d_info->monotonic_error_flag,1); } else d_delta[tid]=0; } else if(valid) d_delta[tid]=0;
}

inline void kcoreGPU(int* h_value, const int* h_row_offsets, const int* h_column_indices,
                     const int* h_column_offsets, const int* h_row_indices, int num_nodes,
                     int num_edges, int k, float alpha, float beta, float threshold) {
    printf("开始 GPU KCORE（阈值判定容错实现）...\n");
    int max_outdegree=compute_max_outdegree_tol(h_row_offsets,num_nodes);
    int *h_active,*h_alive,*h_delta[2]; MonotonicInfo* h_info[2]; CUDA_CHECK(cudaMallocHost(&h_active,num_nodes*sizeof(int))); CUDA_CHECK(cudaMallocHost(&h_alive,num_nodes*sizeof(int)));
    for(int b=0;b<2;++b){CUDA_CHECK(cudaMallocHost(&h_delta[b],num_nodes*sizeof(int))); CUDA_CHECK(cudaMallocHost(&h_info[b],sizeof(MonotonicInfo)));}
    for(int i=0;i<num_nodes;++i){h_active[i]=1; h_alive[i]=1; h_value[i]=h_row_offsets[i+1]-h_row_offsets[i];}
    int *d_values,*d_alive,*d_ro,*d_ci,*d_co,*d_ri,*d_active,*d_update,*d_num_active,*d_delta[2]; MonotonicInfo* d_info[2];
    CUDA_CHECK(cudaMalloc(&d_values,num_nodes*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_alive,num_nodes*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_ro,(num_nodes+1)*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_ci,num_edges*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_co,(num_nodes+1)*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_ri,num_edges*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_active,num_nodes*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_update,num_nodes*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_num_active,sizeof(int)));
    for(int b=0;b<2;++b){CUDA_CHECK(cudaMalloc(&d_delta[b],num_nodes*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_info[b],sizeof(MonotonicInfo)));}
    CUDA_CHECK(cudaMemcpy(d_values,h_value,num_nodes*sizeof(int),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_alive,h_alive,num_nodes*sizeof(int),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_ro,h_row_offsets,(num_nodes+1)*sizeof(int),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_ci,h_column_indices,num_edges*sizeof(int),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_co,h_column_offsets,(num_nodes+1)*sizeof(int),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_ri,h_row_indices,num_edges*sizeof(int),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_active,h_active,num_nodes*sizeof(int),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemset(d_update,0xff,num_nodes*sizeof(int)));
    cudaStream_t stream[2]; cudaEvent_t ev[2],start,stop; for(int b=0;b<2;++b){cudaStreamCreate(&stream[b]); cudaEventCreateWithFlags(&ev[b],cudaEventDisableTiming);} cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    int blocks=(num_nodes+BLOCK_SIZE-1)/BLOCK_SIZE,iter=0,ping=0,active_nodes=num_nodes; CUDA_CHECK(cudaMemset(d_num_active,0,sizeof(int))); scoreAndMarkIntKernel<<<blocks,BLOCK_SIZE>>>(d_active,d_values,d_ro,num_nodes,max_outdegree,alpha,beta,threshold,d_num_active);
    while(iter<1000){++iter; int cur=ping,prev=1-ping; CUDA_CHECK(cudaMemsetAsync(d_info[cur],0,sizeof(MonotonicInfo),stream[cur])); kcorePullDualKernel<<<blocks,BLOCK_SIZE,0,stream[cur]>>>(d_values,d_alive,d_ro,d_ci,d_co,d_ri,d_active,d_update,num_nodes,k,d_delta[cur],d_info[cur],TREND_DEC); CUDA_CHECK(cudaMemcpyAsync(d_active,d_update,num_nodes*sizeof(int),cudaMemcpyDeviceToDevice,stream[cur])); CUDA_CHECK(cudaMemsetAsync(d_update,0xff,num_nodes*sizeof(int),stream[cur])); CUDA_CHECK(cudaMemcpyAsync(h_delta[cur],d_delta[cur],num_nodes*sizeof(int),cudaMemcpyDeviceToHost,stream[cur])); CUDA_CHECK(cudaMemcpyAsync(h_info[cur],d_info[cur],sizeof(MonotonicInfo),cudaMemcpyDeviceToHost,stream[cur])); cudaEventRecord(ev[cur],stream[cur]); if(iter>1&&cudaEventQuery(ev[prev])==cudaSuccess&&(h_info[prev]->dmr_error_flag||h_info[prev]->monotonic_error_flag)) printf("KCORE 检测标记 iter=%d DMR=%d monotonic=%d\n",iter-1,h_info[prev]->dmr_error_flag,h_info[prev]->monotonic_error_flag); CUDA_CHECK(cudaStreamSynchronize(stream[cur])); CUDA_CHECK(cudaMemsetAsync(d_num_active,0,sizeof(int),stream[cur])); scoreAndMarkIntKernel<<<blocks,BLOCK_SIZE,0,stream[cur]>>>(d_active,d_values,d_ro,num_nodes,max_outdegree,alpha,beta,threshold,d_num_active); CUDA_CHECK(cudaMemcpyAsync(&active_nodes,d_num_active,sizeof(int),cudaMemcpyDeviceToHost,stream[cur])); CUDA_CHECK(cudaStreamSynchronize(stream[cur])); if(active_nodes==0) break; ping=prev;}
    CUDA_CHECK(cudaMemcpy(h_value,d_values,num_nodes*sizeof(int),cudaMemcpyDeviceToHost)); cudaEventRecord(stop); cudaEventSynchronize(stop); float ms=0; cudaEventElapsedTime(&ms,start,stop); printf("GPU time: %.4f ms\n",ms);
    cudaFree(d_values);cudaFree(d_alive);cudaFree(d_ro);cudaFree(d_ci);cudaFree(d_co);cudaFree(d_ri);cudaFree(d_active);cudaFree(d_update);cudaFree(d_num_active); for(int b=0;b<2;++b){cudaFree(d_delta[b]);cudaFree(d_info[b]);cudaFreeHost(h_delta[b]);cudaFreeHost(h_info[b]);cudaEventDestroy(ev[b]);cudaStreamDestroy(stream[b]);} cudaFreeHost(h_active);cudaFreeHost(h_alive); cudaEventDestroy(start);cudaEventDestroy(stop); printf("GPU KCORE finished in %d iterations.\n",iter);
}
#endif
