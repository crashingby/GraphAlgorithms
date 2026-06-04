
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "cc_tolerance_compact_queue.cuh"
#include "include/graph.h"
#include "include/output.h"
#define GPU_DEVICE 0

void ccCPU(const CsrGraph &graph, int* value) {
    int n = graph.nodes; for (int i = 0; i < n; i++) value[i] = i;
    bool changed = true; while (changed) { changed = false; for (int u = 0; u < n; u++) { int old = value[u]; for (int j = graph.row_offsets[u]; j < graph.row_offsets[u + 1]; ++j) { int v = graph.column_indices[j]; if (value[v] > value[u]) value[u] = value[v]; } if (value[u] != old) changed = true; } }
}
bool correctTest(int n, const int* ref, const int* gpu) { bool pass=true; int nerr=0; for(int i=0;i<n;i++){ if(ref[i]!=gpu[i]){ if(nerr++<20) printf("Node %d: CPU %d, GPU %d\n",i,ref[i],gpu[i]); pass=false; }} printf("%s\n", pass?"passed":"failed"); return pass; }
void usage(const char* p){ printf("Usage: %s <dataset_id> [-a alpha] [-b beta] [-t threshold]\n", p); }
int main(int argc, char** argv){ float alpha=0.5f,beta=0.5f,threshold=0.3f; bool run_CPU=false; if(argc<2||argv[1][0]=='-'){usage(argv[0]);return 1;} std::string graph_path="dataset/"+std::string(argv[1])+".mtx"; optind=2; int opt; while((opt=getopt(argc,argv,"a:b:t:h"))!=-1){ if(opt=='a') alpha=atof(optarg); else if(opt=='b') beta=atof(optarg); else if(opt=='t') threshold=atof(optarg); else {usage(argv[0]); return opt=='h'?0:1;} }
    printf("加载数据集: %s\n", graph_path.c_str()); printf("参数配置: alpha=%.2f, beta=%.2f, threshold=%.2f\n",alpha,beta,threshold); cudaSetDevice(GPU_DEVICE); CsrGraph g; bool undirected=false; if(BuildMarketGraph(graph_path.c_str(),g,undirected)!=0){fprintf(stderr,"Failed to load graph.\n");return 1;} int* value=(int*)malloc(sizeof(int)*g.nodes); ccGPU(value,g.row_offsets,g.column_indices,g.column_offsets,g.row_indices,g.nodes,g.edges,alpha,beta,threshold); if(run_CPU){int* ref=(int*)malloc(sizeof(int)*g.nodes); ccCPU(g,ref); correctTest(g.nodes,ref,value); free(ref);} FILE* f=fopen(graph_output_path("cc", "info_outcome.txt").c_str(), "w"); if(f){for(int i=0;i<g.nodes;i++) fprintf(f,"%d\n",value[i]); fclose(f);} free(value); cudaDeviceReset(); return 0; }
