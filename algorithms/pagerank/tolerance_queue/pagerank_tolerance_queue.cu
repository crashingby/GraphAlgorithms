
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "pagerank_tolerance_queue.cuh"
#include "include/graph.h"
#include "include/output.h"
#define GPU_DEVICE 0
void usage(const char* p){ printf("Usage: %s <dataset_id> [-a alpha] [-b beta] [-t threshold]\n",p); }
int main(int argc,char** argv){ float alpha=0.5f,beta=0.5f,threshold=0.3f; if(argc<2||argv[1][0]=='-'){usage(argv[0]);return 1;} std::string graph_path="dataset/"+std::string(argv[1])+".mtx"; optind=2; int opt; while((opt=getopt(argc,argv,"a:b:t:h"))!=-1){ if(opt=='a') alpha=atof(optarg); else if(opt=='b') beta=atof(optarg); else if(opt=='t') threshold=atof(optarg); else {usage(argv[0]);return opt=='h'?0:1;} } printf("加载数据集: %s\n",graph_path.c_str()); printf("参数配置: alpha=%.2f beta=%.2f threshold=%.2f\n",alpha,beta,threshold); cudaSetDevice(GPU_DEVICE); CsrGraph g; bool undirected=false; if(BuildMarketGraph(graph_path.c_str(),g,undirected)!=0){fprintf(stderr,"Failed to load graph.\n");return 1;} float* value=(float*)malloc(sizeof(float)*g.nodes); pagerankGPU(value,g.row_offsets,g.column_indices,g.column_offsets,g.row_indices,g.nodes,g.edges,alpha,beta,threshold); FILE* f=fopen(graph_output_path("pagerank", "info_outcome.txt").c_str(), "w"); if(f){for(int i=0;i<g.nodes;i++) fprintf(f,"%f\n",value[i]); fclose(f);} free(value); cudaDeviceReset(); return 0; }
