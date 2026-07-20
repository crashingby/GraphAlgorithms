
/**
 * @file kcore_tolerance_queue.cu
 * @brief CLI entry point for queue-based single-GPU k-core detection.
 */
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "kcore_tolerance_queue.cuh"
#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"
#define GPU_DEVICE 0
void kcoreCPU(const CsrGraph &graph, int* value, int k){ int n=graph.nodes; std::vector<int8_t> alive(n,1); for(int i=0;i<n;i++) value[i]=graph.column_offsets[i+1]-graph.column_offsets[i]; bool changed=true; while(changed){ changed=false; for(int u=0;u<n;u++){ if(alive[u]&&value[u]<k){ alive[u]=0; changed=true; for(int j=graph.row_offsets[u];j<graph.row_offsets[u+1];j++){int v=graph.column_indices[j]; if(alive[v]&&value[v]>0) value[v]--; } } } } }
bool correctTest(int n,const int* ref,const int* gpu,int k){ bool pass=true; int nerr=0; for(int i=0;i<n;i++){ if(ref[i]!=gpu[i]){ if(ref[i]<k&&gpu[i]<k) continue; if(nerr++<20) printf("Node %d: CPU %d, GPU %d\n",i,ref[i],gpu[i]); pass=false; }} printf("%s\n",pass?"passed":"failed"); return pass; }
int main(int argc, char** argv)
{
    GraphCliOptions opts;
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "kcore", opts)) return 1;
    graph_print_config(opts, "kcore", "tolerance_queue", false, true, true);

    cudaSetDevice(GPU_DEVICE);
    CsrGraph g;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), g, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * g.nodes);
    kcoreGPU(value, g.row_offsets, g.column_indices, g.column_offsets, g.row_indices,
             g.nodes, g.edges, opts.k, opts.alpha, opts.beta, opts.threshold);

    if (opts.run_cpu) {
        int* ref = (int*)malloc(sizeof(int) * g.nodes);
        kcoreCPU(g, ref, opts.k);
        correctTest(g.nodes, ref, value, opts.k);
        free(ref);
    }

    FILE* f = fopen(graph_output_path("kcore", "info_outcome.txt").c_str(), "w");
    if (f) {
        for (int i = 0; i < g.nodes; i++) fprintf(f, "%d\n", value[i]);
        fclose(f);
    }
    free(value);
    cudaDeviceReset();
    return 0;
}
