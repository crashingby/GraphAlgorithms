
/**
 * @file cc_tolerance_queue.cu
 * @brief CLI entry point for queue-based single-GPU CC detection.
 */
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "cc_tolerance_queue.cuh"
#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"
#define GPU_DEVICE 0

void ccCPU(const CsrGraph &graph, int* value) {
    int n = graph.nodes; for (int i = 0; i < n; i++) value[i] = i;
    bool changed = true; while (changed) { changed = false; for (int u = 0; u < n; u++) { int old = value[u]; for (int j = graph.row_offsets[u]; j < graph.row_offsets[u + 1]; ++j) { int v = graph.column_indices[j]; if (value[v] > value[u]) value[u] = value[v]; } if (value[u] != old) changed = true; } }
}
bool correctTest(int n, const int* ref, const int* gpu) { bool pass=true; int nerr=0; for(int i=0;i<n;i++){ if(ref[i]!=gpu[i]){ if(nerr++<20) printf("Node %d: CPU %d, GPU %d\n",i,ref[i],gpu[i]); pass=false; }} printf("%s\n", pass?"passed":"failed"); return pass; }
int main(int argc, char** argv)
{
    GraphCliOptions opts;
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "cc", opts)) return 1;
    graph_print_config(opts, "cc", "tolerance_queue", false, false, true);

    cudaSetDevice(GPU_DEVICE);
    CsrGraph g;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), g, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * g.nodes);
    ccGPU(value, g.row_offsets, g.column_indices, g.column_offsets, g.row_indices,
          g.nodes, g.edges, opts.alpha, opts.beta, opts.threshold);

    if (opts.run_cpu) {
        int* ref = (int*)malloc(sizeof(int) * g.nodes);
        ccCPU(g, ref);
        correctTest(g.nodes, ref, value);
        free(ref);
    }

    FILE* f = fopen(graph_output_path("cc", "info_outcome.txt").c_str(), "w");
    if (f) {
        for (int i = 0; i < g.nodes; i++) fprintf(f, "%d\n", value[i]);
        fclose(f);
    }
    free(value);
    cudaDeviceReset();
    return 0;
}
