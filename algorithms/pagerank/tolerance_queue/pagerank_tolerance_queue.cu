
/**
 * @file pagerank_tolerance_queue.cu
 * @brief CLI entry point for queue-based single-GPU PageRank detection.
 */
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "pagerank_tolerance_queue.cuh"
#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"
#define GPU_DEVICE 0
int main(int argc, char** argv)
{
    GraphCliOptions opts;
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "pagerank", opts)) return 1;
    graph_print_config(opts, "pagerank", "tolerance_queue", false, false, true);

    cudaSetDevice(GPU_DEVICE);
    CsrGraph g;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), g, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    float* value = (float*)malloc(sizeof(float) * g.nodes);
    pagerankGPU(value, g.row_offsets, g.column_indices, g.column_offsets, g.row_indices,
                g.nodes, g.edges, opts.alpha, opts.beta, opts.threshold);

    FILE* f = fopen(graph_output_path("pagerank", "info_outcome.txt").c_str(), "w");
    if (f) {
        for (int i = 0; i < g.nodes; i++) fprintf(f, "%f\n", value[i]);
        fclose(f);
    }
    free(value);
    cudaDeviceReset();
    return 0;
}
