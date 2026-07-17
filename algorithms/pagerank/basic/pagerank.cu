/**
 * @file pagerank.cu
 * @brief CLI entry point for the single-GPU PageRank baseline.
 */
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "pagerank_gpu.cuh"
#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"

#define INF 100000
#define GPU_DEVICE 0

//-----------------------------
// CPU PageRank
//-----------------------------
/**
 * @brief Placeholder for a CPU PageRank oracle.
 * @note The current project intentionally leaves this reference empty.
 */
void pagerankCPU(const CsrGraph &graph, float* value)
{
  
}

//-----------------------------
// CPU/GPU 结果正确性检测
//-----------------------------
/** @brief Compare CPU and GPU rank arrays element by element. */
bool correctTest(int n, const float* ref, const float* gpu)
{
    bool pass = true;
    int nerr = 0;
    for (int i = 0; i < n; i++) {
        if (ref[i] != gpu[i]) {
            if (nerr++ < 20)
                printf("Node %d: CPU %f, GPU %f\n", i, ref[i], gpu[i]);
            pass = false;
        }
    }
    printf("%s\n", pass ? "passed" : "failed");
    return pass;
}


//-----------------------------
// 主函数
//-----------------------------
int main(int argc, char** argv)
{
    GraphCliOptions opts;
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "pagerank", opts)) return 1;
    graph_print_config(opts, "pagerank", "basic", false, false, false);

    std::string outFileName = graph_output_path("pagerank", "info_outcome.txt");
    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), csr_graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    float* value = (float*)malloc(sizeof(float) * csr_graph.nodes);
    pagerankGPU(value,
                csr_graph.row_offsets,
                csr_graph.column_indices,
                csr_graph.column_offsets,
                csr_graph.row_indices,
                csr_graph.nodes,
                csr_graph.edges);

    if (opts.run_cpu) {
        float* ref_value = (float*)malloc(sizeof(float) * csr_graph.nodes);
        pagerankCPU(csr_graph, ref_value);
        correctTest(csr_graph.nodes, ref_value, value);
        free(ref_value);
    }

    FILE* f = fopen(outFileName.c_str(), "w");
    if (f) {
        for (int i = 0; i < csr_graph.nodes; i++) fprintf(f, "%f\n", value[i]);
        fclose(f);
    }

    free(value);
    cudaDeviceReset();
    return 0;
}
