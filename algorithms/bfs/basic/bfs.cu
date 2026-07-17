/**
 * @file bfs.cu
 * @brief CLI entry point and optional CPU oracle for single-GPU BFS.
 */
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "bfs_gpu.cuh"
#include "include/graph.h"
#include "include/warmup.cuh"
#include "include/output.h"
#include "include/cli_options.h"
#define INF 100000
#define GPU_DEVICE 0

/**
 * @brief Compute an exact CPU BFS result for optional validation.
 * @param graph Input graph in CSR form.
 * @param dist Output distance array.
 * @param src Source vertex.
 */
void bfsCPU(const CsrGraph &graph, int* dist, int src)
{
    const int n = graph.nodes;
    for (int i = 0; i < n; i++) dist[i] = INF;

    std::vector<int> queue;

    dist[src] = 0;
    queue.push_back(src);

    for (size_t q = 0; q < queue.size(); q++) {
        int u = queue[q];
        for (int j = graph.row_offsets[u]; j < graph.row_offsets[u + 1]; ++j) {
            int v = graph.column_indices[j];
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
}

//-----------------------------
// CPU/GPU 结果正确性检测
//-----------------------------
/** @brief Compare CPU and GPU distance arrays element by element. */
bool correctTest(int n, const int* ref, const int* gpu)
{
    bool pass = true;
    int nerr = 0;
    for (int i = 0; i < n; i++) {
        if (ref[i] != gpu[i]) {
            if (nerr++ < 20)
                printf("Node %d: CPU %d, GPU %d\n", i, ref[i], gpu[i]);
            pass = false;
        }
    }
    printf("%s\n", pass ? "passed" : "failed");
    return pass;
}



//-----------------------------
// 主函数
//-----------------------------
int main(int argc, char* argv[])
{
    GraphCliOptions opts;
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "bfs", opts)) return 1;
    graph_print_config(opts, "bfs", "basic", true, false, false);

    std::string outFileName = graph_output_path("bfs", "info_outcome.txt");
    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), csr_graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * csr_graph.nodes);

    gpu_warmup();
    bfsGPU(value,
           csr_graph.row_offsets,
           csr_graph.column_indices,
           csr_graph.column_offsets,
           csr_graph.row_indices,
           csr_graph.nodes,
           csr_graph.edges,
           opts.src);

    if (opts.run_cpu) {
        int* ref_value = (int*)malloc(sizeof(int) * csr_graph.nodes);
        bfsCPU(csr_graph, ref_value, opts.src);
        correctTest(csr_graph.nodes, ref_value, value);
        free(ref_value);
    }

    FILE* f = fopen(outFileName.c_str(), "w");
    if (f) {
        for (int i = 0; i < csr_graph.nodes; i++) fprintf(f, "%d\n", value[i]);
        fclose(f);
    }

    free(value);
    cudaDeviceReset();
    return 0;
}
