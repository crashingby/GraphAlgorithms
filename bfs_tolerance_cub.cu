#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "bfs_gpu_tolerance_cub.cuh"
#include "include/graph.h"
#include"include/warmup.cuh"
#define INF 100000
#define GPU_DEVICE 0

//-----------------------------
// CPU BFS 单源最短路径（无权图）
//-----------------------------
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
int main()
{
    const char graph_file[] = "dataset/13356.mtx";
    const char outFileName[] = "info_outcome.txt";
    const int src = 0;
    const bool run_CPU = false; 

    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(graph_file, csr_graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * csr_graph.nodes);
    gpu_warmup();
    // GPU BFS
    bfsGPU(value,
            csr_graph.row_offsets,
            csr_graph.column_indices,
            csr_graph.column_offsets,
            csr_graph.row_indices,
            csr_graph.nodes,
            csr_graph.edges,
            src); 

    // CPU 检查
    if (run_CPU) {
        int* ref_value = (int*)malloc(sizeof(int) * csr_graph.nodes);
        bfsCPU(csr_graph, ref_value, src);
        correctTest(csr_graph.nodes, ref_value, value);
        free(ref_value);
    }

    // 输出结果
    FILE* f = fopen(outFileName, "w");
    if (f) {
        for (int i = 0; i < csr_graph.nodes; i++)
            fprintf(f, "%d\n", value[i]);
        fclose(f);
    }

    free(value);
    cudaDeviceReset();
    return 0;
}