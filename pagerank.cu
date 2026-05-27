#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "pagerank_gpu.cuh"
#include "include/graph.h"

#define INF 100000
#define GPU_DEVICE 0

//-----------------------------
// CPU PageRank
//-----------------------------
void pagerankCPU(const CsrGraph &graph, float* value)
{
  
}

//-----------------------------
// CPU/GPU 结果正确性检测
//-----------------------------
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
int main()
{
    const char graph_file[] = "dataset/13356.mtx";
    const char outFileName[] = "info_outcome.txt";
    const bool run_CPU = false;

    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(graph_file, csr_graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    float* value = (float*)malloc(sizeof(float) * csr_graph.nodes);

    // GPU
    pagerankGPU(value,
            csr_graph.row_offsets,
            csr_graph.column_indices,
            csr_graph.column_offsets,
            csr_graph.row_indices,
            csr_graph.nodes,
            csr_graph.edges); 
  

    // CPU 检查
    if (run_CPU) {
        float* ref_value = (float*)malloc(sizeof(float) * csr_graph.nodes);
        pagerankCPU(csr_graph, ref_value);
        correctTest(csr_graph.nodes, ref_value, value);
        free(ref_value);
    }
    // 输出结果
    FILE* f = fopen(outFileName, "w");
    if (f) {
        for (int i = 0; i < csr_graph.nodes; i++)
            fprintf(f, "%f\n", value[i]);
        fclose(f);
    }

    // // 输出结果（按 value 值排序，输出前 20 个最大的顶点）
    // FILE* f = fopen(outFileName, "w");
    // if (f) {
    //     // 将顶点和对应的 value 值组成 pair 数组
    //     std::vector<std::pair<int, float>> vertex_values;
    //     vertex_values.reserve(csr_graph.nodes);
    //     for (int i = 0; i < csr_graph.nodes; i++) {
    //         vertex_values.emplace_back(i, value[i]);
    //     }

    //     // 按 value 从大到小排序
    //     std::sort(vertex_values.begin(), vertex_values.end(),
    //             [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
    //                 return a.second > b.second;  // 改成降序
    //             });

    //     // 输出前 20 个最大的顶点（如果不足 20 个就输出全部）
    //     int limit = std::min(20, (int)vertex_values.size());
    //     for (int i = 0; i < limit; i++) {
    //         fprintf(f, "%d\n", vertex_values[i].first);
    //     }

    //     fclose(f);
    // }



    free(value);
    cudaDeviceReset();
    return 0;
}
