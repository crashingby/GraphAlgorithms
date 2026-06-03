#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "bfs_gpu_tolerance.cuh"
#include "include/graph.h"
#include "include/warmup.cuh"
#include "include/output.h"
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
int main(int argc, char* argv[]) // 修改此处以接收命令行参数
{
    // 1. 处理输入参数
    if (argc < 2) {
        printf("用法: %s <数据编号>\n", argv[0]);
        printf("示例: %s 13356\n", argv[0]);
        return 1;
    }

    // 动态拼接路径，例如输入 13356 得到 "dataset/13356.mtx"
    std::string input_id = argv[1];
    std::string graph_path = "dataset/" + input_id + ".mtx";
    
    const char* graph_file = graph_path.c_str();
    std::string outFileName = graph_output_path("bfs", "info_outcome.txt");
    const int src = 0;
    const bool run_CPU = true;

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
    FILE* f = fopen(outFileName.c_str(), "w");
    if (f) {
        for (int i = 0; i < csr_graph.nodes; i++)
            fprintf(f, "%d\n", value[i]);
        fclose(f);
    }

    // // 输出结果（按 value 值排序，输出前 100 个最小的顶点）
    // FILE* f = fopen(outFileName.c_str(), "w");
    // if (f) {
    //     // 将顶点和对应的 value 值组成 pair 数组
    //     std::vector<std::pair<int, int>> vertex_values;
    //     vertex_values.reserve(csr_graph.nodes);
    //     for (int i = 0; i < csr_graph.nodes; i++) {
    //         vertex_values.emplace_back(i, value[i]);
    //     }

    //     // 按 value 从小到大排序
    //     std::sort(vertex_values.begin(), vertex_values.end(),
    //             [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
    //                 return a.second < b.second;
    //             });

    //     // 输出前 100 个最小的顶点（如果不足 100 个就输出全部）
    //     int limit = std::min(100, (int)vertex_values.size());
    //     for (int i = 0; i < limit; i++) {
    //         fprintf(f, "%d %d\n", vertex_values[i].first, vertex_values[i].second);
    //     }

    //     fclose(f);
    // }


    free(value);
    cudaDeviceReset();
    return 0;
}
