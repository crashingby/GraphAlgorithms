#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h> // 用于 getopt
#include "bfs_gpu_tolerance_threshold.cuh"
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


void print_usage(const char* prog_name) {
    printf("Usage: %s [-a alpha] [-b beta] [-t threshold] [-g graph_file] [-s source_node]\n", prog_name);
    printf("Defaults: alpha=0.5, beta=0.5, threshold=0.3, graph=dataset/13356.mtx, src=0\n");
}

int main(int argc, char **argv)
{
    // --- 默认参数设置 ---
    float alpha = 0.5f;
    float beta = 0.5f;
    float threshold = 0.3f;
    int src = 0;
    std::string outFileName = graph_output_path("bfs", "info_outcome.txt");
    const bool run_CPU = true;
    
    // 用于存放拼接后的路径
    std::string graph_path; 

    // --- 1. 处理位置参数 (数据集编号) ---
    // 我们约定：第一个非选项参数必须是数据集编号
    // 例如：./program 13356 -a 0.8
    if (argc < 2 || argv[1][0] == '-') {
        fprintf(stderr, "错误: 必须在开头提供数据集编号!\n");
        printf("用法: %s <数据集编号> [-a alpha] [-b beta] [-t threshold] [-s src]\n", argv[0]);
        printf("示例: %s 13356 -a 0.7 -s 1\n", argv[0]);
        return 1;
    }

    // 自动拼接路径
    graph_path = "dataset/" + std::string(argv[1]) + ".mtx";
    const char* graph_file = graph_path.c_str();

    // --- 2. 命令行选项解析 ---
    // 注意：我们将 optind 设置为 2，跳过已经处理的数据集编号参数
    int opt;
    optind = 2; 
    while ((opt = getopt(argc, argv, "a:b:t:s:h")) != -1) { // 删掉了 g:，因为改为自动拼接
        switch (opt) {
            case 'a': alpha = atof(optarg); break;
            case 'b': beta = atof(optarg); break;
            case 't': threshold = atof(optarg); break;
            case 's': src = atoi(optarg); break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    // 打印参数确认信息
    printf("加载数据集: %s\n", graph_file);
    printf("参数配置: alpha=%.2f, beta=%.2f, threshold=%.2f, src=%d\n", alpha, beta, threshold, src);

    cudaSetDevice(GPU_DEVICE);

    // --- 图数据加载 ---
    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(graph_file, csr_graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph: %s\n", graph_file);
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * csr_graph.nodes);

    gpu_warmup();

    // --- GPU BFS 调用 (传入解析后的阈值参数) ---
    bfsGPU(value,
           csr_graph.row_offsets,
           csr_graph.column_indices,
           csr_graph.column_offsets,
           csr_graph.row_indices,
           csr_graph.nodes,
           csr_graph.edges,
           src,
           alpha,
           beta,
           threshold); 

    // --- CPU 检查与验证 ---
    if (run_CPU) {
        int* ref_value = (int*)malloc(sizeof(int) * csr_graph.nodes);
        bfsCPU(csr_graph, ref_value, src);
        correctTest(csr_graph.nodes, ref_value, value);
        free(ref_value);
    }

    // --- 结果持久化 ---
    FILE* f = fopen(outFileName.c_str(), "w");
    if (f) {
        for (int i = 0; i < csr_graph.nodes; i++)
            fprintf(f, "%d\n", value[i]);
        fclose(f);
    }

    free(value);
    cudaDeviceReset();
    return 0;
}