#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "kcore_gpu.cuh"
#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"

#define GPU_DEVICE 0

//-----------------------------
// CPU KCORE 连通分量
//-----------------------------
void kcoreCPU(const CsrGraph &graph, int* value, int k)
{
    const int n = graph.nodes;
    // 初始化每个顶点的度
    std::vector<int> degree(n);
    std::vector<int8_t> alive(n, 1);
    for (int i = 0; i < n; i++) {
        degree[i] = graph.row_offsets[i + 1] - graph.row_offsets[i];
        value[i] = degree[i];
    }

    bool changed = true;
    int iter = 0;

    while (changed) {
        changed = false;
        iter++;

        // 遍历所有节点，删除度小于k的
        for (int u = 0; u < n; u++) {
            if (alive[u] && value[u] < k) {
                alive[u] = 0;
                changed = true;
                // 更新邻居的度
                for (int j = graph.row_offsets[u]; j < graph.row_offsets[u + 1]; j++) {
                    int v = graph.column_indices[j];
                    if (alive[v] && value[v] > 0) {
                        value[v]--;
                    }
                }
            }
        }
    }

}

//-----------------------------
// CPU/GPU 结果正确性检测
//-----------------------------
bool correctTest(int n, const int* ref, const int* gpu, int k)
{
    bool pass = true;
    int nerr = 0;
    for (int i = 0; i < n; i++) {
        if (ref[i] != gpu[i]) {
            if(ref[i] < k && gpu[i] < k) continue;
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
int main(int argc, char** argv)
{
    GraphCliOptions opts;
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "kcore", opts)) return 1;
    graph_print_config(opts, "kcore", "basic", false, true, false);

    std::string outFileName = graph_output_path("kcore", "info_outcome.txt");
    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), csr_graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * csr_graph.nodes);
    kcoreGPU(value,
             csr_graph.row_offsets,
             csr_graph.column_indices,
             csr_graph.column_offsets,
             csr_graph.row_indices,
             csr_graph.nodes,
             csr_graph.edges,
             opts.k);

    if (opts.run_cpu) {
        int* ref_value = (int*)malloc(sizeof(int) * csr_graph.nodes);
        kcoreCPU(csr_graph, ref_value, opts.k);
        correctTest(csr_graph.nodes, ref_value, value, opts.k);
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
