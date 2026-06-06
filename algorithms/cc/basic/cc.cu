#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "cc_gpu.cuh"
#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"

#define GPU_DEVICE 0

//-----------------------------
// CPU CC 连通分量
//-----------------------------
void ccCPU(const CsrGraph &graph, int* value)
{
    const int n = graph.nodes;

    // 初始化：每个顶点的value为自身ID
    for (int i = 0; i < n; i++)
        value[i] = i;

    bool changed = true;
    while (changed)
    {
        changed = false;

        // 遍历每个顶点
        for (int u = 0; u < n; u++)
        {
            int old_val = value[u];

            // 遍历邻居
            for (int j = graph.row_offsets[u]; j < graph.row_offsets[u + 1]; ++j)
            {
                int v = graph.column_indices[j];

                // 顶点取邻居的最大值
                if (value[v] > value[u])
                    value[u] = value[v];
            }

            if (value[u] != old_val)
                changed = true;  // 只要有更新就继续迭代
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
int main(int argc, char** argv)
{
    GraphCliOptions opts;
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "cc", opts)) return 1;
    graph_print_config(opts, "cc", "basic", false, false, false);

    std::string outFileName = graph_output_path("cc", "info_outcome.txt");
    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), csr_graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * csr_graph.nodes);
    ccGPU(value,
          csr_graph.row_offsets,
          csr_graph.column_indices,
          csr_graph.column_offsets,
          csr_graph.row_indices,
          csr_graph.nodes,
          csr_graph.edges);

    if (opts.run_cpu) {
        int* ref_value = (int*)malloc(sizeof(int) * csr_graph.nodes);
        ccCPU(csr_graph, ref_value);
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
