#include "kcore_tolerance_multiGPU.cuh"

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "include/graph.h"

void usage(const char* prog) {
    printf("Usage: %s <dataset_id> [-k k] [-a alpha] [-b beta] [-t threshold] [-n]\n", prog);
    printf("  -n  skip CPU correctness check\n");
}

void kcoreCPU(const CsrGraph& graph, int* value, int k) {
    const int n = graph.nodes;
    std::vector<int8_t> alive(n, 1);

    for (int i = 0; i < n; ++i) {
        value[i] = graph.row_offsets[i + 1] - graph.row_offsets[i];
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (int u = 0; u < n; ++u) {
            if (!alive[u] || value[u] >= k) continue;

            alive[u] = 0;
            changed = true;
            for (int e = graph.row_offsets[u]; e < graph.row_offsets[u + 1]; ++e) {
                int v = graph.column_indices[e];
                if (alive[v] && value[v] > 0) --value[v];
            }
        }
    }
}

bool correctTest(int n, const int* ref, const int* gpu, int k) {
    bool pass = true;
    int nerr = 0;
    for (int i = 0; i < n; ++i) {
        if (ref[i] == gpu[i]) continue;
        if (ref[i] < k && gpu[i] < k) continue;

        if (nerr++ < 20) {
            printf("Node %d: CPU %d, GPU %d\n", i, ref[i], gpu[i]);
        }
        pass = false;
    }
    printf("CPU check: %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

int main(int argc, char** argv) {
    float alpha = 0.5f;
    float beta = 0.5f;
    float threshold = 0.3f;
    int k = 5;
    bool run_cpu = true;

    if (argc < 2 || argv[1][0] == '-') {
        usage(argv[0]);
        return 1;
    }

    std::string graph_path = "dataset/" + std::string(argv[1]) + ".mtx";
    optind = 2;

    int opt;
    while ((opt = getopt(argc, argv, "k:a:b:t:nh")) != -1) {
        if (opt == 'k') k = atoi(optarg);
        else if (opt == 'a') alpha = atof(optarg);
        else if (opt == 'b') beta = atof(optarg);
        else if (opt == 't') threshold = atof(optarg);
        else if (opt == 'n') run_cpu = false;
        else {
            usage(argv[0]);
            return opt == 'h' ? 0 : 1;
        }
    }

    printf("加载数据集: %s\n", graph_path.c_str());
    printf("参数配置: k=%d alpha=%.2f beta=%.2f threshold=%.2f CPU_check=%s\n",
           k, alpha, beta, threshold, run_cpu ? "on" : "off");

    CsrGraph graph;
    bool undirected = false;
    if (BuildMarketGraph(graph_path.c_str(), graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * graph.nodes);
    kcoreMultiGPU(value,
                  graph.row_offsets,
                  graph.column_indices,
                  graph.column_offsets,
                  graph.row_indices,
                  graph.nodes,
                  graph.edges,
                  k,
                  alpha,
                  beta,
                  threshold);

    if (run_cpu) {
        int* ref = (int*)malloc(sizeof(int) * graph.nodes);
        kcoreCPU(graph, ref, k);
        correctTest(graph.nodes, ref, value, k);
        free(ref);
    }

    FILE* f = fopen("info_outcome.txt", "w");
    if (f) {
        for (int i = 0; i < graph.nodes; ++i) fprintf(f, "%d\n", value[i]);
        fclose(f);
    }

    free(value);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    for (int d = 0; d < device_count; ++d) {
        cudaSetDevice(d);
        cudaDeviceReset();
    }
    return 0;
}
