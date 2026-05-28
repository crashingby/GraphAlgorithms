#include "pagerank_tolerance_multiGPU.cuh"

#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "include/graph.h"

namespace {
constexpr float kPrAlpha = 0.85f;
constexpr float kPrTol = 1e-3f;
constexpr float kCheckTol = 1e-2f;

void usage(const char* prog) {
    printf("Usage: %s <dataset_id> [-a alpha] [-b beta] [-t threshold] [-n]\n", prog);
    printf("  -n  skip CPU correctness check\n");
}

int computeMaxOutdegree(const CsrGraph& graph) {
    int max_outdegree = 1;
    for (int v = 0; v < graph.nodes; ++v) {
        int outdegree = graph.row_offsets[v + 1] - graph.row_offsets[v];
        if (outdegree > max_outdegree) max_outdegree = outdegree;
    }
    return max_outdegree;
}

int scoreAndMarkCPU(std::vector<int>& active,
                    const std::vector<float>& value,
                    const CsrGraph& graph,
                    int max_outdegree,
                    float alpha,
                    float beta,
                    float threshold) {
    int active_count = 0;
    int safe_max = max_outdegree > 0 ? max_outdegree : 1;

    for (int v = 0; v < graph.nodes; ++v) {
        if (active[v] == -1) continue;

        ++active_count;
        int outdegree = graph.row_offsets[v + 1] - graph.row_offsets[v];
        float value_score = std::fabs(value[v]);
        float score = alpha * ((float)outdegree / (float)safe_max) + beta * value_score;
        active[v] = (score >= threshold) ? 2 : 1;
    }
    return active_count;
}

void pagerankCPU(const CsrGraph& graph,
                 float* out,
                 float alpha,
                 float beta,
                 float threshold) {
    const int n = graph.nodes;
    std::vector<float> value(n, 1.0f / (float)n);
    std::vector<int> active(n, 1);
    std::vector<int> update(n, -1);

    int max_outdegree = computeMaxOutdegree(graph);
    int active_count = scoreAndMarkCPU(active, value, graph, max_outdegree, alpha, beta, threshold);

    for (int iter = 0; iter < 1000 && active_count > 0; ++iter) {
        std::fill(update.begin(), update.end(), -1);

        for (int v = 0; v < n; ++v) {
            if (active[v] == -1) continue;

            float sum = 0.0f;
            for (int e = graph.column_offsets[v]; e < graph.column_offsets[v + 1]; ++e) {
                int src = graph.row_indices[e];
                int outdegree = graph.row_offsets[src + 1] - graph.row_offsets[src];
                if (outdegree > 0) sum += value[src] / (float)outdegree;
            }

            float new_value = (1.0f - kPrAlpha) + kPrAlpha * sum;
            float delta = std::fabs(new_value - value[v]);
            value[v] = new_value;

            if (delta >= kPrTol) {
                for (int e = graph.row_offsets[v]; e < graph.row_offsets[v + 1]; ++e) {
                    update[graph.column_indices[e]] = 1;
                }
            }
        }

        active.swap(update);
        active_count = scoreAndMarkCPU(active, value, graph, max_outdegree, alpha, beta, threshold);
    }

    for (int i = 0; i < n; ++i) out[i] = value[i];
}

bool correctTest(int n, const float* ref, const float* gpu) {
    bool pass = true;
    int nerr = 0;
    float max_abs_err = 0.0f;

    for (int i = 0; i < n; ++i) {
        float err = std::fabs(ref[i] - gpu[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (err > kCheckTol) {
            if (nerr++ < 20) {
                printf("Node %d: CPU %.8f, GPU %.8f, abs_err %.8f\n", i, ref[i], gpu[i], err);
            }
            pass = false;
        }
    }

    printf("CPU check: %s (max_abs_err=%.8f, tol=%.8f)\n",
           pass ? "PASSED" : "FAILED",
           max_abs_err,
           kCheckTol);
    return pass;
}
} // namespace

int main(int argc, char** argv) {
    float alpha = 0.5f;
    float beta = 0.5f;
    float threshold = 0.3f;
    bool run_cpu = true;

    if (argc < 2 || argv[1][0] == '-') {
        usage(argv[0]);
        return 1;
    }

    std::string graph_path = "dataset/" + std::string(argv[1]) + ".mtx";
    optind = 2;

    int opt;
    while ((opt = getopt(argc, argv, "a:b:t:nh")) != -1) {
        if (opt == 'a') alpha = atof(optarg);
        else if (opt == 'b') beta = atof(optarg);
        else if (opt == 't') threshold = atof(optarg);
        else if (opt == 'n') run_cpu = false;
        else {
            usage(argv[0]);
            return opt == 'h' ? 0 : 1;
        }
    }

    printf("加载数据集: %s\n", graph_path.c_str());
    printf("参数配置: alpha=%.2f beta=%.2f threshold=%.2f CPU_check=%s\n",
           alpha, beta, threshold, run_cpu ? "on" : "off");

    CsrGraph graph;
    bool undirected = false;
    if (BuildMarketGraph(graph_path.c_str(), graph, undirected) != 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    float* value = (float*)malloc(sizeof(float) * graph.nodes);
    pagerankMultiGPU(value,
                     graph.row_offsets,
                     graph.column_indices,
                     graph.column_offsets,
                     graph.row_indices,
                     graph.nodes,
                     graph.edges,
                     alpha,
                     beta,
                     threshold);

    if (run_cpu) {
        float* ref = (float*)malloc(sizeof(float) * graph.nodes);
        pagerankCPU(graph, ref, alpha, beta, threshold);
        correctTest(graph.nodes, ref, value);
        free(ref);
    }

    FILE* f = fopen("info_outcome.txt", "w");
    if (f) {
        for (int i = 0; i < graph.nodes; ++i) fprintf(f, "%f\n", value[i]);
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
