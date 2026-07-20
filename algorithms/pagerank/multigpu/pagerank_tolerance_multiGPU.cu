/**
 * @file pagerank_tolerance_multiGPU.cu
 * @brief MPI entry point and CPU oracle for checked distributed PageRank.
 */
#include "pagerank_tolerance_multiGPU.cuh"

#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"

namespace {
constexpr float kPrAlpha = 0.85f;
constexpr float kPrTol = 1e-3f;
constexpr float kCheckAbsTol = 1e-2f;
constexpr float kCheckRelTol = 1e-4f;

/** @brief Compute the normalization denominator for criticality scoring. */
int computeMaxOutdegree(const CsrGraph& graph) {
    int max_outdegree = 1;
    for (int v = 0; v < graph.nodes; ++v) {
        int outdegree = graph.row_offsets[v + 1] - graph.row_offsets[v];
        if (outdegree > max_outdegree) max_outdegree = outdegree;
    }
    return max_outdegree;
}

/** @brief Apply the GPU criticality formula to the CPU reference work set. */
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
        float score = alpha * ((float)outdegree / (float)safe_max) + beta * std::fabs(value[v]);
        active[v] = (score >= threshold) ? 2 : 1;
    }
    return active_count;
}

/** @brief Run the serial reference using the same active-set PageRank formula. */
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

/**
 * @brief Compare serial and distributed in-place PageRank values.
 *
 * Rank scheduling changes the order of in-place floating-point updates. Use a
 * mixed absolute/relative bound so large scores are not rejected for a tiny
 * relative difference after both executions satisfy the same residual limit.
 */
bool correctTest(int n, const float* ref, const float* gpu) {
    bool pass = true;
    int nerr = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float err = std::fabs(ref[i] - gpu[i]);
        const float scale = std::fabs(ref[i]);
        const float allowed =
            kCheckAbsTol + kCheckRelTol * scale;
        const float rel_err = scale > 0.0f ? err / scale : err;
        if (err > max_abs_err) max_abs_err = err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (err > allowed) {
            if (nerr++ < 20) {
                printf(
                    "Node %d: CPU %.8f, GPU %.8f, abs_err %.8f, "
                    "allowed %.8f\n",
                    i, ref[i], gpu[i], err, allowed);
            }
            pass = false;
        }
    }
    printf(
        "CPU check: %s (max_abs_err=%.8f, max_rel_err=%.8f, "
        "abs_tol=%.8f, rel_tol=%.8f)\n",
        pass ? "PASSED" : "FAILED", max_abs_err, max_rel_err,
        kCheckAbsTol, kCheckRelTol);
    return pass;
}
} // namespace

/** @brief Parse CLI options and coordinate checked distributed PageRank. */
int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    GraphCliOptions opts;
    opts.run_cpu = true;
    if (!graph_parse_cli(argc, argv, "pagerank", opts, rank == 0)) {
        MPI_Finalize();
        return 1;
    }
    graph_print_config(opts, "pagerank", "multigpu", false, false, true, rank == 0, world_size);

    CsrGraph graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), graph, undirected) != 0) {
        if (rank == 0) fprintf(stderr, "Failed to load graph.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    float* value = (float*)malloc(sizeof(float) * graph.nodes);
    if (!value) {
        if (rank == 0) fprintf(stderr, "malloc value failed.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    pagerankMultiGPU(value,
                     graph.row_offsets,
                     graph.column_indices,
                     graph.column_offsets,
                     graph.row_indices,
                     graph.nodes,
                     graph.edges,
                     opts.alpha,
                     opts.beta,
                     opts.threshold);

    if (opts.run_cpu && rank == 0) {
        float* ref = (float*)malloc(sizeof(float) * graph.nodes);
        pagerankCPU(graph, ref, opts.alpha, opts.beta, opts.threshold);
        correctTest(graph.nodes, ref, value);
        free(ref);
    }

    if (rank == 0) {
        FILE* f = fopen(graph_output_path("pagerank", "info_outcome.txt").c_str(), "w");
        if (f) {
            for (int i = 0; i < graph.nodes; ++i) fprintf(f, "%f\n", value[i]);
            fclose(f);
        }
    }

    free(value);
    cudaDeviceReset();
    MPI_Finalize();
    return 0;
}
