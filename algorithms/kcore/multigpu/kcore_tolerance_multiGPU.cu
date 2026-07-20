/**
 * @file kcore_tolerance_multiGPU.cu
 * @brief MPI entry point and CPU oracle for checked distributed k-core.
 */
#include "kcore_tolerance_multiGPU.cuh"

#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"

/**
 * @brief Compute the directed peeling fixed point used by the GPU.
 *
 * Support is initialized from incoming CSR; removing a vertex decrements its
 * outgoing destinations. Values below @p k remain classification-equivalent.
 *
 * @param graph Host CSR graph.
 * @param[out] value Final residual degree for every vertex.
 * @param k Requested KCore threshold.
 */
void kcoreCPU(const CsrGraph& graph, int* value, int k) {
    const int n = graph.nodes;
    std::vector<int8_t> alive(n, 1);

    for (int i = 0; i < n; ++i) {
        value[i] =
            graph.column_offsets[i + 1] - graph.column_offsets[i];
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

/**
 * @brief Compare distributed GPU output with the serial KCore classification.
 *
 * Residual degree values below @p k are equivalent because both classify the
 * vertex as peeled; values on opposite sides of @p k are reported as errors.
 *
 * @return true when every vertex has an equivalent KCore classification.
 */
bool correctTest(int n, const int* ref, const int* gpu, int k) {
    bool pass = true;
    int nerr = 0;
    for (int i = 0; i < n; ++i) {
        if (ref[i] == gpu[i]) continue;
        if (ref[i] < k && gpu[i] < k) continue;
        if (nerr++ < 20) printf("Node %d: CPU %d, GPU %d\n", i, ref[i], gpu[i]);
        pass = false;
    }
    printf("CPU check: %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

/**
 * @brief MPI command-line entry point for fault-detecting multi-GPU KCore.
 *
 * Rank zero optionally validates and writes the globally assembled result;
 * all ranks participate in graph loading, KCore execution, and MPI teardown.
 */
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
    if (!graph_parse_cli(argc, argv, "kcore", opts, rank == 0)) {
        MPI_Finalize();
        return 1;
    }
    graph_print_config(opts, "kcore", "multigpu", false, true, true, rank == 0, world_size);

    CsrGraph graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), graph, undirected) != 0) {
        if (rank == 0) fprintf(stderr, "Failed to load graph.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * graph.nodes);
    if (!value) {
        if (rank == 0) fprintf(stderr, "malloc value failed.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    kcoreMultiGPU(value,
                  graph.row_offsets,
                  graph.column_indices,
                  graph.column_offsets,
                  graph.row_indices,
                  graph.nodes,
                  graph.edges,
                  opts.k,
                  opts.alpha,
                  opts.beta,
                  opts.threshold);

    if (opts.run_cpu && rank == 0) {
        int* ref = (int*)malloc(sizeof(int) * graph.nodes);
        kcoreCPU(graph, ref, opts.k);
        correctTest(graph.nodes, ref, value, opts.k);
        free(ref);
    }

    if (rank == 0) {
        FILE* f = fopen(graph_output_path("kcore", "info_outcome.txt").c_str(), "w");
        if (f) {
            for (int i = 0; i < graph.nodes; ++i) fprintf(f, "%d\n", value[i]);
            fclose(f);
        }
    }

    free(value);
    cudaDeviceReset();
    MPI_Finalize();
    return 0;
}
