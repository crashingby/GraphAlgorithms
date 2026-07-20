/**
 * @file bfs_multiGPU_basic.cu
 * @brief MPI entry point and optional CPU oracle for distributed BFS.
 */
#include "bfs_multiGPU_basic.cuh"

#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"

#define INF 100000

void bfsCPU(const CsrGraph& graph, int* dist, int src) {
    const int n = graph.nodes;
    for (int i = 0; i < n; ++i) dist[i] = INF;

    std::vector<int> queue;
    dist[src] = 0;
    queue.push_back(src);

    for (size_t q = 0; q < queue.size(); ++q) {
        int u = queue[q];
        for (int e = graph.row_offsets[u]; e < graph.row_offsets[u + 1]; ++e) {
            int v = graph.column_indices[e];
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
}

bool correctTest(int n, const int* ref, const int* gpu) {
    bool pass = true;
    int nerr = 0;
    for (int i = 0; i < n; ++i) {
        if (ref[i] != gpu[i]) {
            if (nerr++ < 20) printf("Node %d: CPU %d, GPU %d\n", i, ref[i], gpu[i]);
            pass = false;
        }
    }
    printf("%s\n", pass ? "passed" : "failed");
    return pass;
}

/**
 * @brief Initialize the same FUNNELED MPI contract used by the checked build.
 *
 * Only the main thread calls MPI in either variant. Requesting the same thread
 * level removes MPI initialization mode as an experimental confounder.
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
            fprintf(
                stderr,
                "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    GraphCliOptions opts;
    opts.run_cpu = true;
    if (!graph_parse_cli(argc, argv, "bfs", opts, rank == 0)) {
        MPI_Finalize();
        return 1;
    }
    graph_print_config(opts, "bfs", "multigpu_basic", true, false, false, rank == 0, world_size);

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

    bfsMultiGPUBasic(value, graph.row_offsets, graph.column_indices,
                     graph.column_offsets, graph.row_indices,
                     graph.nodes, graph.edges, opts.src);

    if (opts.run_cpu && rank == 0) {
        int* ref = (int*)malloc(sizeof(int) * graph.nodes);
        bfsCPU(graph, ref, opts.src);
        correctTest(graph.nodes, ref, value);
        free(ref);
    }

    if (rank == 0) {
        FILE* f = fopen(graph_output_path("bfs", "info_outcome.txt").c_str(), "w");
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
