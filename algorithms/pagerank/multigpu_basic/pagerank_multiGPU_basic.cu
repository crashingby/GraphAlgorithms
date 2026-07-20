/**
 * @file pagerank_multiGPU_basic.cu
 * @brief MPI entry point for distributed PageRank without checks.
 */
#include "pagerank_multiGPU_basic.cuh"

#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include "include/graph.h"
#include "include/output.h"
#include "include/cli_options.h"

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
    opts.run_cpu = false;
    if (!graph_parse_cli(argc, argv, "pagerank", opts, rank == 0)) {
        MPI_Finalize();
        return 1;
    }
    graph_print_config(opts, "pagerank", "multigpu_basic", false, false, false, rank == 0, world_size);

    CsrGraph graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), graph, undirected) != 0) {
        if (rank == 0) fprintf(stderr, "Failed to load graph.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    float* value = (float*)malloc(sizeof(float) * graph.nodes);
    pagerankMultiGPUBasic(value, graph.row_offsets, graph.column_indices,
                          graph.column_offsets, graph.row_indices, graph.nodes, graph.edges);

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
