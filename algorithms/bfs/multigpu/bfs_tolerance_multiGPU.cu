/**
 * @file bfs_tolerance_multiGPU.cu
 * @brief MPI entry point and optional CPU oracle for checked distributed BFS.
 */
#include "bfs_tolerance_multiGPU.cuh"
#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include<unistd.h>
#include "include/graph.h"
#include "include/warmup.cuh"
#include "include/output.h"
#include "include/cli_options.h"

// 如果你把 bfsMultiGPU 放在 .cuh 里，就 include 对应头文件。
// 如果暂时没有单独拆头文件，也可以直接 include 这个 .cu 做测试。


#define INF 100000

/**
 * @brief Compute an exact CPU BFS result for optional validation.
 * @param graph Input graph in outgoing CSR form.
 * @param dist Output distance array.
 * @param src Source vertex.
 */
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

/** @brief Compare CPU and distributed GPU distances element by element. */
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

/** @brief Initialize funneled MPI and run checked distributed BFS. */
int main(int argc, char **argv)
{
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    GraphCliOptions opts;
    opts.run_cpu = true;
    if (!graph_parse_cli(argc, argv, "bfs", opts, world_rank == 0)) {
        MPI_Finalize();
        return 1;
    }
    graph_print_config(opts, "bfs", "multigpu", true, false, true, world_rank == 0, world_size);

    CsrGraph csr_graph;
    bool undirected = false;
    if (BuildMarketGraph(opts.graph_path.c_str(), csr_graph, undirected) != 0) {
        if (world_rank == 0) fprintf(stderr, "Failed to load graph.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * csr_graph.nodes);
    if (!value) {
        if (world_rank == 0) fprintf(stderr, "malloc value failed.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    bfsMultiGPU(value,
                csr_graph.row_offsets,
                csr_graph.column_indices,
                csr_graph.column_offsets,
                csr_graph.row_indices,
                csr_graph.nodes,
                csr_graph.edges,
                opts.src,
                opts.alpha,
                opts.beta,
                opts.threshold);

    if (opts.run_cpu && world_rank == 0) {
        int* ref_value = (int*)malloc(sizeof(int) * csr_graph.nodes);
        if (!ref_value) {
            fprintf(stderr, "malloc ref_value failed.\n");
            free(value);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }
        bfsCPU(csr_graph, ref_value, opts.src);
        correctTest(csr_graph.nodes, ref_value, value);
        free(ref_value);
    }

    if (world_rank == 0) {
        FILE* f = fopen(graph_output_path("bfs", "info_outcome.txt").c_str(), "w");
        if (f) {
            for (int i = 0; i < csr_graph.nodes; i++) fprintf(f, "%d\n", value[i]);
            fclose(f);
        }
    }

    free(value);
    cudaDeviceReset();
    MPI_Finalize();
    return 0;
}
