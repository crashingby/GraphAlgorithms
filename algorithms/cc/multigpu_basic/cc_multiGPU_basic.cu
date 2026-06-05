#include "cc_multiGPU_basic.cuh"

#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include "include/graph.h"
#include "include/output.h"

void usage(const char* prog) {
    printf("Usage: %s <dataset_id> [-n]\n", prog);
    printf("  -n  skip CPU correctness check\n");
}

void ccCPU(const CsrGraph& graph, int* value) {
    const int n = graph.nodes;
    for (int i = 0; i < n; ++i) value[i] = i;

    bool changed = true;
    while (changed) {
        changed = false;
        for (int u = 0; u < n; ++u) {
            int old = value[u];
            for (int e = graph.row_offsets[u]; e < graph.row_offsets[u + 1]; ++e) {
                int v = graph.column_indices[e];
                if (value[v] > value[u]) value[u] = value[v];
            }
            if (value[u] != old) changed = true;
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
    printf("CPU check: %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    bool run_cpu = true;
    if (argc < 2 || argv[1][0] == '-') {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    std::string graph_path = "dataset/" + std::string(argv[1]) + ".mtx";
    optind = 2;
    int opt = 0;
    while ((opt = getopt(argc, argv, "nh")) != -1) {
        if (opt == 'n') run_cpu = false;
        else {
            if (rank == 0) usage(argv[0]);
            MPI_Finalize();
            return opt == 'h' ? 0 : 1;
        }
    }

    if (rank == 0) {
        printf("加载数据集: %s\n", graph_path.c_str());
        printf("参数配置: ranks=%d, CPU_check=%s\n", world_size, run_cpu ? "on" : "off");
    }

    CsrGraph graph;
    bool undirected = false;
    if (BuildMarketGraph(graph_path.c_str(), graph, undirected) != 0) {
        if (rank == 0) fprintf(stderr, "Failed to load graph.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    int* value = (int*)malloc(sizeof(int) * graph.nodes);
    ccMultiGPUBasic(
        value,
        graph.row_offsets,
        graph.column_indices,
        graph.column_offsets,
        graph.row_indices,
        graph.nodes,
        graph.edges
    );

    if (run_cpu && rank == 0) {
        int* ref = (int*)malloc(sizeof(int) * graph.nodes);
        ccCPU(graph, ref);
        correctTest(graph.nodes, ref, value);
        free(ref);
    }

    if (rank == 0) {
        FILE* f = fopen(graph_output_path("cc", "info_outcome.txt").c_str(), "w");
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
