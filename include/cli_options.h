#ifndef GRAPH_CLI_OPTIONS_H
#define GRAPH_CLI_OPTIONS_H

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

struct GraphCliOptions {
    std::string dataset_id;
    std::string graph_path;
    float alpha = 0.5f;
    float beta = 0.5f;
    float threshold = 0.3f;
    int src = 0;
    int k = 5;
    bool run_cpu = true;
};

inline void graph_print_usage(const char* prog, const char* algorithm, bool rank0 = true) {
    if (!rank0) return;
    printf("Usage: %s <dataset_id> [-s source_node] [-k k] [-a alpha] [-b beta] [-t threshold] [-n] [-h]\n", prog);
    printf("  dataset_id: loads dataset/<dataset_id>.mtx\n");
    printf("  -s: source node, used by BFS; accepted by %s for command-line compatibility\n", algorithm);
    printf("  -k: k-core threshold, used by KCore; accepted by %s for command-line compatibility\n", algorithm);
    printf("  -a/-b/-t: tolerance scoring parameters; accepted by all variants for compatibility\n");
    printf("  -n: skip CPU correctness check when that executable supports CPU checking\n");
}

inline bool graph_parse_cli(
    int argc,
    char** argv,
    const char* algorithm,
    GraphCliOptions& opts,
    bool rank0 = true
) {
    if (argc < 2 || argv[1][0] == '-') {
        graph_print_usage(argv[0], algorithm, rank0);
        return false;
    }

    opts.dataset_id = argv[1];
    opts.graph_path = "dataset/" + opts.dataset_id + ".mtx";

    optind = 2;
    opterr = 0;

    int opt = 0;
    while ((opt = getopt(argc, argv, "s:k:a:b:t:nh")) != -1) {
        switch (opt) {
            case 's':
                opts.src = atoi(optarg);
                break;
            case 'k':
                opts.k = atoi(optarg);
                break;
            case 'a':
                opts.alpha = (float)atof(optarg);
                break;
            case 'b':
                opts.beta = (float)atof(optarg);
                break;
            case 't':
                opts.threshold = (float)atof(optarg);
                break;
            case 'n':
                opts.run_cpu = false;
                break;
            case 'h':
                graph_print_usage(argv[0], algorithm, rank0);
                exit(0);
            default:
                graph_print_usage(argv[0], algorithm, rank0);
                return false;
        }
    }

    return true;
}

inline void graph_print_config(
    const GraphCliOptions& opts,
    const char* algorithm,
    const char* variant,
    bool uses_src,
    bool uses_k,
    bool uses_tolerance,
    bool rank0 = true,
    int ranks = 1
) {
    if (!rank0) return;

    printf("加载数据集: %s\n", opts.graph_path.c_str());
    printf("参数配置: algorithm=%s, variant=%s", algorithm, variant);
    if (uses_src) printf(", src=%d", opts.src);
    else printf(", src=%d(ignored)", opts.src);
    if (uses_k) printf(", k=%d", opts.k);
    else printf(", k=%d(ignored)", opts.k);
    if (uses_tolerance) {
        printf(", alpha=%.2f, beta=%.2f, threshold=%.2f", opts.alpha, opts.beta, opts.threshold);
    } else {
        printf(", alpha=%.2f(ignored), beta=%.2f(ignored), threshold=%.2f(ignored)",
               opts.alpha, opts.beta, opts.threshold);
    }
    if (ranks > 1) printf(", ranks=%d", ranks);
    printf(", CPU_check=%s\n", opts.run_cpu ? "on" : "off");
}

#endif
