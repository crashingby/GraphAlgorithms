#ifndef GRAPH_ALGORITHM_OUTPUT_H
#define GRAPH_ALGORITHM_OUTPUT_H

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>

#ifndef GRAPH_OUTPUT_ROOT
#define GRAPH_OUTPUT_ROOT "outputs"
#endif

inline void graph_ensure_dir(const std::string& path) {
    if (path.empty()) return;
    mkdir(path.c_str(), 0755);
}

inline std::string graph_output_path(const char* algorithm, const char* filename) {
    std::string root = GRAPH_OUTPUT_ROOT;
    std::string dir = root + "/" + algorithm;
    graph_ensure_dir(root);
    graph_ensure_dir(dir);
    return dir + "/" + filename;
}

#endif  // GRAPH_ALGORITHM_OUTPUT_H
