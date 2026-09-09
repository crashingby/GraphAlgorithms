cmake_minimum_required(VERSION 3.18)
project(GraphAlgorithm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(GRAPH_OUTPUT_ROOT "${CMAKE_SOURCE_DIR}/outputs" CACHE PATH "Runtime result output directory")

# -----------------------------
# NCCL manual path
# -----------------------------
set(NCCL_ROOT "/home/huangxy/software/nccl"
    CACHE PATH "NCCL installation root")

find_path(
    NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS
        "${NCCL_ROOT}/include"
    NO_DEFAULT_PATH
)

find_library(
    NCCL_LIBRARY
    NAMES nccl
    PATHS
        "${NCCL_ROOT}/lib"
        "${NCCL_ROOT}/lib64"
        "${NCCL_ROOT}/lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
)

if (NOT NCCL_INCLUDE_DIR)
    message(FATAL_ERROR
        "NCCL header not found under NCCL_ROOT=${NCCL_ROOT}")
endif()

if (NOT NCCL_LIBRARY)
    message(FATAL_ERROR
        "NCCL library not found under NCCL_ROOT=${NCCL_ROOT}")
endif()

message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
message(STATUS "NCCL library: ${NCCL_LIBRARY}")
message(STATUS "Runtime outputs: ${GRAPH_OUTPUT_ROOT}")

function(add_graph_cuda_executable target algorithm)
    add_executable(${target} ${ARGN})
    target_include_directories(${target}
        PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    target_compile_definitions(${target}
        PRIVATE
        GRAPH_OUTPUT_ROOT="${GRAPH_OUTPUT_ROOT}"
    )
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${algorithm}"
    )
endfunction()

function(add_graph_nccl_executable target algorithm)
    add_graph_cuda_executable(${target} ${algorithm} ${ARGN})
    target_include_directories(${target} PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${NCCL_LIBRARY})
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${NCCL_ROOT}/lib"
        INSTALL_RPATH "${NCCL_ROOT}/lib"
    )
endfunction()

function(add_graph_mpi_nccl_executable target algorithm)
    add_graph_nccl_executable(${target} ${algorithm} ${ARGN})
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
endfunction()

add_subdirectory(algorithms/bfs)
add_subdirectory(algorithms/cc)
add_subdirectory(algorithms/kcore)
add_subdirectory(algorithms/pagerank)
