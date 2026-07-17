add_graph_cuda_executable(bfs bfs basic/bfs.cu)
add_graph_cuda_executable(bfs_queue bfs tolerance_queue/bfs_tolerance_queue.cu)
add_graph_mpi_nccl_executable(bfs_multiGPU bfs multigpu/bfs_tolerance_multiGPU.cu)
add_graph_mpi_nccl_executable(bfs_multiGPU_basic bfs multigpu_basic/bfs_multiGPU_basic.cu)

target_link_libraries(bfs_queue PRIVATE CUDA::nvToolsExt)
target_link_libraries(bfs_multiGPU PRIVATE CUDA::nvToolsExt)
