add_graph_cuda_executable(cc cc basic/cc.cu)
add_graph_cuda_executable(cc_queue cc tolerance_queue/cc_tolerance_queue.cu)
add_graph_mpi_nccl_executable(cc_multiGPU cc multigpu/cc_tolerance_multiGPU.cu)
add_graph_mpi_nccl_executable(cc_multiGPU_basic cc multigpu_basic/cc_multiGPU_basic.cu)

target_link_libraries(cc_queue PRIVATE CUDA::nvtx3)
target_link_libraries(cc_multiGPU PRIVATE CUDA::nvtx3)
