cmake_minimum_required(VERSION 3.18)
project(FlashAttention-Research LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# -----------------------------
# NCCL 手动路径
# -----------------------------
set(NCCL_ROOT "/home/huangxy/software/nccl")
set(NCCL_INCLUDE_DIR "${NCCL_ROOT}/include")
set(NCCL_LIBRARY "${NCCL_ROOT}/lib/libnccl.so")

# 检查 NCCL 是否存在
if (NOT EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    message(FATAL_ERROR "NCCL header not found: ${NCCL_INCLUDE_DIR}/nccl.h")
endif()

if (NOT EXISTS "${NCCL_LIBRARY}")
    message(FATAL_ERROR "NCCL library not found: ${NCCL_LIBRARY}")
endif()

message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
message(STATUS "NCCL library: ${NCCL_LIBRARY}")

# -----------------------------
# 可执行文件
# -----------------------------
add_executable(bfs bfs.cu)
add_executable(bfs_tolerance bfs_tolerance.cu)
add_executable(bfs_cub bfs_tolerance_cub.cu)
add_executable(bfs_threshold bfs_tolerance_threshold.cu)
add_executable(bfs_multiGPU bfs_tolerance_multiGPU.cu)

# -----------------------------
# CUDA 链接
# -----------------------------
target_link_libraries(bfs PRIVATE CUDA::cudart)
target_link_libraries(bfs_tolerance PRIVATE CUDA::cudart)
target_link_libraries(bfs_cub PRIVATE CUDA::cudart)
target_link_libraries(bfs_threshold PRIVATE CUDA::cudart)

# -----------------------------
# 多 GPU 版本额外链接 NCCL
# -----------------------------
target_include_directories(bfs_multiGPU PRIVATE ${NCCL_INCLUDE_DIR})

target_link_libraries(bfs_multiGPU
    PRIVATE
    CUDA::cudart
    CUDA::nvToolsExt
    ${NCCL_LIBRARY}
)

# 让运行时能找到 libnccl.so
set_target_properties(bfs_multiGPU PROPERTIES
    BUILD_RPATH "${NCCL_ROOT}/lib"
    INSTALL_RPATH "${NCCL_ROOT}/lib"
)