#include <cuda_runtime.h>

__global__ void warmup_kernel(float* x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] += 1.0f;
}

void gpu_warmup()
{
    const int N = 1 << 20;

    float* d_x;
    cudaMalloc(&d_x, N * sizeof(float));

    cudaMemset(d_x, 0, N * sizeof(float));

    // 多次启动 kernel 预热
    for (int i = 0; i < 10; i++)
    {
        warmup_kernel<<<256, 256>>>(d_x);
    }

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    cudaFree(d_x);
}