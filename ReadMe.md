# GraphAlgorithm 运行说明

本项目按算法划分可执行文件目录：

```text
build/bin/bfs/
build/bin/cc/
build/bin/kcore/
build/bin/pagerank/
```

当前主要维护 4 类方法：

| 方法 | 含义 | 启动方式 |
|---|---|---|
| `basic` | 单节点单 GPU，无容错 | 直接运行 |
| `tolerance_queue` | 单节点单 GPU，队列异步容错检测 | 直接运行 |
| `multigpu_basic` | 多节点/多 GPU，无容错 | `mpirun` 启动 |
| `multigpu` | 多节点/多 GPU，容错检测 | `mpirun` 启动 |

数据集参数只需要传名字，例如 `flickr` 会自动读取：

```text
dataset/flickr.mtx
```

## 构建
**默认系统路径下已有MPI**
```bash
cmake \
    -DNCCL_ROOT=xxx \
    -S . \
    -B build
cmake --build build -j
```

只编译常用 16 个目标：

```bash
cmake --build build --target \
  bfs bfs_queue bfs_multiGPU bfs_multiGPU_basic \
  cc cc_queue cc_multiGPU cc_multiGPU_basic \
  kcore kcore_queue kcore_multiGPU kcore_multiGPU_basic \
  pagerank pagerank_queue pagerank_multiGPU pagerank_multiGPU_basic \
  -j
```

## 通用参数

所有主要入口统一接受下面这些参数：

```text
<dataset_id> [-s src] [-k k] [-a alpha] [-b beta] [-t threshold] [-n] [-h]
```

| 参数 | 含义 |
|---|---|
| `<dataset_id>` | 数据集名，例如 `flickr`，对应 `dataset/flickr.mtx` |
| `-s src` | BFS 源点；其它算法会接受但忽略 |
| `-k k` | KCore 的 k 值；其它算法会接受但忽略 |
| `-a alpha` | 容错阈值打分参数 |
| `-b beta` | 容错阈值打分参数 |
| `-t threshold` | 容错阈值 |
| `-n` | 关闭 CPU correctness check |
| `-h` | 打印帮助 |

## BFS

### 单 GPU 无容错

```bash
./build/bin/bfs/bfs flickr -s 9006 -n
```

### 单 GPU 队列容错

```bash
./build/bin/bfs/bfs_queue flickr -s 9006 -a 0.5 -b 0.5 -t 0.3 -n
```

### 多节点/多 GPU 无容错

本机 2 张 GPU：

```bash
mpirun -np 2 \
  ./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006 -n
```

### 多节点/多 GPU 容错

本机 2 张 GPU：

```bash
mpirun -np 2 \
  ./build/bin/bfs/bfs_multiGPU flickr -s 9006 -a 0.5 -b 0.5 -t 0.3 -n
```

## CC

### 单 GPU 无容错

```bash
./build/bin/cc/cc flickr -n
```

### 单 GPU 队列容错

```bash
./build/bin/cc/cc_queue flickr -a 0.5 -b 0.5 -t 0.3 -n
```

### 多节点/多 GPU 无容错

```bash
mpirun -np 2 \
  ./build/bin/cc/cc_multiGPU_basic flickr -n
```

### 多节点/多 GPU 容错

```bash
mpirun -np 2 \
  ./build/bin/cc/cc_multiGPU flickr -a 0.5 -b 0.5 -t 0.3 -n
```

## KCore

### 单 GPU 无容错

```bash
./build/bin/kcore/kcore flickr -k 5 -n
```

### 单 GPU 队列容错

```bash
./build/bin/kcore/kcore_queue flickr -k 5 -a 0.5 -b 0.5 -t 0.3 -n
```

### 多节点/多 GPU 无容错

```bash
mpirun -np 2 \
  ./build/bin/kcore/kcore_multiGPU_basic flickr -k 5 -n
```

### 多节点/多 GPU 容错

```bash
mpirun -np 2 \
  ./build/bin/kcore/kcore_multiGPU flickr -k 5 -a 0.5 -b 0.5 -t 0.3 -n
```

## PageRank

### 单 GPU 无容错

```bash
./build/bin/pagerank/pagerank flickr -n
```

### 单 GPU 队列容错

```bash
./build/bin/pagerank/pagerank_queue flickr -a 0.5 -b 0.5 -t 0.3 -n
```

### 多节点/多 GPU 无容错

```bash
mpirun -np 2 \
  ./build/bin/pagerank/pagerank_multiGPU_basic flickr -n
```

### 多节点/多 GPU 容错

```bash
mpirun -np 2 \
  ./build/bin/pagerank/pagerank_multiGPU flickr -a 0.5 -b 0.5 -t 0.3 -n
```

## MPI/NCCL 多机模板

本项目的多节点版本采用 rank-per-GPU 模式：1 个 MPI rank 对应 1 张 GPU。

例如两台机器各 1 张 GPU：

```bash
mpirun -np 2 \
  -H 10.4.1.139:1,g2c14:1 \
  --mca plm_rsh_agent ssh \
  --mca psec native \
  --mca pml ob1 \
  --mca btl self,tcp \
  --mca btl_tcp_if_include 10.4.1.0/24 \
  --wdir /tmp/GraphAlgorithm \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_DISABLE=1 \
  -x NCCL_SOCKET_FAMILY=AF_INET \
  -x NCCL_SOCKET_IFNAME=eno1,ens5f0 \
  ./build/bin/bfs/bfs_multiGPU flickr -s 9006 -n
```

两台机器各 2 张 GPU：

```bash
mpirun -np 4 \
  -H 10.4.1.139:2,g2c14:2 \
  --mca plm_rsh_agent ssh \
  --mca psec native \
  --mca pml ob1 \
  --mca btl self,tcp \
  --mca btl_tcp_if_include 10.4.1.0/24 \
  --wdir /tmp/GraphAlgorithm \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_DISABLE=1 \
  -x NCCL_SOCKET_FAMILY=AF_INET \
  -x NCCL_SOCKET_IFNAME=eno1,ens5f0 \
  ./build/bin/bfs/bfs_multiGPU flickr -s 9006 -n
```

注意：多机运行时，每台机器都必须能在相同工作目录下找到可执行文件和数据集。可以用软链接统一路径：

```bash
ln -sfn /home/huangxy/Projects/GraphAlgorithm /tmp/GraphAlgorithm
ssh g2c14 'ln -sfn /workplace/home/huayunpeng/Projects/GraphAlgorithm /tmp/GraphAlgorithm'
```

更多 MPI/NCCL 排错细节见：`MPI_MULTIGPU_USAGE.md`。

## 批量实验与画图

批量实验脚本位于：

```text
scripts/experiments/run_benchmarks.py
```

推荐使用 conda 环境运行：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py --datasets flickr --repeat 3
```

默认会运行 4 个算法的 4 类方法，并从输出中解析 `GPU time`，结果写入：

```text
outputs/experiments/<timestamp>/results.csv
outputs/experiments/<timestamp>/logs/*.log
```

已有 CSV 用独立脚本画图：

```bash
conda run -n test310 python scripts/experiments/plot_results.py \
  outputs/experiments/<timestamp>/results.csv
```

绘图脚本会按算法分别出图，并把耗时量级相近的数据集分到同一张图：

```text
outputs/experiments/<timestamp>/*_gpu_time_group*.png
outputs/experiments/<timestamp>/plot_manifest.csv
```

BFS 的源点会自动从 `dataset/sources/<dataset>_sources.tsv` 里选择第一行的高度数顶点。更多用法见：`scripts/experiments/README.md`。

## 脚本目录

数据下载、格式转换、数据集统计和源点选择脚本位于：

```text
scripts/utils/
```

根目录 `utils` 保留为兼容软链接，指向 `scripts/utils`。

## 输出文件

运行结果默认写到：

```text
outputs/<algorithm>/info_outcome.txt
```

例如：

```text
outputs/bfs/info_outcome.txt
outputs/cc/info_outcome.txt
outputs/kcore/info_outcome.txt
outputs/pagerank/info_outcome.txt
```

## 备注

- `cc_multiGPU`、`kcore_multiGPU`、`pagerank_multiGPU` 已经改为 MPI/NCCL 入口，需要用 `mpirun` 启动。
- 如果只是做性能测试，建议加 `-n` 关闭 CPU correctness check。
- BFS 的 `-s` 源点可以参考 `dataset/sources/*.tsv` 中生成的候选中心节点。
