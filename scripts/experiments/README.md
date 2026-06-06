# Experiments

`run_benchmarks.py` 会批量运行 4 个算法的 4 类方法，并从程序输出里解析 `GPU time: ... ms`，最终生成 `results.csv`。绘图由 `plot_results.py` 单独完成。

## 推荐环境

如果 `matplotlib` 装在 conda 环境 `test310` 里，可以这样运行：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py --datasets flickr --repeat 3
```

或者先进入环境：

```bash
conda activate test310
python scripts/experiments/run_benchmarks.py --datasets flickr --repeat 3
```

## 默认行为

默认会扫描 `dataset/*.mtx`，对每个数据集运行：

- `basic`
- `tolerance_queue`
- `multigpu_basic`
- `multigpu`

覆盖算法：

- `bfs`
- `cc`
- `kcore`
- `pagerank`

BFS 会自动读取 `dataset/sources/<dataset>_sources.tsv` 的第一行，使用度最大的候选源点。

结果输出到：

```text
outputs/experiments/<timestamp>/results.csv
outputs/experiments/<timestamp>/logs/*.log
```

画图脚本会读取已有 CSV，并把耗时量级相近的数据集分到同一张图：

```text
outputs/experiments/<timestamp>/*_gpu_time_group*.png
outputs/experiments/<timestamp>/plot_manifest.csv
```

## 常用命令

只跑 flickr：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py --datasets flickr
```

只跑两个小数据集，重复 3 次：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py --datasets 364 1174 --repeat 3
```

只跑 BFS 和 CC：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py --algorithms bfs cc --datasets flickr
```

只跑单 GPU 方法：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py \
  --methods basic tolerance_queue \
  --datasets flickr
```

只跑 MPI 方法，本机 2 rank：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py \
  --methods multigpu_basic multigpu \
  --mpi-ranks 2 \
  --datasets flickr
```

已有 CSV 画图：

```bash
conda run -n test310 python scripts/experiments/plot_results.py \
  outputs/experiments/<timestamp>/results.csv
```

每张图默认放 8 个数据集；脚本会先按该算法下 4 类方法的最大 GPU 时间排序，再分组：

```bash
conda run -n test310 python scripts/experiments/plot_results.py \
  outputs/experiments/<timestamp>/results.csv \
  --datasets-per-figure 6
```

## 多机 MPI 参数

`--mpirun-args` 后面的内容会插入到 `mpirun -np N` 后面。这个参数要放在命令最后。

例如两台机器各 1 张 GPU：

```bash
conda run -n test310 python scripts/experiments/run_benchmarks.py \
  --methods multigpu_basic multigpu \
  --mpi-ranks 2 \
  --datasets flickr \
  --mpirun-args \
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
  -x NCCL_SOCKET_IFNAME=eno1,ens5f0
```

## 参数

```text
--datasets        数据集名，不带 dataset/ 前缀和 .mtx 后缀；默认全部
--algorithms      bfs cc kcore pagerank；默认全部
--methods         basic tolerance_queue multigpu_basic multigpu；默认全部
--repeat          重复次数；默认 1
--mpi-ranks       MPI rank 数；默认 2
--alpha           容错 alpha；默认 0.5
--beta            容错 beta；默认 0.5
--threshold       容错阈值；默认 0.3
--k               KCore 的 k；默认 5
--timeout         单个命令超时时间，秒；默认 600
--out-dir         指定输出目录
```
