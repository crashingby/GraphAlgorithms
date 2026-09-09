# 容错开销实验流水线

本目录只比较同一资源模式内的容错开销：

| 资源模式 | 基线 | 容错版 |
| --- | --- | --- |
| 1 GPU | `basic` | `tolerance_queue` |
| N GPU / N MPI rank（N ≥ 2） | `multigpu_basic` | `multigpu` |

单 GPU 与 N GPU同时改变资源、分区和通信，不能把它们之间的时间差解释为容错开销。

## 架构

```text
prepare_run.py → run_item.py → collect_results.py → analyze_overhead.py → plot_overhead.py
```

- `prepare_run.py`：冻结配置、metadata/BFS source 快照、数据集和二进制指纹、Git/环境信息与 Slurm work item。
- `run_item.py`：每条 warmup 或测量命令都写独立 UUID raw JSON、stdout 和 stderr；每次执行前后都会检查冻结的数据集（size/mtime）和二进制（含 SHA256），发生漂移即保留为 `input_drift`，不进入结果。超时会终止整个 launcher process group，避免遗留 MPI/GPU rank。
- `collect_results.py`：先验证 raw JSON 的 schema、identity、work item 与 manifest 一致性，再选择 latest-success observation；不合格 raw 仍留在磁盘和报告中，但绝不会静默进入 CSV 或配对。
- `analyze_overhead.py`：独立建立 1 GPU 与配置的 N GPU pair；迭代数或环境不一致的样本写入 `excluded_pairs.csv`，不会静默丢弃。
- `plot_overhead.py`：只读取 `analysis/`，不触碰 raw 数据。

一次运行目录如下：

```text
outputs/experiments/<run-tag>/
├── manifest/     # config、输入快照、数据/二进制 SHA256、环境、work items
├── raw/          # immutable attempt JSON 与 stdout/stderr
├── collected/    # attempts.csv、observations.csv、collection_report.json
├── analysis/     # paired_samples.csv、excluded_pairs.csv、summary、analysis_report.json
├── figures/      # runtime、paired overhead、components 与 figure manifest
└── slurm/        # array/collector scheduler logs
```

`raw/attempts/**/*.json` 与对应 stdout/stderr 是完整证据源；`collected/*.csv` 是有意裁剪的可分析投影，不替代 raw 中的 argv、完整 provenance 和输入 guard。

旧的 `outputs/experiments/local-overhead-v*` 不会被移动或删除，也不能和新 schema 混合。

## 环境与数据准备

脚本默认用本仓库由 `uv` 创建的 `.venv`：

```bash
cd /workplace/home/jiangnan/Projects/GraphAlgorithms
uv venv .venv
uv pip install --python .venv/bin/python matplotlib
```

修改 `dataset/*.mtx` 后，先刷新 metadata 和 BFS source：

```bash
.venv/bin/python scripts/utils/select_source_nodes.py --top-k 10
```

然后构建四算法 × 四方法：

```bash
cmake -S . -B build -DNCCL_ROOT=/path/to/nccl
cmake --build build --target \
  bfs bfs_queue bfs_multiGPU_basic bfs_multiGPU \
  cc cc_queue cc_multiGPU_basic cc_multiGPU \
  kcore kcore_queue kcore_multiGPU_basic kcore_multiGPU \
  pagerank pagerank_queue pagerank_multiGPU_basic pagerank_multiGPU -j
```

完整图集、固定参数和 launcher 写在 [configs/overhead.json](configs/overhead.json)；它显式列出数据集，避免 `dataset/` 中临时文件改变正式实验范围。

## 时间口径

单 GPU内部完成时间为：

```text
gpu_main_ms + cpu_check_tail_ms
```

多 GPU每 rank 内部总时间为：

```text
main_loop + cpu_check_drain + postcheck_total
```

多 GPU实际内部完成延迟必须使用 `rank_algorithm_total_max_ms`。`rank_algorithm_total_avg_ms` 只表示每 rank 平均工作量，用于通信与负载分解；旧 v4 无后缀的 `algorithm_total_ms` 是 rank mean，不能再当作完成时间。

`process_wall_ms` 包含 launcher、读图、初始化和退出，独立保存，不能与内部迭代时间混成同一指标。

`graph_kernel ⊂ gpu_compute ⊂ main_loop`，且 `postcheck_mpi ⊂ postcheck_total`。NCCL、MPI、kernel、CPU drain 和 post-check 都是解释性区间，不能堆叠成总时间。多 GPU组件图只用 rank mean，避免把来自不同 rank 的 maxima 相加。

## 严格配对

pair identity 是：

```text
algorithm + dataset + repeat + GPU mode
```

只有 baseline 和 tolerance 都成功、hostname / `CUDA_VISIBLE_DEVICES` / MPI rank 数，以及有效的 `CUDA_DEVICE_ORDER`、OMP、NCCL 与 OMPI 设置一致，且迭代轮次完全一致时，才计算任何 baseline-vs-tolerance 百分比或通信/imbalance 对照。其余 pair 的绝对时间仍保留，但会带明确原因进入 `analysis/excluded_pairs.csv`。

图中的 `†` 表示至少一个 pair 存在迭代轮次不一致；严格开销图仅用合格 pair，并标注有效样本数 `n`。统计使用每个 repeat 配对结果的中位数与 IQR。

## 图

每个算法按 basic 时间自动分组，每组生成：

1. `*_runtime.png`：四柱绝对内部完成时间（1 GPU basic/tolerance，N GPU basic/tolerance）；N GPU柱是 `max(rank internal total)`。
2. `*_paired_overhead.png`：1 GPU 和配置的 N GPU的严格配对容错开销百分比。这是主结论图。
3. `*_components.png`：核心 kernel 的 paired overhead、多 GPU NCCL/MPI 调用、容错 CPU drain/post-check、rank imbalance。所有 baseline-vs-tolerance panel 只使用严格合格 pair；CPU tail 是 tolerance-only 的绝对观测。各 panel 非堆叠。

## 本地运行

Smoke test：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
CONFIG=scripts/experiments/configs/smoke.json \
MPI_RANKS=2 \
RUN_TAG=smoke_$(date +%Y%m%d_%H%M%S) \
bash scripts/experiments/run_local.sh
```

完整实验：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MPI_RANKS=8 \
RUN_TAG=overhead_$(date +%Y%m%d_%H%M%S) \
REPEAT=5 WARMUP=1 TIMEOUT=3600 \
bash scripts/experiments/run_local.sh
```

选择子集：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DATASETS="cit-HepPh tech-RL-caida" \
ALGORITHMS="bfs cc" \
REPEAT=5 \
bash scripts/experiments/run_local.sh
```

使用 `PREPARE_ONLY=1` 只检查输入和创建 manifest。默认对图做 SHA256；开发 smoke 时可加 `FAST_FINGERPRINTS=1`，只记录文件大小和 mtime。

如需恢复中断任务，使用同一个 `RUN_ROOT`；本地 runner 的 `--resume` 会保留并跳过已有 identity。加 `RETRY_FAILED=1` 才会为只有失败历史的 identity 新建重试 attempt。没有 `RETRY_FAILED=1` 时，保留的测量失败仍会让 runner 以非零状态结束，不能被误报为成功。

`collect_results.py --strict` 只以正式 measurement 的完整性作为退出条件；失败 warmup 会完整保留并标为 warning，不会丢弃已完成的正式测量。`analyze_overhead.py --strict` 则额外要求没有任何被排除的 pair，适合作为论文数据的最终 QC。任何时候都可从 raw 重做派生层：

```bash
.venv/bin/python scripts/experiments/collect_results.py --run-dir outputs/experiments/<run-tag>
.venv/bin/python scripts/experiments/analyze_overhead.py --run-dir outputs/experiments/<run-tag>
.venv/bin/python scripts/experiments/plot_overhead.py --run-dir outputs/experiments/<run-tag> --formats png pdf
```

最终检查可使用 `collect_results.py --strict` 和 `analyze_overhead.py --strict`。

## Slurm

提交器先 prepare，再提交 dataset × algorithm array；所有 worker 结束后，`afterany` collector 自动执行 collect → analyze → plot。`PROJECT_ROOT`、`RUN_ROOT`、`.venv`、`build/` 和 `dataset/` 必须以同一路径出现在所有分配节点与 collector 节点上（通常是共享文件系统）：

```bash
RUN_TAG=overhead_$(date +%Y%m%d_%H%M%S) \
ARRAY_CONCURRENCY=1 \
REPEAT=5 WARMUP=1 TIMEOUT=3600 \
bash scripts/experiments/slurm/submit_overhead.sh
```

提交器不默认添加账号或分区。集群要求时才显式设置：

```bash
PARTITION=<partition> ACCOUNT=<account> QOS=<qos> \
RUN_TAG=overhead_$(date +%Y%m%d_%H%M%S) \
bash scripts/experiments/slurm/submit_overhead.sh
```

若出现账户/分区组合错误，先运行：

```bash
sacctmgr show assoc user="$USER" format=Account,Partition
sinfo -o "%P %a %l %G"
```

默认 launcher 在配置中是 `mpirun -np 2 --bind-to none`。若集群必须使用 `srun`，复制 JSON 配置、修改 `launcher` 后再提交；实际设置会冻结到 manifest。

`mpi_ranks` 是多 GPU 方法的 MPI rank 数，也是本框架假定的一卡一 rank 数；默认配置仍为 2。可直接编辑 JSON 的 `mpi_ranks`，也可通过 `MPI_RANKS=8` 覆盖。`run_local.sh` 会把该值传给 manifest；`submit_overhead.sh` 还会在单节点 worker 上提交 `--ntasks=N --gpus-per-node=N`。请让 `CUDA_VISIBLE_DEVICES` 包含至少 N 张卡；多 GPU 方法要求 `N >= 2`。

提交器会从冻结 manifest 计算一个 work item 的保守时限：`(warmup + repeat) × 方法数 × timeout_s + 10 分钟`，并通过 `--time` 覆盖 worker 脚本默认值。默认正式配置是 `1-00:10:00`；若集群时限更短，可显式设置 `WORKER_TIME=HH:MM:SS`，但这会降低超时覆盖裕量。`COLLECTOR_TIME` 默认为一小时。

重试 Slurm run（manifest 不会重建，旧 raw 不会覆盖）：

```bash
RUN_ROOT=outputs/experiments/<run-tag> \
RESUME=1 RETRY_FAILED=1 ARRAY_CONCURRENCY=1 \
bash scripts/experiments/slurm/submit_overhead.sh
```

自动 collector 会执行 `collect_results.py --strict`：正式 measurement 不完整时 job 明确失败，但仍会留下 collection、analysis 和 figures 供诊断。不要手工修改 `collected/`、`analysis/` 或 `figures/` 来“修数据”；应该从 `raw/` 重跑对应派生步骤。
