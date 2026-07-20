# 容错开销实验

这套实验只研究当前完整容错实现相对对应无容错实现的运行开销，不评价故障是否真的被检测到，也不扫描 `alpha`、`beta`、`threshold` 或 KCore 的 `k`。所有对比都必须在相同数据集、算法、重复编号和 GPU 模式内完成。

## 1. 实验对象与四种方法

实验覆盖 BFS、CC、KCore、PageRank。每个算法保留四个主线方法：

| 方法 | 资源 | 含义 | 容错开销基线 |
| --- | --- | --- | --- |
| `basic` | 1 GPU、1 进程 | 单 GPU 无容错 | — |
| `tolerance_queue` | 1 GPU、1 进程 | 单 GPU 容错，异步 CPU 队列检测 | `basic` |
| `multigpu_basic` | 2 GPU、2 MPI rank | 多 GPU 无容错 | — |
| `multigpu` | 2 GPU、2 MPI rank | 多 GPU 容错，异步 CPU 队列检测 | `multigpu_basic` |

当前 runner 和 Slurm 流程固定多 GPU方法为 2 个 MPI rank、每个 rank 一张 GPU。合法的容错开销配对只有：

```text
1 GPU: tolerance_queue / basic
2 GPU: multigpu / multigpu_basic
```

`basic` 与 `multigpu_basic` 的差异可以描述扩展性，但不能解释成容错开销；它同时改变了 GPU 数量、分区、ghost、同步和通信。

## 2. Python 环境：uv 与本地 .venv

实验脚本要求 Python 3.9 或更高版本。runner 与合并器只使用标准库；绘图需要 Matplotlib。仓库没有自动为计算节点安装依赖，建议在共享文件系统上的项目根目录提前创建 `.venv`：

```bash
uv venv .venv
uv pip install --python .venv/bin/python matplotlib
```

之后本地命令可以使用任一种写法：

```bash
uv run python scripts/experiments/run_benchmarks.py --help
.venv/bin/python scripts/experiments/run_benchmarks.py --help
```

本仓库当前没有 `pyproject.toml`，所以直接使用 `uv run python`；不要添加 `--no-sync`，该选项在项目外没有同步对象，只会产生 “has no effect outside a project” 警告。正式 Slurm 任务仍不应依赖计算节点联网，Matplotlib 等依赖应提前装进共享 `.venv`，或确保 `uv run python` 实际使用的 Python 环境已经具备这些依赖。

Slurm 提交器的 `PYTHON_MODE=auto` 依次选择显式 `PYTHON_BIN`、项目 `.venv/bin/python`、可用的 `uv`。也可显式设置：

- `PYTHON_MODE=venv`：使用项目 `.venv`，也可用 `PYTHON_BIN` 覆盖。
- `PYTHON_MODE=binary`：必须提供可执行的 `PYTHON_BIN`。
- `PYTHON_MODE=uv`：与脚本一致，使用 `UV_BIN run python`。

提交前脚本会检查 Python 版本并执行 `import matplotlib`，因此依赖问题会在提交 array 之前暴露。

## 3. 数据集、metadata 与 BFS 源点

当前 `dataset/` 有 22 个 `.mtx` 图。可以确认数量：

```bash
find dataset -maxdepth 1 -type f -name "*.mtx" | wc -l
```

每次新增、替换或修改图文件后，从项目根目录重新生成元数据和 BFS 源点：

```bash
uv run python scripts/utils/select_source_nodes.py --top-k 10
```

该命令生成或刷新：

```text
dataset/metadata.csv
dataset/sources/<dataset>_sources.tsv
dataset/sources/source_summary.md
```

源点候选按有向图的 `outdegree` 排序，再用 total degree 和顶点 ID 打破平局；不能按无向 total degree 选择，因为高入度汇点可能 `outdegree=0`，会让 BFS 退化成一轮。TSV 和 metadata 同时记录 `bfs_source_outdegree`、`bfs_source_indegree`、`bfs_source_total_degree`；兼容列 `bfs_source_degree` 现在等于 outdegree。该规则保证源点至少有较强的向外扩展能力，但不声称一定最大化完整可达覆盖。

`metadata.csv` 至少包含 `dataset`、`nodes`、`edges`、`valid_edges`、`file_size_bytes` 和 `bfs_source`。runner 使用这些信息做以下工作：

- 校验每个 `.mtx` 都有元数据，节点数、边数和文件大小为正。
- 对 BFS 的四个方法显式传入同一个 `-s <bfs_source>`，不使用可能无代表性的默认顶点 0。
- 把节点数、边数和文件大小写入结果；nominal MTEPS 使用 `edges`。
- 在长任务启动前检查目标数据集、metadata、BFS 源点和可执行文件。

Slurm 提交器还要求 `.mtx` 数量、metadata 数据行数和 `*_sources.tsv` 数量一致。当前应当都是 22；不一致时会要求重新运行上述命令。

## 4. 构建 16 个算法目标

首次配置需要 CMake、CUDA、MPI、NCCL，并指定实际 NCCL 安装位置：

```bash
cmake -S . -B build -DNCCL_ROOT=/path/to/nccl
```

显式构建 4 个算法 × 4 个方法的 16 个目标：

```bash
cmake --build build --target \
  bfs bfs_queue bfs_multiGPU_basic bfs_multiGPU \
  cc cc_queue cc_multiGPU_basic cc_multiGPU \
  kcore kcore_queue kcore_multiGPU_basic kcore_multiGPU \
  pagerank pagerank_queue pagerank_multiGPU_basic pagerank_multiGPU \
  -j
```

可执行文件位于 `build/bin/{bfs,cc,kcore,pagerank}/`。runner 会检查所选算法和方法对应的二进制是否存在且可执行；缺失任一目标会在批量运行前失败。

## 5. 计时字段与总耗时口径

本轮结果使用 CSV schema v4。多 GPU 二进制输出的核心记录示意如下：

```text
BENCHMARK_TIMING
  gpu_main_ms=... main_loop_ms=... gpu_compute_ms=...
  graph_kernel_ms=...
  cpu_check_tail_ms=... cpu_check_drain_ms=...
  nccl_exchange_ms=... mpi_sync_ms=...
  postcheck_total_ms=... postcheck_mpi_ms=...
  communication_ms=...
  rank_*_avg_ms=... rank_graph_kernel_avg_ms=...
  rank_algorithm_total_{avg,max}_ms=...
BENCHMARK_ITERATIONS iterations=...
```

runner 严格要求成功进程恰好各有一条记录，字段缺失、重复、为负或阶段关系不合理都会标记为 `parse_error`。

### 5.1 三个不重叠的总耗时阶段

多 GPU 每个 rank 的内部总耗时定义为：

```text
algorithm_total
  = main_loop
  + cpu_check_drain
  + postcheck_total
```

- `main_loop_ms`：`steady_clock` 包围完整迭代循环，包括 CUDA、NCCL、每轮 host 控制和终止判断 MPI。
- `cpu_check_drain_ms`：主循环结束后发出 worker 停止请求，直到 `join()` 返回。它只表示尚未被主循环隐藏的异步检测工作。
- `postcheck_total_ms`：join 之后按迭代聚合检测结果的完整阶段；无容错版为 0。

兼容字段 `gpu_main_ms` 等于 `main_loop_ms`，`cpu_check_tail_ms` 等于 `cpu_check_drain_ms`。旧结果中的多 GPU tail 曾同时包含 drain 与检测结果 MPI 汇总；schema v3 已将两者分开，schema v4 又加入纯图 kernel 区间。v3 及更早 CSV 不能与 v4 混合作为同一批结果，绘图脚本也会显式拒绝缺少 kernel 字段的旧 CSV。

### 5.2 CUDA、NCCL 和 MPI 解释性子项

- `gpu_compute_ms`：compute stream 上 NCCL 区间之外的 CUDA 时间，包括算法、评分、计数 kernel 及必要 copy；它比旧 `gpu_main_ms` 更接近计算开销，但仍不是纯 kernel time。
- `graph_kernel_ms`：CUDA event 在每轮只包住核心图算法 kernel 后得到的累计设备时间。basic 对应 `*PullKernel`/`*DistributedPullKernel`，容错对应执行主计算和选择性冗余计算的 `*PullDualKernel`/`*DistributedToleranceKernel`。它不包含 score-and-mark、active/update 维护、计数、D2H、NCCL、MPI 或 CPU 检测。
- `rank_graph_kernel_avg_ms`：多 GPU 各 rank 的 `graph_kernel_ms` 算术平均，用于两 GPU柱状图；`graph_kernel_ms` 本身仍是 rank max。
- `nccl_exchange_ms`：每轮 activation/value 的 pack、NCCL send/recv 和 unpack。
- `mpi_sync_ms`：主循环终止判断 `MPI_Allreduce` 的阻塞调用时间。它包含通信与快 rank 等待慢 rank，必须标成 sync/wait。
- `postcheck_mpi_ms`：post-check 阶段内 MPI 调用时间，是 `postcheck_total_ms` 的子集。
- `communication_ms = nccl_exchange + mpi_sync + postcheck_mpi`，也是解释性子项。

这些子项已经包含在 main loop 或 post-check total 中，不能再次加到 algorithm total。`graph_kernel_ms` 又是 `gpu_compute_ms` 的子集，也不能与它相加。详情图中的 post-check total 和 post-check MPI 同样不是堆叠关系。

runner 另外写出 `graph_kernel_per_iter_ms = graph_kernel_ms / iterations`，便于检查平均每轮 kernel 成本；正式百分比仍只使用迭代数一致的成对样本。

核心 kernel 数值使用 CUDA event，而不是从 NVTX 导出。event 与目标 kernel 记录在同一 stream 上，适合稳定写入 CSV；已有 NVTX range 仍可用于 Nsight Systems 交互分析。每轮额外记录两个 event 会给四条路径引入相同类型的轻微测量扰动。该区间在 GPU 被其他进程抢占时仍可能受到调度影响，所以正式实验仍应使用独占 GPU。

主循环还可以得到：

```text
main residual
  = rank_main_loop_avg
  - rank_gpu_compute_avg
  - rank_nccl_exchange_avg
  - rank_mpi_sync_avg
```

它表示未归入 CUDA/NCCL/MPI 调用的 host 控制和计时空隙，适合发现容错版额外的 enqueue、event 和调度成本。

### 5.3 rank max、rank avg 与四柱总览

所有多 GPU 局部字段同时输出 rank max 和真实算术平均 `rank_*_avg_ms`。avg 来自 `MPI_SUM / world_size`，不是用 max 除以 GPU 数。

rank max 的几个组件可能来自不同 rank，因此 max communication 不保证等于几个 max 组件相加。通信分解与四柱总览均使用 rank mean。每个 rank 先计算不重叠 total，再得到：

- `rank_algorithm_total_avg_ms`：四柱图中的一个 rank 平均耗时。
- `rank_algorithm_total_max_ms`：最慢 rank 的内部总耗时。
- `rank_imbalance_pct = (max / avg - 1) × 100%`。

四柱图固定算法，横坐标是按耗时尺度分组的数据集，依次展示：

1. 单 GPU basic 内部时间。
2. 单 GPU tolerance 的主区间 + CPU drain。
3. 两 GPU basic 的 rank-mean 内部时间。
4. 两 GPU tolerance 的 rank-mean main loop + drain + post-check。

单 GPU 程序输出原有 CUDA-event 主区间以及新增的核心 `graph_kernel_ms`；runner 仍将主区间映射为 `main_loop_ms` 和 `gpu_compute_ms`，将 tail 映射为 drain，post-check 置 0。因此开销百分比只在同一 GPU 模式内解释；单 GPU与两 GPU四柱适合总体现象对照，但内部边界不是完全相同的端到端范围。

### 5.4 Python wall time 与排除项

`wall_time_ms` 用 `time.perf_counter()` 包围完整 `subprocess.run()`，包括程序或 mpirun 启动、读图、MPI/NCCL 初始化、内部算法、最终结果收集和退出。

多 GPU内部时间故意排除初始 ghost warmup、最终结果 D2H/`MPI_Allgatherv` 和计时报告自身的 MPI 归约。研究同资源容错开销时优先看配对的内部时间；需要真实任务端到端成本时看 wall time。runner 始终传 `-n`，关闭昂贵的完整 CPU oracle，但不会关闭容错版异步摘要检测。
## 6. 严格配对、迭代一致性与 BFS 注意事项

统计 identity 是 `dataset + algorithm + repeat + method`。绘图只接受 `returncode=0` 的行，并要求同一个 `dataset + algorithm + repeat` 中四个方法都存在且唯一；多 GPU两行还必须报告 2 GPU。重复 identity、缺方法或失败行不会补 0。

测量顺序按 dataset → algorithm → repeat → method。奇数 repeat 使用给定方法顺序，偶数 repeat 反向，从而交替 baseline 与容错版的先后顺序。warmup 日志不进入测量 CSV。

开销百分比只有在同一资源配对的迭代数完全相同时才计算：

```text
1 GPU graph-kernel overhead: tolerance_queue vs basic
2 GPU rank-mean graph-kernel overhead: multigpu vs multigpu_basic
1 GPU CUDA-main/compute-path overhead: tolerance_queue vs basic
2 GPU rank-mean non-NCCL CUDA compute overhead: multigpu vs multigpu_basic
```

如果 pair 内迭代数不同，该 repeat 仍保留绝对内部耗时，但不会进入对应 overhead 百分比样本。统计先在每个合法 repeat 内配对，再对样本计算均值和样本标准差。

只要某个完整 repeat 的四种方法迭代数不全相等，图上的 dataset 名称就加 `†`。`†` 不表示数据无效；它说明绝对时间保留，但 overhead 只使用各自 pair 内迭代相同的 repeats。单 GPU与两 GPU之间迭代数不同也会触发该符号，因为资源模式和执行路径不同。

BFS 的两 GPU基础/容错版现在共享平衡连续分区、去重 activation plan、ghost value plan，并统一以 source + source 出邻居作为 pull BFS 初始工作集。正常情况下 pair 应有相同迭代数；若仍不一致，应按实现 bug、原地更新非确定性或运行异常调查。脚本会保留绝对时间，但拒绝把该 repeat 用于容错开销百分比。

## 7. 本地 smoke test

先在能看到两张 GPU 的节点上，用一个较小图、一次 warmup、一次测量跑四算法和四方法：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python \
  scripts/experiments/run_benchmarks.py \
  --datasets cit-HepPh \
  --repeat 1 \
  --warmup 1 \
  --mpi-ranks 2 \
  --timeout 600 \
  --metadata dataset/metadata.csv \
  --out-dir outputs/experiments/smoke \
  --mpirun-args --bind-to none
```

`--mpirun-args` 使用 `argparse.REMAINDER`，必须放在命令最后。warmup 会为每个 dataset/algorithm 先运行所选四个方法一次，日志保存在 `logs/warmup/`，但不写入 `results.csv`。任一 warmup 失败时，runner 会跳过该 dataset/algorithm 的正式测量，因为后续样本不可比较。

检查 smoke 输出：

```bash
uv run python scripts/experiments/plot_results.py \
  outputs/experiments/smoke/results.csv \
  --out-dir outputs/experiments/smoke/figures
```

同时确认：

- `run_report.json` 的状态为 `complete`。
- `results.csv` 有 4 算法 × 4 方法 = 16 条成功测量行。
- 每条成功行只有一个 `BENCHMARK_TIMING` 和一个 `BENCHMARK_ITERATIONS`。
- `logs/` 与 `logs/warmup/` 中没有 CUDA、NCCL、MPI、timeout 或 parse error。
- `run_config.json` 的 GPU 可见性、固定参数、warmup、repeat 和 rank 数正确。

## 8. 本地完整实验

默认自动发现全部 22 个 `.mtx`。正式运行建议至少 5 次测量，并显式做 1 次 warmup：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python \
  scripts/experiments/run_benchmarks.py \
  --repeat 5 \
  --warmup 1 \
  --mpi-ranks 2 \
  --timeout 3600 \
  --metadata dataset/metadata.csv \
  --out-dir outputs/experiments/local-overhead \
  --mpirun-args --bind-to none

uv run python scripts/experiments/plot_results.py \
  outputs/experiments/local-overhead/results.csv \
  --out-dir outputs/experiments/local-overhead/figures
```

固定参数默认是 KCore `k=5`，容错版 `alpha=0.5`、`beta=0.5`、`threshold=0.3`。本实验不扫描这些参数；整批实验必须保持一致，实际值会写入 `run_config.json`。

一次本地 runner 输出：

```text
<out-dir>/
├── results.csv
├── run_config.json
├── run_report.json
└── logs/
    ├── warmup/*.stdout.log, *.stderr.log
    └── *.stdout.log, *.stderr.log
```

runner 每完成一条测量就 flush CSV，并原子更新 `run_report.json`。超时或单个程序失败不会丢失此前结果，但 runner 最终返回非零。

## 9. Slurm：dataset × algorithm 的 88 个 work items

推荐入口是：

```bash
bash scripts/experiments/slurm/submit_overhead.sh
```

提交器读取当前 22 个数据集并生成：

```text
22 datasets × 4 algorithms = 88 work items
array index: 0-87
```

repeat 和 warmup 不扩大 array；每个 work item 对应一个 algorithm/dataset，并在同一作业中依次运行四个方法的 warmup 与全部 repeats。输出目录是 `runs/<algorithm>/<dataset>/`。这样单个慢算法失败不会阻塞同一数据集的其他算法结果。

常用正式提交：

```bash
RUN_TAG=overhead_20260718 \
ARRAY_CONCURRENCY=2 \
REPEAT=5 \
WARMUP=1 \
TIMEOUT=3600 \
PARTITION=gpu \
ACCOUNT=my_account \
QOS=normal \
PYTHON_MODE=venv \
bash scripts/experiments/slurm/submit_overhead.sh
```

主要环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RUN_TAG` | 当前时间戳 | 本次输出标签 |
| `RUN_ROOT` | `outputs/experiments/<RUN_TAG>` | 可显式指定完整输出目录 |
| `ALLOW_EXISTING_RUN_ROOT` | `0` | 默认拒绝复用已有非空 `work_items.tsv` 的目录；明确重提时设为 `1` |
| `ARRAY_CONCURRENCY` | `1` | 同时运行的 work item 数；每个 item 都申请 2 GPU |
| `REPEAT` | `5` | 每个方法的正式测量次数 |
| `WARMUP` | `1` | 每个方法在该 algorithm/dataset 上的预热次数 |
| `TIMEOUT` | `3600` | 单次可执行程序的超时秒数 |
| `MPI_RANKS` | `2` | 固定为 2，其他值会被拒绝 |
| `PARTITION` | 空 | 可选 partition，同时传给 worker 与 collector |
| `ACCOUNT` | 空 | 可选 Slurm account |
| `QOS` | 空 | 可选 QoS |
| `PYTHON_MODE` | `auto` | `auto`、`venv`、`binary` 或 `uv` |
| `PYTHON_BIN` | 空 | `binary/venv` 使用的解释器路径 |
| `UV_BIN` | 自动发现 | `uv` 模式的可执行文件路径 |

`run_overhead.sbatch` 每个 array task 默认申请：

```text
1 node
2 tasks
2 GPUs per node
4 CPUs per task
64 GiB memory
6 hours
```

两 GPU方法用两个 rank 和两张 GPU；单 GPU方法在各自调用中只用一张 GPU。`ARRAY_CONCURRENCY=2` 意味着最多同时占用 4 张 GPU，不是每个 task 使用 4 张。

worker 设置 `OMP_NUM_THREADS=1`、`CUDA_DEVICE_ORDER=PCI_BUS_ID` 和默认 `NCCL_DEBUG=WARN`，并从提交环境继承其余变量。脚本不会自动执行 `module load`；CUDA、MPI、NCCL、编译器运行库和必要的 `LD_LIBRARY_PATH/PATH` 必须在提交前加载，或由集群统一环境提供。Slurm 负责设置每个 job 的 `CUDA_VISIBLE_DEVICES`。

### account、partition 或 QoS 提交故障

`PARTITION`、`ACCOUNT`、`QOS` 只要非空就会同时传给 array 和 collector。常见错误是当前 shell 继承了无效的 `ACCOUNT`，导致 `sbatch` 报 invalid account 或 account/partition combination，并且由于提交脚本使用 `set -e`，array 提交失败后 collector 也不会提交。

处理方法：

1. 向集群管理员或 `sacctmgr show assoc` 确认可用 account、partition 和 QoS 组合。
2. 需要显式 account 时设置正确的 `ACCOUNT=...`。
3. 集群已有默认 account 时，使用 `unset ACCOUNT`，或通过 `env -u ACCOUNT bash scripts/experiments/slurm/submit_overhead.sh` 防止旧值被传入。
4. 提交失败后检查是否打印了 array/collector job id；没有 job id 就不是运行期失败。默认情况下，只要原目录已有非空 `work_items.tsv`，提交器就会拒绝复用。修正参数后若确实要沿用同一个 `RUN_ROOT`，必须显式设置 `ALLOW_EXISTING_RUN_ROOT=1`；否则使用新的 `RUN_TAG`。允许复用时，提交器会重新生成 `datasets.txt` 和 `work_items.tsv`，并重新提交完整 array 与 collector。

较老 Slurm 若不支持 `#SBATCH --gpus-per-node=2`，应按集群规范把 worker 中该项替换为例如 `#SBATCH --gres=gpu:2`。GPU 型号约束也必须使用本集群支持的格式。

## 10. merge、afterany collector 与 partial 结果

提交器先写入 `datasets.txt` 和 `work_items.tsv`，再提交：

1. `run_overhead.sbatch` array。
2. 依赖 `afterany:<array_job_id>` 的 `collect_overhead.sbatch`。

`afterany` 保证即使部分 work item 失败，collector 仍尝试合并已有结果。collector 先运行 `merge_results.py`，再在顶层 `results.csv` 非空时运行绘图；merge 或 plot 任一返回非零，collector 最终返回 3 并提示检查 partial 输出。

合并器递归读取 `runs/<algorithm>/<dataset>/results.csv`，并严格检查：

- CSV schema 版本、字段集合和字段顺序。
- 行中的 algorithm/dataset 是否与目录一致。
- `dataset + algorithm + repeat + method` identity 是否唯一。
- 成功行的计时、rank 数和状态字段是否合法。
- `work_items.tsv` 声明的 work item 与全部预期测量行是否齐全。

即使发现失败、缺失、重复或 schema 错误，合并器仍会尽可能写出：

```text
results.csv
failed_runs.csv
collection_report.json
```

此时 `collection_report.json` 的 `status` 为 `partial`，合并器返回 3。不要因为顶层 `results.csv` 存在就认为实验完整。报告中的 `missing_work_items`、`missing_rows`、`invalid_inputs`、`duplicate_identities` 和 `manifest_errors` 给出恢复入口。

手工合并与绘图：

```bash
uv run python scripts/experiments/merge_results.py \
  outputs/experiments/<RUN_TAG>

uv run python scripts/experiments/plot_results.py \
  outputs/experiments/<RUN_TAG>/results.csv \
  --out-dir outputs/experiments/<RUN_TAG>/figures
```

若 merge 返回 3，第一条命令仍已写出 partial CSV 和报告。绘图只会使用其中完整、成功且无歧义的四方法 repeat，因此可能为完整子集生成图；缺失项不会补 0。

## 11. 最终生成三套图

旧版七类独立图已经退出。当前每个算法、每个运行时间分组生成三张 PNG：

### 11.1 四方法内部总耗时图

文件名：

```text
<algorithm>_runtime_groupNN.png
```

每个 dataset 有四组均值柱和样本标准差误差条：

1. 1 GPU basic：内部主区间。
2. 1 GPU tolerance：主区间 + CPU drain。
3. 2 GPU basic：真实 mean-rank internal total。
4. 2 GPU tolerance：真实 mean-rank main loop + drain + post-check total。

这张图保留绝对耗时，即使四方法迭代数不同也仍显示，并用 `†` 标记。

### 11.2 可解释开销详情图

文件名：

```text
<algorithm>_overhead_details_groupNN.png
```

一张图包含六个纵向 panel：

1. 同 GPU 模式且 pair 迭代数一致时的 compute-path overhead；两 GPU使用排除 NCCL 的 CUDA 计算时间。
2. 1 GPU 与两 GPU容错版在主循环结束后的真实 CPU checker drain。
3. 两 GPU容错版的 post-check total，以及其中 MPI 调用这一子集。
4. 两 GPU basic/tolerance 的 rank-mean NCCL、主循环 MPI sync/wait，并单列容错版 post-check MPI。
5. 两 GPU main loop 中未归入 CUDA/NCCL/main MPI 的 host/control residual。
6. 两 GPU basic/tolerance 的 rank internal-total 不均衡 `max / mean - 1`。

post-check MPI 是 post-check total 的子集，通信项也是内部时间的解释性子项，图中均不做相加或堆叠。迭代不匹配的 repeat 不会被伪造成 overhead 样本。

### 11.3 核心图算法 kernel 对比图

文件名：

```text
<algorithm>_graph_kernel_groupNN.png
```

该图固定算法与数据集分组，包含两个 panel：

1. 四个核心 kernel 的累计 CUDA-event 时间：1 GPU basic、1 GPU tolerance dual、2 GPU basic rank mean、2 GPU tolerance dual rank mean。
2. 同 GPU 模式下容错 kernel 相对 basic kernel 的成对开销百分比。只有 pair 迭代数相等的 repeat 才进入百分比统计。

这张图回答的是“执行正常图计算的 kernel 与包含选择性冗余计算的 kernel 本身差多少”。它不代表完整容错开销；score-and-mark、active/update 维护、异步检测、通信和 host 控制仍需结合 runtime 图和 details 图判断。绝对 kernel 时间在迭代不同时仍显示并标 `†`，不能直接把其差值解释为单轮冗余成本。

### 11.4 按 4× 耗时自动分组

绘图先为每个 dataset 计算 `scale_score_ms`，即四种内部总耗时均值中的最大值；然后在每个算法内按 score 从小到大排序。新分组在以下任一条件触发：

- 当前组已达到 `--datasets-per-figure`，默认 8 个 dataset。
- 新 dataset 的 score 与当前组最小 score 的比值超过 `--max-scale-ratio`，默认 4。

因此默认每张图同时满足最多 8 个 dataset，并尽量把耗时跨度控制在 4× 内。可以显式调整：

```bash
uv run python scripts/experiments/plot_results.py results.csv \
  --out-dir figures \
  --datasets-per-figure 8 \
  --max-scale-ratio 4
```

`summary.csv` 保存每个 dataset/algorithm 的匹配 repeat、迭代不匹配记录、四方法总耗时、四个核心 kernel 绝对时间、两组 kernel overhead 和六个详情 panel 的均值、样本标准差与样本数。`plot_manifest.csv` 记录每组的数据集、耗时范围、scale ratio、三张图文件名和输入 CSV。

## 12. 输出目录

一次完整 Slurm 运行的目录结构：

```text
outputs/experiments/<RUN_TAG>/
├── datasets.txt
├── work_items.tsv
├── runs/
│   └── <algorithm>/
│       └── <dataset>/
│           ├── results.csv
│           ├── run_config.json
│           ├── run_report.json
│           └── logs/
│               ├── warmup/*.stdout.log, *.stderr.log
│               └── *.stdout.log, *.stderr.log
├── slurm/
│   ├── <array-job>_<task>.out, .err
│   └── collect-<job>.out, .err
├── results.csv
├── failed_runs.csv
├── collection_report.json
└── figures/
    ├── summary.csv
    ├── plot_manifest.csv
    ├── <algorithm>_runtime_groupNN.png
    ├── <algorithm>_overhead_details_groupNN.png
    └── <algorithm>_graph_kernel_groupNN.png
```

最重要的原始证据是 per-work-item `results.csv`、`run_config.json`、`run_report.json`、stdout/stderr 和 Slurm 日志。不要只保留图片。

## 13. 失败代码、检查与恢复

runner 使用以下特殊 returncode：

| returncode | 含义 |
| ---: | --- |
| `-996` | 无法启动进程 |
| `-997` | 进程成功但机器可读 benchmark 记录解析失败 |
| `-998` | 单次命令超过 `TIMEOUT` |
| `-995` | merge 发现 schema、identity 或行状态非法 |

程序本身非零退出时保留其原始 returncode。每次 collector 完成后检查：

```bash
cat outputs/experiments/<RUN_TAG>/collection_report.json
cat outputs/experiments/<RUN_TAG>/failed_runs.csv
find outputs/experiments/<RUN_TAG>/slurm \
  -type f -name "*.err" -size +0 -print
```

还应查看对应的 `runs/<algorithm>/<dataset>/run_report.json` 和 `logs/`。warmup 失败不会写成正式测量行，而是使该 work item 的 report 标记 `warmup_failed` 并跳过测量，因此 merge 会报告 missing rows。

恢复一个失败的 algorithm/dataset 时，建议在重新获得两张 GPU 的节点上重跑整个四方法 work item，并覆盖原目录：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python \
  scripts/experiments/run_benchmarks.py \
  --datasets cit-HepPh \
  --algorithms bfs \
  --methods basic tolerance_queue multigpu_basic multigpu \
  --repeat 5 \
  --warmup 1 \
  --mpi-ranks 2 \
  --timeout 3600 \
  --metadata dataset/metadata.csv \
  --out-dir outputs/experiments/<RUN_TAG>/runs/bfs/cit-HepPh \
  --mpirun-args --bind-to none
```

runner 会以写模式重建该 work item 的 `results.csv`，所以要重跑完整的四方法与全部 repeats，不要只写入一个方法造成新的缺失。完成后重新运行 merge 和 plot，刷新顶层 CSV、partial 报告和三套图。

若整个 array task 在创建 `results.csv` 前退出，报告会列出 `missing_work_items`；若中途退出，runner 已 flush 的行仍可用于诊断，但 `missing_rows` 会指出未完成 identity。修复环境、增大单命令 `TIMEOUT` 或修改 `#SBATCH --time` 后再重跑。

## 14. 正式实验规范

- smoke 的 `repeat=1` 只验证流程；正式统计建议 `REPEAT >= 5`。
- warmup 与正式实验必须使用相同二进制、参数、GPU 型号和软件栈。
- 单 GPU pair 与两 GPU pair 应在同一节点类型上运行；不要混合不同 GPU、驱动、CUDA、MPI 或 NCCL 环境。
- 尽量使用独占节点，避免其他任务引入 GPU、CPU、内存或互联争用。
- `ARRAY_CONCURRENCY` 按可独占 GPU 数决定；每增加 1 都可能再占 2 张 GPU。
- 固定 GPU 时钟、电源模式、CPU 绑定、MPI/NCCL 参数和数据文件位置。
- 结论同时报告绝对内部总耗时、核心 graph kernel 绝对时间与合法 overhead 样本数、iterations、CPU drain、post-check total/MPI 子项、NCCL、主循环 MPI sync 和 rank imbalance。
- BFS 两 GPU结果必须保留 `†` 与迭代轨迹说明，不能把不匹配样本解释为纯容错开销。
- 保留 metadata、manifests、所有 CSV、run report 和原始日志。

即使迭代数相同，容错版也不仅增加 CPU worker；它还包含 score-and-mark、DMR/摘要 kernel、额外 stream、D2H 摘要与不同的活跃集维护。因此本实验衡量的是两套完整实现之间的开销，不是单独 CPU 队列的微基准。
