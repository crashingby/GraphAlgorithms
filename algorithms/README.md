# Algorithms 设计梳理

本文根据当前源码整理 `algorithms/` 的主线实现，重点覆盖四个算法的四类保留版本：

1. `basic`：单进程、单 GPU、无检测；
2. `tolerance_queue`：单 GPU、选择性双模冗余（DMR）与异步检查队列；
3. `multigpu_basic`：MPI rank-per-GPU、NCCL 通信、无检测；
4. `multigpu`：MPI/NCCL 多 GPU、选择性 DMR 与异步 CPU 检查。

当前四个算法的 CMake、批量实验和源码目录都只保留这四条主线。`cub`、`compact_queue`、`legacy`、`threshold` 等历史实验版本已从仓库删除。

## 一句话设计

所有主线实现都是 **pull 计算 + push 激活** 的稀疏迭代：活跃顶点通过入边读取依赖，状态变化后沿出边标记下一轮需要重算的顶点。容错版本在此基础上按重要性选择一部分“关键”顶点，让 block 内闲置线程重算它们，并报告不一致或趋势异常。

```text
dataset/<name>.mtx
        │
        ▼
BuildMarketGraph: COO -> CSR（出边 CSR + 入边 CSR）
        │
        ├── basic / queue：一张 GPU
        └── multigpu_*：owned 顶点 + ghost 顶点
                              │
                每轮计算 → 出边激活同步 → ghost 值同步 → 全局终止判断
```

输入由 `include/graph.h` 读取。项目转换脚本生成的是 **0-based** MatrixMarket；读入时没有把有向边自动对称化。`row_offsets/column_indices` 是出边 CSR，`column_offsets/row_indices` 是入边 CSR。

## 目录、可执行文件与实验状态

| 目录 | 目标文件 | 含义 | 批量实验 |
|---|---|---|---|
| `<alg>/basic` | `<alg>` | 单 GPU 基线 | 是 |
| `<alg>/tolerance_queue` | `<alg>_queue` | 单 GPU：异步标量检查 | 是 |
| `<alg>/multigpu_basic` | `<alg>_multiGPU_basic` | MPI + NCCL，无检测 | 是 |
| `<alg>/multigpu` | `<alg>_multiGPU` | MPI + NCCL，异步 CPU 检测 | 是 |

`<alg>` 分别为 `bfs`、`cc`、`kcore`、`pagerank`。构建规则在四个算法目录中的 `CMakeLists.txt`；运行时应从项目根目录 `GraphAlgorithms/` 启动，因为统一命令行会把数据集名解析为 `dataset/<dataset_id>.mtx`。

除上述主线外，`cub`、`compact_queue`、`legacy`、`threshold` 等历史实验实现已经删除；四个算法目录的 `CMakeLists.txt` 也只注册当前四类目标。

## 四个算法的基线逻辑

所有版本最多迭代 1000 轮。`active` 是工作集：基线中 `0` 表示不活跃、`1` 表示活跃；容错和多 GPU 主线中 `-1` 表示不活跃。

| 算法 | 初始值 | pull 更新（入边） | 何时沿出边激活 | 结果含义 |
|---|---|---|---|---|
| BFS | 源点 `0`，其余 `INF=100000` | `min(value[in-neighbor] + 1)` | 距离变小 | 单源距离近似/收敛值 |
| CC | `value[v]=v` | `max(value[in-neighbor])` | 标签变大 | 可达路径上传播的最大顶点标签 |
| KCore | `value[v]=初始出度`，`alive=1` | 统计仍存活的入邻居数 | 顶点首次低于 `k` 而死亡 | 最后一次统计的度数，不单独输出 `alive` |
| PageRank | `1/N` | `0.15 + 0.85 * sum(rank[in]/outdegree[in])` | 绝对变化 `>=1e-3` | 迭代 rank 值 |

### BFS

`basic/bfs_gpu.cuh` 对活跃顶点执行入边最小化，并在距离变小时激活其出邻居。它用原地更新，不是按层同步的 frontier BFS；因此实现形式更接近异步最短路松弛，但在无权图上以稳定值作为结果。

### CC

`cc` 并非“将边视为无向”的通用弱连通分量实现。它沿有向边传播最大顶点 ID：边 `u -> v` 会把 `u` 已得到的标签继续推向 `v`。若要得到通常意义的无向 CC，输入必须已包含双向边；当前入口均以 `undirected=false` 读图。

### KCore

`kcore` 统计入边中仍存活的邻居数；顶点死亡后激活其出邻居。对于双向图，这对应常见的迭代剥离；对于纯有向图，它是一个由入度支持、沿出边传播删除的有向变体。CPU 检查也把“CPU/GPU 均小于 `k`”视为等价，说明输出重点是 core 留存判定而不是被删除顶点的精确剩余度。

### PageRank

实现是原地 pull 更新，并非双缓冲 Jacobi PageRank；不处理 dangling-node 质量，也使用常数 teleport 项 `1-alpha`，而不是常见的 `(1-alpha)/N`。因此结果应理解为该项目定义的迭代分数，不能直接假定它是归一化的标准 PageRank。基础单 GPU 入口中的 CPU PageRank 参考实现为空，默认也关闭 CPU 检查。

## 单 GPU 容错：`tolerance_queue`

这四个版本共享如下流程。

1. `scoreAndMark*Kernel` 对当前活跃点计数并打分；`-1` 保持不活跃，普通点标成 `1`，关键点标成 `2`。
2. 主 kernel 在每个 CUDA block 内收集关键点和 idle lane（非活跃或越界 lane），把可用 idle lane 依次分配给关键点。
3. 活跃 lane 将 `compute_vertex` 设为自己的 `tid`，获得任务的 idle lane 将其设为对应关键点；两类 lane 随后执行同一段入邻居遍历。角色相关的 warp 分歧只保留在目标选择、结果存放、比较与主结果写回处，不再为主/副计算各执行一套邻接循环。
4. kernel 同时写入少量 `MonotonicInfo`：DMR 标志、单调性标志、变化量的和、变化次数。
5. compute stream 完成后，check stream 异步把该小结构复制到固定页锁定 host 缓冲区；后台 CPU 线程通过 SPSC 队列取任务、读取结果并判断平均变化量是否比上一轮增大。

共同路径消除了 active/idle 角色分支各自遍历邻接表造成的串行化，也去掉了两段计算之间的一次 block barrier。不同顶点的入度仍会造成图结构固有的 lane 分歧；本次优化不改变这一点。queue 与多 GPU 容错 kernel 使用相同结构。

关键点分数为：

```text
score = alpha * (outdegree / max_outdegree) + beta * value_score
```

其中 BFS 使用真实距离 `1/(1+value[v])`，CC、KCore 使用 `1/(1+abs(value))`，PageRank 使用 `abs(value)`。BFS 不能使用 `active[v]`，因为它只是 `-1/1/2` 的工作集编码。因而 `-a/-b/-t` 只控制“哪些点值得重复计算”，**不改变算法本身的数学更新规则**。

队列版本有 64 组 `MonotonicInfo` 缓冲；SPSC 队列分别管理空闲 buffer 与待检查任务。若没有空闲 buffer 或任务队列已满，当前轮仍会计算，但 CPU 检查会跳过，并使用 scratch buffer，因此覆盖并非逐轮保证。

### DMR 与异常指标的真实边界

- DMR 是 block 内、选择性的：每个 idle 线程最多重算一个关键点，且每个 block 的列表最多 256 个关键点。idle 线程不足时，部分关键点没有副本计算；全活跃的早期轮次甚至可能没有 idle 线程。
- 四个算法都原地更新全局状态，主计算与 idle 线程重算之间没有全 grid 快照隔离。不同 block 的调度可能让两次计算观察到不同阶段的邻居值，因此无故障运行也可能出现 DMR 告警；它表示“两个观测不一致”，不能单独证明发生了硬件错误。
- `monotonic` 检查的是算法预期方向：BFS/KCore 不应增大，CC 不应减小；PageRank 没有单调性检查。
- `avg_delta_increase` / `residual_increase` 仅比较相邻轮的平均变化量，是启发式异常信号，不是收敛性证明。
- PageRank 当前用 `fabs(redundant-primary) > epsilon` 判断 DMR；若位翻转产生 NaN，比较可能为假，现有检测并不保证捕获这类浮点异常。
- 检测结果只打印标志和首个异常轮次；没有重算、回滚、隔离故障 rank 或纠正输出的代码。因此这里的“容错”准确说是 **选择性冗余检测**，不是具备恢复能力的容错执行。

## 已归档的实验版本

`tolerance_compact_queue` 曾尝试把每个实际变化的 delta 压紧后传给 host，而 `tolerance_queue` 只传 `sum_abs_delta` 和 `count_update` 两个聚合量。compact 方案虽然让 CPU 能看到变化序列，却要为 16 组 host/device buffer 分别按 `num_nodes` 预分配数组，并额外保留一组 device scratch；大图上的内存代价明显更高。

目前 `compact_queue` 与 `cub`、`legacy`、`threshold` 已从源码树删除。需要回溯历史实验时应通过 git 历史查看；新增功能、基准测试和维护应以四类保留版本为准。

## 多 GPU 共用架构

BFS、CC、KCore、PageRank 的无容错和容错多 GPU 版本全部共用 `include/distributed_partition.cuh`。实际执行路径不再使用 BFS 历史上的 ceil 分区和专用 peer plan。

### 数据划分

- 一个 MPI rank 绑定本机的一张 GPU；本机 rank 由 `MPI_COMM_TYPE_SHARED` 获得，GPU 号为 `local_rank % visible_device_count`。
- 顶点按连续 ID 范围尽量均分为 `owned` 顶点。每个 rank 的局部图还追加计算所需的远端 `ghost` 顶点。
- device 上，`values`/`alive` 包含 owned + ghost；`active` 只覆盖 owned；`update` 可覆盖 owned + ghost，便于跨分区激活。
- 当前实现让**每个 rank 都读完整图，并在 host 上构造所有 rank 的分区**，最后只上传自己的子图。这简化了通信计划构建，但会复制 host 内存，也按顶点而不是按边均衡，可能造成高出度图上的负载不均。

### 每轮通信和终止

```text
owned 顶点 kernel
  → 本地 update 写入 owned/ghost 目标
  → pack remote ghost activation
  → NCCL Send/Recv，owner 将接收标志写入 next_active
  → pack owned values/alive
  → NCCL Send/Recv，更新各 rank 的 ghost cache
  → 统计本地活跃 owned 顶点
  → MPI_Allreduce(SUM) 得到全局活跃数
```

最终所有 rank 通过 `MPI_Allgatherv` 拼回 owned 部分，该最终收集不进入内部算法时间。通信使用每个 peer 的固定索引计划：activation 计划来自跨分区出边，并按远端目标顶点去重——同一 sender 到同一远端顶点无论有多少条跨分区边，每轮只传一个激活标志；value 计划来自 ghost 所属顶点。当前实现每轮同步所需 ghost 值，不做“仅变化值”的压缩。

`multigpu_basic` 只保留上述计算、通信和终止判断，分别同步：

- BFS/CC：`values`（int）；
- KCore：`alive`（int）；
- PageRank：`values`（float）。

## 多 GPU 容错：统一的异步 CPU 检查

BFS、CC、KCore、PageRank 的 `multigpu` 版本现在使用同一种异步检查协议。算法 kernel、NCCL 数据交换和每轮全局活跃数判断仍留在 MPI 主线程；CPU worker 只消费本 rank 的检测摘要，不调用 MPI。

### 每个 rank 内的流水线

1. 主线程从 free SPSC 队列取得一组检测 buffer；若暂时没有空闲项，本轮改用 scratch，仅跳过本轮 CPU 检查，不跳过图算法计算。
2. compute stream 完成打分、容错 kernel 和检测摘要写入后记录 `compute_done` event。
3. check stream 等待该 event，再把一个很小的 `CheckInfo` 异步复制到对应的页锁定 host buffer，并记录 `check_done` event。
4. 主线程把 `{iteration, buffer}` 放入 pending SPSC 队列，然后继续执行 activation/value 通信及下一轮计算。
5. 后台 worker 在无任务时睡眠；收到任务后用 `cudaEventSynchronize(check_done)` 阻塞等待最老的 D2H 完成，保存该轮摘要并归还 buffer。它不再循环调用 `cudaEventQuery` 忙轮询，因此不会持续抢占 MPI 主线程所在 CPU。

每个 rank 预分配 64 组 device/host 摘要 buffer、两组 event 和一组 device scratch。free 队列的消费者与 pending 队列的生产者都是 MPI 主线程，反方向均只有 worker，因此两条队列都保持严格 SPSC。四个入口通过 `MPI_Init_thread` 请求并检查 `MPI_THREAD_FUNNELED`；worker 明确不执行任何 MPI 调用，所以不需要 `MPI_THREAD_MULTIPLE`。

若任一 rank 在某轮拿不到 free buffer，该轮会在该 rank 使用 scratch，最终 `MPI_MIN(done)` 使所有 rank 都跳过整轮检测结果。因此本轮的 DMR、单调性和 delta 证据都会省略；输出 `DMR=0` 只表示所有实际汇总的轮次未报告不一致，不表示 100% 迭代覆盖。程序会同时打印 `checked/total` 与 `skipped`，用于判断本次报告的覆盖范围。

### 结束后的跨 rank 汇总

GPU 迭代结束后，主线程发出停止请求并 join worker；这段单独计为 checker drain。随后主线程才按迭代编号执行 MPI 汇总：只有所有 rank 都完成了某轮检查时，该轮才参与统计，再分别汇总 delta、更新次数、DMR 和单调性标志。这一整段是 post-check aggregation，其中 MPI 调用时间还是一个可解释子集。PageRank 只报告 residual increase，不做单调性判断。

D2H 摘要复制和 CPU 判断不会成为每轮同步点。每轮为了终止判断进行的 compute-stream 同步及 `MPI_Allreduce` 仍存在。四个算法的基础/容错版使用相同分区、peer plan、NCCL 顺序、终止判断和结果收集；容错版额外执行评分、选择性 DMR、摘要 D2H、CPU 消费与结束后汇总。

### Benchmark 计时字段

所有主线入口都输出一条 `BENCHMARK_TIMING` 和一条 `BENCHMARK_ITERATIONS`。多 GPU 记录由 `include/distributed_benchmark.cuh` 统一生成，局部阶段定义如下：

| 字段 | 精确范围 |
|---|---|
| `main_loop_ms` | `steady_clock` 测得的完整迭代循环：CUDA、NCCL、每轮主线程控制和终止 `MPI_Allreduce` |
| `gpu_compute_ms` | CUDA stream 上 NCCL 区间之外的 CUDA 工作累计值；包括算法/评分/计数 kernel 和必要 copy，不等于纯 kernel |
| `graph_kernel_ms` | 每轮用同 stream CUDA event 只包住核心图 kernel 后的累计值：basic 为普通 pull kernel，容错版为包含主计算与选择性冗余计算的 dual/tolerance kernel；不含评分、状态维护、通信或 CPU 检测 |
| `nccl_exchange_ms` | 每轮 activation/value 的 pack、NCCL send/recv、unpack |
| `mpi_sync_ms` | 主循环终止判断 `MPI_Allreduce` 的阻塞调用时间；含快 rank 等待慢 rank |
| `cpu_check_drain_ms` | 主循环结束后，从发出 worker 停止请求到 join 返回；只包含尚未隐藏的检测工作 |
| `postcheck_total_ms` | worker join 后按迭代汇总检测结果的完整阶段 |
| `postcheck_mpi_ms` | post-check 内 MPI 调用时间，是 `postcheck_total_ms` 的子集，不能再次相加 |

兼容字段 `gpu_main_ms == main_loop_ms`，`cpu_check_tail_ms == cpu_check_drain_ms`。schema v4 绘图要求新的 kernel 字段，v3 及更早 CSV 不能与新结果混画。每个 rank 的不重叠内部总耗时是：

```text
algorithm_total = main_loop + cpu_check_drain + postcheck_total
```

基础多 GPU 版的 drain 和 post-check 都为 0。容错版的结果汇总不再伪装成 CPU drain，这正是旧图里“两 GPU drain 明显变大”的主要统计问题。

`communication_ms` 是解释性调用时间：

```text
communication = nccl_exchange + mpi_sync + postcheck_mpi
```

它已经分别处于 main loop 或 post-check total 内，绝不能再叠加到 `algorithm_total`。同理，`postcheck_mpi_ms` 也不能加到 `postcheck_total_ms` 上。

每个局部字段同时输出 rank max 和真实 rank 算术平均 `rank_*_avg_ms`。核心 kernel 因而对应 rank-max `graph_kernel_ms` 和真实均值 `rank_graph_kernel_avg_ms`；后者用于两 GPU kernel 对比图。rank max 的各组件可能来自不同 rank，所以 max communication 不保证等于几个 max 组件之和；作图分解必须使用 rank-mean 字段。内部总耗时先在每个 rank 内相加，再分别计算 `rank_algorithm_total_avg_ms` 与 `rank_algorithm_total_max_ms`，不能把三个独立最大值直接相加。

单 GPU 二进制保留原 CUDA-event 主区间，并新增单独的核心 `graph_kernel_ms`。实验 runner 在 schema v4 中把主区间映射到 `main_loop_ms/gpu_compute_ms`，把原 `cpu_check_tail_ms` 映射为 drain，post-check 置 0。核心 kernel 数值使用 `include/cuda_event_timer.cuh` 中的 CUDA event 累加器；NVTX 只用于 profiler 展示，不作为 CSV 数值来源。严格开销百分比始终只在相同 GPU 模式、相同迭代数的 basic/容错 pair 内计算。四柱绝对时间可用于总体观察，但单 GPU 与多 GPU 内部字段的边界仍不完全相同；完整端到端范围应看 Python 的 `wall_time_ms`。

`graph_kernel_ms` 是 `gpu_compute_ms` 的子集，不能与其相加。多 GPU 初始化 ghost 交换、最终结果 D2H、`MPI_Allgatherv` 和计时报告自身的归约均排除在内部总耗时之外。

### BFS 已与其他算法统一的部分

BFS 两个多 GPU 版本现在都使用共享的平衡连续分区、owned + ghost 布局、去重 activation plan 和 value plan。pull BFS 的首轮工作集也统一为 `source + source 的出邻居`：source 自身保留用于一致性，出邻居才能在第一轮通过入边观察到 source。两版每轮的 activation/value 通信与全局终止判断顺序一致；剩余差异就是容错版的评分、DMR 和异步检测语义。

## 参数与使用建议

主线入口统一接受：

```text
<dataset_id> [-s src] [-k k] [-a alpha] [-b beta] [-t threshold] [-n]
```

- `-s`：仅 BFS 使用；
- `-k`：仅 KCore 使用；
- `-a/-b/-t`：仅容错/检测选择使用；基础版本会接受但忽略；
- `-n`：关闭算法结束后的完整 CPU 参考结果检查；大图 benchmark 应使用。它不会关闭容错版内部的异步摘要检测。

建议的对比顺序是：先用小的、已双向化的数据集验证 `basic` 与 `tolerance_queue` 的结果；再用同一 `mpirun -np`、同一数据集比较 `multigpu_basic` 与 `multigpu`。历史实验版本已退出构建主线，不应直接混入当前性能对比。

## 维护时最值得记住的限制

1. 所有检测版都是检测，不恢复；出现标志不会改变算法结果或返回非零退出码。
2. DMR 覆盖依赖同 block 的 idle 线程，不能视为所有关键顶点都被双算。
3. 多 GPU 通信模型依赖完整图被每个 rank 读入和 host 侧全量分区，超大图的 host 内存与分区构建成本很高。
4. CC/KCore 的通常无向语义要求输入中已有反向边；项目读图不会自动补边。
5. PageRank 的公式、更新方式和 dangling-node 处理不同于标准归一化 PageRank，跨实现或论文基线比较前应先统一定义。
6. BFS 已与其他算法共享分区、通信和初始工作集；若同一两 GPU pair 的迭代数仍不同，应先按 bug 或非确定性调查，实验脚本不会把该 repeat 用于开销百分比。
7. 原地更新会使选择性 DMR 出现无故障不一致；若以后需要把告警解释为可靠故障判据，应改成快照或双缓冲计算，并重新定义检测时序。
8. 公共多 GPU 路径目前假定 `MPI rank 数 <= 顶点数`；否则可能形成空 owned 分区并触发零 block launch。当前实验固定两 rank，数据集规模远大于 2。
