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
2. 主 kernel 仍由所有活跃线程计算正常更新。
3. 同一 CUDA block 内的 idle 线程（非活跃线程或越界线程）依次重算关键点；关键主结果和副结果不同则置 `dmr_error_flag`。
4. kernel 同时写入少量 `MonotonicInfo`：DMR 标志、单调性标志、变化量的和、变化次数。
5. compute stream 完成后，check stream 异步把该小结构复制到固定页锁定 host 缓冲区；后台 CPU 线程通过 SPSC 队列取任务、读取结果并判断平均变化量是否比上一轮增大。

关键点分数为：

```text
score = alpha * (outdegree / max_outdegree) + beta * value_score
```

其中 CC、KCore 使用 `value_score = 1/(1+abs(value))`，PageRank 使用 `abs(value)`；BFS 使用活动数组中的距离型值 `1/(1+active[v])`。因此 `-a/-b/-t` 控制的是“哪些点值得重复计算”，**不改变算法本身的数学更新规则**。

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

除 `bfs/multigpu` 的旧式专用实现外，多 GPU 版本共用 `include/distributed_partition.cuh`。

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

最终所有 rank 通过 `MPI_Allgatherv` 拼回 owned 部分；时间以 CUDA event 测量。通信使用每个 peer 的固定索引计划：activation 计划来自跨分区出边，value 计划来自 ghost 所属顶点。当前实现每轮同步所需 ghost 值，不做“仅变化值”的压缩。

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
5. 后台 worker 查询 `check_done`，保存该轮的 DMR、单调性、变化量之和与变化次数，再把 buffer 归还 free 队列。

每个 rank 预分配 64 组 device/host 摘要 buffer、两组 event 和一组 device scratch。free 队列的消费者与 pending 队列的生产者都是 MPI 主线程，反方向均只有 worker，因此两条队列都保持严格 SPSC。四个入口通过 `MPI_Init_thread` 请求并检查 `MPI_THREAD_FUNNELED`；worker 明确不执行任何 MPI 调用，所以不需要 `MPI_THREAD_MULTIPLE`。

若任一 rank 在某轮拿不到 free buffer，该轮会在该 rank 使用 scratch，最终 `MPI_MIN(done)` 使所有 rank 都跳过整轮检测结果。因此本轮的 DMR、单调性和 delta 证据都会省略；输出 `DMR=0` 只表示所有实际汇总的轮次未报告不一致，不表示 100% 迭代覆盖。程序会同时打印 `checked/total` 与 `skipped`，用于判断本次报告的覆盖范围。

### 结束后的跨 rank 汇总

GPU 迭代结束后，主线程先排空任务并 join worker，再按迭代编号执行 MPI 汇总。只有所有 rank 都完成了某轮检查时，该轮才参与统计；随后分别汇总 delta 和更新次数，并对 DMR、单调性标志做逻辑或。平均变化量是在全局汇总后按轮比较的，PageRank 只报告 residual increase，不做单调性判断。

这使 D2H 摘要复制和 CPU 判断不再成为每轮的同步点。每轮为了终止判断而进行的 compute stream 同步及 `MPI_Allreduce` 仍然存在，NCCL 通信逻辑也没有改变。四个版本统一在 worker 建立后开始计时、在 worker 排空前停止计时；rank 0 打印各 rank 中的最大值。因此 GPU time 表示最慢 rank 的 GPU 主流水线时间，不包含最终 CPU/MPI 检测汇总等待。

### BFS 与其他算法仍有的结构差异

`bfs/multigpu/bfs_tolerance_multiGPU.cuh` 仍使用自己的 `GpuSubgraphHost/PeerPlan` 和按 `ceil(N/ranks)` 划分代码；CC、KCore、PageRank 使用 `include/distributed_partition.cuh`。BFS 检测版还只从源点及其出邻居构造初始 active 集，而 `bfs/multigpu_basic` 首轮将所有 owned 顶点设为 active。此次重构统一的是异步检测协议，没有改变这些既有算法和分区语义。

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
6. BFS 多 GPU 检测版仍有独立的分区与初始 active 语义；比较 BFS 两个多 GPU 版本时，不能把耗时或迭代数差异全部归因于检测开销。
7. 原地更新会使选择性 DMR 出现无故障不一致；若以后需要把告警解释为可靠故障判据，应改成快照或双缓冲计算，并重新定义检测时序。
8. CC/KCore/PageRank 的公共分区路径目前假定 `MPI rank 数 <= 顶点数`；否则可能形成空 owned 分区并触发零 block launch。BFS 专用路径对空分区做了额外保护。
