# MPI/NCCL 多 GPU 启动说明

本项目的多节点/多 GPU 版本采用 **MPI rank-per-GPU** 模式：

- 1 个 MPI rank 对应 1 张 GPU。
- 同一台机器上启动多个 rank，就会使用该机器上的多张 GPU。
- 多台机器一起启动时，MPI 负责拉起进程，NCCL 负责 GPU 间通信。

## 可执行文件

以 BFS 为例：

```bash
./build/bin/bfs/bfs_multiGPU flickr -s 9006          # 多节点容错版本
./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006    # 多节点无容错版本
```

其它算法类似：

```bash
./build/bin/cc/cc_multiGPU
./build/bin/cc/cc_multiGPU_basic

./build/bin/kcore/kcore_multiGPU
./build/bin/kcore/kcore_multiGPU_basic

./build/bin/pagerank/pagerank_multiGPU
./build/bin/pagerank/pagerank_multiGPU_basic
```

## 单机多卡启动

例如本机有 2 张 GPU：

```bash
mpirun -np 2 \
  ./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006
```

这里：

- `-np 2` 表示启动 2 个 MPI rank。
- 每个 rank 自动选择本机上的一张 GPU。
- rank 0 使用 local rank 0 对应的 GPU，rank 1 使用 local rank 1 对应的 GPU。

容错版本：

```bash
mpirun -np 2 \
  ./build/bin/bfs/bfs_multiGPU flickr -s 9006
```

## 多机多卡启动

例如：

- 本机：`10.4.1.139`
- 远端：`g2c14`，IP 为 `10.4.1.114`
- 每台机器使用 1 张 GPU

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
  ./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006
```

如果每台机器 2 张 GPU：

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
  ./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006
```

## 参数解释

`-np 4`

启动 4 个 MPI rank。当前实现里就是使用 4 张 GPU。

`-H 10.4.1.139:2,g2c14:2`

指定 rank 分布。这里表示：

- 在 `10.4.1.139` 上启动 2 个 rank。
- 在 `g2c14` 上启动 2 个 rank。

rank 顺序按 `-H` 里的主机顺序分配，所以这个例子里通常是：

- rank 0、rank 1 在 `10.4.1.139`
- rank 2、rank 3 在 `g2c14`

`--wdir /tmp/GraphAlgorithm`

指定所有机器上的工作目录。多机运行时，每台机器都必须能在这个路径下找到同名可执行文件和数据集。

可以用软链接保证路径一致：

```bash
ln -sfn /home/huangxy/Projects/GraphAlgorithm /tmp/GraphAlgorithm
ssh g2c14 'ln -sfn /workplace/home/huayunpeng/Projects/GraphAlgorithm /tmp/GraphAlgorithm'
```

`--mca pml ob1 --mca btl self,tcp`

强制 MPI 使用 TCP 通信，避免 OpenIB/IB 配置不一致导致 MPI rank 互相不可达。

`--mca btl_tcp_if_include 10.4.1.0/24`

指定 MPI 使用 `10.4.1.x` 这个网段通信。这个比写网卡名更稳，因为两台机器的网卡名可能不同。

`-x NCCL_SOCKET_IFNAME=eno1,ens5f0`

指定 NCCL 使用哪些网卡。你的本机 `10.4.1.139` 在 `eno1` 上，远端 `10.4.1.114` 在 `ens5f0` 上，所以这里写两个网卡名。

`-x NCCL_IB_DISABLE=1`

禁用 NCCL IB，强制走 socket。当前两台机器如果 IB 配置不统一，这个更稳。

## 启动前检查

确认 SSH 能免密或正常登录：

```bash
ssh g2c14 hostname
```

确认两边路径一致：

```bash
ls /tmp/GraphAlgorithm/build/bin/bfs/bfs_multiGPU_basic
ssh g2c14 'ls /tmp/GraphAlgorithm/build/bin/bfs/bfs_multiGPU_basic'
```

确认两边数据集一致：

```bash
ls /tmp/GraphAlgorithm/dataset/flickr.mtx
ssh g2c14 'ls /tmp/GraphAlgorithm/dataset/flickr.mtx'
```

确认 MPI 能拉起远端进程：

```bash
mpirun -np 2 \
  -H 10.4.1.139:1,g2c14:1 \
  --mca plm_rsh_agent ssh \
  bash -c 'echo host=$(hostname) rank=$OMPI_COMM_WORLD_RANK local_rank=$OMPI_COMM_WORLD_LOCAL_RANK'
```

## 常见问题

### 远端找不到可执行文件

报错类似：

```text
mpirun was unable to launch the specified application
Executable: ./build/bin/bfs/bfs_multiGPU
Node: g2c14
```

通常是两台机器的工作目录不同。使用 `--wdir /tmp/GraphAlgorithm`，并在两边建立同名软链接。

### MPI rank 互相不可达

报错类似：

```text
At least one pair of MPI processes are unable to reach each other
BTLs attempted: self openib
```

可以强制 MPI 走 TCP：

```bash
--mca pml ob1 \
--mca btl self,tcp \
--mca btl_tcp_if_include 10.4.1.0/24
```

### NCCL 选错网卡

如果 NCCL 日志里出现了不希望使用的网卡，比如远端用了 `ibs9:10.4.2.114`，可以指定网卡：

```bash
-x NCCL_SOCKET_IFNAME=eno1,ens5f0
```

也可以排除某些网卡：

```bash
-x NCCL_SOCKET_IFNAME=^ibs9,docker0,lo
```

### NCCL 版本不一致

两台机器的 NCCL 版本需要一致。检查方式：

```bash
strings /path/to/nccl/lib/libnccl.so.2 | grep -i "NCCL version" | head
```

如果版本不同，跨机 NCCL 可能初始化失败或通信时报错。

## 推荐命令模板

单机 2 卡：

```bash
mpirun -np 2 \
  ./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006
```

双机各 1 卡：

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
  ./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006
```

双机各 2 卡：

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
  ./build/bin/bfs/bfs_multiGPU_basic flickr -s 9006
```
