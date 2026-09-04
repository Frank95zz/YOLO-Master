# D1 WP8：完整 COCO 2017 三组双卡正式训练方案（待审核）

## 1. 当前状态

WP8 的完整缓存和既有训练工程准备已经完成，正式训练与正式评测尚未启动。本文是新的启动前实验合同：六张 NVIDIA A40 被划分为三组，每组两张 GPU，同时运行三个独立 seed 的完整 COCO 2017 P0 实验。

当前状态：**方案已改为三组双卡并行，但代码和配置仍是旧的六卡单实验合同。在第 9 节所列改造、测试和启动门禁完成并经审核确认前，不得启动正式训练。**

已完成：

- 完整 COCO 2017 train2017/val2017 DINOv3 特征缓存；
- 完整 SHA256 校验、统一索引和本地缓存副本；
- D1 Adapter、三个 LatentMixture、Detect、aux loss、checkpoint 和恢复链路；
- WP7 在线/缓存一致性、32 图过拟合和 COCO8 一轮闭环；
- 旧版六卡训练入口、遥测、汇总和训练后页缓存释放工具。

尚未完成：

- 把正式训练合同从六卡单任务改为每任务两卡；
- 增加 seed 0/1/2 的三任务编排、隔离、恢复和汇总；
- 三组并发短基准和实测 ETA；
- 三个 seed 的 100 epochs 完整训练；
- 三次 COCO val2017 最终评测和跨 seed 汇总。

## 2. 本阶段目标与边界

WP8 完成 D1 的正式 P0 实验：使用冻结 DINOv3 ViT-S/16 生成的离线特征，只训练 Adapter、三个 LatentMixture 和 Detect Head，在完整 COCO val2017 上报告检测精度、训练成本及结果波动。

三个任务仅改变随机种子，不能改变模型、数据、batch、优化器、学习率、epoch、增强、aux 系数或缓存。这样三组结果可以计算均值和标准差，避免把单次随机初始化当成稳定结论。

WP8 不负责：

- P1 的同参数量从零训练检测器；
- 第二数据集对照和“GPU 时降低至少 50%”结论；
- P2 的 DINOv3/SigLIP2 对比或大规模 aux 超参数扫描；
- 根据中途精度临时修改正式配方；
- 部署、导出或实时推理优化。

## 3. 双卡并行方案的含义

### 3.1 固定 batch 口径

本方案中的 `batch` 使用 Ultralytics 标准语义，表示单个 DDP 任务的全局 batch：

```text
每任务 world_size = 2
每卡 batch = 8
每任务 global batch = 2 x 8 = 16
梯度累积 = 1
```

三组同时运行时，服务器每个时刻最多处理 `3 x 16 = 48` 张图片，但三个任务拥有各自的模型、优化器和梯度，不能把它们合并解释成一个 global batch 48 的训练。

### 3.2 与旧六卡方案的区别

| 项目 | 旧方案 | 新方案 |
| --- | ---: | ---: |
| 每个任务使用 GPU | 6 | 2 |
| 每卡 batch | 8 | 8 |
| 每任务 global batch | 48 | 16 |
| 同时运行任务 | 1 | 3 |
| 每任务每 epoch optimizer step | 2,465 | 7,393 |
| 正式 seed | 仅 0 | 0、1、2 |

`ceil(118287 / 16) = 7393`，所以每个新任务在 100 epochs 中约执行 739,300 次同步 optimizer update。global batch 从 48 改成 16 会改变梯度噪声、更新次数和最终优化轨迹，因此新方案是一个新的实验合同，不能与未实际运行的旧方案混写。

### 3.3 “更快”的准确范围

三组双卡并行可以把三个双卡实验从串行变成并行，理想情况下使三 seed 实验墙钟时间缩短到串行双卡方案的约三分之一。但它不会让单个实验比六卡训练更快：每个双卡任务每个 epoch 需要处理同样的 118,287 个样本，却只有两张 GPU，预计单任务时间约为旧六卡方案的三倍，并且 7,393 次 DDP 同步会增加额外开销。

如果目标只是尽快得到一个 seed 的 P0 模型，旧六卡方案更快；如果目标是在固定 global batch 16 下同时得到三个可统计的独立结果，本方案更合适。

## 4. 数据与缓存合同

### 4.1 COCO 2017

| Split | 图片数 | 用途 |
| --- | ---: | --- |
| train2017 | 118,287 | 每个 seed 的完整训练 |
| val2017 | 5,000 | 每 epoch 验证与最终评测 |

使用官方 split，不二次随机划分。输入几何沿用 WP0：640x640、确定性居中 LetterBox、无二次 resize/crop。

### 4.2 DINOv3 缓存

Teacher 固定为 `facebook/dinov3-vits16-pretrain-lvd1689m`，权重 SHA256：

```text
4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d
```

缓存合同 SHA256：

```text
6bfda0e13bde01001c3f3f2d72631a2401fb9a77b146d6fb2794303e379e47a7
```

| Split | 样本 | Tensor | Shard | Cache bytes | 内容摘要 |
| --- | ---: | ---: | ---: | ---: | --- |
| train2017 | 118,287 | 354,861 | 204 | 436,232,631,192 | `0224bbd302faf62b39e914be11f4d3469f7567791ce48994a9fc0f378e070cbe` |
| val2017 | 5,000 | 15,000 | 12 | 18,439,541,408 | `cd5840678d79550482aaf4110b9e68b8c22819f2f6b31e0af40f0e5b7476fa1b` |

合计 123,287 个样本、369,861 个 tensor、216 个 shard，约 423.5 GiB。每个样本包含：

```text
block4  [384,40,40] FP16
block8  [384,40,40] FP16
block12 [384,40,40] FP16
```

完整证据见 [`manifests/wp8-full-cache.json`](manifests/wp8-full-cache.json)。三个任务共享同一份只读缓存，不复制三份。preflight 比对完整校验报告、索引、shard 集合和文件大小，不在每次启动前重新哈希 423.5 GiB。

## 5. 模型与训练边界

模型配置：`ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml`。

训练数据流：

```text
三个缓存 block 特征
  -> DINOFeaturePyramidAdapter
  -> 每尺度三个 P3/P4/P5 候选
  -> P3/P4/P5 LatentMixture
  -> YOLO26 Detect
  -> detection loss + latent aux loss
```

模型共有 3,542,567 个参数，全部属于下游 Adapter、LatentMixture 和 Detect。正式训练不实例化 DINOv3 Teacher，optimizer、EMA 和 checkpoint 均不得包含 `teacher` 或 `dinov3` 参数。

固定 aux 配置：

| 参数 | 值 |
| --- | ---: |
| `balance_loss_coeff` | 0.01 |
| `router_z_loss_coeff` | 0.001 |
| `latent_aux_gain` | 0.1 |
| `mixture_aux_budget` | 3.0 |

## 6. 三组正式实验矩阵

三个任务除 seed 和物理 GPU 位置外完全一致：

| 任务 | seed | 物理 GPU | 进程内逻辑 GPU | DDP 端口 | 运行标识 |
| --- | ---: | --- | --- | ---: | --- |
| A | 0 | 0、1 | 0、1 | 29518 | `wp8-p0-s0-<commit>` |
| B | 1 | 2、3 | 0、1 | 29528 | `wp8-p0-s1-<commit>` |
| C | 2 | 4、5 | 0、1 | 29538 | `wp8-p0-s2-<commit>` |

每组通过独立的 `CUDA_VISIBLE_DEVICES` 只看见自己的两张 A40；每组使用独立 master port、运行目录、报告目录、PID、状态文件和日志。不得让三个任务写入同一个 `results.csv`、checkpoint 目录或 JSON 报告。

seed 控制模型初始化、DistributedSampler 和框架随机源。三个 seed 都是预先登记的正式结果：

- 不因某个 seed 精度较低而删除；
- 不从三个 seed 中只挑最高值作为主结果；
- 每个 seed 分别报告，主汇总报告 `mean +/- std`；
- seed 0 同时作为与既有 WP0-WP7 合同连续的预注册追踪结果。

## 7. 固定训练配置

正式基础配置仍由 [`wp8-formal-coco2017.yaml`](../../ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml) 管理，但启动前必须按第 9 节改为双卡合同。

| 项目 | 固定值 |
| --- | --- |
| 并发任务 | 3 |
| 每任务 GPU | 2 x NVIDIA A40，单机 DDP |
| seed | 任务 A/B/C 分别为 0/1/2 |
| deterministic | true |
| epochs | 100 |
| 每任务 global batch | 16 |
| 每卡 batch | 8 |
| `nbs` | 16 |
| 梯度累积 | 1 |
| workers | 每 rank 4；每任务 8；全机共 24 |
| DataLoader prefetch | 每 worker 1 |
| 每 worker 打开 shard 上限 | 4 |
| precision | AMP |
| AMP init scale | 16 |
| optimizer | AdamW |
| `lr0` / `lrf` | 0.001 / 0.01 |
| scheduler | cosine |
| momentum | 0.9 |
| weight decay | 0.0005 |
| warmup | 3 epochs |
| patience | 100，不主动提前结束 100 epochs 合同 |
| checkpoint | `last.pt`、`best.pt`，每 10 epochs 额外保存 |
| validation | 每 epoch 完整 val2017；结束后严格加载最佳健康 checkpoint 再评测 |
| augmentation | 颜色、仿射、翻转、mosaic、mixup、copy-paste、erasing 和 multi-scale 全部关闭 |
| pretrained student | false |

`nbs=16` 与 global batch 16 相同，因此不做额外梯度累积。`lr0=0.001` 在新合同中保持固定，不按旧 global batch 48 做自动线性缩放；这意味着新旧合同的优化轨迹不可直接等价比较。后续 P1 基线若要公平比较，也必须使用 global batch 16、相同 epoch、优化器、学习率调度和 seed 集合。

## 8. 资源隔离与并发 I/O

### 8.1 GPU、CPU 和端口

- 每张 GPU 只允许一个训练 rank，不在同一卡上叠加任务；
- 三个任务使用不同 DDP master port；
- 启动前读取容器允许的 CPU 列表，再把可用 CPU 尽量均分给三组，保留少量核心给系统和监控；
- 即使不显式绑定 CPU，总 DataLoader worker 也固定为 24，不继续扩大；
- 三组启动间隔 60 秒，避免同时初始化 CUDA、读取索引和创建 worker。

### 8.2 共享缓存

特征缓存以只读方式共享。三组的文件页缓存由同一容器内核统一管理，页面通常不会因为三个进程读取而物理复制三份，但不同 seed 的随机采样顺序会扩大同时活跃的 shard 集合并增加随机 I/O。

监控时必须区分：

- anonymous/RSS：模型、optimizer、worker 和 Python 实际匿名内存；
- file cache：可回收的缓存文件页；
- GPU allocated/reserved：每个 rank 的显存；
- data wait ratio：GPU 等待缓存读取的比例。

不得为了让内存数字看起来更低而在训练中执行全局 `drop_caches` 或内存压力程序。

### 8.3 清理时机

三个任务共享同一份 423.5 GiB 特征缓存，因此不能在某一个任务先结束时立即驱逐共享文件页，否则会降低另外两个任务的读取性能。`run_with_cache_cleanup.py` 应由三任务 campaign supervisor 在三个任务都进入终态后统一调用一次。

如果某个任务失败而另外两个仍在运行，只记录失败并保留其 checkpoint，不执行共享缓存清理。待其他任务完成或明确停止后，再统一释放页缓存。

## 9. 正式启动前必须完成的代码改造

当前 `wp8-formal-coco2017.yaml` 和 `run_wp8_train.py` 仍校验 `world_size=6`、`batch=48`、`nbs=48` 和设备 `0,1,2,3,4,5`。直接按本文命令运行会被旧门禁拒绝，不能绕过校验。启动前必须完成以下改造并单独提交：

1. 将正式合同升级为新的 schema，固定每任务 `world_size=2`、逻辑设备 `0,1`、`batch=16`、`nbs=16`；
2. 允许每个运行从受控参数注入 seed 0/1/2，并把最终 seed 写入 identity，不能只改变目录名；
3. preflight 在任务级验证两张可见 A40，在 campaign 级验证六张物理 A40 的分组完整且互斥；
4. 增加三任务 campaign 编排入口，统一创建目录、启动、等待、恢复、汇总和最终缓存清理；
5. 为三组分配唯一 master port、PID、日志、状态和 checkpoint 路径；
6. 保留单任务恢复入口，只补跑失败 seed，不重跑已通过 seed；
7. 汇总器同时验证三个 run identity，只在模型、配置、缓存和代码 commit 一致且 seed 集合恰为 `{0,1,2}` 时生成跨 seed 报告；
8. 更新测试，覆盖 batch 拆分、GPU 配对、端口与路径无冲突、部分失败恢复、三任务汇总和清理时机；
9. 运行 WP0-WP8、checkpoint、缓存和 recovery 回归测试以及 `git diff --check`；
10. 在干净 commit 上生成 preflight，正式运行期间不得修改训练代码或配置。

## 10. 启动门禁

### 10.1 静态 preflight

正式启动前必须同时满足：

1. Git 工作区干净，三个任务绑定同一完整 commit；
2. 六张 GPU 均为空闲 NVIDIA A40，每张显存满足启动余量；
3. GPU 配对完整且互斥，端口未被占用；
4. train/val 缓存样本数、合同摘要、内容摘要、shard 集合和文件大小与完整校验报告一致；
5. COCO 官方路径列表分别为 118,287 和 5,000；
6. 每个任务的 identity 都包含 seed、global/per-GPU batch、world size、代码和配置 SHA256、缓存摘要；
7. 三个运行目录均为空或与自身 identity 完全一致，绝不覆盖其他实验；
8. 外部工作区有足够空间保存三组 checkpoint、日志和报告；
9. 容器匿名内存没有未知训练进程占用；
10. 配置不包含密码、令牌或只能在当前服务器成立的隐式路径。

### 10.2 三组并发短基准

正式 100 epochs 前，使用与正式运行完全相同的三组 GPU、batch、workers、AMP 和缓存读取路径，运行一个独立的 200-step 并发基准。基准目录不得作为正式 checkpoint 恢复源。

通过条件：

- 六张 GPU 均有且仅有一个训练 rank；
- 三组均完成 200 个 optimizer step，loss 和梯度均有限；
- 无 CUDA OOM、主机 OOM、NCCL、DataLoader、safetensors 或文件描述符错误；
- 每张 GPU 峰值显存至少保留 4 GiB 余量；
- 三组读取同一缓存时吞吐稳定，没有持续增长的 data wait；
- 报告每组 step/s、images/s、data wait、GPU 利用率、显存和匿名/file cache 内存；
- 根据最慢一组的稳态速度计算完整 100 epochs ETA。

该基准预计超过 3 分钟。启动并确认三个任务正常后应退出会话，用户检查完成状态后再继续。若实测 ETA、I/O 或内存不满足要求，只报告瓶颈并等待确认，不自动启动正式训练。

## 11. 审核通过后的目标执行方式

以下命令描述第 9 节代码改造完成后的目标接口，当前旧版入口尚不支持三组 campaign，不能提前执行。

### 11.1 公共路径

```bash
export D1_REPO=/root/yolo-master/repo
export D1_WORKSPACE=/data/yingxi/yolo-master-d1
export D1_CACHE_ROOT=/root/yolo-master/datasets/d1_feature_cache
export D1_COMMIT=$(git -C "$D1_REPO" rev-parse --short HEAD)
export D1_CAMPAIGN="wp8-p0-b16-s012-$D1_COMMIT"
```

### 11.2 生成三个 preflight

```bash
cd "$D1_REPO"

python scripts/d1/run_wp8_campaign.py \
  --workspace "$D1_WORKSPACE" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --campaign "$D1_CAMPAIGN" \
  prepare
```

### 11.3 并发基准

```bash
nohup python scripts/d1/run_wp8_campaign.py \
  --workspace "$D1_WORKSPACE" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --campaign "$D1_CAMPAIGN" \
  benchmark --steps 200 \
  >"$D1_WORKSPACE/logs/$D1_CAMPAIGN-benchmark.log" 2>&1 &

echo $! >"$D1_WORKSPACE/logs/$D1_CAMPAIGN-benchmark.pid"
```

基准完成后先汇报实测 ETA并等待确认。不得由 benchmark 自动串联正式训练。

### 11.4 三组正式训练

```bash
nohup python scripts/d1/run_wp8_campaign.py \
  --workspace "$D1_WORKSPACE" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --campaign "$D1_CAMPAIGN" \
  train \
  >"$D1_WORKSPACE/logs/$D1_CAMPAIGN.log" 2>&1 &

echo $! >"$D1_WORKSPACE/logs/$D1_CAMPAIGN.pid"
```

campaign supervisor 内部固定执行：

```text
A: CUDA_VISIBLE_DEVICES=0,1  seed=0  nproc_per_node=2  master_port=29518
B: CUDA_VISIBLE_DEVICES=2,3  seed=1  nproc_per_node=2  master_port=29528
C: CUDA_VISIBLE_DEVICES=4,5  seed=2  nproc_per_node=2  master_port=29538
```

每组启动间隔 60 秒。supervisor 必须等待所有任务，保存各自退出码；仅在三个任务都退出后调用一次共享缓存页释放。

### 11.5 部分失败恢复

```bash
python scripts/d1/run_wp8_campaign.py \
  --workspace "$D1_WORKSPACE" \
  --campaign "$D1_CAMPAIGN" \
  status

nohup python scripts/d1/run_wp8_campaign.py \
  --workspace "$D1_WORKSPACE" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --campaign "$D1_CAMPAIGN" \
  resume --seeds 1 \
  >"$D1_WORKSPACE/logs/$D1_CAMPAIGN-resume-s1.log" 2>&1 &
```

恢复时只允许更换空闲物理 GPU 对，不得改变 seed、模型、数据、缓存、batch、学习率或代码 commit。`last.pt` 必须先通过健康和 identity 检查。

## 12. 状态检查

campaign 级状态：

```bash
cat "$D1_WORKSPACE/logs/$D1_CAMPAIGN.status"
pid=$(cat "$D1_WORKSPACE/logs/$D1_CAMPAIGN.pid")
kill -0 "$pid" 2>/dev/null && echo RUNNING || echo FINISHED_OR_FAILED
```

三个任务状态：

```bash
python "$D1_REPO/scripts/d1/run_wp8_campaign.py" \
  --workspace "$D1_WORKSPACE" \
  --campaign "$D1_CAMPAIGN" \
  status
```

训练进度和资源：

```bash
tail -n 5 "$D1_WORKSPACE/runs/$D1_CAMPAIGN/seed0/results.csv"
tail -n 5 "$D1_WORKSPACE/runs/$D1_CAMPAIGN/seed1/results.csv"
tail -n 5 "$D1_WORKSPACE/runs/$D1_CAMPAIGN/seed2/results.csv"
nvidia-smi
```

状态汇总必须显示每个 seed 的 PID、epoch、退出码、最后更新时间、GPU 对和 checkpoint 状态。不能仅凭父进程退出判断训练成功，最终以三份验收报告为准。

## 13. 故障处理与恢复规则

- 单个 seed 失败且其他 seed 正常：保留其他任务继续运行，只停止故障任务的进程组；
- 三组同时出现缓存、文件系统、主机 OOM 或相同 NaN：视为系统性故障，停止全部任务并保留证据；
- CUDA OOM：正式配置已通过并发基准后不应发生；发生时记录失败，不能在原 run 中临时减 batch；
- 容器中断或节点回收：恢复后核对 commit、identity、缓存和健康 `last.pt`，再按 seed 单独续跑；
- 某个 seed 已完成 100 epochs：不得重复训练或被失败 seed 的恢复操作覆盖；
- 不允许从较早 checkpoint 重跑后择优拼接曲线；
- `SIGKILL`、容器重启或断电时无法执行退出清理，恢复后可在所有训练停止时单独运行缓存释放工具。

## 14. 评测与报告

### 14.1 每个 seed 必须报告

- COCO `mAP50-95`、`mAP50`、Precision 和 Recall；
- box、class、DFL 和全部 latent/mixture loss 曲线；
- 100 epochs 是否完整、训练/验证墙钟时间和每 epoch 时间；
- 两卡峰值显存、GPU 利用率、step/s、images/s 和 data wait ratio；
- P3/P4/P5 Router 参数变化、平均概率、熵、balance、z-loss 和 residual gain；
- `best.pt`、`last.pt` SHA256、严格重载结果和 Teacher 参数计数；
- seed、代码 commit、配置 SHA256、缓存摘要和环境信息。

### 14.2 跨 seed 汇总

三个 seed 全部通过后，统一报告：

- 每项精度指标的三个原始值、均值和样本标准差；
- 每组总时长、两卡 GPU-hours，以及整个 campaign 的墙钟时间和六卡 GPU-hours；
- 三组吞吐、显存和数据等待的均值、范围与最慢任务；
- Router 行为在不同 seed 间是否一致；
- 失败、恢复和中断次数；
- 不以最高 seed 代替平均结果，不隐去离群结果。

P0 任务书没有绝对 mAP 阈值，因此不使用事后选择的精度门槛判定工程失败。实际精度无论高低均原样报告。

## 15. 时间和资源预算

每个 seed 的训练样本暴露量为：

```text
118,287 images/epoch x 100 epochs = 11,828,700 training images
7,393 optimizer steps/epoch x 100 epochs = 739,300 optimizer steps
```

三个 seed 合计约 35,486,100 次训练图像处理和 2,217,900 次任务内 optimizer update。每 epoch 验证还会让每个任务处理 5,000 张 val 图片。

旧六卡 global batch 48 的启动前估计为 34 至 48 小时。按每卡 batch 同为 8、每个新任务只有三分之一 GPU 粗略外推，单个双卡任务约为 102 至 144 小时；考虑更多同步 step 和三任务并发 I/O，审核前暂按 4.5 至 7 天墙钟窗口预留。三个任务并行时 campaign 墙钟时间取最慢任务，不是三者相加。

该估计不能代替实测。200-step 三组并发基准完成后，使用最慢任务的稳态 step 时间、每 epoch 验证时间和固定 739,300 step 重新计算 ETA。只有实测 ETA、显存和 I/O 报告经确认后，才允许正式启动。

## 16. 验收标准

单个 seed 通过必须满足：

1. 完成 100 epochs，所有 loss 和指标有限；
2. val2017 最终评测覆盖全部 5,000 张图片；
3. `best.pt` 和 `last.pt` 可严格加载，且不包含 Teacher 参数；
4. Adapter、三个 LatentMixture、Detect 和 Router 参数发生有限非零更新；
5. identity 与预注册 seed、global batch 16、代码和缓存完全一致；
6. 日志、metrics、遥测、checkpoint 摘要和异常记录齐全。

WP8 campaign 通过必须满足：

1. seed 集合恰为 `{0,1,2}`，没有缺失、重复或额外试验；
2. 三个 seed 均满足单任务验收；
3. 跨 seed 汇总可重复生成并包含均值、标准差和成本；
4. 三组运行目录和 checkpoint 没有相互覆盖；
5. 三个任务结束后共享缓存页只清理一次；
6. Git 只提交脱敏的小型证据，不提交缓存和 checkpoint 大文件。

## 17. 代码与证据状态

既有训练准备提交：`6e9eb22`；缓存训练性能优化提交：`ead6b59`；训练后页缓存释放提交：`7b4fc75`。

已有文件：

- [`run_wp8.py`](../../scripts/d1/run_wp8.py)：完整缓存调度、基准与合并；
- [`run_wp8_train.py`](../../scripts/d1/run_wp8_train.py)：当前六卡单任务 preflight、训练、恢复、遥测和汇总；
- [`wp8-formal-coco2017.yaml`](../../ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml)：当前六卡合同，尚待改为双卡；
- [`run_with_cache_cleanup.py`](../../scripts/d1/run_with_cache_cleanup.py)：训练结束后定向释放特征文件页；
- [`test_d1_wp8_formal_training.py`](../../tests/test_d1_wp8_formal_training.py)：现有训练合同测试，尚待扩展三组 campaign；
- [`manifests/wp8-full-cache.json`](manifests/wp8-full-cache.json)：完整缓存脱敏证据。

当前测试证明旧入口和 D1 组件可运行，不代表本文三组双卡合同已经实现，也不能写成正式训练通过。

## 18. 审核项

启动前需要明确确认：

1. 接受三个任务各自 global batch 16、每卡 batch 8、两卡 DDP；
2. 接受 seed 0/1/2 三组重复，并以 `mean +/- std` 为主汇总；
3. 接受 `nbs=16`、AdamW、`lr0=0.001`、余弦衰减、3 epoch warmup 和固定 aux 系数；
4. 接受 100 epochs 且每 epoch 完整验证；
5. 接受三组并发短基准后再次报告 ETA，正式训练不会自动衔接启动；
6. 接受暂按 4.5 至 7 天预留墙钟窗口，最终以并发基准为准；
7. 接受正式运行中不调参，异常时仅做同 identity checkpoint 恢复；
8. 接受 P0 不设事后绝对 mAP 门槛，三个 seed 精度全部如实报告；
9. 接受某个任务提前结束时不清理共享缓存，等全部任务终止后统一清理。

只有收到明确确认并完成第 9 至 10 节门禁后，才可以启动三组并发基准；基准结果再次确认后，才能启动三个正式训练任务。
