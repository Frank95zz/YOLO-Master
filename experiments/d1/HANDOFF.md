# D1 项目交接说明

更新时间：2026-09-04（Asia/Shanghai）

本文用于将腾讯犀牛鸟 YOLO-Master D1 课题交接给新的 Codex 会话。接手者应先完整阅读本文，再阅读 `experiments/d1/README.md` 和各阶段 `WP*.md`。本文记录的是当前实际状态；若与较早对话或旧日志冲突，以 Git 当前提交、已提交 manifest 和服务器现场状态为准。

## 1. 当前结论

- 仓库分支：`feat/topic-d1-fengyanqi`。
- 当前提交：`ead6b59503ab1154c3375eafdc52bc5e7f12892d`，远端同名分支已同步。
- WP0-WP7 已完成实现、测试和阶段文档。
- WP8 的完整 COCO 2017 DINOv3 特征缓存已经构建并完成统一索引与校验。
- WP8 正式 100 epochs 六卡训练尚未启动，仍处于训练参数基准和方案审核阶段。
- 最新 batch/workers 六组基准全部因容器 RAM OOM 和 DataLoader 遗留进程而失败，结果无效，不能据此判定 batch 64/128/256 不可用。
- 失败遗留的 56 个指定 WP8 进程已经精准清理。清理后六张 A40 均回到约 266 MiB 基础显存占用，容器内存降到约 9-10 GiB。
- 未获得用户明确批准前，不得启动 WP8 正式训练。

## 2. 两套编号必须区分

### 2.1 课题验收 P0/P1/P2

这是腾讯犀牛鸟任务书中的目标分级：

- `P0`：复用冻结的 DINOv3 teacher 特征，接入 LatentMixture 和检测头，完成训练与评测闭环。
- `P1`：在相同参数量/预算口径下，与从零训练检测器进行精度、显存和训练时长对照；至少覆盖两个数据集，并在至少一个数据集上明确给出精度保留比例且 GPU 时降低不少于 50%。
- `P2`：显式接入 latent aux，扫描 balance/z-loss 等权重或进行 DINOv3 与 SigLIP2 teacher 对比。

当前代码实施重点仍是课题 `P0`。WP8 的正式 COCO 训练是课题 P0 的正式实验，不是课题 P1/P2。

### 2.2 工程实施 WP0-WP8

这是本项目为了完成课题 P0 拆出的工作包：

- `WP0`：实验合同、COCO 数据和 DINOv3 权重准备。
- `WP1`：DINOv3 block 4/8/12 多层输出。
- `WP2`：可复现 safetensors 特征缓存。
- `WP3`：DINOv3 多尺度特征适配器。
- `WP4`：缓存特征检测模型。
- `WP5`：缓存 Dataset、Trainer、Validator。
- `WP6`：Latent aux 损失闭环。
- `WP7`：在线/缓存对齐、32 图过拟合、COCO8 一轮训练。
- `WP8`：完整 COCO 2017 正式训练与评测。

## 3. 服务器与仓库

当前服务器连接方式：

```bash
ssh root@10.210.22.36 -p 30722
```

服务器地址和端口可能随平台变化，应先从平台页面确认。不要把密码、令牌或私钥写入仓库。

关键路径：

```text
仓库                 /root/yolo-master/repo
Python 环境          /root/yolo-master/.conda/d1/bin/python
外部工作区           /data/yingxi/yolo-master-d1
COCO 数据            /data/yingxi/yolo-master-d1/datasets/coco
DINOv3 权重          /data/yingxi/yolo-master-d1/weights/teachers
完整训练缓存         /data/yingxi/yolo-master-d1/feature_cache/coco2017-train2017-d1-cache-v1
完整验证缓存         /data/yingxi/yolo-master-d1/feature_cache/coco2017-val2017-d1-cache-v1
日志                 /data/yingxi/yolo-master-d1/logs
运行输出             /data/yingxi/yolo-master-d1/runs
外部 manifest        /data/yingxi/yolo-master-d1/manifests
```

`/data` 是共享 NFS 工作盘，不是唯一长期备份。代码、小型配置、测试和脱敏 manifest 进入 Git；COCO、模型权重、完整缓存和 checkpoint 不进入 Git。

硬件与限制：

- 6 张 NVIDIA A40，每张约 46 GiB 可用显存。
- 主机可见内存约 503 GiB，但当前容器 cgroup 限制为 `188978561024` bytes，约 176 GiB。
- 无 swap。
- 训练和缓存位于 NFS，吞吐与 page cache 会影响 RAM 和 I/O 表现。
- 不能根据主机 `free -h` 的 503 GiB 判断任务可用内存，必须检查 cgroup 限制和占用。

## 4. 核心技术合同

### 4.1 Teacher 与检测模型的关系

DINOv3 是冻结的外部 foundation teacher，不是 `yolo26-master-latent-n.yaml` 中原有的 YOLO backbone。正式路线先用 DINOv3 离线抽取特征，再用缓存特征训练下游模型，因此训练时不实例化 Teacher，也不把 Teacher 放入 optimizer、EMA 或 checkpoint。

课题材料没有强制指定某一个 DINOv3尺寸。当前正式选择为 ModelScope 可获取并已验证的：

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

这是 DINOv3 ViT-S/16：384 hidden channels、12 blocks、6 attention heads、patch size 16、4 register tokens。

Teacher 权重 SHA256：

```text
4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d
```

### 4.2 固定预处理

```text
LetterBox(640x640, auto=False, scale_fill=False, scaleup=True,
          center=True, stride=32, padding_value=114, INTER_LINEAR)
RGB CHW -> [0,1]
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

640 可被 patch 16 整除，因此 DINOv3 token grid 固定为 `40x40`。不得再做第二次 resize 或 crop。

### 4.3 block 4/8/12 与 P3/P4/P5

`DINOv3Teacher(output_layers=(4, 8, 12))` 使用一基层号，对应实现索引 `3/7/11`，输出：

```text
block4  [B,384,40,40]
block8  [B,384,40,40]
block12 [B,384,40,40]
```

三层都是 stride 16，不能直接称为 P3/P4/P5。WP3 Adapter 对每个 block 分别建立三种尺度分支，共九条独立分支：

```text
每个 block -> P3 candidate: 1x1 Conv 384->64 + GN + SiLU + bilinear 2x
每个 block -> P4 candidate: 1x1 Conv 384->128 + GN + SiLU
每个 block -> P5 candidate: 3x3 stride-2 Conv 384->256 + GN + SiLU
```

因此不是“block4 只转 P3、block8 只转 P4、block12 只转 P5”。正确关系是：

```text
block4/8/12 -> 三个 P3 候选 -> P3 LatentMixture -> P3 [B,64,80,80]
block4/8/12 -> 三个 P4 候选 -> P4 LatentMixture -> P4 [B,128,40,40]
block4/8/12 -> 三个 P5 候选 -> P5 LatentMixture -> P5 [B,256,20,20]
```

P3/P4/P5 分别表示 stride 8/16/32 的检测金字塔尺度；数字越大，空间分辨率越低、语义通常越强。

### 4.4 数据集关系

COCO 与 VisDrone 是两个独立数据集，不是彼此的子集。当前正式 P0 从完整 COCO 2017 开始：`train2017=118,287`、`val2017=5,000`，使用官方 split，不做二次随机划分。VisDrone 主要用于后续课题 P1 的第二数据集对照。

`coco8` 仅是极小工程样例，不代表正式训练配置。WP7 的 COCO8 测试直接从完整 COCO 中选 4 张训练、4 张验证，只验证闭环。

## 5. WP0-WP8 已完成内容

### 5.1 WP0：实验合同与数据准备

关键提交：

```text
c13dc18 feat(d1): define P0 experiment contract
0bc6d66 docs(d1): record WP0 provenance
```

完成内容：

- 固定 COCO 2017、seed 0、确定性预处理、DINOv3 ViT-S/16、block 4/8/12 和缓存合同。
- 下载并校验正式 Teacher 权重、COCO 图片/标签和许可证来源。
- 生成数据列表、环境、权重和实验合同 manifest。
- 正式配置不包含服务器绝对路径，路径由运行参数注入。

主要文件：

```text
experiments/d1/WP0.md
experiments/d1/manifests/p0-experiment-contract.json
experiments/d1/manifests/coco2017-splits.json
experiments/d1/manifests/coco2017-train2017.txt
experiments/d1/manifests/coco2017-val2017.txt
experiments/d1/manifests/dinov3-vits16.json
experiments/d1/manifests/environment.json
experiments/d1/manifests/licenses.md
ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml
scripts/d1/prepare_wp0.py
tests/test_d1_wp0_contract.py
```

### 5.2 WP1：DINOv3 多层输出

关键提交：`3b05836 feat(d1): add DINOv3 multi-layer outputs`。

- `output_layers=None` 保持原有 `dense["p4"]` 兼容行为。
- `output_layers=(4,8,12)` 一次前向返回 `block4/block8/block12`。
- 严格校验层号、stage、batch、channel、grid、有限值和 prefix tokens。
- 每次 encode 都恢复 eval/frozen/inference mode。
- 真实 ViT-S/16、FP16、640 输入已验证三层均为 `[1,384,40,40]`。

主要文件：

```text
experiments/d1/WP1.md
ultralytics/nn/foundation/teachers/dinov3.py
tests/test_d1_wp1_dinov3.py
```

### 5.3 WP2：可复现特征缓存

关键提交：

```text
5cd566d feat(d1): add sharded feature cache
af433a8 docs(d1): record WP2 cache evidence
```

- 缓存版本：`d1-cache-v1`。
- 格式：safetensors、FP16、不可变分片、原子发布。
- 支持 build、verify、compare、断点续跑、索引重建和 checksum。
- 两次独立 100 图缓存均为 100 样本、300 tensor、约 352 MiB。
- 两次耗时 12.755/13.821 秒，吞吐 7.840/7.236 images/s，内容摘要完全一致。

主要文件：

```text
experiments/d1/WP2.md
ultralytics/nn/foundation/cache.py
scripts/d1/cache_features.py
tests/test_d1_wp2_cache_cli.py
tests/test_d1_wp2_feature_cache.py
```

100 图缓存位置：

```text
/data/yingxi/yolo-master-d1/feature_cache/wp2-train100-a
/data/yingxi/yolo-master-d1/feature_cache/wp2-train100-b
```

### 5.4 WP3：多尺度 Adapter

关键提交：

```text
e1c60c2 feat(d1): add multi-scale feature adapter
0a5a471 docs(d1): record WP3 adapter evidence
```

- 实现 `DINOFeaturePyramidAdapter`。
- 九条分支完全独立，使用 GroupNorm，不使用 BatchNorm，不 detach 输入。
- 输出可直接送入三个 `LatentMixture([C,C,C], C)`。
- 真实缓存 FP16/CUDA 路径和梯度已验证。

主要文件：

```text
experiments/d1/WP3.md
ultralytics/nn/modules/foundation_adapter.py
tests/test_d1_wp3_foundation_adapter.py
```

### 5.5 WP4：缓存特征检测模型

关键提交：

```text
a1255c0 feat(d1): add cached-feature detection model
0c9a8d7 docs(d1): record WP4 detection model evidence
```

- 建立缓存 block 特征到 Adapter、LatentMixture、Detect 的模型闭环。
- 正式模型约 3,542,567 个参数，Teacher 参数数必须为 0。
- checkpoint 严格往返，不包含 Teacher。
- 单批 detection loss 和 backward 已验证。

主要文件：

```text
experiments/d1/WP4.md
ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml
ultralytics/nn/foundation_detection_model.py
tests/test_d1_wp4_foundation_detection_model.py
```

### 5.6 WP5：缓存训练与验证链路

关键提交：

```text
81c98b3 feat(d1): add cached training pipeline
3297afa docs(d1): record WP5 cached training evidence
```

- 实现 D1 缓存 Dataset、Trainer 和 Validator。
- 读取缓存特征、COCO 标签和 LetterBox metadata，完成 batch、loss、optimizer、checkpoint 和坐标恢复。
- 缓存训练不读取 RGB 作为模型输入，也不下载普通 YOLO 预训练权重。
- 保留 online 对齐检查模式，但正式训练使用离线缓存。

主要文件：

```text
experiments/d1/WP5.md
ultralytics/data/d1_cache.py
ultralytics/models/yolo/detect/foundation_train.py
ultralytics/models/yolo/detect/foundation_val.py
tests/test_d1_wp5_cached_pipeline.py
```

### 5.7 WP6：latent aux 闭环

关键提交：

```text
c0277eb feat(d1): close latent auxiliary loss loop
315caaa docs(d1): record WP6 auxiliary loss evidence
```

固定起始配置：

```text
balance_loss_coeff = 0.01
router_z_loss_coeff = 0.001
latent_aux_gain = 0.1
mixture_aux_budget = 3.0
```

这些是工程起始点，不是任务书强制值，也不是完成扫描后的最优值。`latent aux` 必须通过 `collect_aux_loss(include_kinds=...)` 显式收集，不能依赖自动发现。已验证三个 Router 的 balance/z-loss、梯度和关闭语义。

主要文件：

```text
experiments/d1/WP6.md
ultralytics/nn/modules/latent_mixture.py
ultralytics/nn/foundation/losses.py
tests/test_d1_wp6_latent_aux.py
```

### 5.8 WP7：最小训练验收

关键提交：

```text
b120185 test(d1): add WP7 minimal training gates
527688a fix(d1): use cache reader root in WP7 identity
bfae5f4 fix(d1): validate AMP without RGB model download
167fe68 fix(d1): retain final training routing evidence
a0a0513 fix(trainer): reset optimizer cursor after recovery
771e453 docs(d1): record WP7 acceptance evidence
```

三个门禁均已通过：

- 真实 DINOv3 在线特征与缓存对齐。
- 32 图过拟合。
- 4 train + 4 val 的 COCO8 一轮训练、验证、checkpoint 保存和严格重载。

总状态见 `experiments/d1/manifests/wp7-summary.json`：`status=passed`，无 failures。WP7 只证明工程闭环，不代表完整 COCO 精度结论。

### 5.9 WP8：完整缓存与正式训练准备

关键提交：

```text
651ffd8 feat(d1): add six-GPU cache orchestration
5b5e4af docs(d1): record WP8 cache benchmark gate
6e9eb22 feat(d1): prepare formal WP8 training
0e62957 docs(d1): define WP8 formal experiment plan
ead6b59 perf(d1): optimize cached feature training
```

已完成：

- 六卡确定性分区、rank 独立 shard、断点恢复和统一 finalize。
- 完整 train2017/val2017 缓存构建与校验。
- 正式训练配置、preflight、六卡 DDP、resume、epoch 遥测和 summarize 入口。
- 缓存读取 LRU、可信缓存模式、可配置 prefetch 和 AMP loss scale 稳定化。

尚未完成：

- 可靠的 batch/workers 最优参数基准。
- 根据基准更新正式配置和 WP8 时间预估。
- 用户审核后的 100 epochs 正式训练。
- 完整 val2017 最终评测和正式结果报告。

## 6. 完整缓存证据

缓存合同 SHA256：

```text
6bfda0e13bde01001c3f3f2d72631a2401fb9a77b146d6fb2794303e379e47a7
```

完整结果：

| Split | 样本 | Tensor | Shard | Cache bytes | 内容 SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| train2017 | 118,287 | 354,861 | 204 | 436,232,631,192 | `0224bbd302faf62b39e914be11f4d3469f7567791ce48994a9fc0f378e070cbe` |
| val2017 | 5,000 | 15,000 | 12 | 18,439,541,408 | `cd5840678d79550482aaf4110b9e68b8c22819f2f6b31e0af40f0e5b7476fa1b` |

合计：123,287 样本、369,861 tensor、216 shard、454,672,172,600 cache bytes，约 423.5 GiB。

构建证据：

- train 六卡聚合吞吐约 69.96 images/s，含最终校验端到端约 54.29 images/s。
- train worker wall time 2178.90 秒，最终校验 1326.90 秒。
- val 六卡聚合吞吐约 78.85 images/s，端到端约 62.56 images/s。
- 完整证据文件：`experiments/d1/manifests/wp8-full-cache.json`。

## 7. 最新性能优化与失败基准

### 7.1 `ead6b59` 已实现的优化

- `FeatureCacheReader(root, max_open_shards=0)` 支持每进程 safetensors handle LRU；默认 0 保持旧行为。
- `D1FeatureCacheDataset` 增加 `trusted_cache`、`max_open_shards` 和 `prefetch_factor`。
- trusted 模式每个 worker 对每个 shard 首次遇到时做有限值校验，之后避免重复全 tensor 扫描；不会修改样本顺序和特征值。
- 已对 16 个真实缓存样本逐 tensor 执行 `torch.equal`，结果完全一致：`REAL_CACHE_EXACT_MATCH=16`。
- AMP 固定初始 scale 16，growth interval 1,000,000，避免默认 scale 在该模型上触发非有限梯度后退回 FP32。
- 当前正式配置包含：`trusted=true`、`max_open_shards_per_worker=4`、`prefetch_factor=1`、`amp_init_scale=16`、`amp_growth_interval=1000000`。
- 最新相关回归测试记录为 `120 passed, 5 skipped`；`git diff --check` 通过。

### 7.2 当前正式配置仍未采用大 batch

`ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml` 当前仍是：

```text
epochs=100
global batch=48
per-GPU batch=8
nbs=48
workers=4 per rank
AMP=true
AdamW, lr0=0.001, lrf=0.01, cosine
weight_decay=0.0005
warmup=3 epochs
patience=100
```

该配置是经过 WP7 闭环验证的保守起点，不是速度最优结论。`nbs=48` 意味着名义 batch 与真实全局 batch 相同，不做额外梯度累积。`workers=4` 是每个 DDP rank 的 DataLoader worker 数，共 24。

### 7.3 batch/workers 基准发生了什么

计划测试：

```text
每卡 batch: 64 / 128 / 256
每 rank workers: 8 / 16
6 张 GPU
```

运行标识：`wp8-batch-worker-ead6b59`。

结果目录：

```text
/data/yingxi/yolo-master-d1/benchmarks/wp8-training/wp8-batch-worker-ead6b59
```

六组都在 56-117 秒内以 `SIGKILL` 退出，`summary.json` 为 `status=failed`、`best=null`。根因不是显存 OOM，而是：

1. 容器 RAM 上限约 176 GiB；
2. 大 batch × 多 worker × prefetch × safetensors/page cache 带来很高主机内存压力；
3. 某个 DDP rank 被杀后，DataLoader worker 被重新挂到 PID 1，没有被 orchestration 清理；
4. 后续候选继续启动，遗留 worker 累积到约 118.5 GiB RSS，造成连续 OOM；
5. cgroup 记录累计 28 次 OOM kill。

最终精准清理了 56 个命令行同时匹配以下条件的进程：

```text
scripts/d1/benchmark_wp8_training.py
run-id = smoke-amp1024 或 wp8-batch-worker-ead6b59
```

清理后残留数为 0，六卡均约 266 MiB。没有删除日志、缓存或代码。

一个较早的短 smoke 曾在每卡 batch 64、workers 8、prefetch 1、LRU 4、AMP scale 16 下跑通 8 steps，聚合约 859.8 images/s、峰值显存约 7.6 GB/卡。但该测试太短，不能作为正式最优配置或总训练时间结论。

## 8. 新账号接手后的优先步骤

### 8.1 先确认现场

```bash
ssh root@10.210.22.36 -p 30722
cd /root/yolo-master/repo
git status --short --branch
git rev-parse HEAD
nvidia-smi
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/memory/memory.usage_in_bytes
```

预期：分支和远端在 `ead6b59`，工作区干净，六卡基本空闲，内存约 9-10 GiB。若不一致，先调查，不要直接重置或删除用户改动。

### 8.2 先修复 benchmark 生命周期管理

修改 `scripts/d1/benchmark_wp8_training.py` 及测试，目标是：

- 每个候选放入独立进程组；
- 候选成功、失败、超时或收到异常时都在 `finally` 中终止整个进程组；
- 额外识别并清理该 run-id 的 `pt_data_worker` 后代/孤儿；
- 每组结束后确认对应进程数为 0，再开始下一组；
- 记录候选启动前后 cgroup RAM、OOM kill 计数和 GPU 显存；
- RAM 接近上限时主动判失败并停止，不让系统 OOM 连锁杀进程；
- 不误杀其他 run-id 或其他用户任务。

必须为正常结束、rank SIGKILL、超时、孤儿 worker 和精确 run-id 隔离补测试。

### 8.3 重新做 batch/workers 基准

用户原始要求仍是测试 batch 64/128/256 与 workers 8/16。建议逐组隔离运行，而不是一次串行启动六组后依赖父进程自动清理：

1. 先验证 batch 64 / workers 8；
2. 确认无残留且 RAM 回落；
3. 再测 batch 64 / workers 16；
4. 同样方式测试 batch 128、256；
5. 若某组已逼近 176 GiB，记录为 RAM 不支持，不继续启动更危险组合；
6. 为工程可用性可额外比较 workers 2/4，但不得用额外候选替代用户要求的结果说明。

每组必须报告：

- 实际每卡/global batch；
- 实际每 rank workers；
- AMP 是否全程保持开启、是否发生 fallback；
- 是否自动缩小 batch、worker cap 或 epoch retry；
- 训练 step 吞吐和数据等待比例；
- 每卡峰值显存；
- cgroup 峰值 RAM 与 OOM 计数变化；
- 启动开销和稳定运行时间；
- 基于完整 epoch 的保守 ETA。

短测试超过 3 分钟时必须后台运行，确认正常启动一次后退出会话，并把 PID、状态、日志和结果命令告诉用户，不持续轮询。

### 8.4 基准后更新正式方案，但不要启动训练

选出满足以下门禁且吞吐最高的配置：

- 精确使用请求的 batch/workers；
- 无 CUDA OOM、cgroup OOM、AMP fallback、自动 batch reduction、worker cap、epoch retry；
- loss 有限；
- 各 rank 正常同步；
- RAM 和显存保留安全余量。

然后：

1. 更新 `wp8-formal-coco2017.yaml` 的 `batch/nbs/workers`；
2. 将正式 global batch 同时写入 `nbs`，避免无意梯度累积和学习率缩放语义变化；
3. 更新 `experiments/d1/WP8.md`，加入真实吞吐、峰值资源和 ETA；
4. 提交并推送代码、配置和脱敏 benchmark 摘要；
5. 将新的 WP8.md 交给用户审核；
6. 等待用户明确说可以开始正式训练。

### 8.5 获得明确批准后才启动正式训练

先执行 `prepare` preflight，再以六卡 DDP 后台启动。现有入口：

```bash
export D1_REPO=/root/yolo-master/repo
export D1_WORKSPACE=/data/yingxi/yolo-master-d1
export D1_COMMIT=$(git -C "$D1_REPO" rev-parse --short HEAD)
export D1_RUN_ROOT="$D1_WORKSPACE/runs/wp8-formal-$D1_COMMIT"
export D1_REPORT_DIR="$D1_WORKSPACE/manifests/wp8-formal-$D1_COMMIT"

cd "$D1_REPO"
/root/yolo-master/.conda/d1/bin/python scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  prepare
```

正式 DDP 核心命令见 `experiments/d1/WP8.md`。实际启动必须使用后台日志/PID/status 包装。启动后只检查一次六个 rank、GPU 占用、早期 loss 和错误，然后结束会话。

## 9. 正式训练验收与汇报内容

正式 WP8 报告至少包含：

- COCO `mAP50-95`、`mAP50`、Precision、Recall；
- box、class、DFL 和 latent/mixture aux loss 曲线；
- 总墙钟时间、每 epoch 时间、六卡峰值显存；
- 数据等待比例与 NFS/cache 读取瓶颈；
- P3/P4/P5 Router 参数变化、平均概率、熵、balance、z-loss 和 residual gain；
- best/last checkpoint SHA256；
- checkpoint 严格重载、Teacher 参数计数为 0；
- 训练代码 commit、配置 SHA256、缓存合同和内容摘要；
- 完整 val2017 的 5,000 张覆盖证明。

课题 P0 没有强制绝对 mAP 阈值。正式结果必须如实报告，不能根据中间精度临时修改本次已锁定配方。若需要调参，应作为新的独立实验 ID 和合同。

## 10. 用户协作约定

- 预计超过 3 分钟的命令：后台启动，确认进程正常挂起后退出会话，不持续检查。
- 用户会自行检查状态并通知继续。
- 正式训练前：必须先把详细实验安排写入 `WP8.md` 供审核，收到明确批准才可启动。
- 不要修改 smoke 配置来代替正式 D1 配置。
- 不要把 `/data` 绝对路径、密码或令牌写入 tracked 配置。
- 不要把 COCO、Teacher 权重、完整缓存、checkpoint 或大日志提交到 Git。
- 服务器不是固定可用资源，重要外部内容必须备份或至少有可验证 manifest。
- 工作区可能包含用户修改；不得使用 `git reset --hard` 或覆盖未确认改动。

## 11. 备份策略

交大云盘建议结构：

```text
YOLO-Master-D1/
├── datasets/coco2017-source/
├── weights/dinov3-vits16/
├── feature_cache/dinov3-vits16-coco2017-640-fp16/
├── checkpoints/
├── results/
└── manifests/
```

备份职责：

- COCO 官方源压缩包和 Teacher 权重：必须备份并保存 SHA256。
- 完整特征缓存：建议备份；约 423.5 GiB，上传前确认云盘空间，不要在 `/data` 再生成一份完整压缩副本。
- 正式 `best.pt`、`last.pt`、最终评测 checkpoint：必须备份。
- 配置、代码、测试、小型 manifest 和脱敏摘要：Git + 云盘。
- staging 和可重建临时 `.part`：不作为长期备份。

恢复顺序：检出记录的 Git commit -> 恢复并校验 COCO/权重 -> 恢复或重建缓存 -> 校验缓存摘要 -> 恢复 checkpoint -> 严格验证。

## 12. 主要文档和代码入口

阶段文档：

```text
experiments/d1/README.md
experiments/d1/WP0.md ... WP8.md
```

主要脚本：

```text
scripts/d1/prepare_wp0.py
scripts/d1/cache_features.py
scripts/d1/run_wp7.py
scripts/d1/run_wp8.py
scripts/d1/run_wp8_train.py
scripts/d1/benchmark_wp8_training.py
```

主要实现：

```text
ultralytics/nn/foundation/teachers/dinov3.py
ultralytics/nn/foundation/cache.py
ultralytics/nn/modules/foundation_adapter.py
ultralytics/nn/modules/latent_mixture.py
ultralytics/nn/foundation_detection_model.py
ultralytics/data/d1_cache.py
ultralytics/models/yolo/detect/foundation_train.py
ultralytics/models/yolo/detect/foundation_val.py
```

正式配置：

```text
ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml
ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml
ultralytics/cfg/experiments/d1/wp7-minimal-tests.yaml
ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml
```

测试：

```text
tests/test_d1_wp0_contract.py
tests/test_d1_wp1_dinov3.py
tests/test_d1_wp2_cache_cli.py
tests/test_d1_wp2_feature_cache.py
tests/test_d1_wp3_foundation_adapter.py
tests/test_d1_wp4_foundation_detection_model.py
tests/test_d1_wp5_cached_pipeline.py
tests/test_d1_wp6_latent_aux.py
tests/test_d1_wp7_acceptance.py
tests/test_d1_wp8_multi_gpu_cache.py
tests/test_d1_wp8_formal_training.py
tests/test_d1_wp8_training_benchmark.py
```

## 13. 重要提交索引

```text
c13dc18  WP0 实验合同
0bc6d66  WP0 provenance
3b05836  WP1 DINOv3 多层输出
5cd566d  WP2 分片缓存
af433a8  WP2 缓存证据
d800a81  WP0/WP1/WP2 阶段文档
1b378ad  D1 scripts/tests 目录整理
e1c60c2  WP3 Adapter
0a5a471  WP3 文档
a1255c0  WP4 检测模型
0c9a8d7  WP4 文档
81c98b3  WP5 缓存训练链路
3297afa  WP5 文档
c0277eb  WP6 aux 闭环
315caaa  WP6 文档
b120185  WP7 验收工具
527688a/bfae5f4/167fe68/a0a0513  WP7 修复
771e453  WP7 验收证据
651ffd8  WP8 六卡缓存
5b5e4af  WP8 缓存基准证据
6e9eb22  WP8 正式训练准备
0e62957  WP8 正式实验方案
ead6b59  WP8 缓存读取与 AMP 性能优化
```

## 14. 接手者第一条建议指令

```text
先阅读 experiments/d1/HANDOFF.md、README.md 和 WP8.md。只读核对服务器、Git、GPU、cgroup RAM、完整缓存和失败 benchmark 状态。随后修复 benchmark 在 DDP rank 被 SIGKILL 后遗留 pt_data_worker 的生命周期问题，补测试并推送。不要启动正式训练；修复后按 HANDOFF.md 逐组重新测 batch/workers，更新 WP8.md 和预计训练时间，等待我明确批准。
```

