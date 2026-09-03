# D1 P0：冻结 DINOv3 × LatentMixture 检测闭环

## 1. 文档定位

本文档是 D1 课题正式研发与复现的入口，当前只定义 P0 的实现范围、技术方案、代码归属、测试门槛和交付证据。实施过程中应持续更新任务状态和最终命令。

- 正式研发方案与实验入口：`experiments/d1/`
- 准入检查及历史证据：`smoke/d1/`
- 实验配置：`ultralytics/cfg/experiments/d1/`
- 模型结构配置：`ultralytics/cfg/models/26/`
- Foundation 教师与缓存：`ultralytics/nn/foundation/`
- LatentMixture 等网络模块：`ultralytics/nn/modules/`
- 检测训练与评测入口：`ultralytics/models/yolo/detect/`
- D1 数据准备与缓存命令：`scripts/d1/`
- 自动化测试：`tests/`

### 阶段完成报告

| 阶段 | 报告 | 状态 |
| --- | --- | --- |
| WP0：实验合同、正式数据与 Teacher 准备 | [WP0.md](WP0.md) | 已完成 |
| WP1：DINOv3 多层特征输出 | [WP1.md](WP1.md) | 已完成 |
| WP2：可复现特征缓存 | [WP2.md](WP2.md) | 已完成 |
| WP3：DINOv3 多尺度特征适配器 | [WP3.md](WP3.md) | 已完成 |
| WP4：缓存特征检测模型 | [WP4.md](WP4.md) | 已完成 |
| WP5：缓存 Dataset、Trainer 与 Validator | [WP5.md](WP5.md) | 已完成 |
| WP6：Latent aux 损失闭环 | [WP6.md](WP6.md) | 已完成 |
| WP7：最小训练与工程验收 | [WP7.md](WP7.md) | 已完成 |
| WP8：六卡缓存准备与启动门禁 | [WP8.md](WP8.md) | 门禁未通过，完整缓存未启动 |

阶段报告记录已经完成的实现、实测结果和复现证据；本文继续维护总体技术方案、WP8 路线及完整背景。后续阶段沿用 `WPn.md` 的方式补充完成报告。

`smoke/d1/` 不再承载正式 P0 实现。数据集、DINOv3 权重、特征缓存和训练 checkpoint 不提交 Git，只提交 manifest、校验值、配置、日志摘要和云盘说明。

## 2. 课题边界

P0 要完成：

> 冻结基础模型抽取多层特征，接入 LatentMixture 与检测头，并完成训练和评测。

本实现必须满足：

1. DINOv3 仅作为冻结特征来源，不进入优化器。
2. 训练对象仅包括多尺度适配器、LatentMixture 和 Detect Head。
3. 不蒸馏 YOLO backbone，不使用教师特征监督 Router。
4. 不把 `foundation_distill_model.py` 的 D2/F11 蒸馏路径当作 D1。
5. 不直接使用 `yolo26-master-latent-n.yaml` 的 YOLO Backbone；只复用 LatentMixture、Detect 和统一 aux loss 基础设施。

P0 暂不包含：

- COCO 与 VisDrone 的同预算成本对照；
- 训练成本降低 50% 的 P1 结论；
- DINOv3 尺寸、SigLIP2 teacher 或大规模 aux 权重扫描；
- 部署、导出或实时推理优化。

## 3. 当前代码基础

已有能力：

- `DINOv3Teacher`：加载、冻结、预处理并一次返回 block 4/8/12；
- `FoundationTeacher` / `FoundationFeatures`：统一教师协议；
- `FeatureCacheReader` / `FeatureCacheWriter`：可复现的分片特征缓存；
- `DINOFeaturePyramidAdapter`：生成 P3/P4/P5 三组多层候选特征；
- `D1FoundationDetectionModel`：连接 Adapter、三个 LatentMixture、Detect 和 CompositeCriterion；
- `LatentMixture`：同尺度多特征融合、Router、balance loss 和 z-loss；
- `Detect`：P3/P4/P5 检测头；
- `CompositeCriterion`：将 routed aux loss 加入检测损失；
- `collect_aux_loss(... include_kinds=(..., "latent"))`：统一收集 latent aux。

当前缺口：

- 尚未构建完整 COCO train2017/val2017 特征缓存；
- 尚未执行正式完整训练、COCO val2017 精度评测和成本对照实验。

## 4. P0 固定技术方案

| 项目 | P0 默认值 |
| --- | --- |
| Teacher | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| Teacher 架构 | DINOv3 ViT-S/16 |
| Teacher 层 | Block 4/8/12；实现索引 `3/7/11` |
| 数据集 | COCO 2017：完整 `train2017`（118,287 张）+ `val2017`（5,000 张） |
| 输入 | `640×640`，确定性 letterbox |
| 缓存 | FP16、分片 safetensors、JSONL/index manifest |
| P3/P4/P5 通道 | 64 / 128 / 256 |
| 检测尺度 | stride 8 / 16 / 32 |
| LatentMixture | 每个尺度一个，共三个 |
| 检测头 | YOLO26 `Detect`，COCO `nc=80` |
| 训练精度 | AMP；缓存保存 FP16 |
| 数据增强 | P0 禁用随机空间增强和多尺度训练 |

模型 ID、权重路径、权重 SHA256、许可证、数据 split、代码 commit 和环境版本必须写入实验 manifest，不依赖隐式默认值。

P0 的正式训练和评测不使用 COCO-mini 或其他人为抽取子集。固定 100 图缓存、32 图过拟合和 COCO8 一 epoch 只作为工程正确性测试，不作为正式实验结果。

## 5. 目标数据流

### 5.1 离线特征抽取

```text
COCO image + annotation
  -> deterministic letterbox 640x640
  -> frozen DINOv3 ViT-S/16
  -> block4 / block8 / block12
  -> remove CLS and register tokens
  -> F4 / F8 / F12: 384x40x40
  -> cast to FP16
  -> feature cache + manifest
```

### 5.2 缓存训练

```text
F4 / F8 / F12
  -> DINOFeaturePyramidAdapter
       P3 candidates: 3 x (64x80x80)
       P4 candidates: 3 x (128x40x40)
       P5 candidates: 3 x (256x20x20)
  -> LatentMixture-P3 / P4 / P5
  -> P3' / P4' / P5'
  -> Detect([P3', P4', P5'])
  -> detection loss + latent aux loss
```

ViT 的多层输出具有不同语义深度，但原生空间分辨率都为 stride 16。P3/P5 必须由可训练适配器生成，不能把不同 Transformer 层直接命名为 P3/P4/P5。

### 5.3 WP1 Teacher API

默认调用保持 Foundation 蒸馏链路兼容，只返回最终层：

```python
teacher = DINOv3Teacher(weights_path=weights_dir)
features = teacher.encode(images)
assert tuple(features.dense) == ("p4",)
```

D1 离线抽取器显式请求一基 block 4/8/12；Teacher 通过 Transformers Backbone 的公开 stage 选择接口一次前向返回三层：

```python
teacher = DINOv3Teacher(
    weights_path=weights_dir,
    output_layers=(4, 8, 12),
)
features = teacher.encode(images)
assert tuple(features.dense) == ("block4", "block8", "block12")
```

正式 ViT-S/16 在 640×640 输入下，三个张量均为 `[B, 384, 40, 40]`。metadata 记录一基层号 `(4, 8, 12)`、实现索引 `(3, 7, 11)`、backbone stage、feature name、patch size、grid、hidden dim 和 prefix token 数量。多层特征不命名为 P3/P4/P5；尺度转换留给 WP3 Adapter。Teacher 每次编码前重新进入 eval 并冻结参数，输出处于 inference mode，且现有 wrapper 继续保证 Teacher 不进入 student optimizer、DDP、EMA 或 checkpoint。

### 5.4 WP2 特征缓存

`ultralytics.nn.foundation.cache` 定义 `d1-cache-v1` 协议。cache key 由原始图像 SHA256、WP0 预处理 SHA256、Teacher 权重 SHA256、输出层、dtype 和 schema version 共同决定。每个样本在 `samples.jsonl` 中记录 split、相对图像路径、cache key、shard、tensor key、shape、dtype、字节数和 tensor SHA256；`index.json` 记录合同、内容摘要和每个 shard 的文件 SHA256。

分片先写入隐藏的 `.part` 文件，通过 safetensors header 校验后再原子改名。已提交 shard 的完整样本记录同时嵌入 header，因此 `index.json` 丢失或在更新前中断时可自动重建。续跑会校验当前图像和合同后跳过已提交样本；`verify` 子命令不加载 Teacher，只检查 manifest、shard 和 tensor。

固定 100 图的两次独立构建命令：

```bash
python scripts/d1/cache_features.py build \
  --workspace /data/yingxi/yolo-master-d1 \
  --cache-dir /data/yingxi/yolo-master-d1/feature_cache/wp2-train100-a \
  --split train2017 --limit 100 --batch-size 8 --device 0 \
  --report /data/yingxi/yolo-master-d1/manifests/wp2-train100-a.json

python scripts/d1/cache_features.py build \
  --workspace /data/yingxi/yolo-master-d1 \
  --cache-dir /data/yingxi/yolo-master-d1/feature_cache/wp2-train100-b \
  --split train2017 --limit 100 --batch-size 8 --device 0 \
  --report /data/yingxi/yolo-master-d1/manifests/wp2-train100-b.json

python scripts/d1/cache_features.py compare \
  --cache-dir /data/yingxi/yolo-master-d1/feature_cache/wp2-train100-a \
  --other-cache-dir /data/yingxi/yolo-master-d1/feature_cache/wp2-train100-b \
  --first-report /data/yingxi/yolo-master-d1/manifests/wp2-train100-a.json \
  --second-report /data/yingxi/yolo-master-d1/manifests/wp2-train100-b.json \
  --report /data/yingxi/yolo-master-d1/manifests/wp2-train100-reproducibility.json
```

独立只校验命令：

```bash
python scripts/d1/cache_features.py verify \
  --cache-dir /data/yingxi/yolo-master-d1/feature_cache/wp2-train100-a
```

两次构建的一致性以 cache contract、样本 cache key 和逐 tensor SHA256 为准。safetensors metadata 映射的容器字节顺序不保证规范化，因此不同构建的 shard 文件 SHA256 可以不同；每次构建仍必须用各自 `index.json` 中记录的 shard SHA256 完整校验。

WP2 固定 100 图验收已在代码 commit `5cd566dcdb1e47a9227a81852296ade76d9ba20f` 上完成：

| 指标 | Build A | Build B |
| --- | ---: | ---: |
| 样本 / tensor | 100 / 300 | 100 / 300 |
| cache bytes | 368,791,336 | 368,791,336 |
| 抽取时间 | 12.755 s | 13.821 s |
| 抽取吞吐 | 7.840 images/s | 7.236 images/s |
| 峰值 GPU 显存 | 320,947,200 bytes | 320,947,200 bytes |
| 热缓存读取吞吐 | 2,778.5 MiB/s | 2,759.7 MiB/s |

两次构建的 contract SHA256 均为 `6bfda0e13bde01001c3f3f2d72631a2401fb9a77b146d6fb2794303e379e47a7`，tensor 内容摘要均为 `0102f2e707b369ca8bfb3d996d54fa0d8ba9eba79787501bd5fe6542f90c1a8d`。固定路径列表 SHA256 为 `d2c16d9021e923f4435c14af706048784e558ca57c2addb676d67a46352bb080`。读取吞吐是 100 图单 shard 在操作系统热页缓存下的微基准，不代表完整训练时的冷缓存或多 worker 吞吐；WP8 仍需记录正式训练的数据等待比例。

Git 证据位于：

- `experiments/d1/manifests/wp2-cache-100-reproducibility.json`
- `experiments/d1/manifests/wp2-cache-100-index-a.json`
- `experiments/d1/manifests/wp2-cache-100-index-b.json`
- `experiments/d1/manifests/wp2-cache-100-samples.jsonl`

## 6. 实现工作包

### WP0：实验合同与数据划分

- [x] 新建 `ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml`。
- [x] 锁定完整 COCO 2017 `train2017`/`val2017` manifest，不二次随机划分。
- [x] 记录数据许可、教师许可、commit、环境和权重哈希。
- [x] 明确输入尺寸、letterbox、归一化、层编号和缓存版本。

### WP1：DINOv3 多层特征

- [x] 扩展 `DINOv3Teacher`，支持显式 `output_layers`。
- [x] 保持默认单层行为兼容现有 Foundation 蒸馏测试。
- [x] 返回 block4/block8/block12 的 BCHW 特征。
- [x] 在 metadata 中记录层编号、patch size、grid、hidden dim 和 prefix token 数量。
- [x] 断言教师始终为 eval、无梯度且不在 optimizer。

### WP2：特征缓存

- [x] 新增 `ultralytics/nn/foundation/cache.py` 或等价职责模块。
- [x] 支持分片写入、原子提交、中断续跑、只校验模式。
- [x] cache key 覆盖图像哈希、预处理、教师权重、层编号和 dtype。
- [x] manifest 记录 sample ID、split、shard、tensor key、shape 和 checksum。
- [x] 完成固定 100 张图的两次独立构建与一致性验证。
- [x] 记录磁盘占用、抽取耗时、峰值显存和读取吞吐。

### WP3：多尺度适配器

- [x] 新增 `DINOFeaturePyramidAdapter`。
- [x] 每个 DINO 层分别投影到 P3/P4/P5 对应通道。
- [x] P3 使用 2 倍上采样，P4 保持 stride 16，P5 使用 stride-2 卷积。
- [x] 使用适合小 batch 的 GroupNorm。
- [x] 输出三组同尺度候选特征供 LatentMixture 融合。

### WP4：D1 检测模型

- [x] 新增独立的 D1 Foundation Detection Model，输入为特征字典而非 RGB 图像。
- [x] 连接 Adapter、三个 LatentMixture 和 Detect。
- [x] 显式设置 Detect stride `[8, 16, 32]`，不使用 RGB dummy forward 推断。
- [x] 使用现有 E2E/Detect loss 和 `CompositeCriterion`。
- [x] checkpoint 只保存下游模型参数、配置和教师引用信息。

### WP5：缓存 Dataset、Trainer 与 Validator

- [x] Dataset 根据 `im_file`/sample ID 查询缓存，同时保留 COCO 标签与几何元数据。
- [x] Trainer 将缓存特征移动到设备，不再对特征执行 RGB `/255` 处理。
- [x] Validator 使用相同缓存并正确恢复预测框坐标。
- [x] 保留 online 模式，仅用于验证在线特征与缓存特征一致。
- [x] 正式 P0 训练和评测统一使用 cache 模式。

### WP6：Latent aux 闭环

- [x] 确认三个 LatentMixture 均发布 `kind="latent"`。
- [x] 确认 D1 criterion 未绕过 `CompositeCriterion`。
- [x] 记录 balance loss、z-loss、latent aux 和总 aux。
- [x] 证明 latent aux 保持计算图连接且 Router 获得非零梯度。
- [x] 确认 aux 关闭时结果为 0，且每个模块每步只收集一次。

### WP7：测试与最小训练

- [x] 教师多层形状、特殊 token、冻结和确定性测试。
- [x] 缓存 manifest、失效条件、续跑和 checksum 测试。
- [x] 在线特征与缓存特征的 FP16 容差测试。
- [x] Adapter 输出 P3/P4/P5 形状测试。
- [x] 单 batch 前向、loss、反向和有限值测试。
- [x] 32 图过拟合测试。
- [x] COCO8 一 epoch 端到端测试。

### WP8：完整 COCO 2017 正式 P0 运行

- [ ] 固定单 seed 训练，不在正式运行中继续调参。
- [ ] 保存训练配置、日志、checkpoint、metrics JSON 和环境信息。
- [ ] 输出 COCO mAP50-95、mAP50、训练时长和峰值显存。
- [ ] 输出缓存大小、抽取耗时、读取吞吐和数据等待比例。
- [ ] 补齐训练配方、缓存说明、接口维度表和已知局限。

## 7. 建议文件布局

```text
experiments/d1/
  README.md                         # 本文档，研发与复现入口
  WP0.md ... WP8.md                # 各阶段完成报告与当前门禁状态
  manifests/                       # 小型数据/缓存/运行 manifest
  results/                         # CSV/JSON 汇总，不放大 checkpoint

scripts/d1/
  prepare_wp0.py                    # WP0 数据、权重与 manifest 准备
  cache_features.py                 # WP2 缓存构建、校验与比较
  run_wp7.py                        # WP7 parity、最小训练和验收汇总
  run_wp8.py                        # WP8 六卡缓存调度、基准与合并

ultralytics/cfg/experiments/d1/
  p0-dinov3-vits16-coco2017.yaml

ultralytics/cfg/models/26/
  yolo26-d1-dinov3-latent-n.yaml   # 下游 Adapter/Latent/Detect 配置

ultralytics/nn/foundation/
  teachers/dinov3.py               # 扩展多层输出
  cache.py                         # 新增缓存与 manifest 协议

ultralytics/nn/
  foundation_detection_model.py    # D1 缓存特征检测模型

ultralytics/nn/modules/
  foundation_adapter.py            # DINO 多尺度适配器

ultralytics/models/yolo/detect/
  foundation_train.py              # 名称按最终扩展方式确定
  foundation_val.py

tests/
  test_d1_wp0_contract.py
  test_d1_wp1_dinov3.py
  test_d1_wp2_feature_cache.py
  test_d1_wp2_cache_cli.py
  test_d1_wp3_foundation_adapter.py
  test_d1_wp4_foundation_detection_model.py
  test_d1_wp5_cached_pipeline.py
  test_d1_wp6_latent_aux.py
```

文件名是当前建议，实施时可以依据现有模块边界微调，但职责不得混入 `smoke/d1/`。

## 8. 接口维度表

以输入 `640×640`、DINOv3 ViT-S/16 为准：

| 阶段 | 名称 | 形状 | stride | 是否训练 |
| --- | --- | --- | --- | --- |
| Teacher | block4 | `B×384×40×40` | 16 | 否 |
| Teacher | block8 | `B×384×40×40` | 16 | 否 |
| Teacher | block12 | `B×384×40×40` | 16 | 否 |
| Adapter | P3 candidates | `3 × B×64×80×80` | 8 | 是 |
| Adapter | P4 candidates | `3 × B×128×40×40` | 16 | 是 |
| Adapter | P5 candidates | `3 × B×256×20×20` | 32 | 是 |
| Latent | P3' | `B×64×80×80` | 8 | 是 |
| Latent | P4' | `B×128×40×40` | 16 | 是 |
| Latent | P5' | `B×256×20×20` | 32 | 是 |
| Detect | predictions | 由 `nc=80`、`reg_max` 和 E2E 模式决定 | 8/16/32 | 是 |

## 9. 缓存容量预算

三层 FP16 原始特征的理论大小：

```text
3 × 384 × 40 × 40 × 2 bytes ≈ 3.5 MiB / image
100 images ≈ 350 MiB
train2017（118,287 images）≈ 406 GiB
val2017（5,000 images）≈ 17 GiB
完整 COCO 2017 原始三层特征 ≈ 423 GiB
```

实际结果还包括 safetensors 索引、文件系统开销和元数据，建议为特征缓存预留至少 500 GiB 本地 SSD。P0 不通过减少到单层或抽取 COCO 子集来规避容量问题；磁盘不足时应增加存储、优化分片或在不破坏特征精度的前提下采用经过验证的压缩方案。

## 10. 服务器存储与备份

### 10.1 当前工作区

当前服务器的大文件工作区为：

```text
/data/yingxi/yolo-master-d1/
├── datasets/coco/
├── weights/teachers/
├── feature_cache/dinov3-vits16-coco2017-640-fp16/
├── runs/
├── staging/
└── manifests/
```

`/data` 挂载自共享 NFS NVMe 池 `10.210.22.253:/nvme-pool`。建立工作区时全池剩余约 2.1 TiB、使用率 98%，因此它只作为临时工作盘，不作为重要内容的唯一副本。D1 工作区应控制在约 600 GiB 内，并在缓存生成期间持续检查：

```bash
du -sh /data/yingxi/yolo-master-d1
df -h /data
```

训练代码和提交配置不得写死该绝对路径；通过运行参数、环境变量或本地未提交配置注入数据、缓存和输出目录。

### 10.2 备份职责

| 目录/内容 | 是否备份 | 长期保存位置 | 说明 |
|---|---|---|---|
| `datasets/coco` | 备份源文件 | 交大云盘 | 保存 COCO 官方压缩包和校验值，不重复备份解压副本 |
| `weights/teachers` | 必须 | 交大云盘 | 保存 teacher 权重、来源、版本和 SHA256 |
| `feature_cache` | 建议 | 交大云盘 | 生成成本低时只保存 manifest；成本高时备份完整分片 |
| `runs` 中的重要 checkpoint | 必须 | 交大云盘 | 至少保存 best、last 和最终评测 checkpoint |
| 指标、配置、日志摘要、图表 | 必须 | Git + 交大云盘 | 不向 Git 提交大型日志和 checkpoint |
| `manifests` | 必须 | Git + 交大云盘 | 包括数据、权重、缓存和结果的校验清单 |
| `staging` | 不备份 | 无 | 未完成的临时分片，可随时重建 |

当前使用新版 Pan 交大云盘 `https://pan.sjtu.edu.cn/`，建议建立：

```text
YOLO-Master-D1/
├── datasets/coco2017-source/
├── weights/dinov3-vits16/
├── feature_cache/dinov3-vits16-coco2017-640-fp16/
├── checkpoints/
├── results/
└── manifests/
```

完整备份特征缓存前，必须确认云盘可用空间大于预计上传量。不要为了上传而在 `/data` 中生成第二份完整压缩包。

### 10.3 缓存分片与校验

全量缓存不得保存为单个约 423 GiB 文件，也不建议每张图片一个小文件。缓存生成器应输出 1--4 GiB 的不可变分片，并用 `index.json` 记录样本到分片的映射：

```text
feature_cache/dinov3-vits16-coco2017-640-fp16/
├── train-00000.safetensors
├── train-00001.safetensors
├── ...
├── val-00000.safetensors
└── index.json
```

缓存完成后生成校验清单：

```bash
cd /data/yingxi/yolo-master-d1
find feature_cache/dinov3-vits16-coco2017-640-fp16 \
  -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > manifests/feature-cache.sha256
```

恢复后验证：

```bash
cd /data/yingxi/yolo-master-d1
sha256sum -c manifests/feature-cache.sha256
```

数据源、teacher 权重和关键 checkpoint 也应生成对应 SHA256 清单。每个分片完成并原子改名后即可上传，不必等待全部分片结束。

### 10.4 上传与恢复

服务器没有官方 Pan Linux 同步客户端。使用 Windows 交大云盘客户端同步时，通过 WinSCP/SFTP 连接当前服务器，从 `/data/yingxi/yolo-master-d1` 将完成的分片、权重和 checkpoint 下载到 Pan 同步目录。服务器 IP 和端口以平台当前页面为准，凭据不得写入仓库。

恢复顺序：

1. clone 仓库并检出实验记录的 commit；
2. 恢复 COCO 源文件和 teacher 权重并验证 SHA256；
3. 有缓存备份时恢复分片并运行 `sha256sum -c`；
4. 无缓存备份时按 manifest 和固定配置重新抽取；
5. 恢复 checkpoint，运行验证命令确认指标和环境契约。

正式训练期间遵守“先生成 manifest 和校验值，再上传大文件，最后提交 Git 证据”的顺序。

## 11. 测试门槛

进入下一阶段前必须满足：

1. **Teacher gate**：三层输出正确，教师参数无梯度，训练前后权重哈希不变。
2. **Cache gate**：100 张图缓存可重复生成，在线/缓存特征在 FP16 容差内一致。
3. **Shape gate**：P3/P4/P5 严格为 stride 8/16/32，通道与 Detect 一致。
4. **Loss gate**：检测损失和 latent aux 有限，`counts_by_kind["latent"] == 3`。
5. **Gradient gate**：Adapter、LatentMixture、Detect 有梯度，DINOv3 无梯度。
6. **Overfit gate**：32 图训练损失明显下降，预测从随机状态转为有效拟合。
7. **Pipeline gate**：COCO8 一 epoch 完成训练、验证、保存和重新加载。
8. **P0 gate**：完整 COCO 2017 `train2017` 训练和 `val2017` 评测完成，产物与成本记录齐全。

## 12. P0 交付证据

Git 中应包含：

- 代码与单测；
- P0 配置和确定性数据 split manifest；
- 100 图缓存 manifest、SHA256 和容量/I/O 报告；
- 接口维度表；
- 完整复现命令；
- 训练日志摘要、metrics JSON/CSV、成本统计；
- checkpoint 和大缓存的云盘索引及校验值；
- 已知局限和失败记录。

Git 中不得包含：

- COCO/VisDrone 原始数据；
- DINOv3 权重；
- 大规模特征缓存；
- 完整训练 checkpoint；
- 仅对当前服务器有效的绝对路径。

## 13. 风险与降级

- DINOv3 权重不可获得或许可不明确：按任务书降级到许可明确的 DINOv2，并在配置和报告中显式声明。
- 多层输出接口不稳定：优先使用模型官方 `feature_maps`/hidden states 接口，必要时使用受测试保护的 hooks。
- 缓存 I/O 成为瓶颈：增加 shard 大小、预取和 pinned memory，缓存放本地 SSD。
- FP16 特征误差过大：仅对问题层改用 BF16/FP32，并重新记录容量与性能。
- 自定义模型 stride 初始化失败：由接口元数据显式设置 `[8,16,32]`，增加 Detect 解码测试。
- VisDrone 小目标在 stride-16 教师特征中丢失：属于 P1 数据域问题，P0 不通过伪造 P3 细节解决。

## 14. 建议执行顺序

```text
WP0 实验合同
 -> WP1 DINO 多层输出
 -> WP2 100 图缓存
 -> WP3 Adapter
 -> WP4 D1 Model
 -> WP5 Trainer/Validator
 -> WP6 Aux 闭环
 -> WP7 最小训练测试
 -> WP8 完整 COCO 2017 正式 P0
```

每个工作包完成后应立即提交对应测试和文档，不等到 P0 结束后一次性补齐。

## 15. P0 完成定义

P0 只有在以下条件全部满足后才完成：

- [ ] 冻结 DINOv3 可重复抽取三层特征；
- [ ] 100 图缓存可复现并有磁盘/I/O/显存估算；
- [ ] 缓存特征可生成标准 P3/P4/P5；
- [ ] LatentMixture 与 Detect 完成训练和评测；
- [ ] latent aux 确实进入总损失并向 Router 反传；
- [ ] 完整 COCO 2017 有可复现的训练结果和 `val2017` 指标；
- [ ] 配置、命令、日志、测试、manifest 和云盘索引齐全；
- [ ] 代码不依赖当前服务器的临时目录或绝对路径。
