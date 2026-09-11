# D1：冻结 DINOv3 多层特征检测

冻结 DINOv3 ViT-S/16，复用一次性特征缓存，只训练多尺度 Adapter、LatentMixture 和 YOLO26 Detect。本文是代码阅读、最小复现和结果索引入口；历史开发日志不再放在当前 PR 的主目录。

## 1. 当前结果与边界

| 项目 | 已完成内容 | 结论边界 |
| --- | --- | --- |
| P0 | 多层 Teacher、缓存、Adapter、训练/验证、aux、严格重载与恢复；已有完整 COCO 训练评测 | 训练闭环完成不等于成本目标达成 |
| 架构筛选 | COCO 固定第 50 轮：BASE AP 28.870、DW 28.593、BN64 29.745 | 单 seed，不能作为正式同参 P1 结论 |
| P2 aux | VisDrone 两阶段共 36 次独立训练，三个 seed，固定第 60 轮官方评分 | gain0.1 平均 AP 8.132；相对 gain0 仅 +0.110 点，不足以证明稳定收益 |
| P1 | 已选总参数匹配的 YOLO26-L width0.9375 | 完整配对训练、显存和成本验收未完成，不宣称降低 50% |
| Teacher 尺寸对比 | 未完成 | 不计为已完成成果 |

后续冻结架构使用 BN64、weighted_sum；E3 按预先规则选定 balance=0.1、z=0、gain=0.1、budget=3.0。历史运行使用各自记录的配置，不用新配置重标旧结果。

## 2. 实现边界

```text
RGB -> 固定 LetterBox 640 + RGB [0,1] + Teacher normalization
    -> 冻结 DINOv3 blocks 4/8/12，三层均为 [B,384,40,40]
    -> 可校验的 FP16 safetensors / NPY 缓存
    -> 九条独立 Adapter 分支
    -> P3/P4/P5 各三组候选
    -> 三个 LatentMixture
    -> YOLO26 end-to-end Detect
```

- block4/8/12 都是 stride16，不是原生 P3/P4/P5。
- 默认 Teacher API 保留 dense["p4"]；显式 output_layers=(4,8,12) 才返回 block4/block8/block12。
- P3/P4/P5 输出尺寸为 80/40/20；下游通道为 64/128/256。
- BN64 的 P5 是 384->64 的 1x1 投影，再用 3x3 stride2 卷积变为 256 通道；BN64 不是 BatchNorm。
- Teacher 始终冻结、eval/inference，不进入 student optimizer、DDP、EMA 或 checkpoint。
- collect_aux_loss 显式包含 latent；总损失只计入一次模型级 aux，报告 raw/effective aux。
- 固定缓存不支持随意打开几何增强、Mosaic 或随机多尺度，否则特征与标签不再对应。

### 核心代码

| 文件 | 职责 |
| --- | --- |
| [dinov3.py](../../ultralytics/nn/foundation/teachers/dinov3.py) | 兼容默认模式的多层 Teacher 输出 |
| [cache.py](../../ultralytics/nn/foundation/cache.py)、[npy_cache.py](../../ultralytics/nn/foundation/npy_cache.py) | 分片、摘要、断点与 NPY 读取 |
| [foundation_adapter.py](../../ultralytics/nn/modules/foundation_adapter.py) | 多层到多尺度的独立可训练分支 |
| [foundation_detection_model.py](../../ultralytics/nn/foundation_detection_model.py) | 缓存特征检测模型与 aux 日志 |
| [d1_cache.py](../../ultralytics/data/d1_cache.py) | 缓存与标签配对 Dataset |
| [foundation_train.py](../../ultralytics/models/yolo/detect/foundation_train.py)、[foundation_val.py](../../ultralytics/models/yolo/detect/foundation_val.py) | Trainer / Validator 接入 |
| [mixture_loss.py](../../ultralytics/nn/mixture_loss.py) | 统一损失收集和一次性标量 aux |
| [模型配置目录](../../ultralytics/cfg/models/26/) | BASE、DW、BN64 和历史/总参数 Scratch |

正式入口与历史复现实验共用部分脚本函数，因此保留实际依赖的脚本，不按文件名中的 WP 数字直接删除代码。当前 PR 的分类清单见 [PR_SCOPE.md](PR_SCOPE.md)。

## 3. 环境与数据

D1 实测环境使用 Python 3.11；Foundation 依赖需 Python >=3.10。仓库其他路径的最低 Python 支持范围不因此改变。

```bash
pip install -e ".[dev,foundation]"
export D1_WORK=/path/to/d1-work
```

查看 [Teacher 来源与 SHA256](manifests/dinov3-vits16.json)、[预处理合同](manifests/p0-experiment-contract.json)、[COCO 划分摘要](manifests/coco2017-splits.json)、[许可证来源](manifests/licenses.md)。

首次准备完整 COCO 和 Teacher（涉及大文件下载和校验）：

```bash
python -m scripts.d1.prepare_wp0 --workspace "$D1_WORK" --download
```

已有完整 COCO 时，只恢复本地路径列表，不下载、不加载 Teacher、不改写发布 manifest：

```bash
python -m scripts.d1.prepare_wp0 --workspace "$D1_WORK" \
  --materialize-splits-from /path/to/coco
```

COCO 根目录须包含 images/train2017 与 images/val2017。两份列表不再提交 Git；生成时检查 118,287/5,000 数量、互斥性、排序及已发布 SHA256。此模式只验证划分身份，不替代图像内容/压缩包完整性校验。`--download` 或 `--verify-only` 会重新生成 provenance；历史 manifest 不能以新机器输出覆盖后仍声称是旧实验环境。

VisDrone 使用完整 DET train=6,471、val=548；数据转换、ignore 语义和官方 MATLAB 评测见 [E2.md](E2.md)。

## 4. 最小训练闭环

以下只构建 100 图工程缓存并运行 4 图训练/4 图验证的 1 epoch；不是完整精度结论。首次准备已将数据与 Teacher 放在 D1_WORK 的默认子目录，其他布局使用 CLI 的 --data-root/--weights-dir 显式传入。

```bash
python -m scripts.d1.cache_features build --workspace "$D1_WORK" \
  --cache-dir "$D1_WORK/feature_cache/repro100" --split train2017 \
  --limit 100 --batch-size 8 --device 0

python -m scripts.d1.cache_features verify \
  --cache-dir "$D1_WORK/feature_cache/repro100"

python -m scripts.d1.run_wp7 --workspace "$D1_WORK" \
  --cache-dir "$D1_WORK/feature_cache/repro100" --device cuda:0 parity

python -m scripts.d1.run_wp7 --workspace "$D1_WORK" \
  --cache-dir "$D1_WORK/feature_cache/repro100" --device cuda:0 train --profile coco8
```

工程 profile 使用其固定合同，不等同于最新 BN64 的正式配方。全量缓存入口为 [run_wp8.py](../../scripts/d1/run_wp8.py)，完整训练入口为 [run_wp8_train.py](../../scripts/d1/run_wp8_train.py)；分别运行 --help 查看显式路径与子命令。

NPY 转换入口 [convert_cache_to_npy.py](../../scripts/d1/convert_cache_to_npy.py) 是逐片校验后退役源分片的迁移工具，必须先理解其 --retire-verified-source 行为，不应当作无副作用的复制命令。完整缓存和 RGB 数据建议放在有足够容量的本地高速存储；临时盘的持久性由运行环境决定。

## 5. 研究结果与复现身份

| 文档/证据 | 内容 |
| --- | --- |
| [P5_FAST_RUN_20260909.md](P5_FAST_RUN_20260909.md) | BASE/DW/BN64 的配对初始化、精度、实际成本及资源干扰 |
| [E3.md](E3.md) | balance/z/gain 两阶段三种子消融与原始训练身份 |
| [E3 第一阶段](manifests/e3-stage1-official-20260911.json)、[第二阶段](manifests/e3-stage2-official-20260911.json) | 固定末轮官方结果、逐 seed 差值及成本 |
| [SCRATCH_TOTAL_PARAMETER_BASELINE.md](SCRATCH_TOTAL_PARAMETER_BASELINE.md) | 总参数基线的结构、参数量和前向验收 |
| [P1_P2_EXPERIMENT_PLAN.md](P1_P2_EXPERIMENT_PLAN.md) | 当前后续实验合同；计划不冒充已完成结果 |
| [P0 完整评测](manifests/final-evaluation-20260907.json)、[发布校验](manifests/publication-20260907.json) | 早期 100 epoch 运行的结果与代码关系 |
| [最小工程验收](manifests/wp7-summary.json) | 100 图 parity、32 图过拟合与 1 epoch 闭环 |
| [历史归档](ARCHIVE.md) | 旧阶段文档、准入资料与一次性诊断的固定版本索引 |

早期 P0 的 COCO AP 11.972 是独立历史配置；BN64 的 AP 29.745 是另一个固定 50 轮筛选，不能混为同一次训练或直接归因于单个变化。

历史重放使用记录的训练 commit、配置和工作区身份。不要在改过源码的工作区直接 resume 旧 run；新版本需新建 run 并完成门禁。执行提交、归档提交和最终 PR 提交分别记录。

## 6. 测试与交付

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_d1_*.py \
  tests/test_foundation_dinov3.py tests/test_foundation_distill_model.py \
  tests/test_latent_mixture.py tests/test_mixture_loss_composition.py
git diff --check
```

真实 Teacher/CUDA 集成测试按各测试中显式环境变量启用；普通离线 CI 不下载模型。实际清理和上游兼容验证结果记录在 PR_SCOPE.md，不把历史测试数量累加成一次新测试结果。

Git 保留实现、配置、必要回归、小型证据和结果报告。数据集、Teacher 权重、缓存、checkpoint、完整预测、完整运行日志留在外部工作区；不随 PR 提交。P1 的正式精度/成本与 Teacher 尺寸对照尚未完成，不能据工程测试宣称验收达标。
