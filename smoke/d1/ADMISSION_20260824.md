# D1｜冻结基础模型 x Latent Mixture 适配头：8.24 准入检查

> 本文记录腾讯犀牛鸟 YOLO-Master D1 课题的 2026-08-24 准入结果。准入目标是验证
> **100 张图冻结特征缓存可复现、资源成本可量化、接口维度可审计，以及现有 LatentMixture
> 最小训练/推理链路可运行**。这不是 P0 精度报告，也不代表 Foundation 特征到
> P3/P4/P5 的完整适配接线已经完成。

## 1. 基本信息

| 字段 | 内容 |
| --- | --- |
| 项目 | 腾讯犀牛鸟开源人才计划 2026 · YOLO-Master |
| 课题 | D1 · 冻结基础模型 x Latent Mixture 适配头 |
| 负责人 | fengyanqi / Frank95zz |
| 仓库 | `https://github.com/Frank95zz/YOLO-Master` |
| 工作分支 | `feat/topic-d1-fengyanqi` |
| 准入执行 commit | `511b26754fd1c4d76407dadf26dc96539925d5ed` |
| 执行日期 | 2026-08-24 |
| 准入结论 | **PASS**；允许进入 D1 P0 接线和正式实验阶段 |

执行准入时工作树干净，分支相对 `origin/feat/topic-d1-fengyanqi` 为 `ahead 3`。因此本页所列
commit 和证据首先以服务器本地仓库为准，推送后再补固定 GitHub 链接。

## 2. 课题定位与准入边界

D1 的正式目标是冻结 DINO 系列视觉基础模型，提取多层特征，接入 Latent Mixture 与检测头，
并完成 train/predict。当前仓库已经具备 LatentMixture、辅助损失、Foundation teacher 和蒸馏包装器；
本次准入新增的是可审计的 100 图缓存与复现工具。

| 范围 | 本次验证内容 | 状态 |
| --- | --- | --- |
| 冻结 Foundation backbone | DINOv2 ViT-S/14 共 `22,056,576` 个参数，冻结数同为 `22,056,576` | 已验证 |
| 多层特征 | 缓存 block 3/6/9/12 空间特征及最终 CLS 特征 | 已验证 |
| 缓存复现 | 两次独立运行逐图比较源图与特征张量 SHA256 | 已验证，`100/100` 一致 |
| 资源边界 | 实测编码吞吐、缓存体积、读取吞吐、峰值显存和剩余磁盘 | 已验证 |
| LatentMixture 最小链路 | `coco8` 训练 1 epoch，并由 `best.pt` 预测 4 张图 | 已验证 |
| Foundation -> P3/P4/P5 适配 | resize、通道投影及与 LatentMixture/Head 的正式接线 | **未完成，属于 P0** |
| 正式精度结论 | COCO-mini 与 VisDrone/工业小目标数据上的同预算对照 | **未开展，属于 P0/P1** |

本次训练 smoke 的日志明确记录 `foundation_enabled=False`。因此训练结果只证明现有
LatentMixture train/predict 链路可用；Foundation 缓存和 LatentMixture 训练是两个分别通过的
准入边界，不在本文中合并表述为端到端 D1 已完成。

## 3. 8.24 准入状态

| 检查项 | 实现与证据 | 结果 |
| --- | --- | --- |
| 环境安装 | Python、PyTorch、CUDA、Ultralytics 和 GPU 信息均写入日志/缓存摘要 | PASS |
| 基线/最小任务 | 固定 commit；COCO test2017 mini-100 缓存；`coco8` 1 epoch train/predict | PASS |
| 复现入口 | `smoke/d1/run_admission.sh` 一键执行，工作树非干净时 fail closed | PASS |
| 数据锁定 | 100 张不同官方 COCO test2017 图片；记录 URL、文件 SHA256 和 manifest SHA256 | PASS |
| 模型锁定 | 记录实际 teacher、权重大小、SHA256、参数总数和冻结参数数 | PASS |
| 缓存复现 | 两次缓存 `100/100` tensor hash 一致，无 mismatch index | PASS |
| 接口维度 | 输入、四层空间特征、pooled 特征和 P3/P4/P5 目标维度均有表格 | PASS |
| 资源评估 | 100 图缓存 `75.34 MiB`；估算 10 万图 `73.58 GiB` | PASS |
| D1 测试 | 5 个测试文件共 `59 passed in 3.63s` | PASS |
| 训练/预测 | 1 epoch 完成，3 个 checkpoint 生成，4 张预测图输出 | PASS |
| 精度结论 | smoke mAP 为 0，仅用于链路验证，不作为精度结论 | N/A |

## 4. 锁定环境与输入

### 4.1 环境矩阵

| 项目 | 实测值 |
| --- | --- |
| 服务器 | `root@10.210.22.36:30722` |
| 工作区 | `/root/yolo-master` |
| 代码仓库 | `/root/yolo-master/repo` |
| Conda 环境 | `/root/yolo-master/.conda/d1` |
| Python | `3.11.15` |
| Ultralytics | `8.4.101`，editable install 指向当前仓库 |
| PyTorch | `2.6.0+cu124` |
| CUDA runtime / driver | `12.4` / `550.54.15` |
| GPU | `6 x NVIDIA A40`；本次使用 GPU 0，单卡显存 `45,619 MiB` |
| 训练配置 | `ultralytics/cfg/models/26/yolo26-master-latent-n.yaml` |

### 4.2 数据与模型锁定

| 对象 | 实际输入 | 完整性证据 |
| --- | --- | --- |
| 数据 | COCO test2017 官方 image-info 固定排序后的前 100 张不同图片 | `image_count=100`；source manifest SHA256 `6942b9c6ae77d4c899a64e990745be3912029615ab03e463d11a05c23c95791a` |
| COCO image-info | `image_info_test2017.zip` | SHA256 `e52f412dd7195ac8f98d782b44c6dd30ea10241e9f42521f67610fbe055a74f8` |
| 目标 teacher | DINOv3 ViT-S/16 | 官方权重为 gated，准入阶段不可直接取得 |
| 实际 teacher | Meta DINOv2 ViT-S/14 官方预训练权重 | SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| 权重文件 | `dinov2_vits14_pretrain.pth`，`88,283,115` bytes | 参数总数与冻结参数数均为 `22,056,576` |

DINOv2 权重来源：

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

本次按任务书允许的降级路径使用 DINOv2，只验证冻结、多层特征、缓存、接口与资源成本，
不声称获得 DINOv3 实验结果。DINOv3 适配器的冻结、预处理和输出协议由
`tests/test_foundation_dinov3.py` 覆盖。

## 5. 代码复用与新增入口

### 5.1 仓库已有能力

| 代码入口 | 作用 |
| --- | --- |
| `ultralytics/nn/modules/latent_mixture.py` | LatentMixture 多输入路由与融合 |
| `ultralytics/nn/mixture_loss.py` | 路由辅助损失收集与预算归一化 |
| `ultralytics/nn/modules/routing_protocol.py` | 路由状态和训练期协议 |
| `ultralytics/nn/foundation/teachers/dinov3.py` | DINOv3 teacher 适配器 |
| `ultralytics/nn/foundation_distill_model.py` | Foundation teacher/student 蒸馏包装器 |
| `ultralytics/cfg/models/26/yolo26-master-latent-n.yaml` | 本次 LatentMixture smoke 模型 |

### 5.2 本次准入工具

| 文件 | 作用 |
| --- | --- |
| `smoke/d1/prepare_coco_mini.py` | 从官方 COCO 元数据准备并校验 100 图固定样本 |
| `smoke/d1/build_feature_cache.py` | 冻结 teacher，抽取多层特征，写入 manifest/summary/resource report |
| `smoke/d1/compare_feature_caches.py` | 比较两次缓存的图片与张量哈希，差异时返回失败 |
| `smoke/d1/run_admission.sh` | 串联数据、测试、双缓存、比较、训练和预测 |

## 6. 复现命令

### 6.1 一键复现完整准入

```bash
ssh root@10.210.22.36 -p 30722
cd /root/yolo-master/repo
bash smoke/d1/run_admission.sh
```

脚本会执行以下步骤：

1. 从 COCO 官方源准备并校验 100 张不同图片。
2. 拒绝脏工作树并锁定当前 Git commit。
3. 记录数据元信息、teacher 权重、GPU 和磁盘状态。
4. 运行 5 个 D1 测试文件。
5. 独立执行两次特征缓存并比较逐图 tensor SHA256。
6. 运行 `coco8` 1 epoch LatentMixture 训练和 checkpoint predict。
7. 写入日志、缓存和 run 的 latest 指针以及最终 `admission-result.txt`。

### 6.2 独立核验已有结果

```bash
LOG_DIR=/root/yolo-master/logs/d1-admission-511b26754fd1-20260824T191535
CACHE_RUN1=/root/yolo-master/feature_cache/d1-admission-511b26754fd1-20260824T191535-run1
CACHE_RUN2=/root/yolo-master/feature_cache/d1-admission-511b26754fd1-20260824T191535-run2

cat "$LOG_DIR/admission-result.txt"
cat "$CACHE_RUN1/summary.json"
cat "$CACHE_RUN1/dimension_table.md"
cat "$CACHE_RUN1/resource_report.md"

/root/yolo-master/.conda/d1/bin/python smoke/d1/compare_feature_caches.py \
  "$CACHE_RUN1" "$CACHE_RUN2" \
  --output "$LOG_DIR/repeatability-report-recheck.json"
```

测试、训练和预测的完整参数以 `smoke/d1/run_admission.sh` 为唯一口径，避免 README 命令与
实际执行脚本漂移。

## 7. 结果证据

### 7.1 特征与接口维度

| Stage | Tensor shape | Dtype | 说明 |
| --- | --- | --- | --- |
| Input | `B x 3 x 224 x 224` | FP32 | resize + ImageNet normalize |
| DINOv2 block 3 | `B x 384 x 16 x 16` | FP16 cache | patch size 14 |
| DINOv2 block 6 | `B x 384 x 16 x 16` | FP16 cache | 多层空间特征 |
| DINOv2 block 9 | `B x 384 x 16 x 16` | FP16 cache | 多层空间特征 |
| DINOv2 block 12 | `B x 384 x 16 x 16` | FP16 cache | 最终空间特征 |
| Global pooled | `B x 384` | FP16 cache | 归一化 CLS token |
| P3 adapter target | `B x 256 x 28 x 28` | FP32/AMP | P0 需 resize + channel projection |
| P4 adapter target | `B x 512 x 14 x 14` | FP32/AMP | P0 需 resize + channel projection |
| P5 adapter target | `B x 1024 x 7 x 7` | FP32/AMP | P0 需 resize + channel projection |

### 7.2 复现性与资源实测

| 指标 | 实测值 |
| --- | --- |
| 独立缓存次数 | 2 |
| 两次缓存图片数 | `100 / 100` |
| 一致 tensor hash | `100 / 100`，mismatch indices 为空 |
| 聚合特征 SHA256 | `38e33873159d69389c1761ed4c128261f5dcb90e6826b22a60d8d740d3ea90ed` |
| 编码耗时 / 吞吐 | `2.293 s` / `43.61 images/s`，batch 8 |
| 100 图源文件体积 | `15.40 MiB` |
| 100 图缓存体积 | `75.34 MiB`，即 `0.753 MiB/image` |
| 10 万图缓存估算 | `73.58 GiB` |
| 缓存读取吞吐 | `585.32 MiB/s` |
| 峰值 CUDA allocated / reserved | `151.75 MiB` / `192.00 MiB` |
| 运行后剩余磁盘 | `643.07 GiB` |

### 7.3 测试、训练与预测

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| D1 单元测试 | `59 passed in 3.63s` | `pytest-d1.txt` |
| LatentMixture train | `coco8`、1 epoch、GPU 0，完成且无 traceback | `train-smoke.txt` |
| 辅助损失可观测 | 日志列包含 `mixture_aux_loss`，本次值为 `3` | `train-smoke.txt` |
| Checkpoint | `best.pt`、`last.pt`、`last_healthy.pt` | train run 的 `weights/` |
| Predict | `best.pt` 成功处理 4 张 val 图片并保存 4 个 JPG | `predict-smoke.txt` 和 predict run |
| Smoke 指标 | mAP50 与 mAP50-95 均为 `0` | 只训练 1 epoch 的链路检查，不作精度结论 |

## 8. 证据索引

本次准入 run id：

```text
d1-admission-511b26754fd1-20260824T191535
```

| 证据 | 路径 |
| --- | --- |
| 完整日志 | `/root/yolo-master/logs/d1-admission-511b26754fd1-20260824T191535` |
| 缓存 run 1 | `/root/yolo-master/feature_cache/d1-admission-511b26754fd1-20260824T191535-run1` |
| 缓存 run 2 | `/root/yolo-master/feature_cache/d1-admission-511b26754fd1-20260824T191535-run2` |
| Train run | `/root/yolo-master/runs/d1-admission-511b26754fd1-20260824T191535-train` |
| Predict run | `/root/yolo-master/runs/d1-admission-511b26754fd1-20260824T191535-predict` |
| 最新日志指针 | `/root/yolo-master/logs/latest-d1-admission.txt` |
| 最新主缓存指针 | `/root/yolo-master/feature_cache/latest-d1-admission.txt` |

日志目录中的关键文件：

```text
admission-result.txt
git-commit.txt
git-status.txt
coco-image-info-sha256.txt
dinov2-weights-sha256.txt
pytest-d1.txt
feature-cache-run1.txt
feature-cache-run2.txt
repeatability-report.json
train-smoke.txt
predict-smoke.txt
nvidia-smi-before.txt
nvidia-smi-after.txt
disk-before.txt
disk-after.txt
```

缓存目录中的关键文件：

```text
features/*.pt
manifest.jsonl
summary.json
dimension_table.md
resource_report.md
```

## 9. 风险与降级

| 风险 | 当前处理 | P0/P1 动作 |
| --- | --- | --- |
| DINOv3 权重受 gated license 限制 | 使用官方 DINOv2 ViT-S/14，锁定来源与 SHA256 | 权限获批后替换 DINOv3 并重跑同一准入脚本 |
| 多尺度特征与检测金字塔不对齐 | 已锁定输入/输出与 P3/P4/P5 目标维度 | 实现显式 resize/channel projection，并补 shape/gradient 测试 |
| Foundation 与 LatentMixture 尚未端到端接线 | 两条链路分别测试，文档不作完成声明 | 先完成 P4 最小闭环，再扩展 P3/P5 |
| 10 万图缓存约 `73.58 GiB` | FP16 分图缓存，run id 隔离 | 增加容量预检、分片和清理策略，必要时改在线 teacher |
| Smoke 数据与预算不足以评价精度 | mAP=0 不用于排名或收益判断 | 在同数据、seed、epoch、imgsz 下做 baseline/on 对照 |
| 服务器本地证据尚未推送 | commit、日志和哈希均已锁定 | 推送分支并补 GitHub 固定 commit 链接 |

## 10. Go / No-Go 与下一步

### Go 条件

- [x] Owner、分支、仓库和执行 commit 已锁定。
- [x] Python/CUDA/PyTorch/Ultralytics/GPU 环境已记录。
- [x] 100 张官方 COCO 图片可读取，来源与哈希可审计。
- [x] Foundation 参数全部冻结，四层空间特征和 pooled 特征可缓存。
- [x] 两次独立缓存 `100/100` tensor hash 一致。
- [x] 缓存体积、I/O、GPU 峰值显存和接口维度已量化。
- [x] D1 测试通过，LatentMixture train/predict 最小链路通过。
- [x] 风险触发条件与降级方案已写入本文。

### No-Go 条件

以下任一情况发生时不得进入正式对照实验：teacher 权重/数据来源不可审计、冻结参数数不符、
双缓存哈希不一致、P3/P4/P5 接口维度不匹配、loss/gradient 不可观测，或同预算配置除目标开关外
仍存在未解释差异。

### P0 下一步

1. 实现 `B x 384 x 16 x 16` 到 P3/P4/P5 的 resize 与 channel projection，先锁定 P4 最小闭环。
2. 将冻结 Foundation 特征显式接入 LatentMixture/检测头，补输入 -> 融合 -> loss -> gradient 的测试。
3. 在同数据、seed、epoch、imgsz 和优化器下运行 Foundation off/on 对照，并自动审计配置差异。
4. 扩展到 COCO-mini 与 VisDrone/工业小目标数据；报告精度、吞吐、显存和缓存成本，不只报告 smoke 成功。

**准入结论：PASS。** 该结论仅表示 8.24 准入项已经具备可复现证据；D1 P0 的正式判定以
Foundation -> adapter -> LatentMixture/Head 端到端训练与预测结果为准。
