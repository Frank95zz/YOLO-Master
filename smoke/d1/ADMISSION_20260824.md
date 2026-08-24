# D1｜冻结基础模型 x Latent Mixture 适配头：8.24 准入检查

## 1. 课题定位与准入边界

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


## 2. 8.24 准入状态

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

## 3. 锁定环境与输入

### 3.1 环境矩阵

| 项目 | 实测值 |
| --- | --- |
| Python | `3.11.15` |
| Ultralytics | `8.4.101`，editable install 指向当前仓库 |
| PyTorch | `2.6.0+cu124` |
| CUDA runtime / driver | `12.4` / `550.54.15` |
| GPU | `6 x NVIDIA A40`；本次使用 GPU 0，单卡显存 `45,619 MiB` |
| 训练配置 | `ultralytics/cfg/models/26/yolo26-master-latent-n.yaml` |

### 3.2 数据与模型锁定

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

本次按任务书允许的降级路径使用 DINOv2，只验证冻结、多层特征、缓存、接口与资源成本。

## 4. 代码复用与新增入口

### 4.1 仓库已有能力

| 代码入口 | 作用 |
| --- | --- |
| `ultralytics/nn/modules/latent_mixture.py` | LatentMixture 多输入路由与融合 |
| `ultralytics/nn/mixture_loss.py` | 路由辅助损失收集与预算归一化 |
| `ultralytics/nn/modules/routing_protocol.py` | 路由状态和训练期协议 |
| `ultralytics/nn/foundation/teachers/dinov3.py` | DINOv3 teacher 适配器 |
| `ultralytics/nn/foundation_distill_model.py` | Foundation teacher/student 蒸馏包装器 |
| `ultralytics/cfg/models/26/yolo26-master-latent-n.yaml` | 本次 LatentMixture smoke 模型 |

### 4.2 本次准入工具

| 文件 | 作用 |
| --- | --- |
| `smoke/d1/prepare_coco_mini.py` | 从官方 COCO 元数据准备并校验 100 图固定样本 |
| `smoke/d1/build_feature_cache.py` | 冻结 teacher，抽取多层特征，写入 manifest/summary/resource report |
| `smoke/d1/compare_feature_caches.py` | 比较两次缓存的图片与张量哈希，差异时返回失败 |
| `smoke/d1/run_admission.sh` | 串联数据、测试、双缓存、比较、训练和预测 |

## 5. 复现命令

### 5.1 从克隆仓库开始复现

```bash
# 1. 克隆课题分支
git clone --branch feat/topic-d1-fengyanqi --single-branch \
  https://github.com/Frank95zz/YOLO-Master.git
cd YOLO-Master

# 2. 创建与准入测试一致的 Python/CUDA 环境
conda create -n yolo-master-d1 python=3.11 -y
conda activate yolo-master-d1
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
pip install pytest

# 3. 一键执行完整准入测试
bash smoke/d1/run_admission.sh
```

脚本默认将数据、日志、缓存和训练产物写入仓库同级的 `yolo-master-d1-work/`，不会依赖作者的
服务器目录。需要使用其他数据盘时，可显式覆盖工作目录：

```bash
D1_ROOT=/path/to/yolo-master-d1-work bash smoke/d1/run_admission.sh
```

首次运行需要联网下载 COCO image-info、100 张 COCO test2017 图片、`coco8`，以及 Meta 官方
DINOv2 源码与 ViT-S/14 权重。脚本会校验 DINOv2 权重 SHA256，GPU 0 为默认执行设备。

脚本会执行以下步骤：

1. 从 COCO 官方源准备并校验 100 张不同图片。
2. 拒绝脏工作树并锁定当前 Git commit。
3. 记录数据元信息、teacher 权重、GPU 和磁盘状态。
4. 运行 5 个 D1 测试文件。
5. 独立执行两次特征缓存并比较逐图 tensor SHA256。
6. 运行 `coco8` 1 epoch LatentMixture 训练和 checkpoint predict。
7. 写入日志、缓存和 run 的 latest 指针以及最终 `admission-result.txt`。

### 5.2 独立核验本机结果

```bash
D1_ROOT=${D1_ROOT:-"$(dirname "$PWD")/yolo-master-d1-work"}
LOG_DIR=$(cat "$D1_ROOT/logs/latest-d1-admission.txt")
CACHE_RUN1=$(cat "$D1_ROOT/feature_cache/latest-d1-admission.txt")
CACHE_RUN2=$(sed -n 's/^cache_run2=//p' "$LOG_DIR/admission-result.txt")

cat "$LOG_DIR/admission-result.txt"
cat "$CACHE_RUN1/summary.json"
cat "$CACHE_RUN1/dimension_table.md"
cat "$CACHE_RUN1/resource_report.md"

python smoke/d1/compare_feature_caches.py \
  "$CACHE_RUN1" "$CACHE_RUN2" \
  --output "$LOG_DIR/repeatability-report-recheck.json"
```

测试、训练和预测的完整参数以 `smoke/d1/run_admission.sh` 为唯一口径，避免 README 命令与
实际执行脚本漂移。

## 6. 结果证据

### 6.1 特征与接口维度

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

### 6.2 复现性与资源实测

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

### 6.3 测试、训练与预测

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| D1 单元测试 | `59 passed in 3.63s` | [`pytest-d1.txt`](evidence/d1-admission-511b26754fd1-20260824T191535/logs/pytest-d1.txt) |
| LatentMixture train | `coco8`、1 epoch、GPU 0，完成且无 traceback | [`train-smoke.txt`](evidence/d1-admission-511b26754fd1-20260824T191535/logs/train-smoke.txt) |
| 辅助损失可观测 | 日志列包含 `mixture_aux_loss`，本次值为 `3` | [`train-smoke.txt`](evidence/d1-admission-511b26754fd1-20260824T191535/logs/train-smoke.txt) |
| Checkpoint | 已生成 `best.pt`、`last.pt`、`last_healthy.pt` | 生成记录见 [`train-smoke.txt`](evidence/d1-admission-511b26754fd1-20260824T191535/logs/train-smoke.txt)；权重二进制不提交 Git |
| Predict | `best.pt` 成功处理 4 张 val 图片并保存 4 个 JPG | [`predict-smoke.txt`](evidence/d1-admission-511b26754fd1-20260824T191535/logs/predict-smoke.txt) 和 [`predict/`](evidence/d1-admission-511b26754fd1-20260824T191535/predict/) |
| Smoke 指标 | mAP50 与 mAP50-95 均为 `0` | 只训练 1 epoch 的链路检查，不作精度结论 |

## 7. 证据索引

本次准入 run id：

```text
d1-admission-511b26754fd1-20260824T191535
```

| 证据 | 仓库内路径 |
| --- | --- |
| 证据根目录 | [`evidence/d1-admission-511b26754fd1-20260824T191535/`](evidence/d1-admission-511b26754fd1-20260824T191535/) |
| 完整文本日志 | [`logs/`](evidence/d1-admission-511b26754fd1-20260824T191535/logs/) |
| 缓存 run 1 元数据 | [`cache-run1/`](evidence/d1-admission-511b26754fd1-20260824T191535/cache-run1/) |
| 缓存 run 2 元数据 | [`cache-run2/`](evidence/d1-admission-511b26754fd1-20260824T191535/cache-run2/) |
| 训练参数与结果表 | [`train/`](evidence/d1-admission-511b26754fd1-20260824T191535/train/) |
| 预测结果图片 | [`predict/`](evidence/d1-admission-511b26754fd1-20260824T191535/predict/) |

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

每份缓存元数据目录包含：

```text
manifest.jsonl
summary.json
dimension_table.md
resource_report.md
```

训练结果目录包含 `args.yaml` 和 `results.csv`，预测结果目录包含 4 张 JPG。两份
`features/*.pt` 缓存和 `best.pt`、`last.pt`、`last_healthy.pt` checkpoint 均可由准入脚本重新生成，
因体积较大未复制到 Git；其文件生成记录、模型/数据哈希、张量哈希和资源统计已包含在上述日志与元数据中。

## 8. 风险与降级

| 风险 | 当前处理 | P0/P1 动作 |
| --- | --- | --- |
| DINOv3 权重受 gated license 限制 | 使用官方 DINOv2 ViT-S/14，锁定来源与 SHA256 | 权限获批后替换 DINOv3 并重跑同一准入脚本 |
| 多尺度特征与检测金字塔不对齐 | 已锁定输入/输出与 P3/P4/P5 目标维度 | 实现显式 resize/channel projection，并补 shape/gradient 测试 |
| Foundation 与 LatentMixture 尚未端到端接线 | 两条链路分别测试，文档不作完成声明 | 先完成 P4 最小闭环，再扩展 P3/P5 |
| 10 万图缓存约 `73.58 GiB` | FP16 分图缓存，run id 隔离 | 增加容量预检、分片和清理策略，必要时改在线 teacher |
| Smoke 数据与预算不足以评价精度 | mAP=0 不用于排名或收益判断 | 在同数据、seed、epoch、imgsz 下做 baseline/on 对照 |
## 9. P0 下一步

1. 实现 `B x 384 x 16 x 16` 到 P3/P4/P5 的 resize 与 channel projection，先锁定 P4 最小闭环。
2. 将冻结 Foundation 特征显式接入 LatentMixture/检测头，补输入 -> 融合 -> loss -> gradient 的测试。
3. 在同数据、seed、epoch、imgsz 和优化器下运行 Foundation off/on 对照，并自动审计配置差异。
4. 扩展到 COCO-mini 与 VisDrone/工业小目标数据；报告精度、吞吐、显存和缓存成本，不只报告 smoke 成功。
