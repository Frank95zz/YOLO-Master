# D1 Smoke Test / 准入测试报告

## 基本信息

| 字段 | 内容 |
| --- | --- |
| 项目 | 腾讯犀牛鸟 YOLO-Master |
| 课题 | D1 - 冻结 DINOv3 / Foundation 表征迁移与 LatentMixture 适配 |
| 负责人 | fengyanqi / Frank95zz |
| 代码分支 | `feat/topic-d1-fengyanqi` |
| Smoke 名称 | `fengyanqi-d1-smoke-test` |
| 命名规则 | `姓名-课题编号-smoke-test`；代码分支 `feat/topic-编号-姓名` |
| Smoke 执行 commit | `1b81dc966bb44066a7762af6635d49e20fd79f92` |

仓库链接：

```text
https://github.com/Frank95zz/YOLO-Master/tree/feat/topic-d1-fengyanqi
```

## 准入测试范围

本次 smoke test 用于验证 D1 课题在正式开展前的最小可运行链路，覆盖以下内容：

1. 环境和 GPU 可正常使用。
2. D1 相关单元测试通过。
3. LatentMixture 模型可以完成最小 `coco8` 训练。
4. 训练、验证、EMA、recovery checkpoint、checkpoint 保存链路可以完整走通。
5. 训练生成的 `best.pt` 可以继续执行 `predict` 推理。

本次 smoke test 不用于评价精度。由于数据集是 `coco8`，并且只训练 1 个 epoch，mAP 为 0 或没有检测框属于正常现象。准入判断重点是链路是否稳定、日志是否完整、结果是否可复现。

## 环境安装

项目工作区：

```bash
/root/yolo-master
```

代码仓库：

```bash
/root/yolo-master/repo
```

Conda 环境：

```bash
/root/yolo-master/.conda/d1
```

环境激活方式：

```bash
source /root/yolo-master/env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/yolo-master/.conda/d1
cd /root/yolo-master/repo
```

`yolo checks` 已验证的关键环境版本：

| 项目 | 版本 / 信息 |
| --- | --- |
| Python | 3.11.15 |
| Ultralytics | 8.4.101，editable install from `/root/yolo-master/repo` |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime | 12.4 |
| GPU | 6 x NVIDIA A40，本次 smoke 使用 `CUDA_VISIBLE_DEVICES=0` |
| CPU | 128 logical CPUs |
| 内存 | 约 503 GiB |

## 基线 / 最小任务

Smoke 使用的模型配置：

```bash
ultralytics/cfg/models/26/yolo26-master-latent-n.yaml
```

Smoke 使用的数据集：

```bash
coco8.yaml
```

训练输出目录：

```bash
/root/yolo-master/runs/fengyanqi-d1-smoke-test
```

预测输出目录：

```bash
/root/yolo-master/runs/fengyanqi-d1-smoke-test-predict
```

## 复现命令

### 1. 环境检查 / Doctor

```bash
RUN_NAME=fengyanqi-d1-smoke-test
TS=$(date +%Y%m%dT%H%M%S)
LOG_DIR=/root/yolo-master/logs/${RUN_NAME}-${TS}
mkdir -p "$LOG_DIR"

git rev-parse HEAD | tee "$LOG_DIR/git-commit.txt"
git status --short --branch | tee "$LOG_DIR/git-status.txt"
nvidia-smi | tee "$LOG_DIR/nvidia-smi-before.txt"
yolo checks 2>&1 | tee "$LOG_DIR/yolo-checks.txt"
python -m pip freeze > "$LOG_DIR/pip-freeze.txt"
```

### 2. D1 单元测试

```bash
python -m pytest -q \
  tests/test_latent_mixture.py \
  tests/test_mixture_aux_loss.py \
  tests/test_foundation_dinov3.py \
  tests/test_foundation_distill_model.py \
  tests/test_foundation_mixture_interaction.py \
  2>&1 | tee "$LOG_DIR/pytest-d1.txt"
```

实测结果：

```text
59 passed
```

### 3. coco8 最小训练 Smoke

```bash
CUDA_VISIBLE_DEVICES=0 yolo detect train \
  model=ultralytics/cfg/models/26/yolo26-master-latent-n.yaml \
  data=coco8.yaml \
  epochs=1 \
  imgsz=160 \
  batch=2 \
  workers=2 \
  device=0 \
  project=/root/yolo-master/runs \
  name=fengyanqi-d1-smoke-test \
  exist_ok=True \
  plots=False \
  2>&1 | tee "$LOG_DIR/train-smoke.txt"
```

实测结果：

```text
TRAIN_EXIT_CODE=0
Traceback count=0
1 epochs completed
完成 best.pt 最终验证
```

生成的 checkpoint：

```bash
/root/yolo-master/runs/fengyanqi-d1-smoke-test/weights/best.pt
/root/yolo-master/runs/fengyanqi-d1-smoke-test/weights/last.pt
/root/yolo-master/runs/fengyanqi-d1-smoke-test/weights/last_healthy.pt
```

### 4. Predict Smoke

```bash
CUDA_VISIBLE_DEVICES=0 yolo detect predict \
  model=/root/yolo-master/runs/fengyanqi-d1-smoke-test/weights/best.pt \
  source=/root/yolo-master/datasets/coco8/images/val \
  imgsz=160 \
  device=0 \
  project=/root/yolo-master/runs \
  name=fengyanqi-d1-smoke-test-predict \
  exist_ok=True \
  save=True \
  2>&1 | tee "$LOG_DIR/predict-smoke.txt"
```

实测结果：

```text
PREDICT_EXIT_CODE=0
Traceback count=0
处理 coco8 validation 图片 4 张
结果保存到 /root/yolo-master/runs/fengyanqi-d1-smoke-test-predict
```

生成的预测文件：

```bash
/root/yolo-master/runs/fengyanqi-d1-smoke-test-predict/000000000036.jpg
/root/yolo-master/runs/fengyanqi-d1-smoke-test-predict/000000000042.jpg
/root/yolo-master/runs/fengyanqi-d1-smoke-test-predict/000000000049.jpg
/root/yolo-master/runs/fengyanqi-d1-smoke-test-predict/000000000061.jpg
```

## 配置文件

本次 smoke test 使用的核心模型配置：

```bash
ultralytics/cfg/models/26/yolo26-master-latent-n.yaml
```

准入前已检查的 D1 相关代码入口：

```bash
ultralytics/nn/modules/latent_mixture.py
ultralytics/nn/mixture_loss.py
ultralytics/nn/modules/routing_protocol.py
ultralytics/nn/foundation/teachers/dinov3.py
ultralytics/nn/foundation_distill_model.py
```

D1 相关测试文件：

```bash
tests/test_latent_mixture.py
tests/test_mixture_aux_loss.py
tests/test_foundation_dinov3.py
tests/test_foundation_distill_model.py
tests/test_foundation_mixture_interaction.py
```

## 完整日志

最终 smoke 日志目录：

```bash
/root/yolo-master/logs/fengyanqi-d1-smoke-test-20260824T174402
```

关键日志文件：

```bash
git-commit.txt
git-status.txt
nvidia-smi-before.txt
nvidia-smi-after.txt
yolo-checks.txt
pip-freeze.txt
train-smoke.txt
train-exit-code.txt
predict-smoke.txt
predict-exit-code.txt
predict-files.txt
```

最终 D1 单元测试日志保存在 `/root/yolo-master/logs/`，包括：

```bash
pytest-d1-after-recovery-cache-fix.txt
pytest-model-ema-routing-cache-fix.txt
```

## 结果证据

| 检查项 | 结果 |
| --- | --- |
| Git 分支 | `feat/topic-d1-fengyanqi` |
| Smoke 执行 commit | `1b81dc966bb44066a7762af6635d49e20fd79f92` |
| D1 单元测试 | `59 passed` |
| Train smoke | 通过，exit code 0 |
| Train traceback count | 0 |
| 最终验证 | 已基于 `best.pt` 完成 |
| Predict smoke | 通过，exit code 0 |
| Predict traceback count | 0 |
| Checkpoint | 已生成 `best.pt`、`last.pt`、`last_healthy.pt` |
| Predict 产物 | 已生成 4 张 validation 图片预测结果 |

## 设计说明

D1 课题的核心是复用已有 Foundation / DINOv3 teacher stack，并验证其与 LatentMixture 路由适配路径的结合。本次准入 smoke 选择 `yolo26-master-latent-n.yaml`，因为该配置可以直接覆盖 LatentMixture 模块和 `latent` auxiliary loss 路径，同时计算量足够小，适合单卡快速验证。

准入过程中还修复了 PyTorch 2.6 下与 runtime routing cache 相关的若干问题，确保 LatentMixture 在保留训练期 graph-connected routing tensor 的同时，不会破坏 EMA、recovery checkpoint 和模型 deepcopy 流程。这些修复均限制在运行期状态清理和无参数 EMA 设备初始化逻辑上，不改变 D1 模型结构和训练目标。

## 风险与降级方案

| 风险 | 降级方案 |
| --- | --- |
| 完整 DINOv3 online teacher 计算开销过高 | 先使用离线 teacher feature cache，并从 `coco8` / COCO-mini 开始 |
| 多尺度对齐不稳定 | 优先固定单尺度接口，例如 P4 |
| Foundation feature cache 占用磁盘过大 | 统一放入 `/root/yolo-master/feature_cache`，按 run id 管理和清理 |
| Smoke 指标不能代表真实精度 | Smoke 只作为链路验证，正式精度结论放到后续 P0/P1 实验 |
| GPU 资源冲突 | 使用 `CUDA_VISIBLE_DEVICES` 指定单卡，并在日志中记录 GPU、commit、Owner 和 run name |

## 准入结论

D1 smoke test 已通过。当前环境、D1 单元测试、LatentMixture train path、validation、EMA/recovery checkpoint、checkpoint 保存和 predict 推理链路均已在 1 张 NVIDIA A40 上完整走通。该课题可以从准入 smoke 阶段进入正式 D1 P0 实验阶段。
