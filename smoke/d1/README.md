# D1 Smoke Test / Admission Report

## Basic Information

| Field | Value |
| --- | --- |
| Project | Tencent Rhino-Bird YOLO-Master |
| Topic | D1 - Frozen DINOv3 / Foundation representation adaptation with LatentMixture |
| Owner | fengyanqi / Frank95zz |
| Branch | `feat/topic-d1-fengyanqi` |
| Smoke name | `fengyanqi-d1-smoke-test` |
| Naming rule | `name-topicId-smoke-test`; branch `feat/topic-topicId-name` |
| Smoke execution commit | `1b81dc966bb44066a7762af6635d49e20fd79f92` |

Repository link:

```text
https://github.com/Frank95zz/YOLO-Master/tree/feat/topic-d1-fengyanqi
```

## Admission Scope

This smoke test verifies the D1 minimum runnable path before formal topic work:

1. Environment and GPU are usable.
2. D1-related unit tests pass.
3. LatentMixture model can complete a minimal `coco8` training run.
4. Validation, EMA, recovery checkpoint, and checkpoint saving complete without crash.
5. The produced checkpoint can run `predict` on `coco8` validation images.

The smoke test is not intended to validate accuracy. The dataset is too small and the run is only 1 epoch, so zero mAP / no detections are acceptable for admission. The pass criterion is end-to-end execution and reproducible evidence.

## Environment Installation

Project workspace:

```bash
/root/yolo-master
```

Repository:

```bash
/root/yolo-master/repo
```

Conda environment:

```bash
/root/yolo-master/.conda/d1
```

Activation:

```bash
source /root/yolo-master/env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/yolo-master/.conda/d1
cd /root/yolo-master/repo
```

Key environment versions verified by `yolo checks`:

| Item | Value |
| --- | --- |
| Python | 3.11.15 |
| Ultralytics | 8.4.101, editable install from `/root/yolo-master/repo` |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime | 12.4 |
| GPU | 6 x NVIDIA A40, smoke used `CUDA_VISIBLE_DEVICES=0` |
| CPU | 128 logical CPUs |
| RAM | about 503 GiB |

## Baseline / Minimum Task

Smoke model:

```bash
ultralytics/cfg/models/26/yolo26-master-latent-n.yaml
```

Dataset:

```bash
coco8.yaml
```

Output directory:

```bash
/root/yolo-master/runs/fengyanqi-d1-smoke-test
```

Prediction output directory:

```bash
/root/yolo-master/runs/fengyanqi-d1-smoke-test-predict
```

## Reproduction Commands

### 1. Doctor / Environment Check

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

### 2. D1 Unit Tests

```bash
python -m pytest -q \
  tests/test_latent_mixture.py \
  tests/test_mixture_aux_loss.py \
  tests/test_foundation_dinov3.py \
  tests/test_foundation_distill_model.py \
  tests/test_foundation_mixture_interaction.py \
  2>&1 | tee "$LOG_DIR/pytest-d1.txt"
```

Observed result:

```text
59 passed
```

### 3. coco8 Train Smoke

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

Observed result:

```text
TRAIN_EXIT_CODE=0
Traceback count=0
1 epochs completed
Final validation completed on best.pt
```

Generated checkpoints:

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
