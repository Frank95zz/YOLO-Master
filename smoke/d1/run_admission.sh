#!/usr/bin/env bash
set -euo pipefail

D1_ROOT=/root/yolo-master
D1_REPO=${D1_ROOT}/repo
D1_PYTHON=${D1_ROOT}/.conda/d1/bin/python
D1_YOLO=${D1_ROOT}/.conda/d1/bin/yolo
D1_DATASET=${D1_ROOT}/datasets/coco128
D1_ARCHIVE=${D1_ROOT}/tmp/coco128.zip
D1_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main
D1_DINOV2_WEIGHTS=/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth

mkdir -p "${D1_ROOT}/tmp" "${D1_ROOT}/datasets" "${D1_ROOT}/feature_cache" "${D1_ROOT}/logs" "${D1_ROOT}/manifests"

if [[ ! -s "${D1_ARCHIVE}" ]]; then
  curl --http1.1 -L --fail --retry 5 --retry-delay 2 --retry-all-errors \
    --output "${D1_ARCHIVE}" \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
fi
unzip -tq "${D1_ARCHIVE}" >/dev/null
if [[ ! -d "${D1_DATASET}/images/train2017" ]]; then
  unzip -q -n "${D1_ARCHIVE}" -d "${D1_ROOT}/datasets"
fi

cd "${D1_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR: git worktree must be clean" >&2
  git status --short >&2
  exit 2
fi

D1_COMMIT=$(git rev-parse HEAD)
D1_SHORT=${D1_COMMIT:0:12}
D1_TIMESTAMP=$(date +%Y%m%dT%H%M%S)
D1_RUN_ID=d1-admission-${D1_SHORT}-${D1_TIMESTAMP}
D1_LOG_DIR=${D1_ROOT}/logs/${D1_RUN_ID}
D1_CACHE_ONE=${D1_ROOT}/feature_cache/${D1_RUN_ID}-run1
D1_CACHE_TWO=${D1_ROOT}/feature_cache/${D1_RUN_ID}-run2
D1_TRAIN_NAME=${D1_RUN_ID}-train
D1_PREDICT_NAME=${D1_RUN_ID}-predict
mkdir -p "${D1_LOG_DIR}"

git rev-parse HEAD | tee "${D1_LOG_DIR}/git-commit.txt"
git status --short --branch | tee "${D1_LOG_DIR}/git-status.txt"
sha256sum "${D1_ARCHIVE}" | tee "${D1_LOG_DIR}/coco128-archive-sha256.txt"
sha256sum "${D1_DINOV2_WEIGHTS}" | tee "${D1_LOG_DIR}/dinov2-weights-sha256.txt"
find "${D1_DATASET}/images/train2017" -type f | sort | head -n 100 > "${D1_LOG_DIR}/source-images-100.txt"
test "$(wc -l < "${D1_LOG_DIR}/source-images-100.txt")" -eq 100
nvidia-smi | tee "${D1_LOG_DIR}/nvidia-smi-before.txt"
df -h "${D1_ROOT}" | tee "${D1_LOG_DIR}/disk-before.txt"

"${D1_PYTHON}" -m pytest -q \
  tests/test_latent_mixture.py \
  tests/test_mixture_aux_loss.py \
  tests/test_foundation_dinov3.py \
  tests/test_foundation_distill_model.py \
  tests/test_foundation_mixture_interaction.py \
  2>&1 | tee "${D1_LOG_DIR}/pytest-d1.txt"

CUDA_VISIBLE_DEVICES=0 "${D1_PYTHON}" smoke/d1/build_feature_cache.py \
  --images "${D1_DATASET}/images/train2017" \
  --output "${D1_CACHE_ONE}" \
  --repo "${D1_REPO}" \
  --dinov2-repo "${D1_DINOV2_REPO}" \
  --weights "${D1_DINOV2_WEIGHTS}" \
  --device cuda:0 \
  --limit 100 \
  --batch 8 \
  --imgsz 224 \
  2>&1 | tee "${D1_LOG_DIR}/feature-cache-run1.txt"

CUDA_VISIBLE_DEVICES=0 "${D1_PYTHON}" smoke/d1/build_feature_cache.py \
  --images "${D1_DATASET}/images/train2017" \
  --output "${D1_CACHE_TWO}" \
  --repo "${D1_REPO}" \
  --dinov2-repo "${D1_DINOV2_REPO}" \
  --weights "${D1_DINOV2_WEIGHTS}" \
  --device cuda:0 \
  --limit 100 \
  --batch 8 \
  --imgsz 224 \
  2>&1 | tee "${D1_LOG_DIR}/feature-cache-run2.txt"

"${D1_PYTHON}" smoke/d1/compare_feature_caches.py \
  "${D1_CACHE_ONE}" \
  "${D1_CACHE_TWO}" \
  --output "${D1_LOG_DIR}/repeatability-report.json" \
  2>&1 | tee "${D1_LOG_DIR}/repeatability.txt"

CUDA_VISIBLE_DEVICES=0 "${D1_YOLO}" detect train \
  model=ultralytics/cfg/models/26/yolo26-master-latent-n.yaml \
  data=coco8.yaml \
  epochs=1 \
  imgsz=160 \
  batch=2 \
  workers=2 \
  device=0 \
  project="${D1_ROOT}/runs" \
  name="${D1_TRAIN_NAME}" \
  exist_ok=False \
  plots=False \
  2>&1 | tee "${D1_LOG_DIR}/train-smoke.txt"

CUDA_VISIBLE_DEVICES=0 "${D1_YOLO}" detect predict \
  model="${D1_ROOT}/runs/${D1_TRAIN_NAME}/weights/best.pt" \
  source="${D1_ROOT}/datasets/coco8/images/val" \
  imgsz=160 \
  device=0 \
  project="${D1_ROOT}/runs" \
  name="${D1_PREDICT_NAME}" \
  exist_ok=False \
  save=True \
  2>&1 | tee "${D1_LOG_DIR}/predict-smoke.txt"

nvidia-smi | tee "${D1_LOG_DIR}/nvidia-smi-after.txt"
df -h "${D1_ROOT}" | tee "${D1_LOG_DIR}/disk-after.txt"
printf '%s\n' "${D1_COMMIT}" > "${D1_ROOT}/manifests/git-commit.txt"
git status --short --branch > "${D1_ROOT}/manifests/git-status.txt"
printf '%s\n' "${D1_LOG_DIR}" > "${D1_ROOT}/logs/latest-d1-admission.txt"
printf '%s\n' "${D1_CACHE_ONE}" > "${D1_ROOT}/feature_cache/latest-d1-admission.txt"

{
  echo "result=PASS"
  echo "commit=${D1_COMMIT}"
  echo "log_dir=${D1_LOG_DIR}"
  echo "cache_run1=${D1_CACHE_ONE}"
  echo "cache_run2=${D1_CACHE_TWO}"
  echo "train_run=${D1_ROOT}/runs/${D1_TRAIN_NAME}"
  echo "predict_run=${D1_ROOT}/runs/${D1_PREDICT_NAME}"
} | tee "${D1_LOG_DIR}/admission-result.txt"
