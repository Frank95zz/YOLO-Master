# D1 WP5：缓存 Dataset、Trainer 与 Validator

## 1. 阶段结论

WP5 已打通从 WP2 特征缓存到 WP4 检测模型的训练与评测链路。`D1FeatureCacheDataset` 按 COCO 图像路径恢复 sample ID，读取 `block4/block8/block12`，并在不解码训练图像的情况下提供检测标签与几何元数据；专用 Trainer 和 Validator 直接搬运特征张量，不执行 RGB `/255` 归一化。

真实 WP2 缓存上的最小闭环已经完成一次参数更新、训练期验证、checkpoint 保存与恢复、最终验证。该验收证明训练基础设施可运行，不代表模型已经收敛，也不是 COCO 精度结果。

## 2. 实现文件

| 文件 | 职责 |
| --- | --- |
| [`ultralytics/data/d1_cache.py`](../../ultralytics/data/d1_cache.py) | 缓存 Dataset、特征 batch、标签几何和设备迁移 |
| [`ultralytics/models/yolo/detect/foundation_train.py`](../../ultralytics/models/yolo/detect/foundation_train.py) | D1 专用 Trainer、模型构造和缓存 checkpoint 最终验证 |
| [`ultralytics/models/yolo/detect/foundation_val.py`](../../ultralytics/models/yolo/detect/foundation_val.py) | D1 专用 Validator 和 COCO 坐标恢复 |
| [`tests/test_d1_wp5_cached_pipeline.py`](../../tests/test_d1_wp5_cached_pipeline.py) | 离线合同、真实缓存 CUDA 和最小训练闭环测试 |

`D1FeatureCacheDataset`、`D1FoundationDetectionTrainer` 和 `D1FoundationDetectionValidator` 已从各自模块入口导出。通用恢复控制器只增加了可选的 `checkpoint_smoke_inputs()` 扩展点；其他模型未实现该方法时继续使用原有 RGB smoke 输入。

## 3. Dataset 数据合同

Dataset 输入仍是标准 COCO 图片列表和标签，但样本前向数据来自 `d1-cache-v1`：

```text
COCO image path
  -> train2017/<stem> 或 val2017/<stem>
  -> FeatureCacheReader
  -> block4/block8/block12: 3 x [384,40,40], CPU FP16

COCO label
  -> 原图 normalized xywh
  -> WP0 centered LetterBox(640)
  -> 640 坐标系 normalized xywh
```

构造时会检查每个数据样本均存在缓存记录，并核对缓存中的 split、相对图片路径、输出层、特征名、shape 和 dtype。缺失样本、重复 sample ID、错误路径、非法缓存合同或非有限特征都会立即失败。

训练取样不调用 `load_image()`，也不运行 RGB 数据增强。标签缓存首次生成时，Ultralytics 仍可能读取图片以核验原始尺寸和文件完整性；这属于数据索引准备，不进入每步训练前向。

## 4. Batch 与训练链路

三层特征分别堆叠为 `[B,384,40,40]`。`D1FeatureBatch` 对通用 YOLO 循环暴露虚拟的 `[B,3,640,640]` shape，但不分配 RGB 张量：

```python
batch["features"] == {
    "block4": Tensor[B, 384, 40, 40],
    "block8": Tensor[B, 384, 40, 40],
    "block12": Tensor[B, 384, 40, 40],
}
batch["img"] is batch["features"]
```

Trainer 的数据路径为：

```text
FeatureCacheReader
  -> CPU FP16 batch
  -> device transfer（CUDA AMP 时保持 FP16，否则转 FP32）
  -> D1FoundationDetectionModel
  -> detection loss + mixture_aux_loss
  -> backward / optimizer / EMA / checkpoint
```

该路径不会把特征除以 255，也不会创建 Teacher。checkpoint 健康检查使用两组确定性的 DINO 特征映射，避免通用恢复逻辑向 D1 模型传入 RGB dummy tensor。

## 5. 训练参数约束

WP5 为缓存训练设置以下严格条件：

- `imgsz=640`、显式整数 batch、`rect=False`、`multi_scale=0`；
- RGB cache、compile 和全部 WP0 禁用的数据增强必须关闭；
- 训练 split 固定使用 cache 模式；
- 验证默认使用 cache 模式；
- online 模式只允许用于验证缓存一致性，且要求 `workers=0`；
- train 和 val 缓存目录必须显式传入，不写入模型或实验 YAML。

这些约束保证标签几何与 WP2 特征抽取时的 640 居中 LetterBox 一致，避免对离线特征应用无法同步的随机空间增强。

## 6. Validator 与坐标恢复

Validator 复用 Dataset 保存的：

- `ori_shape`：原始 COCO 图像高宽；
- `resized_shape=(640,640)`；
- `ratio_pad=((gain,gain),(left,top))`。

因此预测框可通过现有 DetectionValidator 逻辑从 640 letterbox 坐标恢复到原图坐标。D1 Validator 必须由 D1 Trainer 驱动；独立 AutoBackend 验证会执行 RGB warmup，与特征字典接口不兼容，因此被显式拒绝。

训练结束时，D1 Trainer 直接加载筛选后的下游 checkpoint，再通过缓存 Validator 完成最终评估。Teacher 和 RGB 图像均不参与该过程。

## 7. Online 一致性模式

`validation_feature_mode="online"` 是可选诊断模式。调用方必须提供在线 Teacher extractor；Dataset 可逐样本比较在线输出与缓存 tensor，并报告每层最大绝对误差。

正式 P0 训练和最终评测统一使用 cache 模式。online 模式不作为正式训练后端，也不允许隐式回退：Teacher 不可用或输出不满足三层 FP16 合同时直接失败。

## 8. 使用方式

准备主机无关的路径变量：

```bash
export D1_TRAIN_CACHE=/path/to/train-cache
export D1_VAL_CACHE=/path/to/val-cache
export D1_DATA_YAML=/path/to/coco.yaml
```

Python 训练入口：

```python
import os

from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
from ultralytics.utils import YAML

experiment = YAML.load("ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml")
trainer = D1FoundationDetectionTrainer(
    overrides={**experiment, "data": os.environ["D1_DATA_YAML"]},
    feature_caches={
        "train": os.environ["D1_TRAIN_CACHE"],
        "val": os.environ["D1_VAL_CACHE"],
    },
)
trainer.train()
```

当前 100 图 WP2 缓存只覆盖 train2017 的固定样本，不能替代正式 train/val 全量缓存。正式命令只能在两个 split 的缓存均构建并校验完成后执行。

## 9. 测试证据

验收基于代码提交 `81c98b334eb7a696561753164695fe30eb38531f`。

离线和真实缓存测试覆盖：

- sample ID 映射、完整 cache coverage 和无训练期 RGB 读取；
- LetterBox 标签变换及预测框原图坐标恢复；
- batch 键、shape、dtype、device 和有限值检查；
- 训练预处理不执行 RGB `/255`；
- online/cache 完全一致与不一致时的失败路径；
- 真实 FP16 缓存到 CUDA WP4 模型的 `[2,300,6]` 输出；
- 一批次训练、四项 loss、训练期验证、checkpoint 恢复和最终验证；
- 通用 checkpoint recovery 未受 D1 扩展影响。

相关回归命令：

```bash
export D1_WP2_CACHE=/path/to/wp2-train100-a
export D1_COCO_ROOT=/path/to/coco

python -m pytest -q \
  tests/test_d1_wp0_contract.py \
  tests/test_d1_wp1_dinov3.py \
  tests/test_d1_wp2_cache_cli.py \
  tests/test_d1_wp2_feature_cache.py \
  tests/test_d1_wp3_foundation_adapter.py \
  tests/test_d1_wp4_foundation_detection_model.py \
  tests/test_d1_wp5_cached_pipeline.py \
  tests/test_latent_mixture.py \
  tests/test_foundation_checkpoint.py \
  tests/test_foundation_distill_model.py \
  tests/test_foundation_teacher_protocol.py \
  tests/test_ddp_lifecycle_ema_nan.py
```

结果为 `176 passed, 2 skipped`。其中真实缓存路径、真实 CUDA 前向和一批次完整训练测试均已执行；两个 skip 来自与本机可选条件无关的既有测试。`py_compile`、`git diff --check` 和 staged diff check 通过；服务器环境未安装 Ruff。

## 10. 阶段边界

WP5 已完成缓存数据进入训练基础设施的闭环，但尚未完成：

- 完整 COCO train2017/val2017 特征缓存；
- WP6 的 balance loss、z-loss 和 latent aux 分项记录与开关证据；
- 32 图过拟合和正式 COCO 训练；
- 收敛精度、训练成本、I/O 等待比例和训练成本降低结论。

下一阶段 WP6 将验证三个 LatentMixture 的 aux 收集、日志、开关和 Router 梯度闭环。完整缓存构建和正式训练留在 WP8，不能用本阶段的两图一批次结果替代。
