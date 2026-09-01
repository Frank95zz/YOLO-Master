# D1 WP1：DINOv3 多层特征输出

## 1. 阶段结论

WP1 已将 `DINOv3Teacher` 从默认单层输出扩展为一次前向返回指定 Transformer blocks，同时保持现有 Foundation 蒸馏调用完全兼容。真实 DINOv3 ViT-S/16 在 `640×640` 输入下已验证 block 4/8/12 均输出 `[1,384,40,40]`。

WP1 只扩展冻结 Teacher API，不实现特征缓存、尺度转换、LatentMixture 或检测头。

## 2. 接口与兼容行为

### 2.1 默认模式

未指定 `output_layers` 时保持原有行为：

```python
teacher = DINOv3Teacher(weights_path=weights_dir)
features = teacher.encode(images)

assert tuple(features.dense) == ("p4",)
```

默认 Foundation 蒸馏消费者仍读取 `dense["p4"]`，无需修改现有配置和 wrapper。

### 2.2 D1 多层模式

```python
teacher = DINOv3Teacher(
    weights_path=weights_dir,
    output_layers=(4, 8, 12),
)
features = teacher.encode(images)

assert tuple(features.dense) == ("block4", "block8", "block12")
```

`output_layers` 使用一基 block 编号，通过 Transformers Backbone 的公开 `out_features` stage 选择机制完成，不使用 forward hooks 或私有 encoder 结构。

| 一基层号 | 实现索引 | Backbone stage | 输出键 | 640 输入形状 |
| ---: | ---: | --- | --- | --- |
| 4 | 3 | `stage4` | `block4` | `B×384×40×40` |
| 8 | 7 | `stage8` | `block8` | `B×384×40×40` |
| 12 | 11 | `stage12` | `block12` | `B×384×40×40` |

## 3. 已完成工作

### 3.1 多层输出解析

[dinov3.py](../../ultralytics/nn/foundation/teachers/dinov3.py) 已支持：

- 一次 backbone 前向返回多个 feature maps；
- BCHW feature map 和 token 测试替身的统一解析；
- 按配置移除 1 个 CLS token 和 4 个 register tokens；
- 默认模式与多层模式共用 batch、通道、grid、pooled 和有限值校验；
- 未选择最终 block 时，通过公开 hidden-state 输出保持 `pooled` 来自最终层。

### 3.2 严格输入与输出校验

`output_layers` 必须是非空、严格递增、无重复的正整数序列。以下情况会立即报错：

- 字符串、布尔值、无序集合或非整数；
- 零、负数、重复或乱序层号；
- 超出 `num_hidden_layers` 或不存在的 `stageN`；
- 返回层数不足；
- 任意层 batch、通道或 grid 不匹配；
- 任意 dense 或 pooled 特征包含 NaN/Inf。

实现不会在异常时静默回退到最后一层。

### 3.3 Metadata 合同

多层模式固定记录：

- `output_layers=(4,8,12)`；
- `output_layer_indices=(3,7,11)`；
- `backbone_stages=("stage4","stage8","stage12")`；
- `feature_names=("block4","block8","block12")`；
- `patch_size`、`grid_size`、`hidden_dim`；
- `num_register_tokens`、`prefix_tokens`；
- 原始输入尺寸、padding 后尺寸、model ID 和 backend。

### 3.4 冻结与训练边界

每次 `encode()` 前都会重新执行冻结和 eval 约束，即使外部直接调用过 `teacher.model.train()` 也不会启用 dropout 或梯度。Teacher 前向和输出解析位于 `torch.inference_mode()`。

现有 Foundation wrapper 继续将 Teacher 放在注册模块树之外，因此 Teacher 不进入：

- student optimizer；
- DDP；
- EMA；
- student state dict 或 checkpoint。

## 4. 验收结果

真实权重集成测试使用 DINOv3 ViT-S/16、CUDA、FP16 和 `640×640` 输入：

| 验收项 | 结果 |
| --- | --- |
| 输出键 | `block4`、`block8`、`block12` |
| 三层形状 | 均为 `[1,384,40,40]` |
| prefix tokens | 5 |
| 重复前向 | 同一输入结果逐元素一致 |
| 参数状态 | 全部 `requires_grad=False`，模型保持 eval |
| 普通回归 | 101 passed，1 个无权重环境下的可选测试 skipped |
| 真实权重测试 | 1 passed |

主要测试位于：

- [test_foundation_dinov3.py](../../tests/test_foundation_dinov3.py)
- [test_d1_wp1_dinov3.py](../../tests/test_d1_wp1_dinov3.py)
- [test_foundation_distill_model.py](../../tests/test_foundation_distill_model.py)

## 5. 复现与校验

```bash
# 离线与 Foundation 兼容性回归
python -m pytest -q \
  tests/test_foundation_dinov3.py \
  tests/test_foundation_teacher_protocol.py \
  tests/test_foundation_distill_model.py \
  tests/test_foundation_checkpoint.py

# 真实 ViT-S/16、FP16、640 输入集成测试
export D1_DINOV3_WEIGHTS=/path/to/dinov3-vits16-pretrain-lvd1689m
python -m pytest -q tests/test_d1_wp1_dinov3.py
```

## 6. 提交与边界

- `3b05836 feat(d1): add DINOv3 multi-layer outputs`

block 4/8/12 表示不同语义深度，但原始空间分辨率均为 stride 16。它们不能直接命名为 P3/P4/P5；stride 8/16/32 的尺度转换属于后续 WP3 Adapter。特征持久化由 [WP2](WP2.md) 完成。
