# D1 WP3：DINOv3 多尺度特征适配器

## 1. 阶段结论

WP3 已实现独立的 `DINOFeaturePyramidAdapter`，将 WP2 缓存中的 block 4/8/12 三组 stride-16 特征转换为 P3/P4/P5 三个尺度的候选特征。每个 DINO block 在每个尺度都有独立可训练分支，共九条分支；输出可直接分别交给三个单尺度 `LatentMixture`。

WP3 只完成特征尺度与通道适配，不包含 D1 检测模型、Detect、缓存 Dataset、Trainer、Validator 或正式训练。

## 2. 接口合同

适配器由 [foundation_adapter.py](../../ultralytics/nn/modules/foundation_adapter.py) 提供，并从 `ultralytics.nn.modules` 公开导出：

```python
adapter = DINOFeaturePyramidAdapter(
    in_channels=384,
    source_names=("block4", "block8", "block12"),
    pyramid_channels=(64, 128, 256),
    norm_groups=8,
)
outputs = adapter(features)
```

输入必须是只包含 `block4`、`block8`、`block12` 的映射，三者必须具有相同 batch、device、dtype 和偶数空间网格。输出固定为：

| 输出键 | 候选顺序 | 通道 | stride | 正式 40×40 输入 |
| --- | --- | ---: | ---: | --- |
| `p3` | block 4/8/12 | 64 | 8 | 三组 `[B,64,80,80]` |
| `p4` | block 4/8/12 | 128 | 16 | 三组 `[B,128,40,40]` |
| `p5` | block 4/8/12 | 256 | 32 | 三组 `[B,256,20,20]` |

公开属性 `source_names`、`pyramid_names`、`out_channels=(64,128,256)` 和 `strides=(8,16,32)` 为 WP4 构建检测模型提供明确合同。

## 3. 九条适配分支

| 尺度 | 每个 DINO block 的独立分支 |
| --- | --- |
| P3 | `1×1 Conv(384→64) + GroupNorm + SiLU + bilinear 2× upsample` |
| P4 | `1×1 Conv(384→128) + GroupNorm + SiLU` |
| P5 | `3×3 stride-2 Conv(384→256) + GroupNorm + SiLU` |

GroupNorm 通过 `get_safe_groups(channels, 8)` 选择可整除组数。模块不包含 BatchNorm，不共享不同来源或不同尺度的参数，也不会 detach 输入；Adapter 参数和输入梯度均保持计算图连接。

## 4. 校验与失败条件

以下情况会立即失败，不进行静默补齐或重排：

- 输入不是映射，或存在缺失、额外、重复或非法来源名称；
- 任意输入不是浮点 BCHW tensor；
- 通道不是 384，或 batch、device、dtype、空间尺寸不一致；
- 空 batch、空空间维度或不能精确二分的奇数网格；
- 构造参数不是三个合法来源名称、三个正输出通道或正整数归一化组数。

Adapter 支持严格 state dict 往返。WP3 不注册 YAML parser 特殊逻辑，也不设置 Detect stride。

## 5. 真实缓存验收

验收基于代码提交 `e1c60c27c2a57285a384048ce9e379035f2a046a`，使用 WP2 的 `wp2-train100-a` 缓存和样本 `train2017/000000000009`：

| 项目 | 结果 |
| --- | --- |
| 缓存输入 | block 4/8/12 均为 `torch.float16 [384,40,40]` |
| P3 输出 | 三组 `[1,64,80,80]` |
| P4 输出 | 三组 `[1,128,40,40]` |
| P5 输出 | 三组 `[1,256,20,20]` |
| AMP 输出 dtype | 当前 A40/PyTorch 环境下九组均为 `torch.float32` |
| 可训练参数 | 2,878,080 |
| 反向传播 | 九条分支权重梯度均有限且非零 |
| LatentMixture 兼容 | 三个尺度均可直接融合并反向传播 |

缓存输入保持 FP16；当前 autocast 下 GroupNorm 按数值稳定策略产生 FP32 输出。接口只要求同一批候选使用一致浮点 dtype，不强制将归一化结果重新压回 FP16。

## 6. 测试与复现

离线 Adapter 测试：

```bash
python -m pytest -q tests/test_d1_wp3_foundation_adapter.py
```

使用已有缓存运行真实 CUDA 验收：

```bash
export D1_WP2_CACHE=/path/to/wp2-train100-a
python -m pytest -q \
  tests/test_d1_wp3_foundation_adapter.py::test_real_wp2_cache_cuda_fp16
```

WP0-WP3 与 LatentMixture 最终相关回归：

```bash
D1_WP2_CACHE=/path/to/wp2-train100-a python -m pytest -q \
  tests/test_d1_wp0_contract.py \
  tests/test_d1_wp1_dinov3.py \
  tests/test_d1_wp2_feature_cache.py \
  tests/test_d1_wp2_cache_cli.py \
  tests/test_d1_wp3_foundation_adapter.py \
  tests/test_latent_mixture.py
```

结果为 `77 passed, 1 skipped`。跳过项是未配置 `D1_DINOV3_WEIGHTS` 时的可选真实 Teacher 集成测试；WP3 真实缓存 CUDA 测试已经通过。服务器环境未安装 Ruff，因此未运行 Ruff；`py_compile` 和 `git diff --check` 均通过。

## 7. 提交与阶段边界

- 代码提交：`e1c60c2 feat(d1): add multi-scale feature adapter`

WP3 尚未完成：

- 未实例化三个正式 LatentMixture 或 Detect；
- 未新增 D1 模型 YAML、Dataset、Trainer 或 Validator；
- 未执行 32 图过拟合、COCO8 或完整 COCO 训练；
- 未把 block 4/8/12 伪装为原生 P3/P4/P5；尺度信息由可训练 Adapter 显式生成。

下一阶段由 WP4 构建独立的 D1 Foundation Detection Model，并显式连接 Adapter、三个 LatentMixture 和 Detect。
