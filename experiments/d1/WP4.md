# D1 WP4：缓存特征检测模型

## 1. 阶段结论

WP4 已实现独立的 `D1FoundationDetectionModel`。模型输入是 WP2 协议下的 `block4/block8/block12` 特征字典，不接受 RGB 图像；内部完整连接 WP3 Adapter、三个单尺度 `LatentMixture` 和 YOLO26 `Detect`，可以执行训练前向、E2E 检测损失、latent aux 组合、反向传播和检测结果解码。

WP4 只组装下游检测模型，不包含缓存 Dataset、Trainer、Validator、完整 COCO 缓存或正式训练。

## 2. 模型链路

模型由 [foundation_detection_model.py](../../ultralytics/nn/foundation_detection_model.py) 提供，并从 `ultralytics.nn` 公开导出：

```python
from ultralytics.nn import D1FoundationDetectionModel

model = D1FoundationDetectionModel()
predictions = model(
    {
        "block4": block4,
        "block8": block8,
        "block12": block12,
    }
)
```

完整前向链路为：

```text
block4/block8/block12
        -> DINOFeaturePyramidAdapter
        -> p3: 3 candidates -> LatentMixture-p3 -> [B,  64, 80, 80]
        -> p4: 3 candidates -> LatentMixture-p4 -> [B, 128, 40, 40]
        -> p5: 3 candidates -> LatentMixture-p5 -> [B, 256, 20, 20]
        -> Detect(nc=80, reg_max=1, end2end=True)
```

三个 LatentMixture 参数互不共享，分别发布 `kind="latent"` 的 routed aux。模型训练参数只包括 Adapter、LatentMixture 和 Detect，不注册 DINOv3 Teacher。

## 3. 配置与 stride 合同

正式配置为 [yolo26-d1-dinov3-latent-n.yaml](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml)，也是 WP0 实验配置引用的模型文件。配置固定：

- 输入通道 384，来源顺序 block 4/8/12；
- P3/P4/P5 通道 64/128/256；
- 三个 LatentMixture，每个 4 个专家；
- COCO `nc=80`、`reg_max=1`、`end2end=True`；
- Detect stride 显式设置为 `[8,16,32]`。

构造模型时不会执行 RGB dummy forward 推断 stride。输入不是精确特征映射、配置 stride 不匹配、非法通道或未知 LatentMixture 参数时立即失败。

## 4. 损失与反向传播

训练 batch 使用 `batch["features"]` 传入缓存特征。模型根据 Detect 模式构造现有 `E2ELoss` 或 `v8DetectionLoss`，随后通过现有 `CompositeCriterion` 加入 routed aux，没有新增或复制检测损失实现。

单 batch 离线验收已经证明：

- E2E one-to-many 和 one-to-one 分支均产生合法预测；
- 检测 loss 与 latent aux 均为有限值；
- loss items 在 box/cls/dfl 后附加 latent balance、z、原始 aux 和最终 mixture aux；
- Adapter、Detect 和三个 Router 均获得有限非零梯度。

上述分项日志、关闭语义和单 step 三次发布证据现已由 [WP6](WP6.md) 补齐。

## 5. Checkpoint 合同

`checkpoint_payload()` 使用 `d1-downstream-v1`，只包含：

```text
schema_version
state_dict             # Adapter、三个 LatentMixture、Detect 及下游运行状态
config                 # 主机无关的 D1 模型配置
teacher_reference      # Teacher ID、层号、缓存协议和权重 SHA256
```

Teacher 引用固定到 `facebook/dinov3-vits16-pretrain-lvd1689m`，权重 SHA256 为 `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`。checkpoint 不包含 Teacher 模块或权重；配置与 Teacher 引用不一致时严格恢复会失败。

## 6. 参数量

| 模块 | 可训练参数 |
| --- | ---: |
| Adapter | 2,878,080 |
| LatentMixture-P3 | 17,925 |
| LatentMixture-P4 | 68,613 |
| LatentMixture-P5 | 268,293 |
| Detect | 309,656 |
| 合计 | 3,542,567 |

DINOv3 Teacher 不在该统计和下游 `state_dict` 中。

## 7. 真实缓存验收

验收基于代码提交 `a1255c09b989d20f4f130ddf55e29614811f5fc2`，使用 `wp2-train100-a` 的 FP16 缓存：

| 项目 | 结果 |
| --- | --- |
| 输入 | block 4/8/12，各 `[1,384,40,40]` |
| Detect 输入 P3 | `[1,64,80,80]` |
| Detect 输入 P4 | `[1,128,40,40]` |
| Detect 输入 P5 | `[1,256,20,20]` |
| 解码输出 | `[1,300,6]`，全部有限 |
| WP4 真实缓存测试 | `12 passed` |

真实缓存测试只验证完整 CUDA 前向；loss 与反向使用小网格离线 batch 验证。正式缓存 Dataset 和 COCO 标签接入属于 WP5。

## 8. 测试与复现

WP4 离线测试：

```bash
python -m pytest -q tests/test_d1_wp4_foundation_detection_model.py
```

使用已有 WP2 缓存运行 CUDA 验收：

```bash
export D1_WP2_CACHE=/path/to/wp2-train100-a
python -m pytest -q tests/test_d1_wp4_foundation_detection_model.py
```

WP0-WP4、LatentMixture、CompositeCriterion、路由和 checkpoint 相关回归结果为 `110 passed, 1 skipped`。模型配置完整性测试另为 `13 passed`。`py_compile` 和 `git diff --check` 均通过；服务器环境未安装 Ruff。

## 9. 提交与阶段边界

- 代码提交：`a1255c0 feat(d1): add cached-feature detection model`

WP4 提交时尚未完成：

- 未根据 COCO sample ID 从缓存读取特征并同时返回标签与几何元数据，后由 [WP5](WP5.md) 完成；
- 未接入专用 Trainer 和 Validator，后由 [WP5](WP5.md) 完成；
- 未运行 32 图过拟合、COCO8 或完整 COCO 训练；
- 未完成 latent aux 指标记录和逐步收集证明，后由 [WP6](WP6.md) 完成。

下一阶段 WP5 将实现缓存 Dataset、Trainer 和 Validator，使该模型能够进入正式训练与评测流程。
