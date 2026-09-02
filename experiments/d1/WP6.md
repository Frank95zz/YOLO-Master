# D1 WP6：Latent aux 损失闭环

## 1. 阶段结论

WP6 已完成三个 `LatentMixture` 的辅助损失闭环。P3、P4、P5 Router 每次训练前向各发布一次 `kind="latent"` 的图连接标量，现有 `CompositeCriterion` 统一收集并加入检测损失；D1 Trainer 同时把 balance loss、z-loss、原始 latent aux 和最终 mixture aux 写入标准训练指标。

真实 WP2 缓存的一批次训练、训练期验证、checkpoint 恢复和最终验证均已通过，生成的 `results.csv` 包含四个 WP6 指标列。本阶段证明损失链路和记录机制正确，不代表 Router 已在完整 COCO 上学到有效路由策略。

## 2. 三个 Router 的发布合同

D1 模型包含三个相互独立的单尺度 `LatentMixture`：

| 名称 | 输入候选 | 发布类型 | 每个训练 step 的发布数 |
| --- | --- | --- | ---: |
| P3 Router | 3 个 `[B,64,80,80]` | `latent` | 1 |
| P4 Router | 3 个 `[B,128,40,40]` | `latent` | 1 |
| P5 Router | 3 个 `[B,256,20,20]` | `latent` | 1 |

每次 `D1FoundationDetectionModel.predict()` 开始前都会清空旧 runtime 记录并推进 aux step。三个 Router 完成前向后，统一 collector 只接受当前 step、训练模式且保持 autograd 的记录。

D1 在 loss 阶段严格断言：

- `counts_by_kind["latent"] == 3`；
- stale、eval 和 duplicate 跳过数均为 0；
- 三条 collector 值与 P3/P4/P5 快照中的 aux 值一致；
- 所有分项均为有限标量。

发布缺失、重复、来自错误 step 或数值不一致时立即失败，不静默继续训练。

## 3. 损失公式

对每个尺度的 Router，设专家数为 `E`，batch 平均路由概率为 `p_e`，Router logits 为 `z`：

```text
balance = E * sum_e(p_e^2) - 1
z_loss  = mean(logsumexp(z)^2)
level_latent_aux = balance_loss_coeff * balance
                 + router_z_loss_coeff * z_loss
```

正式配置固定：

```text
balance_loss_coeff = 0.01
router_z_loss_coeff = 0.001
latent_aux_gain = 0.1
mixture_aux_budget = 3.0
```

三个尺度的 `level_latent_aux` 相加得到原始 `latent_aux_loss`。现有 mixture loss 基础设施随后使用持久化 EMA 尺度归一化，乘以 `latent_aux_gain`，并受 `mixture_aux_budget` 约束，得到最终加入检测 criterion 的 `mixture_aux_loss`。

WP6 没有另写检测损失，也没有绕过 `CompositeCriterion`。D1 的原生 E2E box/class/DFL loss 仍由现有 `E2ELoss` 计算。

## 4. 训练记录

D1 每个 epoch 在 `results.csv` 中记录：

| CSV 字段 | 含义 |
| --- | --- |
| `train/latent_balance_loss` | P3/P4/P5 未乘模块系数的 balance 之和 |
| `train/latent_z_loss` | P3/P4/P5 未乘模块系数的 z-loss 之和 |
| `train/latent_aux_loss` | 三层乘各自模块系数后的原始 aux 之和 |
| `train/mixture_aux_loss` | 归一化、gain 和 budget 后接入检测损失的 aux |

检测损失的完整记录顺序为：

```text
box_loss
cls_loss
dfl_loss
latent_balance_loss
latent_z_loss
latent_aux_loss
mixture_aux_loss
```

`model.last_latent_aux_metrics` 还保留 P3/P4/P5 各自的 balance、z 和 aux，以及 aux step 和发布数量，供调试和后续 WP7/WP8 结果审计使用。这些值全部 detach，不会额外保留计算图。

## 5. 梯度闭环

collector 保存的三条 latent aux 是图连接 Tensor，只有日志副本和路由快照被 detach。测试对总损失执行反向传播后确认：

- P3 Router 的输出层 bias 获得有限非零梯度；
- P4 Router 的输出层 bias 获得有限非零梯度；
- P5 Router 的输出层 bias 获得有限非零梯度；
- Adapter 和 Detect 原有梯度链路保持有效。

这证明 latent aux 不只是日志数值，而是实际参与优化的正则项。

## 6. 关闭语义

WP6 分别验证两级关闭：

1. `latent_aux_gain=0`：三层 balance、z 和原始 latent aux 仍被计算并记录，但最终 `mixture_aux_loss` 精确为 0，Router 不从 aux 获得非零梯度。
2. `balance_loss_coeff=0` 且 `router_z_loss_coeff=0`：三个 Router 仍发布图连接零值，四项 aux 指标全部精确为 0，Router 梯度存在但数值精确为 0。

保留图连接零值可以维持稳定的 DDP/optimizer 参数使用合同，同时保证关闭 aux 后不改变目标函数。

## 7. 实现与测试

主要代码：

- [`ultralytics/nn/foundation_detection_model.py`](../../ultralytics/nn/foundation_detection_model.py)：D1 三发布严格检查、分项聚合和调试指标；
- [`ultralytics/nn/mixture_loss.py`](../../ultralytics/nn/mixture_loss.py)：可选模型报告钩子，非 D1 模型行为不变；
- [`ultralytics/models/yolo/detect/foundation_train.py`](../../ultralytics/models/yolo/detect/foundation_train.py)：七项训练/验证指标名称；
- [`tests/test_d1_wp6_latent_aux.py`](../../tests/test_d1_wp6_latent_aux.py)：WP6 专用闭环测试。

代码提交：`c0277eb528c1ea65e37a99e851b29732ec1e7806`。

运行 WP6 专用测试：

```bash
python -m pytest -q tests/test_d1_wp6_latent_aux.py
```

使用现有缓存运行真实训练闭环：

```bash
export D1_WP2_CACHE=/path/to/wp2-train100-a
export D1_COCO_ROOT=/path/to/coco

python -m pytest -q \
  tests/test_d1_wp5_cached_pipeline.py::test_real_wp2_cache_one_batch_train_and_validate
```

最终 WP0-WP6、LatentMixture、CompositeCriterion、routing protocol、Foundation loss、checkpoint 和 recovery 回归结果为：

```text
218 passed, 2 skipped, 1 warning
```

两个 skip 是未启用的可选环境测试；warning 来自既有 MoA 测试自动调整 head 数。`py_compile`、`git diff --check` 和非 D1 CompositeCriterion 兼容测试均通过；服务器环境未安装 Ruff。

## 8. 阶段边界

WP6 已证明 aux 的计算、收集、日志、开关和梯度闭环，但尚未证明：

- Router 在小数据上能够随训练产生可解释的非均匀路由；
- 32 图过拟合时检测损失和 aux 曲线符合预期；
- 完整 COCO 训练中的路由稳定性和精度收益；
- 不同 balance/z/gain 配置的消融结论。

下一阶段 WP7 应完成最小训练正确性测试和小数据过拟合，检查 loss 下降、checkpoint 重载和路由指标随 step 的变化。正式全量 COCO 训练与成本结论仍属于 WP8。
