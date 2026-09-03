# D1 WP7：最小训练与工程验收

## 1. 阶段结论

WP7 已完成 WP0-WP6 组件的集中工程验收。真实 DINOv3 在线特征与 WP2 缓存逐层一致，32 图过拟合达到损失、精度、Router 更新和 checkpoint 门槛，COCO8 完成一轮训练、验证、保存和严格重载闭环。三项 gate 及总验收均为 `passed`。

本阶段证明缓存特征训练链路可运行、可恢复、可度量，不代表完整 COCO 2017 上的正式精度或训练成本结论。完整 train2017/val2017 缓存、正式训练和对照实验属于 WP8。

## 2. 固定验收合同

验收配置由 [`wp7-minimal-tests.yaml`](../../ultralytics/cfg/experiments/d1/wp7-minimal-tests.yaml) 固定，不包含服务器绝对路径：

| Gate | 数据 | 训练设置 | 通过条件 |
| --- | --- | --- | --- |
| 在线缓存对齐 | `wp2-train100-a` 前 8 图 | DINOv3 ViT-S/16，FP16，640 输入 | 三层有限，`rtol=atol=1e-3` |
| 32 图过拟合 | 缓存中排序后的前 32 图，训练/验证同集 | 100 epochs，batch 8，AdamW，AMP，无增强 | 损失比不高于 0.5，mAP50 不低于 0.20，mAP50-95 不低于 0.05，至少 400 次更新 |
| COCO8 闭环 | 前 4 图训练、后 4 图验证 | 1 epoch，batch 4，AMP，无增强 | 至少一次更新、验证 4 图、checkpoint 严格重载，不设精度门槛 |

所有 gate 固定 seed 0、确定性执行、`workers=0`，数据列表、运行目录和报告均由 [`run_wp7.py`](../../scripts/d1/run_wp7.py) 生成。已通过且代码 commit、缓存合同和 Teacher 权重摘要完全一致的报告可安全复用；身份不一致时立即拒绝。

## 3. 在线特征与缓存对齐

使用正式 DINOv3 ViT-S/16 权重重新抽取 8 张图片，并与 `wp2-train100-a` 比较：

| 层 | shape | dtype | 最大绝对误差 | 平均绝对误差 |
| --- | --- | --- | ---: | ---: |
| block4 | `[384,40,40]` | FP16 | 0.0 | 0.0 |
| block8 | `[384,40,40]` | FP16 | 0.0 | 0.0 |
| block12 | `[384,40,40]` | FP16 | 0.0 | 0.0 |

Teacher 在验收期间保持冻结和 eval。缓存完整校验覆盖 100 个样本、300 个 tensor，内容摘要为 `0102f2e707b369ca8bfb3d996d54fa0d8ba9eba79787501bd5fe6542f90c1a8d`。

详细证据：[`wp7-parity.json`](manifests/wp7-parity.json)。

## 4. 32 图过拟合

100 epochs 实际耗时 192.586 秒，共完成 400 次有效 optimizer 更新。最终结果：

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| 初始检测损失 | 11.90014 | - |
| 最终检测损失 | 1.19775 | - |
| 前 10 轮损失中位数 | 8.069325 | - |
| 后 10 轮损失中位数 | 1.23077 | - |
| 后期/前期损失比 | 0.152525 | 不高于 0.5 |
| mAP50 | 0.94225 | 不低于 0.20 |
| mAP50-95 | 0.78591 | 不低于 0.05 |

P3、P4、P5 的 Router 参数与 residual gain 均发生有限非零变化：

| 尺度 | Router 最大绝对变化 | residual gain 最大绝对变化 |
| --- | ---: | ---: |
| P3 | 0.026199 | 0.170898 |
| P4 | 0.017395 | 0.084839 |
| P5 | 0.014122 | 0.028870 |

最终路由诊断全部有限，checkpoint 严格重载成功，state dict 不包含 Teacher 参数。最佳 checkpoint 对应较早 epoch，因此其中记录 388 次更新；验收报告使用完整训练结束时的 400 次有效更新判定通过。

详细证据：[`wp7-overfit32.json`](manifests/wp7-overfit32.json)。

## 5. COCO8 一轮闭环

COCO8 直接复用完整 COCO 2017 与 WP2 缓存中的 8 张图片，不进行额外下载：

```text
train: 000000000009, 000000000025, 000000000030, 000000000034
val:   000000000036, 000000000042, 000000000049, 000000000061
```

一轮训练耗时 7.913 秒，完成 1 次 optimizer 更新和 4 张验证图片评测。checkpoint 严格重载及最终验证均通过，Teacher 参数数为 0，P3/P4/P5 Router 与 residual gain 均发生非零更新。该 gate 不设置精度门槛，`mAP=0` 只表示单步训练尚未形成检测能力，不影响工程闭环结论。

详细证据：[`wp7-coco8.json`](manifests/wp7-coco8.json)。

## 6. AMP 恢复修复

首次 32 图运行暴露了训练器恢复路径的步进游标问题：AMP 初始动态缩放检测到非有限梯度后，恢复控制器正确回滚并关闭 AMP，但局部 `last_opt_step` 仍保留恢复前的全局 batch 索引，导致重启 epoch 的前三批没有执行 optimizer 更新，最终只有 397 次更新。

修复后，训练开始和两条恢复路径都把游标设为“当前 epoch 第一批之前”的全局索引。验收门槛仍保持 400，没有通过降低门槛掩盖问题。新增回归测试覆盖非零 epoch 的游标计算。

修复提交：`a0a05135a3da7228aeabb662003ce125ffe774dc`。

## 7. 复现命令

外部工作区需包含 WP0 的 COCO、Teacher 权重和 WP2 缓存：

```bash
export D1_WORKSPACE=/path/to/yolo-master-d1
export D1_RUN_ROOT="$D1_WORKSPACE/runs/wp7-$(git rev-parse --short HEAD)"
export D1_REPORT_DIR="$D1_WORKSPACE/manifests/wp7-$(git rev-parse --short HEAD)"

python scripts/d1/run_wp7.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  --device cuda:0 parity

python scripts/d1/run_wp7.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  --device cuda:0 train --profile coco8

python scripts/d1/run_wp7.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  --device cuda:0 train --profile overfit32

python scripts/d1/run_wp7.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  --device cuda:0 summarize
```

也可以使用 `all` 依次执行全部 gate。运行日志、完整 `results.csv`、routing trace 和 checkpoint 只保留在外部工作区，不进入 Git。

## 8. 实现、测试与证据

主要文件：

- [`scripts/d1/run_wp7.py`](../../scripts/d1/run_wp7.py)：parity、训练、恢复和汇总入口；
- [`tests/test_d1_wp7_acceptance.py`](../../tests/test_d1_wp7_acceptance.py)：WP7 合同、样本选择、报告和失败条件测试；
- [`ultralytics/engine/trainer.py`](../../ultralytics/engine/trainer.py)：D1 AMP 兼容检查扩展点及恢复游标修复；
- [`ultralytics/models/yolo/detect/foundation_train.py`](../../ultralytics/models/yolo/detect/foundation_train.py)：缓存特征 AMP 一致性检查。

代码提交链：

```text
b120185  test(d1): add WP7 minimal training gates
527688a  fix(d1): use cache reader root in WP7 identity
bfae5f4  fix(d1): validate AMP without RGB model download
167fe68  fix(d1): retain final training routing evidence
a0a0513  fix(trainer): reset optimizer cursor after recovery
```

最终相关 D1、Foundation、LatentMixture、checkpoint 和 recovery 回归结果：

```text
387 passed, 6 skipped
```

`py_compile` 和 `git diff --check` 通过；服务器未安装 Ruff。总验收证据见 [`wp7-summary.json`](manifests/wp7-summary.json)，其中 parity、COCO8、overfit32 三项均为 `passed`。

## 9. 阶段边界

WP7 已证明：

- Teacher 在线输出与缓存内容一致；
- Adapter、Router、Detect、criterion 和 Trainer 可完成真实训练闭环；
- 小样本可被模型记忆，损失和精度达到固定门槛；
- Router、residual gain、checkpoint 和恢复机制实际生效。

WP7 尚未证明：

- 完整 COCO train2017/val2017 的正式精度；
- 相同训练预算下相对零训练检测器的精度保留比例；
- 冻结 Teacher 方案相对基线的 GPU 时间、显存和存储收益；
- 不同 block、balance/z-loss 和缓存策略的消融结论。

以上内容进入 WP8 完整 COCO 训练与性能报告。
