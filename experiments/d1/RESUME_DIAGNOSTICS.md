# D1 断点恢复差异诊断

日期：2026-09-09。范围：六卡隔离诊断、必要的恢复修复及证据归档，不启动正式实验。

## 1. 结论

本轮完成了恢复差异的定位，并修复了各 rank 缓冲区保存不完整的问题；**连续训练与中断恢复的逐位一致性仍未通过**，不能把本轮结果写成“严格恢复完全一致”。

三个架构 BASE、DW、BN64 的结果一致：在补齐各 rank 缓冲区后，恢复首批的输入、模型和优化器等状态、前向损失及归约前的本地梯度均逐位一致；第一处数值差异发生在 DDP/NCCL 汇总六卡梯度之后。

这将残余差异定位在分布式浮点归约路径，而不是特征缓存读取、EMA 批处理、学习率恢复、AMP scale 丢失或优化器状态没有加载。此前的 [EMA 批处理验收](EMA_BATCHING_RESULTS.md) 中，新旧 EMA 在相同执行路径下逐位一致的结论没有改变。

小样本数值差异不能直接换算成最终 AP 差异，也不能据此声称完整 COCO 训练的精度完全不受影响。

## 2. 诊断配置

| 字段 | 本轮设置 |
| --- | --- |
| experiment_id | D1-RESUME-20260909 |
| git_ref | `feat/topic-d1-fengyanqi`，基准 HEAD `6f88a422f426b5fe9bd013aa25a7ef6a95570231` |
| 实际代码身份 | 基准 HEAD 加未提交改动，具体文件 SHA256 见证据，不声称来自干净 commit |
| 配置来源 | 已准备的 P5 matrix 和原始合同；独立工作区记录 matrix、输入文件及代码哈希 |
| 数据 | 已有 COCO 2017，排序取 389 张 train2017 和 13 张 val2017，使用既有 NPY 特征缓存 |
| 硬件 | 6 张 A40；PyTorch 2.6.0+cu124、CUDA 12.4、NCCL 2.21.5 |
| 预算 | 保留原 100 epoch 调度，只执行前 2 个 epoch；不执行正式 100 轮实验 |
| batch | 全局 384，每 rank 64；每 rank 每轮 2 个 batch，第二个为 1 张图的短批次 |
| workers | 请求每 rank 4，短数据加载器实际限制为每 rank 2 |
| seed | 0 |
| 精度与更新 | AMP FP16，初始 scale 16，每 batch 更新一次，全部诊断训练 AMP 重试为 0 |
| EMA | `foreach-v1`，保留上一轮验证过的数学运算顺序 |
| 恢复策略 | `resume_temperature_policy=epoch-boundary-v1`；最终三架构诊断启用 `resume_rank_buffers=true` |
| status | 诊断完成；各 rank 缓冲区修复通过；连续/恢复逐位一致检查失败 |

每种架构分别运行：连续训练 2 轮、训练 1 轮后停止、从自己的新 checkpoint 恢复第 2 轮。最终验收共 9 个六卡短任务。验证图片共 13 张，每条完整路径累计 4 次 optimizer/EMA 更新。

图片和标签采用独立工作区内的逐文件符号链接，子集标签索引不会覆盖完整 COCO 数据目录中的索引。日志、快照、checkpoint 和完整结果均留在外部工作区。

## 3. 如何定位第一处差异

新增 [`diagnose_resume.py`](../../scripts/d1/diagnose_resume.py)，沿真实 P5 Trainer 逐阶段记录第 2 轮的两个 batch：

1. 输入图像 ID、特征和目标张量的形状、dtype、内容 SHA256。
2. 前向前的模型、optimizer、scaler、scheduler、EMA、更新计数、criterion 调度与模块 train/eval 状态。
3. 前向返回的损失及损失分量。
4. 参数 hook 捕获的本地梯度，发生在跨卡归约之前。
5. optimizer 更新前，DDP 已汇总后的梯度。
6. optimizer/EMA 更新后的状态。

hook 只克隆数值，不替换返回值或梯度。每轮结束时移除 hook，不将诊断对象带入 checkpoint。快照只保存张量和基本类型，分析时使用 `weights_only=True`。

“checkpoint 可 strict load”和“与不中断训练逐位相同”是不同检查：前者检查结构与状态能够加载，后者检查数值轨迹。本轮前者通过、后者未通过。

## 4. 数值结果

下表对应修复各 rank 缓冲区后的最终诊断，均比较同一种架构的连续路径与恢复路径。

| 检查位置 | BASE | DW | BN64 |
| --- | --- | --- | --- |
| 首批输入，全部 6 rank | 逐位一致 | 逐位一致 | 逐位一致 |
| 首批前的全部记录状态，全部 6 rank | 逐位一致 | 逐位一致 | 逐位一致 |
| 首批前向损失，全部 6 rank | 逐位一致 | 逐位一致 | 逐位一致 |
| 首批本地梯度，全部 6 rank | 逐位一致 | 逐位一致 | 逐位一致 |
| 首批归约后梯度 | 存在差异 | 存在差异 | 存在差异 |
| 首批归约后梯度最大绝对差，AMP scaled | 6.103515625e-5 | 1.52587890625e-5 | 3.0517578125e-5 |
| 除以 scale=16 后的最大绝对差 | 3.814697266e-6 | 9.536743164e-7 | 1.907348633e-6 |
| 第 2 轮两个更新后，rank 0 浮点模型状态最大绝对差 | 9.742714465e-4 | 1.600816846e-3 | 7.362670149e-4 |
| 同一位置浮点模型状态相对 L2 差 | 8.729077440e-6 | 1.444024240e-5 | 8.128546433e-6 |

相对 L2 定义为 `||continuous - resumed||2 / ||continuous||2`，统计 rank 0 模型 state dict 中全部浮点张量，包括浮点缓冲区；不是 AP 误差，也不是每个张量最大相对误差。

在 BASE 中，原诊断的首批归约后共有 238 个梯度张量出现差异；该次参数更新后最大参数差约为 `2.384185791e-7`。下一短批次继续经过 AMP 前向、反向和 AdamW 更新后，模型状态最大绝对差扩大到约 `9.74e-4`。因此不能只检查首批打印到四位小数的 loss。

三个架构的连续、停止和恢复 checkpoint 均可严格加载，均没有 Teacher 参数，具体 checksum 见 [summary.json](manifests/resume-diagnostics-20260909/summary.json)。停止时 checkpoint 的 checksum 是当时采集的值；恢复后同一运行目录的 `last.pt` 会正常更新，不代表保留了停止时文件的副本。

## 5. 原因与必要修复

### 5.1 梯度通信路径

当前 Trainer 使用 `static_graph=True`、`broadcast_buffers=False` 的 DDP。重新启动后会重新构建 DDP reducer，其运行阶段和通信分桶不是模型/optimizer state dict 的组成部分。

PyTorch 2.6.0 的实现会在训练开始阶段重建通信分桶，静态图首轮也有专门的归约路径。实验中连续路径与恢复路径的 reducer 迭代阶段、分桶布局不同；配合“本地梯度逐位相同、归约后不同”的直接证据，可以把差异定位到归约环节。浮点数的加法结合顺序会影响末位舍入。[PyTorch 2.6.0 reducer 实现](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/distributed/c10d/reducer.cpp)

本轮没有分别强制 NCCL 算法和协议，也没有逐项证明每一种分桶、通道和归约顺序各贡献多少误差，不将原因进一步简化为单个 NCCL 内核的缺陷。

### 5.2 各 rank 的 BatchNorm 统计遗漏

修复前，`resume.pt` 保存主卡模型状态和各卡 RNG，却没有保存各卡独立的模型缓冲区。由于 `broadcast_buffers=False`，检测头的 BatchNorm 运行均值和方差会因各卡样本不同而不同；恢复时却都从主卡状态开始。

Adapter 仍使用 GroupNorm，这里的 BatchNorm 位于检测头。在训练态下，BN 使用当前 batch 的统计，因此原诊断中其他 rank 的运行统计虽不同，首批本地梯度仍然一致；但这并不意味着可以遗漏恢复状态。

已在 [`p1p2_runtime.py`](../../scripts/d1/p1p2_runtime.py) 中补齐：

- 在既有各 rank 状态中保存 `model_buffers` 和 `ema_buffers`，包括已注册的非持久缓冲区。
- 恢复时校验完整键集合、形状、dtype 和有限性，通过全部校验后再复制。
- 策略必须与运行 identity 一致；需要新策略但 checkpoint 缺字段时明确失败，不静默降级。
- 未声明新字段的旧配置继续使用原恢复行为，不改写或自动升级旧 checkpoint。

[`run_p5_ablation.py`](../../scripts/d1/run_p5_ablation.py) 新准备的工作区记录 `resume_rank_buffers=true`。本轮没有修改已经准备好的正式工作区；以后启动正式实验前需要按确定后的代码重新准备新工作区，不能绕过代码身份校验。

修复后的 BASE 连续路径与修复前对照比较，6 rank × 2 batch 的全部记录训练状态均逐位一致。修复不改变不中断训练的计算配方，也不新增每步通信；附加内容在既有 epoch 结束 checkpoint 汇总中保存。

## 6. 没有接入的方案

另做了 BASE 的两次无参数更新 DDP 预热对照，保存并恢复缓冲区、criterion 和随机状态，不推进 optimizer、EMA 或 scaler。

预热后训练状态不变，但首批归约后仍有 160 个梯度张量不同，第二批更新后最大模型差仍约为 `5.83e-4`；重建后的分桶布局也未与连续路径完全相同。

因此 `--prime-ddp` 仅保留在隔离诊断脚本中，**没有接入正式 Trainer**。本轮也没有采用固定顺序 all-gather 或 FP64 梯度归约来强制追求逐位一致，因为它们会改变通信实现、数值路径和性能，需要单独的正确性与开销对照。

## 7. 测试

| 项目 | 结果 |
| --- | --- |
| 新增诊断、快照、误差、缓冲区与兼容性测试 | 29 passed |
| D1、LatentMixture、Foundation/checkpoint、DDP checkpoint 相关回归 | 549 passed，58 skipped；53.64 秒 |
| EMA 与恢复诊断测试，CUDA 可用 | 63 passed；9.49 秒，包含 AMP 溢出恢复测试 |
| 三架构真实六卡短流程 | 9 个子任务完成，0 AMP 重试，每条完整路径 4 次更新 |
| 改动 Python 文件 Ruff、py_compile、codespell | 通过 |
| 新增 Python 文件 Ruff format | 通过 |
| git diff --check | 通过 |

58 项跳过包含 CPU 回归环境下的 CUDA/本地资产可选测试，不能写成所有可选集成测试均已运行。没有顺带修复仓库已有的全局 lint 问题。JUnit 文件保留在外部工作区，摘要中记录其 SHA256。

## 8. 复现与证据

在仓库根目录运行以下隔离诊断。`SOURCE_WORKSPACE` 指向已有、来源校验有效的 P5 准备工作区；`DIAGNOSTIC_WORKSPACE` 必须是新的外部目录，不能指向仓库或旧实验目录。

```bash
export SOURCE_WORKSPACE=/path/to/prepared-p5-workspace
export DIAGNOSTIC_WORKSPACE=/path/to/new-resume-diagnostic

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$PWD" \
python -m scripts.d1.diagnose_resume run \
  --source-workspace "$SOURCE_WORKSPACE" \
  --output "$DIAGNOSTIC_WORKSPACE" \
  --variant BASE --rank-buffers

python -m scripts.d1.diagnose_resume summarize \
  --output "$DIAGNOSTIC_WORKSPACE" --variant BASE
```

DW 和 BN64 分别使用 `--variant DW`、`--variant BN64`，每次指定不同的新输出目录。省略 `--rank-buffers` 可复现旧的主卡缓冲区恢复语义；只有显式添加 `--prime-ddp` 才执行实验性预热。

脚本每个子任务限时 300 秒，失败会回收本脚本创建的进程组并写失败状态，不处理其他用户的进程。上述命令会执行小样本训练，不是完整 COCO 实验。快照拷贝和分析存在额外开销，不能用这轮耗时推算正式训练吞吐或 GPU-hours。

仓库小型证据：

- [summary.json](manifests/resume-diagnostics-20260909/summary.json)：三架构数值结果、checkpoint checksum、代码哈希、连续路径对照和预热对照。
- [validation.json](manifests/resume-diagnostics-20260909/validation.json)：测试数量、JUnit 哈希、旧文件保护校验、进程与 GPU 退出状态。

外部工作区目录名：`resume-diagnostics-20260909-d-base`、`resume-diagnostics-20260909-d-dw`、`resume-diagnostics-20260909-d-bn64`。每个目录包含 `reports/`、逐 rank 快照、3 份训练日志及 `diagnostics.json`。BASE 目录的 `archive/` 额外保存归档摘要和 JUnit。前置诊断与预热对照分别留在 `resume-diagnostics-20260909-a`、`resume-diagnostics-20260909-c`。

## 9. 边界与后续

目前可以确认正常的状态恢复、checkpoint 严格加载及 EMA 等价性；**不能确认跨进程重启后的训练数值轨迹逐位等于不中断训练，也没有验证最终 AP 差异**。

下一步若仍要求逐位一致，应单独设计固定通信布局/顺序的对照，测量其精度与速度代价；若采用数值一致性与多 seed 统计作为实验标准，则需要先明确容差、精度门槛和恢复次数，不在看到结果后临时放宽。本轮不擅自替换现有实验验收标准。

没有启动正式 BASE/DW/BN64 实验、改动已有 checkpoint、修改 `P1_P2_EXPERIMENT_PLAN.md`、提交 commit 或推送 GitHub。旧 DW `last.pt`、`resume.pt` 及实验计划 SHA256 均与操作前相同。全部本轮任务已退出，GPU 已回到诊断前占用水平。
