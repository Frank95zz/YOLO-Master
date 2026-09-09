# D1 EMA 批处理接入与闭环验收

日期：2026-09-09。基座提交：`6f88a422f426b5fe9bd013aa25a7ef6a95570231`。

本轮在该基座的工作区上实现和测试，代码尚未提交/推送；实际运行文件 SHA256 见[闭环证据](manifests/ema-integration-20260909/loop-summary.json)。不将基座提交冒充已经包含本轮修改的干净代码版本。

## 1. 结论与范围

- 已接入 D1 专用 `foreach-v1` EMA，默认仍为 `scalar-v1`。不替换全局 `ModelEMA`，不影响未显式启用的其他实验。
- BASE、DW、BN64 在相同执行路径下，新旧 EMA 的模型、EMA、优化器、scaler、scheduler、计数器和 criterion 状态均逐位一致。
- 三种架构均完成真实缓存训练、验证、保存、严格重载、停一轮再恢复；无 AMP 重试、无漏更新。
- **不能声称跨进程恢复与不间断训练逐位一致。** 原版和新版均存在同样的残余差异，详见第5节。总验收状态为 `ema_passed_with_resume_limitation`，而非无条件全部通过。
- 未启动第6.5节的正式三架构筛选，未加入 P3 编译、GPU 预取或 Router 统计合并；未更改原实验计划和旧 checkpoint。

## 2. 正式代码接口

| 文件 | 职责 |
| --- | --- |
| [ema.py](../../scripts/d1/ema.py) | `D1ModelEMA`、实现名称校验和局部安装 |
| [run_p5_ablation.py](../../scripts/d1/run_p5_ablation.py) | prepare 时登记实现，setup/resume 后安装，身份一致性检查 |
| [run_p5_suite.py](../../scripts/d1/run_p5_suite.py) | 队列启动记录 EMA 实现和恢复策略 |
| [check_ema_runtime.py](../../scripts/d1/check_ema_runtime.py) | 独立小样本六卡闭环，不调用正式实验队列 |
| [test_d1_ema.py](../../tests/test_d1_ema.py) | 算术、状态、dtype、接口、AMP与恢复边界测试 |

实现保留原先先乘再加的算术顺序，不使用会改变舍入顺序的 fused alpha-add。参数和持久浮点 buffer 按原规则平均，整数及指定运行时 buffer 按原规则复制；不额外更新非持久 buffer。每次读取当前注册张量，不缓存过期 tensor 引用，也不在每步序列化诊断 extra state。

仅 FP32 来源/目标使用批处理。FP16、BF16、FP64和混合 dtype 使用原有逐张量计算，避免低精度 foreach 标量提升差异。AMP 前向为混合精度，不等于 EMA 必须为 FP16。

通过已有 EMA 实例安装，保留 EMA 模型、`updates`、`decay` 和 enabled 状态；不重新复制模型、不重置恢复计数。没有 EMA 的 rank 仍保持原样。拒绝非 D1 模型、未知实现、形状/设备不一致和带 state_dict hook 的来源。

新矩阵记录 `ema_implementation` 和 `resume_temperature_policy`，进入运行身份及恢复校验。旧矩阵缺字段时分别保持 `scalar-v1` 和 `legacy`，不静默升级旧实验。

## 3. 小样本闭环合同

| 项目 | 实际设置 |
| --- | --- |
| 架构 | BASE、DW、BN64 |
| GPU | 6张 A40 |
| 数据 | 现有 NVMe COCO2017 RGB/标签及 NPY 特征缓存 |
| 训练/验证图数 | 389 / 13；均来自各自官方 split，不混用 |
| batch | 全局384，每卡64；每轮两个 batch，尾 batch 每卡1 |
| sampler | 六卡对齐到390样本，填充一个样本；不是增加一张独立图 |
| workers | 配置4/rank；框架受两步小子集限制，实际2/rank，共12 |
| 训练窗口 | 两轮，共4次 optimizer/EMA 更新；保留100轮调度参数 |
| 精度与增强 | AMP、初始scale16、seed0、确定性、无增强 |
| 共同 P3 | `separable_bilinear2x`，不是 torch.compile 版本 |

每种架构运行四组配对路径：原版连续两轮、新版连续两轮、原版停一轮再恢复、新版停一轮再恢复。合计18个有界六卡子任务；每个子任务最长300秒，超时仅清理自身进程组。

测试使用独立的图像/标签链接目录，标签扫描缓存、checkpoint和日志仅写到本轮外部工作区，不覆盖完整数据集的标签缓存。不下载、不转换、不重建大缓存。

## 4. 已通过的检查

| 架构 | 新旧连续训练状态 | 新旧恢复训练状态 | 验证图数 | optimizer/EMA更新 | AMP重试 |
| --- | --- | --- | ---: | ---: | ---: |
| BASE | 逐位一致 | 逐位一致 | 13 | 4 / 4 | 0 |
| DW | 逐位一致 | 逐位一致 | 13 | 4 / 4 | 0 |
| BN64 | 逐位一致 | 逐位一致 | 13 | 4 / 4 | 0 |

比较采用 `rtol=0, atol=0`，覆盖模型、EMA、优化器、scaler、scheduler、EMA/optimizer计数和criterion。checkpoint可严格重载且不包含Teacher参数。这里的13图验证只验工程闭环，不给出正式 COCO AP 结论。

相关 D1、LatentMixture、Foundation checkpoint、DDP checkpoint、兼容性回归：**520 passed, 58 skipped，52.58秒**。普通CPU回归的跳过项包括可选CUDA、真实权重和缓存集成条件；新增EMA测试另以CUDA执行，**34 passed，9.31秒**。见[测试记录](manifests/ema-integration-20260909/validation.json)。

本轮五个Python文件的Ruff检查、新建文件的格式检查、拼写检查及 `git diff --check` 通过。全库检查仍不干净：基座与当前均有8项Ruff问题、27个待格式化文件、85行拼写告警输出；新增三个Python文件未增加这些数量。保留已有无关问题，没有顺带重排或修复其他代码。

## 5. 恢复边界与剩余局限

发现共享控制器在 `epoch == start_epoch` 时不执行温度衰减。连续训练第二轮温度为0.97，而旧恢复路径仍为1.00。新登记的 P5 使用 `epoch-boundary-v1`：恢复完整状态后补一次应有的温度衰减，再进入恢复轮。不改全局控制器、不追溯更新旧结果；新旧策略均有测试。

温度边界修正后，跨进程恢复相对于连续训练仍有以下差异。**同一架构中，原版和新版的偏差数值完全相同。**

| 架构 | 模型参数/浮点状态最大绝对差 | 整体相对L2差 |
| --- | ---: | ---: |
| BASE | 9.742714e-4 | 8.729077e-6 |
| DW | 1.600817e-3 | 1.444024e-5 |
| BN64 | 7.362670e-4 | 8.128546e-6 |

在BASE的进一步读取中，恢复轮第一次 backward 后仅少数参数的梯度范数出现很小差别，学习率、scaler、scheduler、criterion及更新次数一致。跨进程重建后具体是哪一步造成剩余差异尚未完全定位；不能直接断言是磁盘、EMA或数据错误，也未测定其对最终AP的影响。

本轮证明的是 **EMA替换不增加已观察到的恢复差异**，而不是严格证明任意断点恢复都复现连续训练的全部浮点轨迹。正式报告若涉及续训，应记录断点与执行路径，不能省略该局限。

## 6. 使用方式

在仓库根目录检查配置，下面命令不启动训练：

```bash
python -m scripts.d1.run_p5_ablation inspect \
  --p3-upsample-mode separable_bilinear2x \
  --ema-implementation foreach-v1
```

后续获准实验后，应在最终代码版本上重新 prepare 一个空的外部工作区，不能手改旧矩阵来绕过 provenance 检查：

```bash
python -m scripts.d1.run_p5_ablation prepare \
  --source-workspace "$E1_SOURCE_WORKSPACE" \
  --workspace "$NEW_P5_WORKSPACE" \
  --p3-upsample-mode separable_bilinear2x \
  --ema-implementation foreach-v1
```

这仍然只准备输入，不授权训练。train/evaluate禁止临时传入EMA实现；它们读取已登记矩阵。

独立复现本轮小样本验收，需要已有正式缓存和配对初始权重，输出目录必须不存在：

```bash
python -m scripts.d1.check_ema_runtime run \
  --source-workspace "$P5_SOURCE_WORKSPACE" \
  --output "$NEW_EMA_ACCEPTANCE_WORKSPACE"
```

## 7. 性能解释与后续

上一轮隔离原型中，EMA批处理使完整步骤中位耗时减少约2.9%-4.5%。本轮主要验正确性、接口和恢复，不把两步小子集耗时作为正式训练ETA；也未重新证明生产实现的完整COCO提速比例。

正式长实验仍暂停。若继续追求跨进程位级复现，应单独诊断恢复后第一次forward/backward及DDP归约路径，不能通过放宽本轮EMA对照容限来掩盖差异。正式成本比较需使用同一实现、同一数据与预算，仍不声称满足P1的GPU-hours降低50%。
