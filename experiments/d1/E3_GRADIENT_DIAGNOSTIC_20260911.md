# E3 独立梯度诊断精度修复

## 结论与边界

2026-09-11，E3 第一阶段已完成 21/27 组。第 22 组
`E3-screen-b0.1-z0.001-g0.1-s0` 在完成第 18 轮后、第 19 轮训练开始前，
被独立诊断的梯度分解检查拦截。原因是诊断副本中 FP32 Softmax 反向归约的舍入误差，
不是训练 OOM、AMP 溢出或数据损坏。该诊断早已禁用 TF32，不能归因为 TF32。

修复仅应用于独立诊断副本，不改变正式训练的 Teacher、Adapter、LatentMixture、
Detect、loss、AMP、优化器、EMA、数据或调度。所有 `ultralytics/` 代码保持原样。
不调整 `rtol=2e-4, atol=2e-5`，不跳过梯度检查，不修改已完成的实验结果。

## 精确复现

- 原科学代码：`993aee4d32168616fb96cedb2b524379c09d35fd`。
- 恢复点：完成 18 轮，1224 次 optimizer/EMA 更新，AMP scale 16，重试 0。
- 恢复点 SHA256：`2facae865c3dc8384d8a5b40860f2c99e39c9fa653e1b8551a81084f299686e9`。
- 按 checkpoint 恢复各 rank 缓冲区、criterion 状态及原 epoch-boundary 温度规则。
- 每卡使用原排序列表中 `2*rank, 2*rank+1` 两张图；rank 5 即第 10、11 索引。
- 六卡共同执行原辅助损失归约。两次独立重复得到完全相同的原始梯度和偏差。
- 唯一超差参数：`model.mixtures.p3.router.expert_head.weight`，5/256 个元素。

数学上 `grad(det) + grad(aux) = grad(det + aux)`；但 FP32 反向归约不保证两种计算
顺序逐元素等价。Softmax 的反向公式含有
`p * (g - sum(g * p))`，减法消去与归约舍入会传到 Router 权重。
例如一个超差元素的检测梯度约 -0.1621803、aux 梯度约 0.1392528，合成后只有约
-0.0229550；严格逐元素比较会放大相对影响。该观察不代表存在真实的非线性梯度分解。

## 修复实现

新增 `scripts/d1/e3_probe_precision.py`，版本 `softmax-vjp-fp64-v1`：

1. 只在 `probe()` 创建的独立模型副本中安装临时 Router hook。
2. 前向仍调用 FP32 `torch.softmax`，每次都断言其输出与原 Router 输出逐元素完全一致。
3. 一阶反向使用保存的同一组 FP32 概率，在 FP64 中计算上述小型 VJP，再转回 FP32。
4. 前向 hook 在退出或异常时移除；正式训练模型不安装 hook。
5. 原 `separated_gradients` 实现和严格容限保持不变，NaN/Inf 和真实分解错误仍直接报错。

这里不是把整个模型或训练切换到 FP64，仅提高诊断中每张图 4 个路由概率的反向归约精度。
新旧诊断的梯度范数存在正常的末位差异，因此报告明确记录诊断精度版本。

## 验证结果

| 项目 | 修复前 | 修复后 |
| --- | --- | --- |
| rank 5 出错权重最大分解误差 | 2.810359e-5 | 4.053116e-6 |
| rank 5 超差元素 | 5/256 | 0/256 |
| rank 5 所有参数最大误差/容限比 | 超过 1 | 0.15854 |
| 六卡严格原检查 | rank 5 失败 | 全部通过 |
| 诊断前向、routing、loss、aux EMA | 对照 | 与原诊断完全一致 |
| 源模型参数及 buffers | 对照 | 全部保持不变 |
| 两次独立诊断的梯度值 | 完全一致 | 完全一致 |

回归测试：161 passed，包括 E3、新增梯度精度、VisDrone、P1/P2 合同和恢复诊断测试。
新增测试涵盖精确前向、FP64 VJP 参考、hook 隔离和异常清理、前向漂移拒绝、非有限梯度、
人为错误梯度仍失败，以及恢复代码版本校验。真实六卡验证既记录误差，也实际运行未改动的严格检查。

小型证据：[e3-gradient-diagnostic-20260911.json](manifests/e3-gradient-diagnostic-20260911.json)。
完整逐参数梯度、日志及各 rank 报告保存在外部诊断工作区，不进入 Git。

## 版本登记与恢复

`run_e3.py` 新增严格的诊断修复登记：只允许显式批准、原代码后继的干净提交，
且变更文件必须属于该诊断修复白名单；训练核心或配置变化一律拒绝。
恢复登记绑定原科学 identity、新 execution identity 和实际验收证据 SHA256。
原 matrix、初始状态、已有结果和 checkpoint 的科学 identity 不变。
后续 checkpoint 和执行成本记录另存实际 execution identity，避免把旧实验伪装成新代码生成。

恢复前保全原第 18 轮恢复点及 `last.pt`，从第 19 轮继续原第 22 组；前 21 组按原校验跳过。
保留原失败尝试的运行耗时，不从成本账本删除。本次修复不代表全部 E3 或官方 MATLAB 评测已完成。

隔离复现入口：

```bash
PYTHONPATH="$ORIGINAL_CODE" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=6 \
  "$REPAIRED_CODE/scripts/d1/diagnose_e3_gradients.py" \
  --workspace "$E3_WORKSPACE" --output "$DIAGNOSTIC_OUTPUT" \
  --repair-dir "$REPAIRED_CODE/scripts/d1"
```

`ORIGINAL_CODE` 为原干净提交目录，`REPAIRED_CODE` 为本次修复提交目录，
`E3_WORKSPACE` 为原实验工作区，`DIAGNOSTIC_OUTPUT` 必须使用单独目录。
此命令只做诊断，不更新模型；不加 `--repair-dir` 可在原代码环境重现原偏差。
