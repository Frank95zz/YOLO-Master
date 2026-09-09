# P3 确定性双线性上采样优化验证

> 文档迁移与执行更新：第 1-7 节保留隔离诊断时的完整结论和边界；第 8 节登记用户随后批准的仓库接入与第 6.5 节三架构实验。历史数值证据不改写为新训练结果。

## 1. 结论与范围

2026-09-09，针对 D1 Adapter 的 P3 上采样完成隔离数值验证、确定性微基准和六卡完整训练步骤对照。

**结论：保持 `deterministic=True`、2 倍双线性插值和 `align_corners=False`，候选实现将完整训练步骤耗时降低约 22%–24%。**

候选实现目前仅位于外部诊断脚本，不在正式仓库默认路径中生效。本次没有恢复、启动或覆盖正式实验，没有修改主仓库中已有的实验计划变更，也没有提交或推送代码。

代码基座：`3dfe0f6e72022b62a2f169e693b4996ac0361f7e`。
服务器外部工作区：`/root/yolo-master/yolo-master-d1-work`。
原版 P5、DW、BN64 是 P5 架构对照；本次只改变三者共同的 P3 上采样实现，不改变 P5。

## 2. 优化内容

原版三个 P3 分支均使用：

`1x1 Conv -> GroupNorm -> SiLU -> nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)`

服务器安装的 PyTorch 2.6 在 CUDA 确定性模式下，将该双线性上采样转入分解实现。Profiler 将高耗时的 `UnsafeIndexBackward` 追溯到 Adapter，其 FP32 索引反向 kernel 累计约 116 ms/步。

候选 `SeparableBilinear2x` 先沿宽、再沿高进行插值，用基本切片、拼接和交错排列代替高级索引。单轴数学表达为：

- `y[2i] = 0.25*x[i-1] + 0.75*x[i]`；
- `y[2i+1] = 0.75*x[i] + 0.25*x[i+1]`；
- 越界时复制端点，与原版边界规则一致。

实际代码使用与 PyTorch 相同的 `a + (b-a)*weight` 运算顺序。FP16 输入先提升至 FP32 完成两轴插值，再转回 FP16。不引入参数或缓冲区，不改变输出尺寸、通道、loss、AMP、随机种子或优化器。

适用范围是浮点 BCHW、固定 2 倍、`align_corners=False`、当前 eager 执行合同；不宣称是任意缩放倍率或 `torch.compile` 的通用替代。

## 3. 数值验证

覆盖 45 组 CUDA 测试：FP64 / FP32 / FP16，普通连续、channels-last、转置布局，单行、单列、单像素、非方形和正式 40x40 网格。

| 数据类型 | 前向最大绝对差 | 输入梯度最大绝对差 |
| --- | ---: | ---: |
| FP64 | 0 | 1.7764e-15 |
| FP32 | 0 | 9.5367e-7 |
| FP16 | 0 | 4.8828e-4 |

45 组前向均逐元素完全一致；候选自身重复运行的输出和输入梯度均逐元素一致。另有 2 组 CPU FP64 gradcheck、2 组 gradgradcheck、3 组非法输入检查通过。

完整模型使用真实缓存的 COCO 图片 `000000000009/25/30/34`，对 BASE、DW、BN64 分别执行 FP32 和 AMP 单步，共 6 组：

- loss 与已报告的损失分项均逐元素完全一致；
- AMP 下参数梯度相对 L2 偏差最高约 1.7004e-7，最大绝对偏差 3.8147e-6；
- AMP 下单次 AdamW 更新的最大绝对偏差最高约 5.8375e-6；
- 参数量、state dict 键不变，严格加载通过；
- 梯度并非与原实现逐位一致。短测通过不等于长期轨迹或最终 AP 已证明完全一致。

完整数值记录见 [numerical-status.json](manifests/p3-deterministic-probe-20260909/numerical-status.json)。

## 4. 确定性微基准

单张 A40，batch 64，正式三组 `[64,384,40,40]` FP16 输入，完整 Adapter 前向和反向，5 次预热、15 次测量，中位数：

| P5 架构 | 原版前向 ms | 候选前向 ms | 原版反向 ms | 候选反向 ms |
| --- | ---: | ---: | ---: | ---: |
| BASE | 27.04 | 26.60 | 137.69 | 30.33 |
| DW | 28.50 | 28.11 | 139.61 | 32.14 |
| BN64 | 27.87 | 27.48 | 139.17 | 31.76 |

这里两种实现均明确开启确定性，不能与此前未明确匹配确定性设置的单独 micro 结果混用。

证据：[deterministic-micro.json](manifests/p3-deterministic-probe-20260909/deterministic-micro.json)。

## 5. 六卡完整步骤

配置：6 张 A40，每卡 batch 64，全局 batch 384，AMP，确定性开启，每 rank 4 个 DataLoader workers、prefetch 1。复用原有 Trainer、完整检测和辅助损失、backward、优化器、数值检查与 EMA；每组从相同的配对初始权重开始。

每种数据模式 6 次预热、24 次测量；计时取每一步六个 rank 中最慢者，再取中位数。Profiler 的另外 3 步不混入无 profiler 计时。每个进程共 63 步，6 组测试均无 AMP 重试。

### 正常读取真实 COCO batch

| P5 架构 | 原版 ms/step | 候选 ms/step | 耗时减少 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| BASE | 441.91 | 334.77 | 24.25% | 1.320x |
| DW | 437.03 | 339.63 | 22.29% | 1.287x |
| BN64 | 435.44 | 336.58 | 22.70% | 1.294x |

按完整 COCO 每轮 309 步外推，纯训练部分约从 134.5–136.6 秒降至 103.4–104.9 秒。不是实测完整 epoch，不包含验证、周期独立评测、保存和初始化；50 轮纯训练约节省 25–28 分钟。不能据此宣称已满足相对 scratch 的 P1 GPU-hours 降低 50% 门槛。

### 复用 GPU batch

| P5 架构 | 原版 ms/step | 候选 ms/step | 耗时减少 |
| --- | ---: | ---: | ---: |
| BASE | 424.17 | 310.27 | 26.85% |
| DW | 421.27 | 313.47 | 25.59% |
| BN64 | 418.29 | 311.89 | 25.44% |

当前脚本通过浅复制 batch 字典避免预处理原地覆盖原始 CPU batch，并断言其仍在 CPU。此前有缺陷的 CPU replay 结果不用于本报告。本次正式对照只使用 live 和 resident 两种模式，二者样本变化不同，不能直接将其差额全部归因于磁盘或传输。

Profiler 中每 3 步的 36 次 `UnsafeIndexBackward` 在候选实现中消失，相关 FP32 索引反向 kernel 的累计 GPU 时间从约 116.0 ms/步降至 0.15 ms/步。该数字是 kernel 累计时间，不与 CPU 同步等待时间重复相加。

rank 0 峰值 allocated 显存约 7.1–7.3 GiB，替换前后相近；这里不是六卡显存总和，也不是 nvidia-smi 的 reserved 显存。

证据：[comparison-summary.json](manifests/p3-deterministic-probe-20260909/comparison-summary.json)、[comparison-launch.json](manifests/p3-deterministic-probe-20260909/comparison-launch.json)、[comparison-status.json](manifests/p3-deterministic-probe-20260909/comparison-status.json)。
完整 traces、operators 和分 rank 耗时保留在服务器各 `BASE/DW/BN64-reference/separable` 子目录。

## 6. 回归与安全边界

在独立 CPU 测试进程中，临时替换 Adapter 构造后的三个 P3 上采样模块，运行已有测试，结果 **91 passed, 5 skipped，13.10 秒**：

- `tests/test_d1_wp3_foundation_adapter.py`
- `tests/test_d1_wp4_foundation_detection_model.py`
- `tests/test_d1_wp6_latent_aux.py`
- `tests/test_d1_p5_ablation.py`
- `tests/test_latent_mixture.py`

GPU 可选用例在该 CPU 回归进程中跳过；本报告另有真实 CUDA 数值验证和六卡性能对照，不将两类结果混淆。
5 个新脚本通过 py_compile；代码基座 `git diff --check` 通过且工作树仍干净。对照任务结束时验证 DW 的 `last.pt`、`resume.pt` 与主仓库实验计划文件 SHA256 均未变化。

证据：[regression.log](manifests/p3-deterministic-probe-20260909/regression.log)。

## 7. 文件与后续接入

外部工作区和本地诊断目录中的新增文件：

- `bilinear2x_probe.py`：无参数候选上采样和局部安装函数。
- `test_bilinear2x_probe.py`：数值、模型单步与确定性微基准。
- `profile_p3_runtime.py`：修正 CPU batch 复用后的六卡测量入口。
- `run_p3_comparison.py`：受控的六组对照执行器，超时只清理自己的进程组。
- `run_p3_regressions.py`：现有回归用例的进程内候选替换入口。

执行器默认拒绝覆盖已有对照启动记录。不要直接重复启动到同一证据目录；重新测试时应使用新的输出目录，并保留代码和配置版本。

建议下一步正式接入无参数上采样模块、增加仓库内测试与显式实验实现版本，再进行小规模训练一致性核对。由于浮点求和顺序不同，不应将替换后的运行标记为与旧训练逐位相同的续跑。本次没有启动新的正式训练，也没有证明最终精度不变。

## 8. 按第 6.5 节开展提速后三架构实验

用户已明确批准本轮三组实验和每 30 分钟一次的状态检查。本轮遵循 [实验计划第 6.5 节](P1_P2_EXPERIMENT_PLAN.md#65-p5-adapter-轻量化对照)，共同启用已验证的 P3 实现，再比较 P5 结构；不混用旧 E1-B 或旧暂停 DW 的成本。

### 8.1 实现与兼容边界

- 正式模块位于 `ultralytics/nn/modules/foundation_adapter.py`，通过 `p3_upsample_mode=separable_bilinear2x` 显式启用。
- 旧配置默认仍为 `bilinear`；旧模型和 checkpoint 的行为不变。新配置和实际模块类型均记录在新运行中。
- 三组 P3/P4 数学定义、通道、Teacher、缓存、LatentMixture、Detect、loss 相同，只保留第 6.5 节的 P5 结构差异。
- 三组重新从配对初始权重开始。旧 DW 第 14 轮检查点及恢复状态保持原样，不用于新运行初始化。

### 8.2 固定安排

| 项目 | 本轮固定值 |
| --- | --- |
| 顺序 | BASE -> DW -> BN64，串行独占六张 A40 |
| 数据 | 完整 COCO2017，train=118287，val=5000；RGB 与 NPY 均使用已验证 NVMe 路径 |
| Teacher / 输入 | 冻结 DINOv3 ViT-S/16，block4/8/12；640；无增强 |
| 预算 | seed0，100 轮余弦调度的前 50 轮；每组 15450 次有效更新 |
| batch | 每卡 64，全局 384，nbs384，无额外梯度累积 |
| 数据加载 | 每 rank workers4、prefetch1 |
| 优化器 | AdamW，lr0=0.001，lrf=0.01，weight_decay=0.0005，warmup3 |
| 精度与随机性 | AMP，初始 scale16，growth_interval=1000000；deterministic=True |
| 融合与 aux | weighted_sum，固定归一化层权重 [1,1,1]；balance0.01、z0.001、gain0.1、budget3.0 |
| 评测 | 每 5 轮独立标准 AP；固定第 50 轮为主结果；最终 last 与 standard-best 均严格重载评测 |

短测推算每组纯训练约 1.44-1.46 小时；加上内部验证、周期标准评测、最终评测与初始化，初步按每组 1.7-2 小时、三组合计约 5-6 小时安排。实际 ETA 按已完成完整 epoch 和剩余评测动态更新，不将短测外推写成实测训练成本。

每组独立保存运行参数、matrix/model/initial SHA256、完整日志、分 rank 时间与显存、loss/aux/Router、checkpoint 和最终标准评测。训练作业及最终评测的 GPU-hours 分别计量，实际作业 GPU-hours 与六卡整段预留时间同时报告。

筛选仍按第 6.5 节：相对本轮 BASE，第 50 轮 AP 下降不超过 0.5 点，AP75、AP_large 各下降不超过 1.0 点，且实际 GPU-hours 更低；AP_small 单列。单 seed 只支持候选筛选，不证明统计等效或 P1/P2 已完成。

### 8.3 入口与检查

路径由运行参数提供；在干净的固定代码提交上执行：

```bash
python -m scripts.d1.run_p5_ablation inspect --p3-upsample-mode separable_bilinear2x
python -m scripts.d1.run_p5_ablation prepare --source-workspace "$E1_WORKSPACE" --workspace "$P5_WORKSPACE" --p3-upsample-mode separable_bilinear2x
python -m scripts.d1.run_p5_suite --workspace "$P5_WORKSPACE" --approved
```

`prepare` 不训练且拒绝覆盖非空目录。suite 拒绝旧启动记录或已有实验状态，依序完成三组训练及最终评测；失败或安全暂停后停止队列，不自动改预算或重开同名实验。

每 30 分钟检查 `status.json`、六 rank progress、CSV、最新标准 AP、AMP 重试、checkpoint、GPU/CPU/存储竞争与 ETA。服务器任务自行运行，Codex 检查依赖本机及 VPN 可用；SSH 失联不代表训练停止，也不能据此重启训练。

新回归入口：`python -m pytest -q tests/test_d1_p3_upsample.py tests/test_d1_p5_ablation.py`。正式队列在接入测试通过、代码提交和实验矩阵登记后启动，具体执行 commit、PID 与开始时间以外部 `suite-launch.json`、`status.json` 为准。

### 8.4 仓库接入验收

2026-09-09，在服务器正式模块接入后执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_d1_*.py tests/test_latent_mixture.py tests/test_master_model_configs.py tests/test_default_config_integrity.py -q --tb=short --color=no
```

结果为 **525 passed, 8 skipped，36.80 秒**。包括 45 组 CUDA 确定性数值检查、三种架构的 loss/backward 与严格 checkpoint 往返、配对初始化、默认兼容行为、模式登记防篡改和三组队列顺序/拒绝覆盖测试。跳过项为未启用的可选集成测试；已有 MoA 配置产生一条 heads 自动调整提示。

六个变更 Python 文件的 Ruff check、py_compile 与 `git diff --check` 通过；新增脚本和测试经过 Ruff format。已有文件的历史格式差异不作无关重排。Ruff/codespell 安装在外部独立工具目录，未更改训练环境依赖。完整接入日志保留在外部工作区 `p3-integration-all-d1.log`。

本轮代码不改写第 1-7 节的诊断测量值；第 50 轮精度及真实训练成本须等待三组新实验完成，不能用旧测量代替。
