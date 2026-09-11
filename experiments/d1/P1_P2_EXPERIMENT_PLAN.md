# D1 P1/P2 后续实验实施方案

制定日期：2026-09-08。代码基线：`c44025eb36e63e4e858306b6e138a9d3c0f3aebd`。

**状态（2026-09-11）：E0/E1/E2 已完成既有工程与阶段验收；E3 两阶段36次训练及固定第60轮官方评分已完成，AUX_STAR锁定为balance=0.1、z=0、gain=0.1、budget=3.0。E4及后续正式实验未启动。** E3只是候选筛选，尚不足以证明aux稳定收益，也不代表P1训练成本目标达成；最新结果见第8.7节。
实际实现、验收、运行入口及结果边界见 [E0/E1 执行记录](E0_E1.md)和 [E2 执行记录](E2.md)。这不代表已经达到 P1/P2 收益目标。

本次按 E1 最终结果更新状态、精度、筛选与成本表，证据见 [E1 完成摘要](manifests/e1-completion-20260908.json)。保留 E1 已登记的固定轮数、评测和可训练参数匹配合同，以及后续多 seed 和成本判据；不改写已完成实验，也不以更新文档授权启动后续训练。

**2026-09-08 参数口径补充：正式 Scratch 基线暂不选定，待最终冻结模型架构确定后再设计。** 当前 E1 仅作“同可训练参数量”的阶段性对照，不等同于已满足课题页的“同参数量”要求。第 4.3 节区分完整系统总参数与可训练参数，第 9 节列出正式对照前的确认门槛；本次不重建基线、不改训练代码或历史结果。

**2026-09-10 E3 方案修订：确定 BN64 为冻结模型基座，aux 消融仅在 VisDrone 开展。** balance/z 的 9 个组合各跑 seed 0、1、2，再对选中的非零组合比较 4 个 gain，每个 gain 同样跑三个 seed；复用 gain0.1 的三次结果后，共 36 个独立有效运行，每次为 300 轮调度的前 60 轮。用户已授权验收通过后启动第一阶段 27 次；第二阶段 9 次待首阶段官方结果与 ETA 汇报后确认。

已有成果见 [双周报告](BIWEEKLY_20260907.md)及其 [Issue #266](https://github.com/Tencent/YOLO-Master/issues/266)。
本方案在现有冻结特征检测闭环上继续实施，不恢复旧实验、不改写历史结果，不用内部阶段编号代替课题 P1/P2 验收。

## 1. 要求与交付对应关系

依据用户提供的《实战课题任务书（细化版，2026-08-22）》D1“目标分级”及课题页：

| 要求 | 本方案落实方式 | 必须提交的证据 |
| --- | --- | --- |
| P1：至少两个数据集的精度、显存、GPU-hours 三维对照 | 完整 COCO 2017、VisDrone2019-DET；每个数据集均设置冻结 ViT-S 与从零训练检测器 | 两个数据集的配方、参数统计、完整训练与独立评测、逐 seed 结果 |
| P1：课题页要求同参数量，细化任务书要求同预算 | 先确定最终冻结架构，再确认匹配总参数还是可训练参数并设计基线；当前 E1 只匹配可训练参数；配对数据、预算、评测和硬件另行保持一致 | 参数口径确认记录、完整系统/冻结/可训练参数审计、正式基线配置、实际更新次数和时间轴 |
| P1：至少一个数据集训练成本降低至少 50% | 分别验证固定轮数和匹配精度下的成本，冷启动成本为主、缓存摊销另列 | 成本分解、精度保留率、达标成本及多 seed 统计 |
| P2：latent aux 注册并证明实际生效 | 复用已修复的统一收集与 scalar-once-v1，补充扫描配置的 loss/梯度验证 | 正向与反向测试、真实 batch 日志、EMA 和预算缩放记录 |
| P2：扫描 latent aux 权重 | 仅 VisDrone：balance/z 的 9 组和随后 4 个 gain 均覆盖三个 seed；按均值选参 | 全部候选三 seed 明细、均值/标准差、负结果、路由/梯度及成本表 |
| P2：扫描底座尺寸 | VisDrone 上比较 DINOv3 ViT-S/16 与 ViT-B/16，配方保持一致 | 权重与缓存合同、维度、冻结/可训练参数、抽取与训练成本 |
| 课题页可选扩展：DINOv3 对比 SigLIP2 | 不作为本轮必跑项；先完成 aux 和尺寸对照 | 如另行开展，新增 Teacher 协议与单独实验身份，不沿用 DINO 缓存 |

P1/P2 是项目内部分级，不替代官方最终评审。任务书允许有证据的负结果：
实验覆盖完成与“精度/成本目标达到”分别标记，不能因跑完实验就宣称降低 50%。

以下数值是本方案预注册判据，不是官方原文阈值：

- 当前 E1 的可训练参数差异绝对值不超过 **1%**；这是阶段性工程约定，不是课题书对参数口径或容差的明文定义。正式对照的口径与容差按第 4.3、9 节确认后锁定。
- “保留足够精度”的工作判据为 **AP 保留率至少 90%**；始终同时报告实际比例和 AP 差值。
- E3 的每个候选使用 **seed 0、1、2**；E4/E5 正式对照也使用 **seed 0、1、2**。E6 保留已约定的独立 seed 4、5、6，不随本次缩减调整，确认运行数不变。
- 筛选实验只支持候选选择；不得将单 seed 的最佳结果当作正式多 seed 结论。

## 2. 已有基础与当前缺口

### 2.1 可以复用的内容

- `DINOv3Teacher(output_layers=(4,8,12))`：ViT-S 三层输出均为 `[B,384,40,40]`。
- `DINOFeaturePyramidAdapter`：九条独立分支；P3/P4/P5 输出通道为 64/128/256。
- `D1FoundationDetectionModel`：Adapter、三个 LatentMixture、YOLO26 Detect、CompositeCriterion。
- `value_fusion_mode` 已支持 `router_only` 和等权 `weighted_sum`；后者不是待新造的模型功能。
- 已完成的 COCO 2017 FP16 NPY 缓存、NVMe 读取、训练/评测、checkpoint 和 AMP 修复。
- 已完成的 VisDrone2019-DET 10 类数据协议、7,019 图 ViT-S NPY 缓存、训练续跑与官方评分闭环；这些是 E2 工程证据，不是正式收敛对照。
- 80 类 ViT-S 下游可训练参数 3,542,567；当前 E1 Scratch 可训练参数 3,510,624，相差 -0.902%。这不包含冻结 DINOv3，不是完整系统总参数量匹配。
- 已发布 100 轮冻结结果用于背景分析；旧 30 轮冻结/Scratch 结果不替代本方案的新对照。

已有代码入口：

- [冻结训练入口](../../scripts/d1/run_wp8_train.py)
- [Scratch 训练入口](../../scripts/d1/run_wp8_p1_control.py)
- [后续实验准备](../../scripts/d1/inspect_wp8_followup.py)
- [当前 A/B/C 准备配置](../../ultralytics/cfg/experiments/d1/wp8-followup.yaml)
- [冻结模型](../../ultralytics/nn/foundation_detection_model.py)
- [辅助损失组合](../../ultralytics/nn/mixture_loss.py)
- [NPY 读取器](../../ultralytics/nn/foundation/npy_cache.py)

### 2.2 不能仅靠修改命令完成的部分

| 历史限制与当前状态 | 已完成适配及后续工作 |
| --- | --- |
| 旧正式入口固定 COCO 样本数、80 类、100 epochs、六卡 batch384 等条件 | 新建版本化 P1/P2 合同和入口，显式支持数据集、模型、seed、训练窗口和评测协议 |
| 旧 Scratch 入口固定 seed0、30 轮窗口、80 类和特定参数量 | 支持完整预算与多 seed，按实际 `nc` 重新构建并校验参数，不移除旧入口的保护 |
| `cache_features.py`、NPY 转换和部分校验存在 ViT-S 形状/COCO 假设 | 将新实验维度和 split 从新合同读取；保留旧缓存格式和旧测试兼容性 |
| `wp8-followup.yaml` 明确仅为准备配置 | 生成可执行的模型配置、完整训练合同与 run manifest，不能把它直接当训练配方 |
| VisDrone 数据、缓存与工程闭环已通过 E2 | 标签转换、ignore 侧记录、ViT-S 缓存和官方评分均可复用；完整训练、多 seed 和正式配对对照仍待后续阶段 |
| 旧 NPY 迁移入口仍保留 `/root` 限制及原有删除源分片行为 | E2 已使用独立非破坏转换路径，在批准的 NVMe 根目录写入并保留源件；未放宽旧入口保护 |
| 旧 `diagnose_wp8.py` 和部分报告偏向 COCO | E2 新入口已分派 VisDrone 推理导出与官方评测，保留原始图片 ID；后续正式实验仍需绑定各自合同 |

E0 已新增独立的 COCO seed0 四组入口、NVMe 数据准备、恢复和评测，未移除旧入口保护。
E2 已补齐 VisDrone 数据、非破坏缓存转换和短训练评测闭环；多 seed、其他 Teacher 与正式对照仍待对应阶段实施。
当前可运行命令以 [E0/E1 执行记录](E0_E1.md)和 [E2 执行记录](E2.md)为准，不应把后续接口设想当成已经实现。

## 3. 实验执行顺序

| 顺序 | 工作 | 输出与继续条件 |
| --- | --- | --- |
| E0 | 合同、数据接口、日志和训练入口准备 | 小样本闭环、配置拒绝规则、参数匹配和恢复测试通过 |
| E1 | 完整 COCO 的 A/B/C + Scratch，seed0，执行完整 100 轮预算的前 50%（50 轮） | 判断空间融合及 aux 开关作用，选择固定融合方式 |
| E2 | VisDrone 数据协议、评测和 ViT-S 缓存；短基准 | 10 类小样本闭环、官方评测对齐、资源与 ETA 记录 |
| E3 | 仅 VisDrone：BN64，9 组 balance/z 各三 seed，再比较 4 个 gain 各三 seed | 27 + 9 = 36 个独立运行，按三 seed 平均 AP 锁定 aux；不新增 COCO aux 消融 |
| E4 | 最终冻结架构与正式基线确认后的两数据集多 seed 对照 | 先通过第 9 节基线门槛；每个数据集暂拟 6 个完整运行，形成 P1 三维表 |
| E5 | VisDrone ViT-B 与 ViT-S 尺寸对照 | ViT-B 新缓存和 3 个完整运行，形成 P2 尺寸表 |
| E6 | 汇总成本曲线；有希望的一个数据集做匹配精度停止确认 | 判断是否真正达到 50%，不足则明确报告 |
| E7 | 固定配方后评测 VisDrone test-dev，归档结果及复现说明 | test-dev 只作最终保留集，不反向调参 |

E2 的工程准备可先于 E1 的全部训练结束，但不能并行占用正式对照的 GPU 或制造磁盘竞争。
每个长任务单独报告预算并获得启动确认；上一任务完成不自动授权整个矩阵。

### 3.1 根据 E1 阶段结果调整投入优先级

1. **E1 已完成并关闭证据链。** 四组统一按第 50 轮标准 AP、15,450 次有效更新及实际成本比较，标准 best 仅作补充；不重复启动已完成任务。
2. **后续以 weighted_sum 为主方向。** A 保留为机制对照，不再默认增加 router_only 的大规模训练；B/C 均保留为 aux 候选，不将 seed0 的小幅 AP 差定为正式胜负。
3. **先完成 E2，再分批开展 E3。** VisDrone 的协议、NVMe 数据/缓存、小样本闭环通过后，完成每个候选三 seed 的两阶段 aux 扫描，锁定 AUX_STAR；不在 COCO 新增 aux 扫描或开关复核。
4. **最终冻结架构和正式基线确认后，再满足门禁进入 E4。** 参数口径、基线结构及预算暂不在本次选定；形成两数据集、三 seed 的正式配对结果前，须通过第 9 节确认。如果触发第 6 节的 80% 暂停线，仍先执行第 8 节的解除条件，不自动展开 COCO 长训练。
5. **重点分析达到共同 AP 的成本。** 用第 11 节的同目标、连续达标规则判断早期收敛能否转化为成本收益；E5 的 VisDrone 底座尺寸对照仍保留，完整 COCO 的更大 Teacher 缓存不在本轮默认范围内。

结构改进仅作为第 6.4 节的条件性候选：先有精度缺口和诊断证据，再确认小预算实验；不同时改变 Teacher、Adapter、融合和检测头，不挤占当前 S 的训练资源。

## 4. 统一实验合同

### 4.1 数据、模型和随机性

- COCO 使用官方 train2017 118,287 张、val2017 5,000 张，80 类。
- VisDrone 使用官方 train 6,471 张、val 548 张，10 类；test-dev 1,610 张不参与筛选。
- 不引入 COCO-mini，不二次随机划分完整 COCO；32 图仅用于工程测试。
- 后续 RGB 训练和独立评测统一使用本地 NVMe 上的图片、标签及标注副本；Frozen 的特征缓存也使用本地 NVMe，不将存储差异混入模型对照。
- 输入固定 640×640；复用原 LetterBox、RGB 和 DINO 归一化合同，不启用二次缩放/裁剪。
- 关闭颜色、翻转、几何、mosaic、mixup、copy-paste、cutmix、erasing 和多尺度训练。
- Frozen 从头初始化下游；Scratch 全网随机初始化；不加载旧训练 checkpoint 或预训练检测器。
- Teacher 始终冻结、eval、inference mode，不进入 student optimizer、EMA 或 checkpoint。
- 每个配对 seed 的初始化规则、采样顺序、DDP 分区和尾 batch 规则固定并记录。
- A/B/C 参数形状相同，使用同一份初始 state dict，记录其 SHA256；不同网络只要求各自初始化可复现。
- 保留所有样本；DDP 尾部补齐产生的重复样本数量必须记录，不静默 drop_last。

### 4.2 训练配方

| 参数 | COCO | VisDrone | 说明 |
| --- | --- | --- | --- |
| 完整训练上限 | 100 epochs | 300 epochs | 本方案选择，不声称来自官方最优配方 |
| 筛选窗口 | E1：前 50/100 epochs（50%） | E3：前 60/300 epochs | 保留完整调度的前缀，不压缩余弦周期 |
| GPU | 6×A40 | 6×A40 | 每个运行独占同一组卡，组间串行 |
| 数据存储 | RGB 与特征缓存均为本地 NVMe | 同左 | 训练/验证路径均需校验；原始数据保留，不回退到慢盘 |
| 每卡/global batch | 64 / 384 | 16 / 96 | VisDrone 较小，避免每轮仅约 17 次更新 |
| nbs / 累积 | 384 / 1 | 96 / 1 | 每个有效 batch 更新一次，包括 warmup |
| workers / prefetch | 每 rank 4 / 1 | 每 rank 4 / 1 | Frozen/Scratch 相同；性能偏离先做共同基准 |
| optimizer | AdamW | AdamW | lr0=0.001，lrf=0.01，beta1=0.9 |
| weight decay | 0.0005 | 0.0005 | 同时记录实际缩放后的值及 beta2 |
| schedule | cosine，warmup 3 epochs | cosine，warmup 3 epochs | 记录实际 warmup 更新数；禁止隐式累积 |
| AMP | 开启 | 开启 | 初始 scale16，growth_interval=1,000,000 |
| loss | box7.5 / cls0.5 / dfl1.5 | 相同 | Detect 均为 end2end=True、reg_max=1 |
| patience | 不提前停止 | 不提前停止 | 固定预算表跑满；E6 的停止规则单独定义 |
| 内置验证 | 每 epoch | 每 epoch | 训练监控，不混同于独立标准 AP |
| checkpoint | last/best，每 5 epochs 保留一份 | 相同 | 断点恢复与成本曲线使用 |
| 标准评测 | 每 5 epochs + 最终 last/best | 相同 | 在同一时点执行，成本计入，两组同规则 |

VisDrone 的 batch96 对应每 rank 1,079 个采样位置、约 68 次更新/epoch，
300 轮约 20,400 次更新；实际值由 DataLoader 记录，不以估算代替证据。
固定 300 轮是为了给小数据集足够更新，不保证收敛或最优。
如短基准发现 OOM 或明显不合理的吞吐，先修订并提交该数据集的配对合同，两组重新开始；
不得只调整其中一组的 batch、workers、GPU 数或学习率。

`scalar-once-v1` 在新 batch/卡数下必须重新核验实际检测/aux 梯度比例。
aux 只在 VisDrone 按第 8 节选参；COCO 不新增 aux 消融或开关复核。E4 仍使用锁定的 AUX_STAR 做两数据集 Frozen/Scratch 对照；该系统对照不能单独证明 aux 在 COCO 上有收益。

### 4.3 参数口径与正式基线待定

#### 4.3.1 完整系统与可训练部分

课题页原文为“与同参数量从零训练检测器做精度/显存/时长对照”，没有明确限定总参数量还是可训练参数量。
此前直接采用“同可训练参数量”是本项目的实验设计选择，不能当作已经确认的课题要求；正式报告前需要与导师/课题负责人确认口径并记录依据。

| 统计项 | Frozen 方案 | Scratch 方案 | 报告边界 |
| --- | --- | --- | --- |
| 完整系统总参数 | 冻结 DINOv3 + 最终 Adapter/融合模块/检测头的全部参数 | 从 RGB 到检测输出的完整模型参数 | 总参数匹配时必须计入冻结 DINOv3，不能因缓存或 checkpoint 隔离而排除 |
| 可训练参数 | 实际参与学习的下游参数 | 实际参与学习的完整检测器参数 | 当前 E1 匹配的是这一项，不代表总参数匹配 |
| 冻结参数 | 包括冻结 DINOv3，按实际模型审计 | 按实际冻结设置审计；当前全网从零训练 | 总参数、可训练参数和冻结参数分栏，按唯一参数对象计数，不把 buffer 混入参数 |
| 前向/训练计算量 | 分开记录 RGB 端到端路径与缓存输入的下游训练路径 | 记录 RGB 前向和训练路径 | 参数元素数不是权重文件字节数或 FLOPs；同参数不等于同计算量，也不自动等于同架构 |

离线缓存只把 Teacher 前向提前执行，不使其从完整检测系统中消失。新 RGB 图像的端到端推理仍需 Teacher；
冷启动成本仍含特征抽取，缓存复用只影响摊销成本，不改变完整系统总参数量。
当前 Scratch 是小型 CNN，不是在从零训练 DINOv3；不能用“省去了 DINOv3 反向”直接推导其相对当前 Scratch 必然节省 50%。

#### 4.3.2 当前 E1 的阶段性参照

E1 Scratch 保留 [当前结构](../../ultralytics/cfg/models/26/yolo26-d1-scratch-matched-n.yaml)：
Conv/C3k2 主干、SPPF/C2PSA、上采样/拼接和自底向上 neck、YOLO26 Detect；没有 Teacher、缓存输入或 LatentMixture。
现有 E1 训练、指标、成本和配置继续有效，统一标为“同可训练参数量的阶段性对照”，不因本次口径澄清重跑或改写历史。

E1 已登记的计算方式保留为：

```text
parameter_delta = (P_scratch_trainable - P_frozen_downstream_trainable)
                  / P_frozen_downstream_trainable
require abs(parameter_delta) <= 0.01
```

该公式及 1% 容差不自动成为 E4 的正式总参数匹配规则。COCO 当前 3,542,567 对 3,510,624 的数字只用于 E1；
VisDrone 的 10 类模型和任何新架构均须重新统计，不能沿用这组数字。

#### 4.3.3 正式基线的延后决策

按照本次用户决定，先完成冻结方案的架构研究，待 Teacher 规格、Adapter、LatentMixture/融合、Neck（如有）和 Detect 全部锁定后，再设计正式对照基线。
到时确认总参数匹配还是可训练参数匹配、是否另需同架构参照以及容差；本次不预选更大的 CNN、随机初始化 DINO 或其他正式基线。
若要求总参数匹配，目标必须包含冻结 Teacher；若确认匹配可训练参数，则明确命名并同时披露完整系统总参数，不隐去预训练底座容量。
现有 E1 可作为机制筛选或补充对照，但不自动替代最终 P1 主表；新基线启用前补齐第 9 节审计、合同和预算确认。

仅使用真实参与计算的层做合法匹配，不添加闲置参数凑数，不为了得到 50% 降本结论而挑选低效或精度弱的基线。
候选规则在正式训练前预注册，不依据候选精度或成本事后挑选有利参照。新基线通过工程测试后再提交配置并开展正式实验。

### 4.4 指标范围、单位与读取顺序

本节按当前 E0/E1 的实际实现解释指标，核对基线为 `9347a8b9541790c980d5371ae9cf79630ecc606d`；
仅补充评价口径，不修改正在运行的模型、训练配方或筛选门槛。
“已记录”指当前运行产物确实保存的字段；“派生”需要从同口径记录计算；“待补齐”不能当成现有成果。

| 类别 | 当前有哪些 | 用途与边界 |
| --- | --- | --- |
| 独立 COCO 精度 | 6 项：AP、AP50、AP75、APsmall、APmedium、APlarge | 模型效果的主要证据，每 5 轮及最终 checkpoint 评测 |
| 逐轮精度监控 | 4 项：Precision、Recall、内置 mAP50、内置 mAP50-95 | 判断学习趋势；后两项与标准 AP 名称相近，但实现口径不同，不混表 |
| loss 监控 | Frozen 为 7 项训练 loss、7 项验证 loss；Scratch 为 3 项检测 loss 及对应验证项 | 解释优化过程，不能代替标准 AP；验证 aux 的 4 项为 0 |
| 机制与稳定性 | Router 概率/熵、balance/z、残差增益、特征幅值、梯度范数、AMP 与有效更新数 | 回答融合和 aux 是否、如何起作用，不是一组精度分数 |
| 资源与成本 | 参数量、显存、分段耗时、数据等待；由记录派生吞吐和 GPU-hours | P1 的成本对照；冷启动、摊销、磁盘/主存等完整账本仍需 E4/E6 补齐 |
| 配对与统计 | AP 保留率、AP 差、成本节省率、达标时间、均值/标准差/置信区间 | E1 只做单 seed 筛选，不提前宣称正式多 seed 收益达标 |

当前 A 组 `results.csv` 实际有 **25 列**：epoch/time 2 列、训练 loss 7 列、精度 4 列、验证 loss 7 列、学习率 5 列。
这不是“25 个独立评价指标”：进度和优化诊断也在同一文件里，6 项独立 COCO AP 则保存在评测 JSON 中。
下文 `AP` 未加“内置”限定时，均指对应数据集的独立标准评测结果。

- AP、Precision、Recall 统一保存为 `0..1`；展示时可乘 100。例如 `AP=0.25` 是 25.0 AP 点，不是“25% 的图片全部检测正确”。
- AP 从 0.25 到 0.27 是增加 **2.0 AP 点**，相对增加 8%；二者不可混称。
- 秒为计时基本单位；`GPUh` 为 GPU-hours；`MiB=2^20 bytes`、`GiB=2^30 bytes`，不与十进制 GB 混用。
- 未实现、缺失或无有效分母的指标写 `N/A`/未知并说明原因，不填 0。评测器的无有效 GT 占位值不当成实际负精度。

### 4.5 检测精度：检测对不对、漏不漏、框得准不准

**基础概念。** `IoU = 预测框与真值框的交集面积 / 并集面积`，范围 0..1，越大表示框越重合。
在固定类别、IoU 门槛和置信度门槛下，正确匹配为 TP，错误类别/未匹配或重复预测通常为 FP，未被检出的有效真值为 FN；
ignore/crowd 等特殊目标交给标准评测器处理，不能机械套用普通一对一计数。

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

例如有 100 个有效目标，预测 80 个框，其中 60 个正确、20 个误检，则 Precision=75%、Recall=60%。
提高置信度门槛通常会减少误检但增加漏检，所以一个阈值下的 P/R 不能完整反映模型质量。
AP 对某类目标按置信度排序，汇总不同召回率下的插值 Precision；mAP 再对有效类别平均。
COCO 通常也把这个跨类别平均结果简称为 AP。它不是最后一个 batch 的准确率，也不是所有类别 TP 汇总后的一个 Precision。

| 指标 / 实际字段 | 含义 | 如何使用 |
| --- | --- | --- |
| AP / `official.AP_all` | 对 IoU=0.50、0.55、…、0.95 共 10 个门槛以及有效类别平均，综合分类、检出和定位能力 | **主指标，越高越好**；E1 固定第 50 轮选组，E4 用固定末轮比较 |
| AP50 / `official.AP_50` | 只要求 IoU≥0.50 的 AP，定位要求相对宽松 | 判断是否大致找到目标；高 AP50 但低 AP75 常提示精细定位不足，不能仅据此确定原因 |
| AP75 / `official.AP_75` | 只要求 IoU≥0.75 的 AP，框位置和大小要求更严格 | 检查定位质量，通常不高于 AP50 |
| APsmall / `official.AP_small` | 标准 small 面积范围的 AP，仍跨 10 个 IoU 门槛 | 检查小目标能力，与高分辨率特征是否有帮助相关 |
| APmedium / `official.AP_medium` | 标准 medium 面积范围的 AP | 检查中等目标能力 |
| APlarge / `official.AP_large` | 标准 large 面积范围的 AP | 检查大目标能力；不能只改善大目标就宣称所有尺度均改善 |
| Precision / `metrics/precision(B)` | 内置 Validator 的类别平均查准率，即报告阈值下预测有多可靠 | 越高通常误检越少，须同时看 Recall；`(B)` 表示 bounding boxes，不是 B 实验组 |
| Recall / `metrics/recall(B)` | 内置 Validator 的类别平均查全率，即报告阈值下找回多少有效目标 | 越高通常漏检越少，须同时看 Precision |
| 内置 mAP50 / `metrics/mAP50(B)` | 内置匹配与积分实现得到的 IoU=0.50 类别平均 AP | 每轮监控，不直接替代 `official.AP_50` |
| 内置 mAP50-95 / `metrics/mAP50-95(B)` | 内置实现的 10 个 IoU 门槛平均 AP | 每轮监控，并作为当前内部 `fitness`；不直接替代 `official.AP_all` |

面积按原始标注/标准工具的 `area` 计算，不按 640 LetterBox 后的框重新分桶，也不是 P3/P4/P5 的名称映射。
当前 COCO 工具范围是 small `[0,32^2]`、medium `[32^2,96^2]`、large `[96^2,1e10]` 像素平方；
精确边界处理沿用工具，不能自行改成互斥分桶后仍声称完全相同的标准 AP。
COCO 参数定义可核对 [官方评测实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。

**必须区分的两套精度口径：**

1. 当前独立评测使用 `faster-coco-eval==1.8.0`、`iouType=bbox`、完整 val2017 的 5,000 图及官方标注，保留类别语义；
   IoU 为 0.50:0.05:0.95，召回率采样为 0:0.01:1，`maxDets=[1,10,100]`，上述 6 项 AP 使用 100 档。
   预测导出的 `conf=0.001, max_det=300` 是输入候选截断，**不等于标准评测改成 maxDets=300**；标准工具继续按其规则截断。
2. 内置 P/R 在 IoU=0.50 下，选取平滑后的类别平均 F1 曲线最高点对应的共同置信度，再分别平均各类 P/R；
   不是固定 `conf=0.25`，也不是直接在 `conf=0.001` 处读取。不同 epoch 的所选阈值可能不同。
   内置匹配、AP 积分和有效标注/ignore 处理与独立标准评测并非完全相同，因此同一个 checkpoint 两套数值可能有差异。
3. F1 是 P/R 的调和平均，用来解释二者平衡；当前不单独保存 F1 标量列。
   对类别平均后的 P/R 再算 F1，不一定等于逐类 F1 的平均。`fitness` 当前等于内置 mAP50-95，不是额外的新评价维度。
4. `best.pt` 按逐轮内部 fitness 选取；`standard-best.pt` 按每 5 轮独立 AP 选取，并列保留更早者；
   `last.pt` 是当前末轮。报告必须带 checkpoint epoch 和 SHA256；“标准 best”不代表所有 epoch 都做过标准评测。
   E1 仍按固定第 50 轮 AP 筛选，不能用某组 best 对另一组 last。

类别 AP 可定位具体类别的失败，但当前 E1 小型独立报告只保存上述 6 项聚合 AP；完整类别表属于 E4 待补齐输出。
VisDrone 的 AP/AP50/AP75 及 AR@1/10/100/500 应按第 7.2 节锁定的官方工具实现：
AR 是在给定预测数上限下对 IoU 门槛/类别平均的最大召回率，不是内置 Recall 的别名。
该工具 AP 默认使用 maxDets=500；其输出为百分制，后续适配器须除以 100 转成统一 `0..1` 口径，并保留原值和工具版本。
E2 已完成短训练 checkpoint 的 VisDrone 官方评分，结果见第 7.4 节；正式收敛实验尚未开展，COCO AR 也尚未持久化到 E1 摘要，不能将后两项写成已有结果。
VisDrone 指标与 ignore 语义以 [官方检测工具](https://github.com/VisDrone/VisDrone2018-DET-toolkit)为准，不与 COCO 数值混表。

### 4.6 训练与验证 loss：模型正在优化什么

loss 衡量当前优化目标的误差，通常希望训练中下降，但它与 AP 不是同一个量：loss 降低不保证验证 AP 提高。
以下为 Frozen 的 7 个实际字段；CSV 中分别加 `train/` 或 `val/` 前缀。
三个检测项已经乘过 `box=7.5、cls=0.5、dfl=1.5`，不能读取后再次乘这些系数。

| 字段 | 当前代码中的实际含义 | 正确解读 |
| --- | --- | --- |
| `box_loss` | 按分配目标质量加权并归一化的 `1-CIoU` 定位损失；CIoU 同时考虑重叠、中心距离和长宽比 | 越低通常定位误差越小；不是 `1-AP`，也不只是 `1-IoU` |
| `cls_loss` | 分类 logits 与分配器生成的目标分数之间的 BCEWithLogitsLoss，按目标分数总和归一化 | 衡量类别/置信度预测误差；当前没有另列 `obj_loss`，不能凭旧版 YOLO 习惯补造一项 |
| `dfl_loss` | **当前 reg_max=1，实际是归一化 left/top/right/bottom 距离的加权 L1 回归损失**，不是离散分布 DFL | 字段沿用历史名称，值非零正常；不能称为“DFL 已关闭所以此列应为 0” |
| `latent_balance_loss` | P3/P4/P5 三个 LatentMixture 的未乘系数 balance 项之和 | 检查专家平均概率是否失衡；越低越均匀，不代表检测精度必然越好 |
| `latent_z_loss` | 三个尺度的未乘系数 Router z-loss 之和，即每尺度对 `logsumexp(logits)^2` 求均值 | 约束路由 logits 的数值尺度，不是框回归，也不是某个方向的坐标损失 |
| `latent_aux_loss` | 三尺度 `c_balance * balance + c_z * z_loss` 的总和，已乘局部系数，但未做全局 gain/EMA/预算缩放 | 当前默认等于 `0.01 * latent_balance_loss + 0.001 * latent_z_loss`；raw 指全局归一化之前，不是完全未加权 |
| `mixture_aux_loss` | 经过 EMA 归一化、`latent_aux_gain` 和全局预算缩放后，实际加入模型 loss 的 scalar aux | 这是有效辅助项；不能再加上前面三个诊断项，否则重复计数 |

单尺度的 balance 公式为 `E * sum_e(mean(p_e)^2) - 1`，并截断到非负数；`E=4`，均匀使用专家时为 0。
balance 启用时按当前 DDP 实现同步平均概率；z-loss 为本地 batch 的路由统计，二者不能假设都是同一种全局归约。
设三尺度加权和为 `raw_latent`，D1 当前仅启用 latent routed loss 时：

```text
normalized_aux = latent_aux_gain * raw_latent / clamped_EMA_latent
budget_scale   = min(1, mixture_aux_budget / max(abs(normalized_aux), 1e-4))
effective_aux  = normalized_aux * budget_scale
L_native      = o2m_weight * sum(L_one2many) + o2o_weight * sum(L_one2one)
L_total       = L_native + effective_aux
```

EMA 是辅助损失幅值的指数移动平均，用于归一化；它不是用于评测的模型权重 EMA。
当前 aux EMA 衰减为 0.99，latent 初值为 0.1，预算为 3.0；分母和缩放系数不建立额外梯度路径。
`scalar-once-v1` 保证 aux 对原生 loss 向量求和时只加入一次；原生检测项保留本地 batch 缩放，Trainer 还会作 DDP 乘数和 AMP loss scaling。
因此改变 batch 可能改变检测与 aux 的相对作用，不能只比较配置中的 0.1。

**不能把 CSV 七列相加当作实际反传 loss。** 当前 `E2ELoss` 同时训练 one-to-many 与 one-to-one 分支，权重随调度变化；
CSV 的三个检测 loss 只返回 one-to-one 的报告项，不是两分支加权后的全部原生 loss，也未包含上述 batch/DDP 缩放。
判断 aux 强弱时，用同一真实 batch 的 `applied_aux / abs(native_loss)` 以及第 4.7 节的梯度比例；
分母为 0 时标记无定义，不能用不同日志列或不同 batch 的数值硬算。

- `train/*` 是 rank0 对本轮各 batch 报告项的算术平均，不是六卡全部目标逐个加权的全局 loss；跨组比较须保持 batch、归一化及报告方式一致。
- `val/*` 是验证集损失，用于观察泛化趋势，但训练/验证模式和数据不同，不要求两个数值相等。
- 验证时 LatentMixture 不发布训练 aux，四个 `val/latent_*`、`val/mixture_aux_loss` 字段为 0；不能据此断言训练未启用 aux。
- C 组 gain=0 时有效 `mixture_aux_loss=0`，raw balance/z/aux 仍可能非零；Scratch 没有这些模块，该项记不适用。

课题页“latent aux 仍需显式加入 collect_aux_loss 的 include_kinds”指损失收集白名单：
[collect_aux_loss](../../ultralytics/nn/modules/routing_protocol.py) 的默认种类为 `moe/moa/mot/molora`，不含 `latent`；
LatentMixture 发布 `kind="latent"` 的辅助损失后，训练损失调用方必须显式允许该种类，否则该调用会跳过它。
当前 [统一 mixture 收集器](../../ultralytics/nn/mixture_loss.py) 已传入 `include_kinds=("moe", "moa", "mot", "molora", "latent")`，
由 CompositeCriterion 完成 EMA 归一化、gain/预算控制和总损失单次加入。A/B 的 gain=0.1 启用该项，C 的 gain=0 是有意关闭其有效贡献，
不是漏注册，也不会关闭 LatentMixture 检测路径。无需为解释这句话再次修改训练代码。
- 长期训练 loss 下降而验证 AP 停滞是排查线索，不足以单凭一条曲线确定过拟合或宣布某模块失效。

实现依据：[原生检测 loss](../../ultralytics/utils/loss.py)、[D1 报告字段](../../ultralytics/nn/foundation_detection_model.py)、
[统一辅助损失组合](../../ultralytics/nn/mixture_loss.py)。

### 4.7 路由、梯度与数值稳定性：辅助机制是否真正生效

这些是 P2 的机制证据，不能替代 AP/成本对照，也不是一律越大越好或越小越好。

| 字段 / 派生指标 | 定义与用途 | 当前采集边界 |
| --- | --- | --- |
| `mean_router_probs`、`expert_usage` | 每个专家的平均路由概率，总和约为 1；判断是否集中在少数专家 | 当前 `expert_usage` 就是平均软概率，不是实际被派发样本数 |
| `entropy` | `mean(-sum_e p_e * ln(p_e))`，4 专家时范围约 0..ln(4)=1.386；低表示单样本偏好更集中，高表示更均匀 | 高熵不等于高 AP；平均负载均匀也不意味着每个样本都是均匀路由 |
| `executed_experts`、`mean_active_experts_per_sample`、`batch_expert_union`、`kernel_calls` | 模块报告的专家执行数/覆盖数，用来核对 dense 或 sparse 语义 | 当前训练和评测均执行全部 4 专家；`kernel_calls` 是模块级计数，不是 profiler 测得的全部 CUDA kernel 数 |
| `mean_router_logits`、`temperature`、`noise_std` | 路由打分均值、softmax 温度和路由噪声幅值，解释概率分布及随机性 | logits 大小本身不是准确率；与 z-loss、entropy 一起看 |
| `residual_gain`、`residual_gain_magnitude` | 可训练残差缩放系数及其最大绝对值，决定专家残差对输出的作用幅度 | 接近 0 提示残差贡献可能小，须结合残差特征和梯度；不是“百分之多少信息来自专家” |
| `router_output_head_magnitude`、`identity_cold_start` | Router 输出头参数最大绝对值，以及增益/输出头均为零的冷启动标志 | 检查路由是否离开初始状态，不单独作为收益结论 |
| 特征 `rms`、`abs_max` | `sqrt(mean(x^2))` 与 `max(abs(x))`，分别衡量整体幅值和极端值 | 对比九条 Adapter 候选与融合后 P3/P4/P5，寻找信号衰减、爆炸或层间失衡 |
| 梯度 `detection`、`aux`、`total` | 对同一参数分别求 `L_native`、`L_aux`、`L_total` 的梯度 L2 范数 | 当前探针覆盖 Adapter 分支、Router、residual gain；零梯度须结合计算路径/初始化解释 |
| 梯度比例（派生） | `norm(grad L_aux) / norm(grad L_native)`，解释 aux 相对检测监督的强度 | 同参数、同 batch、同精度；分母为零单独标记，范数不能揭示两者是否方向相反 |
| 梯度可加性检查 | 逐元素检查 `g_detection + g_aux` 与 `g_total` 满足 `abs(delta) <= 2e-5 + 2e-4 * abs(g_total)` | 工程门槛，不是 AP；独立 FP32 探针关闭 autocast/TF32，不改训练模型/RNG/EMA |
| `latent_publications`、`aux_step`、`aux_ema` | 当前步三尺度 aux 发布数、所属步号及归一化历史状态 | 发布数应为 3；缺失、陈旧、重复、非有限值不能当正常正则结果 |
| `amp_scale`、`amp_retries` | 梯度缩放倍率，以及为完成有效更新而发生的 AMP 重试次数 | scale16 不是精度/速度分数；重试不等于新增有效更新，不允许静默跳过样本 |
| `optimizer_steps`、`batches`、finite/NaN/Inf | 有效参数更新数、已消费 batch 数及数值有限性 | E1 每轮 309 次、50 轮 15,450 次有效更新，所有 rank 一致；失败则不作为有效配对结果 |
| `lr/pg0` 至 `lr/pg4` | 当前 Frozen 的 5 个 optimizer 参数组实际学习率 | 反映 warmup、调度与组别倍率，不是 5 组实验；含义以 optimizer 参数归属为准 |

当前逐 rank 的 epoch 路由 JSON 是**本轮最后一个训练 batch 的快照**，不是全轮概率/熵平均；
独立梯度/幅值探针在第 1、25、50 轮开始时，各 rank 取两张真实样本，不是每个训练 batch 的统计，也不是这些轮完成后的全验证集结果。
`aux_zero/balance_only/z_only` 是探针副本上的诊断用例，不代表额外完成了三次训练。
E3 要求的逐轮全局统计、显式预算缩放/梯度比例时间序列仍需补齐，不能把稀疏快照冒充完整曲线。
当前没有启用 sparse 推理，`inference_calibration_*` 等能力字段不代表已做过稀疏加速对照。
实现依据：[LatentMixture](../../ultralytics/nn/modules/latent_mixture.py)、[E1 运行与诊断](../../scripts/d1/p1p2_runtime.py)。

### 4.8 参数、显存、吞吐与时间：省在哪里

| 指标 | 定义与单位 | 对照时的约束 |
| --- | --- | --- |
| 可训练参数 / `parameters` | 实际参与学习且 `requires_grad=True` 的参数元素个数 | 当前 E1 COCO 为 3,542,567 对 3,510,624，仅匹配下游可训练参数；E4 主匹配口径待第 4.3、9 节确认 |
| 冻结参数、总参数、FLOPs | Teacher 冻结参数单列；系统总参数包含 Teacher；FLOPs 为指定输入下计算量 | `teacher_parameters=0` 只证明训练模型隔离，不代表整个方法不依赖 Teacher；FLOPs 要说明 MAC 计数、输入/分支及未覆盖算子，不用参数量估算 |
| `peak_allocated_bytes` | PyTorch 活跃张量的峰值显存，换算 GiB | 每 rank 在 epoch 开始重置、训练结束读取；训练峰值取所有 epoch、rank 的最大值，不把六卡相加作为单卡需求 |
| `peak_reserved_bytes` | PyTorch allocator 向 CUDA 保留的峰值内存，包括可复用的分配器缓存 | 不与 allocated 相加；reserved 较大不等于都被活跃张量使用 |
| `sampled_device_peak`、`teacher_peak_bytes` | 设备采样峰值、Teacher 抽取阶段峰值 | 完整三维表待补齐；设备采样受 CUDA 上下文/其他进程影响。当前训练峰值不覆盖 epoch 后验证或独立探针，不能冒充全流程峰值 |
| GPU 利用率 | 设备采样周期内 GPU 执行任务的活跃程度百分比，用于判断是否持续有工作 | 不是显存占用率，也不是达到理论算力的百分比；状态快照不等于整次训练平均值，完整时间序列待补齐 |
| `step_seconds` | 一轮各 batch 开始/结束回调间墙钟累计，包括区间内预处理、前反向及更新等 | 不是纯 GPU kernel 时间；DDP 通信/同步等待可能包含其中 |
| `data_wait_seconds` | 上个 batch 结束至下个 batch 开始的时间累计 | 反映主线程可见的数据等待，含调度等开销；不是磁盘服务时间，也不包含被计算隐藏的全部预取开销 |
| 数据等待占比（派生） | `wait / (wait + step)`，按单 rank 同一轮计算 | 越低通常阻塞越少；不能把六卡秒数相加后当成墙钟时间 |
| 吞吐 images/s（派生） | 同一测量区间内实际处理样本数 / 墙钟秒数 | 明确是否含 DDP 尾部补齐，同时报有效唯一图片数；不固定用 batch384 乘所有 batch 冒充准确样本数 |
| `epoch_wall_seconds` | 当前回调测量的本轮训练、内置验证、保存恢复状态等耗时 | **不含本轮随后启动的每 5 轮独立评测**；此前独立探针也不在此计时区间 |
| `seconds` / `train_val_seconds` / `final_eval_seconds` | 各作业或阶段墙钟时间，秒；必须带 scope | E1 评测 `report.json.seconds` 不含所有启动/加载；外层评测 `cost.json.seconds` 更完整，不随意互换 |
| GPU-hours / `GPUh` | `sum(实际占用 GPU 数 * 阶段秒数 / 3600)` | 6 卡占用 1 小时为 6 GPUh，即使有等数据/评测等待；不能按利用率打折 |
| 磁盘读写吞吐、`storage_bytes` | 同期实际磁盘 bytes/s、缓存/权重/预测等占用字节数 | 用磁盘采样和文件实测；逻辑读文件量不等于物理读盘量，页缓存命中会改变差异；缓存热读不能代替训练吞吐 |
| CPU/容器主存及页缓存 | CPU 使用、cgroup 上限/占用、匿名内存、文件缓存与 shared memory 等 | 辅助资源指标，当前未完整逐 run 采集；容器限额不是宿主机 `free` 总内存，shmem 可能已包含在 cache 中，不能重复相加 |
| ETA | 剩余训练轮数、评测、收尾及排队组别的实测耗时外推 | 是排期估计，不是已发生训练成本，需要随当前组真实速度更新 |

CSV 的 `time` 是本次 Trainer 进程启动后的累计秒数，跨进程 resume 可能重置；不能直接拿末行作为整个实验总成本。
当前 E1 `cost.json` 汇总已记录训练尝试，包含训练中等待独立评测的时间，不含停机间隔和最终独立评测；
每 5 轮评测是其子区间，不能再加一次。最终评测单独计费并以实际保留 GPU 数为准。
尝试/重跑开销保持可追踪，正式方法成本和总研发成本按第 11 节分栏，不静默删除，也不重复计费。

```text
H_cold         = H_extract + H_train_val + H_eval
H_amortized(K) = H_extract / K + H_train_val + H_eval
```

`H_extract` 是 train/val Teacher 特征抽取及该阶段 GPU 校验成本；`H_eval` 只计未包含在训练账本中的独立评测。
`K` 是实际完成且确实复用同一缓存的运行数，不能用计划未来跑很多次来压低摊销成本。
CPU 转换/复制不虚构 GPUh，但计入端到端墙钟及 CPU/RAM/存储资源；已有缓存不等于冷启动抽取成本为零。
冻结下游的训练速度、验证速度不等于原始 RGB 到最终检测的在线推理速度；没有把 Teacher、预处理和后处理一起测量，就不报告端到端 FPS。

### 4.9 P1/P2 收益、统计与有效性判据

以下必须在同数据集、同 seed、同预算及同一独立评测协议下计算，未知分母不产生结论。

| 指标 | 计算与含义 | 判读 |
| --- | --- | --- |
| 参数差异 `parameter_delta` | `(P_scratch_trainable - P_frozen_trainable) / P_frozen_trainable` | 保留符号，要求绝对值≤1%；只是配对资格，不是精度收益 |
| AP 保留率 `retention` | `AP_frozen / AP_scratch`，Scratch AP 必须>0 | 例如 0.27/0.30=90%；可以超过100%。正式工作判据≥90%，不是“AP 至少90分” |
| AP 差 `AP_drop` | `AP_scratch - AP_frozen`；乘100得到 AP 点 | 正数表示 Frozen 落后，负数表示超过参照；与保留率同时报告 |
| 成本节省率 `saving` | `1 - H_frozen/H_scratch`，两边 H 采用同一口径且参照>0 | 50%表示成本减半；负值表示更贵，不将负结果截断为0 |
| 加速比（派生） | `T_scratch/T_frozen` 或同口径 `H_scratch/H_frozen` | 2倍才对应成本节省50%；“速度快50%”即1.5倍只对应约33.3%节省，不能混用 |
| 同 GPU 预算 AP | 在同一累计 GPUh 下比较标准评测点的 AP | 按第11.3节不插值的共同可比较范围报告，不外推未到达的精度 |
| 达标时间/成本 | 第11.3节共同 `AP_target` 首次连续两个标准评测点达标，并计到第二个点结束 | 未达标记 `target_not_reached`；潜在早停曲线不能改写已实际跑完的支出 |
| 均值、样本标准差 | 按各阶段固定 seed 集计算 AP、成本、配对差和 saving；E3/E4/E5 均为 seed0/1/2；`s=sqrt(sum((x-mean)^2)/(n-1))` | 均值描述典型结果，标准差描述跨训练种子波动，不是同一训练的 epoch 波动 |
| 配对差及95% t区间 | 先按相同 seed 得到差值/比例，再算 `mean ± t(0.975,n-1)*s/sqrt(n)`；E3/E4/E5 的 n=3，t≈4.303 | 小样本区间不稳定；E3 使用同一批结果选参，区间只作探索性描述，不作为独立确认或多重比较校正后的显著性结论 |
| 完整性与复现门槛 | `seen`/ID覆盖、有限值、有效更新、`strict_reload`、checkpoint/预测 SHA256、代码/合同/数据身份 | COCO 必须完整5,000验证图且无缺漏；SHA256是内容身份，不是精度分数；工程通过不等于收益达标 |

E1 的 **80% 保留率暂停线**和 **0.5 AP 点并列线**只用于筛选，正式 P1 仍按精度保留≥90%且同口径成本节省≥50%判断。
逐 seed saving 的均值不一定等于 `1-两组平均成本之比`，两种统计明确命名，不混作同一个结果。
F1/PR 曲线、Router 曲线、loss 曲线可支持解释，但不能替代多 seed、两数据集及标准 AP 的正式证据。
当前已完成 E1 seed0 四组筛选，多 seed 置信区间、VisDrone 收益和尺寸扫描均未完成，不能从现有单次结果生成这些结论。

### 4.10 指标文件索引与待补齐项

下列路径都相对于运行参数指定的外部 `workspace`；`<run-id>` 为 `E1-A/B/C/S`，`NNN` 为三位 epoch。
不把当前服务器绝对路径写入统一合同，不把全部日志复制进 Git。

| 产物 | 主要内容 |
| --- | --- |
| `runs/<run-id>/results.csv` | 逐轮训练/验证 loss、4项内部精度、学习率及进度时间 |
| `reports/<run-id>/progress-rank-*.json` | 最近已写入 batch 的 loss、raw aux、EMA、AMP、有效更新，不是完整逐batch历史 |
| `reports/<run-id>/epochs/rank-*-epoch-NNN.json` | 本轮数据等待/step计时、训练峰值显存、末batch路由快照 |
| `reports/<run-id>/validation/epoch-NNN.json` | 内部验证结果、seen、本轮墙钟 |
| `reports/<run-id>/official/epoch-NNN/report.json` | 每5轮独立COCO六项AP、重载/覆盖、预测及checkpoint摘要 |
| `reports/<run-id>/official/epoch-NNN/cost.json` | 对应独立评测的外层耗时和保留GPU数，属于训练作业的已包含子区间 |
| `reports/<run-id>/mechanism/rank-*-epoch-NNN.json` | 第1/25/50轮开始时的独立FP32梯度/幅值探针 |
| `reports/<run-id>/standard-best.json` | 标准best的epoch、AP和checkpoint摘要 |
| `reports/<run-id>/final/{last,standard-best}/report.json` | 最终两个checkpoint严格重载后的独立重评结果 |
| `reports/<run-id>/training-result.json`、`reports/<run-id>/cost.json`、`jobs/` | 完成状态、实际训练尝试和作业分段成本 |

当前已有：完整 COCO E1 入口、六项标准 AP、逐轮监控、训练峰值显存、有效更新/AMP记录、稀疏机制探针和作业成本分段。
当前仍待补齐：VisDrone 官方 AP/AR 与类别表、逐 epoch 全局机制统计、全阶段设备/主存/磁盘采样、
可审计的冷启动/摊销成本总表、正式多 seed 统计、底座尺寸比较及匹配精度停止确认。
本节定义这些项不代表相应采集器已经实现；后续按 E2-E6 逐项实现与验收。

数值来源以 [E1 编排/独立评测](../../scripts/d1/run_p1p2.py)、[COCO 评测封装](../../scripts/d1/launch_wp8_p1.py)、
[内部指标实现](../../ultralytics/utils/metrics.py)、[训练器](../../ultralytics/engine/trainer.py)及上述 loss/路由实现为准。
本节与第 6、8、9、11 节共同约束报告，不能为获得更好数字临时更换评测器、checkpoint选择或成本口径。

## 5. E0 工程实施清单

以下是阶段职责分配。`run_p1p2.py`、COCO 合同和 `test_d1_p1p2_contract.py` 已实现；
另有 `p1p2_runtime.py`、`p1p2_data.py` 及数据测试。VisDrone 等后续阶段文件仍为约定名称，
aux/梯度与恢复目前由新合同测试及既有相关回归覆盖，不为凑目录重复创建同职责测试。

| 拟新增文件 | 职责 |
| --- | --- |
| `scripts/d1/run_p1p2.py` | 薄编排层：prepare、benchmark、train、evaluate、summarize；复用现有 Dataset/Trainer，不复制训练引擎 |
| `scripts/d1/prepare_visdrone.py` | 原始标注转换、ignore 侧记录、split 与类别 manifest；不联网重下载已有数据 |
| `scripts/d1/evaluate_visdrone.py` | 预测格式、官方 ignore 语义与评测结果适配，不自造新的 AP 定义 |
| `ultralytics/cfg/experiments/d1/p1p2/` | 版本化的 COCO、VisDrone、A/B/C、aux-grid、Teacher-size 合同 |
| `tests/test_d1_p1p2_contract.py` | 参数/预算/样本数、seed、拒绝未登记配置、resume 身份检查 |
| `tests/test_d1_visdrone_protocol.py` | 类别、空标注、ignore、边界框及评测一致性 |
| `tests/test_d1_p1p2_aux.py` | aux 配置、EMA/预算缩放、DDP 梯度、固定初始化和非有限值失败规则 |
| `tests/test_d1_p1p2_cost.py` | 计时不重算、成本摊销、失败/未达标、精度阈值和配对汇总 |

已有文件按所有权小范围扩展：Teacher/cache/NPY 负责数据合同；检测训练/评测负责 batch 与标签；
新 runner 负责实验编排。保留旧 COCO 配置的严格校验，不用全局替换常量破坏历史入口。
NVMe 转换新增模式需要显式根目录白名单、路径越界拒绝、原子文件、收据和中断恢复。
原料保持不变；转完源件删除属于单独动作，不在默认转换内执行。

E0 完成门槛：

1. 新旧合同都可解析；未知字段、错 Teacher、错 nc/shape/split/seed 均立即失败。
2. 32 图闭环验证 Frozen/Scratch 两条路径，至少一次有效更新，loss 与梯度有限，checkpoint 严格重载。
3. aux=0/非零、balance-only、z-only 均有对应计算图证据；额外日志不得改变梯度。
4. 三尺度 Router、residual gain、Adapter 梯度均可追踪；区分检测梯度和 aux 梯度。
5. 恢复训练保留 optimizer、scheduler、scaler、采样 RNG、aux EMA 和数据身份。
6. 新 runner 发现任何 aux nonfinite 隔离、静默跳更新或数据缺失时标记无效运行并安全停止。
7. 原 D1/Foundation/LatentMixture/CompositeCriterion/checkpoint 回归、相关真实缓存测试和 diff 检查通过。
8. 新训练代码先提交并推送，在干净代码 commit 上生成实际 run manifest。
9. RGB 副本复制校验完成；预检确认图片、标签、标注和列表展开后的实际路径均在指定 NVMe 文件系统上，拒绝指回慢盘/NFS 的软链接、旧绝对路径和静默回退；补充相应拒绝规则测试。

## 6. E1 完整 COCO 的空间融合筛选

### 6.1 固定合同与原有筛选规则

完整 train2017/val2017，不重建现有 ViT-S 缓存；seed0，四组统一执行完整训练预算的 **50%**：
完整调度仍为 100 epochs，筛选在第 50 轮验证与保存完成后停止，不将余弦调度压缩成 50 轮。
本次比例调整只针对 E1；VisDrone 的 E3 aux 筛选仍为前 60/300 轮，E4 正式预算不变。

| ID | 模型/融合 | balance / z | latent_aux_gain | 唯一主要对照 |
| --- | --- | --- | --- | --- |
| COCO-A50 | ViT-S，router_only | 0.01 / 0.001 | 0.1 | 新代码基线 |
| COCO-B50 | ViT-S，weighted_sum，权重 [1,1,1] 归一化 | 0.01 / 0.001 | 0.1 | 与 A 比较空间融合 |
| COCO-C50 | 同 B | 0.01 / 0.001 | 0 | 与 B 比较 aux 开关 |
| COCO-S50 | 同可训练参数量 Scratch（阶段性） | 不适用 | 0 | E1 同预算从零训练参照，不预定为最终 P1 基线 |

四组均新建运行。旧 A100 或旧 Scratch30 不直接混入此表。
固定推理语义：LatentMixture 训练和评测都使用全部 4 个专家，不启用 top-k 剪枝/稀疏回退。

筛选依据为第 50 轮独立 COCO AP；每 5 轮保留曲线，窗口内 best 仅作补充，不替代固定轮数的主筛选指标。
50 轮覆盖旧冻结运行最佳第 42 轮附近的观察区间，但不保证新配方也会在此前达到最佳值。
出现 NaN、数据错位、丢样本、静默跳更新则工程失败，
修复后所有受影响组重新开始，不能仅补跑赢家。
AP 差不超过 0.5 个百分点视为筛选并列，优先实际训练成本更低者；仍并列且 A 在并列集合中时保留 A。
如果只有 B/C 并列且可比成本也无法区分，则保留两个 aux 候选交由 E3 复核，不退回明显落后的 A。
这只是候选选择规则，不是统计等效检验；实际成本受到外部争用影响时，保留原账本并披露混杂，不把成本差归因于 aux。
最佳 Frozen 若低于同轮 Scratch AP 的 80%，不立即投入整套 COCO 多 seed 长训练，
先完成 VisDrone/aux 筛选并分析空间梯度；若仍无改善，提交负结果及新方案，不无上限堆预算。
该 80% 是节省探索预算的触发线，不是 P1 成功标准。

A/B 的机制解释至少包括：block4/8/12 各分支的检测梯度、融合前后特征幅值、
各尺度 Router 概率、专家负载和 residual gain；仅“B 比 A 高几点”不作为完整机制结论。
不在同一个 A/B 对照里同时更改 aux、专家数、输入尺度、优化器或训练时长。

### 6.2 已核验的阶段结果与边界

2026-09-08 最终结果：以下来自已完成批次 `e01-20260908/attempt-04`，seed0，完整 COCO val2017 的 5,000 张图片。
AP 为独立标准评测的 `AP_all`，下表乘 100 展示；不是内置 Validator 的 mAP，也不是官方权重成绩。

| 组别 | 融合 / aux gain | 固定第 50 轮 AP | 每 5 轮评测中的最高 AP | 标准 best 所在轮 |
| --- | --- | ---: | ---: | ---: |
| A | router_only / 0.1 | 11.3757 | 11.3983 | 40 |
| B | weighted_sum / 0.1 | 28.8346 | 29.3134 | 30 |
| C | weighted_sum / 0 | 28.6197 | 29.0189 | 30 |
| S | 同可训练参数量 Scratch / 不适用 | 22.7463 | 23.8814 | 30 |

A/B/C/S 均完成 50 轮、15,450 次有效 optimizer 更新；8 份最终 last/standard-best 报告均通过严格重载及全量评测，checkpoint 中 Teacher 参数数目为 0。
完成核验再次校验了 8 个 checkpoint 的实际 SHA256、200 行 CSV 数值有限性、40 个定期标准 AP 点与作业成本账本；不重复训练或评测。
脱敏证据见 [E1 完成摘要](manifests/e1-completion-20260908.json)，包含分组实际执行身份、checkpoint/报告摘要、精度曲线与分段成本。
完整证据位于外部工作区 `reports/E1-{A,B,C,S}/`、`jobs/` 和 `E1-summary.json`；详细归档见 [E0/E1 执行记录](E0_E1.md)，不把 checkpoint 或完整预测放入 Git。

A/B/C 训练及评测记录包含 `9347a8b9541790c980d5371ae9cf79630ecc606d` 与 `d9151b49188651b9c3562fc6b7877a5b3f1626e9`，相关源码 SHA256 均为 `f4d263706445eb649ba845995e0bb9e9edaa3e0372eb8af8ecdfe6b591733d09`。
S 恢复后的训练及最终评测绑定 `ab48913c74c0534dfbf32f5e6ec7bb07cf4c0131`，源码 SHA256 为 `cc17abf13c4236331be7cbd64c3c2fd28bf4dd03607404d61b11871efd7bb170`；恢复兼容修复的范围与验收见 E0/E1 执行记录。
四组合同 SHA256 均为 `619bc71e155f3fe291179189edc656f6a8399b50f7bf911f123fc7b5e18d0322`，模型、loss、数据、batch 和训练调度未因恢复改变。
每份报告仍保留原始 identity 与实际 execution_identity，不用本次文档提交覆盖运行身份。

目前支持的结论：

- B 相比 A 的固定第 50 轮 AP 增加 **17.4588 个点**。weighted_sum 是当前有直接对照支持的改进方向，但单 seed 结果仍不替代正式统计。
- B/C 相比 S 的固定第 50 轮 AP 分别高 **6.0883 / 5.8734 个点**，保留率为 **126.77% / 125.82%**；A 仅为 **50.01%**。结论限于本次同可训练参数量、seed0 的阶段性参照。
- B 比 C 仅高 **0.2149 个 AP 点**，属于预设的筛选并列区间。自动摘要按并列后的记录成本选择 C、融合方式为 weighted_sum；B 的外部 CPU 争用影响成本，不能据此证明 C/关闭 aux 更优。B/C 均保留为后续候选，尚未锁定 AUX_STAR。
- B/C/S 的 standard-best 都在第 30 轮，至第 50 轮分别下降约 **0.4788 / 0.3992 / 1.1351 个 AP 点**。这是固定采样点上的曲线现象，不足以单独证明过拟合；不能据此把正式预算改为 30 轮，或用 best 替代第 50 轮主结果。
- B/C 第 50 轮 AP_small 分别约 **12.79 / 12.87**。小目标和定位值得诊断，但不同尺寸目标的难度不同，不能仅凭 AP_small 低就认定 Adapter 存在缺陷。
- 关闭 aux 不等于关闭 LatentMixture、Router 或专家。当前仍执行完整检测前向/反向和 raw aux 诊断；实际进入总损失的 aux 为零。更低的路由正则指标也不自动等于更高 AP。

### 6.3 E1 完成后的决策与交付

S 按原初始化、100 轮调度前缀、50 轮窗口、batch384 和标准评测频率完成，四组均已生成最终报告。
`E1-summary.json` 为 completed、fusion=weighted_sum、selected=C、pause_full_coco=false；保留率字段对应 C，为 125.82%。

| 结论项 | 当前结果 | 下一步边界 |
| --- | --- | --- |
| 空间融合 | B/C 均明显高于 A 和当前 S | 以 weighted_sum 为后续研究方向，A 保留作机制参照 |
| aux 选择 | B/C 相差 0.2149 AP，属于筛选并列 | 保留启用/关闭 aux 两条候选，E3 再确定 AUX_STAR，不将自动选择 C 当作统计显著胜出 |
| COCO 80% 暂停线 | 最佳 Frozen 与选中的 C 均超过当前 S 的 90%，未触发 80% 暂停线 | 仅通过本次筛选门槛；正式架构、参数口径和基线仍按第 4.3、9 节另行确认 |
| 固定轮数成本 | B/C 训练作业 GPU-hours 分别降低 4.44%/12.36%；加最终评测后为 4.37%/12.20% | 未达到 50%，尚不含 Teacher 抽取，不能宣称冷启动达标 |
| 实验覆盖 | 单 seed、COCO 一套阶段性参照；E2 VisDrone 工程验收已完成 | E3 及最终架构研究仍待确认实施；不自动启动 E4 或将现有 S 指定为正式基线 |

固定第 50 轮 AP 为主结果，标准 best 和早期收敛作为补充；不外推第 100 轮成绩。
E1 每小时监控已暂停，完成状态不授权重跑或扩展矩阵。E4 仍需按确认后的正式基线形成两数据集、多 seed 和同口径成本结论。

### 6.4 条件性结构改进候选

本节为待确认的独立小预算研究，不属于已经启动或默认必跑的矩阵。
先完成 E1 和机制诊断，必要时在 E3 后、E4 配方锁定前另行确认；不把新增结构混入原 A/B/C 因果对照。

| 优先级 | 单项改动 | 检验问题 | 必须同步观察 |
| --- | --- | --- | --- |
| 1 | 每个尺度的三层融合权重可学习；softmax、均匀初始化，共 9 个标量参数 | 固定 1/3 融合是否限制不同层的利用 | 三尺度层权重、分支检测梯度、AP/AP_small/AP75、成本 |
| 2 | P3 上采样后增加轻量 3×3 空间细化 | 插值后的细节加工是否改善小目标与定位 | AP_small/AP75、参数、显存、单轮耗时；不宣称现有专家/检测头没有空间卷积 |
| 3 | 轻量跨尺度 Neck，先验证简单自顶向下融合 | 当前三尺度之间缺少显式直接交互是否限制精度 | 多尺寸 AP、增量 FLOPs/参数和 GPU-hours；不直接上复杂整套 Neck |
| 后续 | 检测头容量或更大的 Teacher，分开开展 | 小改动不足时判断定位容量或底座能力限制 | 单独合同和预算；ViT-B 优先沿用 E5 的 VisDrone 设计 |

每个候选只改变一个因素，保留等权 weighted_sum 对照、数据、seed、aux、batch、优化器和完整调度前缀。
具体筛选窗口和最多运行数须在启动前登记，不边跑边延长；同初始化身份的基线可复用，否则补配对基线。
在查看候选结果前固定入选规则，进入正式对照前补独立 seed 复核和源码/配置/测试证据。
如果采用新结构，则新建实验身份并重新统计两数据集总参数、冻结参数、可训练参数和计算量；不再自动要求每个结构候选立即重建 Scratch。
正式基线待最终冻结架构确定后按第 4.3、9 节统一设计，不能沿用旧同参声明或把旧参照成绩当作新架构的正式匹配结果。
不同时更换 Teacher、融合、Neck 和检测头；可选结构实验不替代 E3 的 aux 扫描、E5 的尺寸对照或 P1 的两数据集证据。

### 6.5 P5 Adapter 轻量化对照

**初始执行边界：实现阶段仅批准方案、代码和工程验收。2026-09-09 用户另行批准后三组已完成，结果见第 6.5.6 节；本次归档不启动新实验。**
不改变已经完成的 E1 记录，不启动 E2/E3/E4 训练，不以参数减少代替 GPU-hours 收益。

#### 6.5.1 问题与三组结构

当前 P5 为三个 DINO 来源各自使用独立的普通 `3x3 Conv(384->256, stride=2)`，
三条分支含 GroupNorm 共 2,655,744 参数，占 COCO 下游 3,542,567 可训练参数的 74.97%。
这里的分母不包含冻结 Teacher；P5 参数占比也不是计算量或实际训练耗时占比。

三个组都保持 block4/8/12 顺序、独立参数及 `[B,256,20,20]` 输出；P3/P4、LatentMixture、Detect 不变。
所有卷积不带 bias；每个卷积后均接 `GroupNorm(get_safe_groups(C,8)) + SiLU`。

| 实验 ID | P5 每条分支 | 三条 P5 参数 | COCO 下游可训练参数 |
| --- | --- | ---: | ---: |
| P5-BASE | 普通 `3x3 384->256, stride=2, padding=1` | 2,655,744 | 3,542,567 |
| P5-DW | `3x3 depthwise 384->384, groups=384, stride=2, padding=1`，再 `1x1 384->256` | 309,120 | 1,195,943 |
| P5-BN64 | `1x1 384->64`，再普通 `3x3 64->256, stride=2, padding=1` | 518,016 | 1,404,839 |

P5-DW 将空间下采样与通道混合分开；P5-BN64 先压缩通道再做普通空间卷积。
BN64 表示 64 通道瓶颈，不是 BatchNorm；两种结构都只使用 GroupNorm。
本轮不实现平均池化方案，不同时增加可学习层权重、P3 细化或跨尺度 Neck。
FP16 三层缓存及其 key/索引完全复用，不重抽 Teacher，也不修改缓存内容。

#### 6.5.2 配方与配对初始化

三组均采用 E1-B 的 weighted_sum，固定三层权重 `[1,1,1]` 归一化；
`balance_loss_coeff=0.01`、`router_z_loss_coeff=0.001`、`latent_aux_gain=0.1`、`mixture_aux_budget=3.0`。
保留 aux 是为了只研究 P5 结构，不代表已经确定 AUX_STAR；B/C 的后续 aux 比较仍按 E3 执行。

- 数据：完整 COCO 2017 train2017=118,287、val2017=5,000，640 输入，无增强；只用已验证 NVMe RGB/NPY 路径。
- 预算：seed0，保留 100 epoch 余弦调度，固定前 50 epoch 筛选；最多三组各一次，不自动延长或增加 seed。
- 硬件：六张 A40，每卡 batch64、全局 batch384、nbs384、每 rank workers4、prefetch1；不额外累积梯度。
- 优化：AdamW、lr0=0.001、lrf=0.01、weight_decay=0.0005、warmup3；AMP 初始 scale16、growth_interval=1,000,000。
- 每 5 轮保存并独立评测标准 AP；第 50 轮为主结果，standard-best 仅作补充；最终 last/standard-best 严格重载评测。
- 使用三个独立实验目录、优化器和恢复状态，严禁把原 P5 权重以宽松加载方式塞入新 P5，或从旧 best 接着微调。

复用已验证 E1 `initial-frozen.pt` 中所有非 P5 初始张量，逐项核验相同键、形状、dtype 和数值；
BASE 还复用原 P5 初始张量，DW/BN64 的新 P5 以 seed0 初始化并分别保存摘要。
这样 Router、专家、检测头等不会因为新 P5 消耗随机数的数量不同而改变初始化。
原 E1-B50 精度可作历史参照，但跨时间吞吐、争用和代码身份不同，不能替代本轮配对 BASE 的成本测量。
正式启动前先报告三组短基准与预计时间，用户确认后按 BASE、DW、BN64 顺序串行运行，不和其他 GPU 作业争用。

#### 6.5.3 指标、判据与边界

记录标准 AP/AP50/AP75/AP_small/AP_medium/AP_large、检测及 aux loss、P5 各卷积梯度、
Router 概率和 residual gain、参数量、卷积 MACs、显存、数据等待/计算/验证时间与实际 GPU-hours。
理论上 DW 的 P5 卷积 MACs 约减少 88.50%，BN64 约减少 72.22%；仅指正式 40x40 输入下三条 P5 的卷积乘加，
不含归一化、激活、其他模块、反向传播或 I/O；不是整网实测速率。

预登记的探索性保留线：相对新 BASE，第 50 轮整体 AP 下降不超过 0.5 点，
AP75 与 AP_large 各下降不超过 1.0 点，且实际训练加最终评测的 GPU-hours 更低。
AP_small 必须单列，不能假定压缩 P5 自动修复小目标；单 seed 通过仅保留候选，不作为统计等效或 P1 成功结论。
若两组均通过，先比较实测成本；成本受争用污染或差异不足时保留并列，待确认预算后用独立 seed 复核。
若均未通过，则保留原 P5，不自动增加瓶颈宽度、Teacher 大小或训练轮数。
本次不是新的 Scratch 对照；最终参数口径、正式基线和两数据集统计仍按第 4.3、9 节，待冻结架构确定后处理。

#### 6.5.4 实现入口与启动门禁

- Adapter 通过 `p5_mode=conv/depthwise/bottleneck` 选择结构；仅 bottleneck 接受 `p5_bottleneck_channels=64`。
- 新模型配置：[DW](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-dw-n.yaml)、[BN64](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml)。
- 独立合同：[P5 COCO](../../ultralytics/cfg/experiments/d1/p1p2/p5-coco2017.yaml)。
- 独立入口：[run_p5_ablation.py](../../scripts/d1/run_p5_ablation.py)，复用现有 E1 训练策略和标准评测，不修改 E1 的组别和合同。
- 新旧架构只能严格按各自配置恢复；旧配置没有 P5 字段时仍构建原始结构，不改变旧 state_dict 键。

工程检查和准备命令如下；运行参数注入外部路径，不把服务器路径写入 Git 配置：

```bash
python -m scripts.d1.run_p5_ablation inspect
python -m scripts.d1.run_p5_ablation prepare --source-workspace "$E1_WORKSPACE" --workspace "$P5_WORKSPACE"
```

`prepare` 只核验来源并生成模型配置、配对初始化、哈希及实验登记，不训练；要求代码已提交且工作树干净。
确认正式训练后，才能对单组执行以下模板；没有 `--approved` 必须直接拒绝：

```bash
torchrun --standalone --nproc_per_node=6 -m scripts.d1.run_p5_ablation train --workspace "$P5_WORKSPACE" --variant DW --approved
python -m scripts.d1.run_p5_ablation evaluate --workspace "$P5_WORKSPACE" --run-id P5-DW --checkpoint "$P5_WORKSPACE/runs/P5-DW/weights/last.pt" --output "$P5_WORKSPACE/reports/P5-DW/final/last"
```

BASE/BN64 使用各自 variant 与目录；恢复需同时提供 `--resume --approved`，并通过配置、来源和恢复状态身份检查。
预计超过 3 分钟的基准、训练或全量评测按已有约定后台挂载，保存 PID/日志，只确认启动一次后退出并给出状态命令。
本节模板本身不是新的启动授权；以下保留实现阶段验收记录，后续已批准运行的结果见第 6.5.6 节，不改写旧 E1 成绩。

#### 6.5.5 工程验收状态

截至 2026-09-09，两种轻量结构、独立模型 YAML、实验合同及准备/训练/恢复/评测入口已实现。
新增测试 29 项通过，其中两项使用真实 COCO 单图的 NPY 特征与检测标注，在 CPU 上完成 loss、
反向传播、AdamW 更新及严格 checkpoint 重载；其余覆盖形状、参数独立性、配对初始化、非法输入和启动门禁。
相关回归共 **284 passed, 8 skipped**；跳过项为未启用的可选集成测试。
原始默认 Adapter 的参数布局及旧 checkpoint 保持兼容，E1 原有组别与训练合同不变。

上述初始工程验收阶段没有启动正式训练，也没有执行六卡 AMP/DDP 吞吐基准；不能只据此报告 AP、实际加速比或新训练 ETA。
正式启动前仍需独占 GPU 的短基准与用户确认。服务器未安装 Ruff，静态检查采用 `py_compile` 与 `git diff --check`。
真实单样本验收可由 `D1_P5_CACHE_ROOT` 和 `D1_P5_RGB_ROOT` 指向本机缓存与 RGB 根目录后运行
`python -m pytest -q tests/test_d1_p5_ablation.py`；不设置这两个变量时只跳过真实数据测试，不下载数据。

#### 6.5.6 提速后三组完成结果（2026-09-09）

三组已使用同一干净代码 `7a0ad4d`，统一采用 `separable_bilinear2x` P3、
`foreach-v1` EMA 和已验收的恢复实现，从配对初始化完成 50/100 轮筛选。
每 rank 15,450 次有效更新，AMP 重试均为 0；最终 last/standard-best 各完成 5,000 图标准评测与严格重载。
完整报告、成本定义与证据见 [P5 完成报告](P5_FAST_RUN_20260909.md)。

| 组别 | 第 50 轮标准 AP | AP75 | AP_small | AP_large | 活动作业 GPUh | 原筛选门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BASE | 28.870 | 30.113 | 12.921 | 40.228 | 10.278431 | 本轮参照 |
| DW | 28.593 | 29.723 | 13.033 | 41.559 | 10.363558 | 精度通过，成本未通过 |
| BN64 | 29.745 | 31.203 | 12.965 | 42.992 | 10.424189 | 精度通过，成本未通过 |

AP 按 0-100 展示，成本包含六卡训练作业与两次单卡最终评测，不包含 Teacher 离线抽取。
BN64 比 BASE 提升 0.875 AP，但其训练期间与四段 FinsSim 短任务窗口重叠，取并集约 12 分 52 秒。
没有无争用配对重跑，不能把其耗时差完全归因于结构，也不能直接扣除重叠时长。
按已登记规则，DW/BN64 均未证明“同时更省 GPUh”，不自动替换原 P5；
BN64 可保留为精度候选，最终架构与后续复核预算仍须确认。
本节不改动 E1 旧结果、不锁定 AUX_STAR、不确定正式 Scratch 基线，也不启动 E3-E6。

## 7. E2 VisDrone 数据和评测协议

实际实现、数据统计、评测验收和缓存复现入口见 [E2 执行记录](E2.md)。
**2026-09-09 E2 已完成。** 独立标签与 ignore sidecar、7,019 图完整缓存、299 图原 batch 在线对齐、
10 类小样本训练及严格恢复、3 轮全集短基准、实际模型预测的官方 MATLAB 评分全部通过。
执行版本为 `f328a10907f57c66c2efb92d7b506be7cc836b8d`，完整判定见 [E2 验收摘要](manifests/e2-acceptance.json)。
不将工具夹具或三轮短训练当成正式精度结论，也不自动启动 E3/E4。

### 7.1 数据准备

复用已下载的 VisDrone2019-DET 原始图像、标注和 ZIP 校验记录。
保留原始文件，不执行现有 [VisDrone.yaml](../../ultralytics/cfg/datasets/VisDrone.yaml) 中会移动图片并清理原目录的自动下载块。
生成独立 YOLO 标签目录、train/val/test-dev 相对路径列表、原标注 SHA256 和 ignore 侧记录。

有效检测类别为原始 1..10，映射到 YOLO 0..9。仅有效类别且 GT score=1 的框进入训练目标；
score=0、类别 0/11 等原始信息完整保留在 sidecar，不直接减一后写入非法类别。
不因遮挡程度而额外删除合法目标。拒绝非有限坐标/负宽高，裁剪到图像边界并记录裁剪/剔除计数，
保留无有效目标的图片，不能为凑样本数重复图像。

本轮训练沿用现有 YOLO 检测损失，两组均使用相同有效目标集；未另行实现 ignore 区域背景梯度屏蔽，
这一训练语义必须披露。如后续加入 ignore-aware loss，要新建协议并重做配对实验。

### 7.2 正式评测

使用 [VisDrone 官方数据与工具入口](https://github.com/VisDrone/VisDrone-Dataset)及其链接的
[VisDrone2019-DET Toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)。
工具仓库名称含 2018，但其 README 标明服务于 VisDrone2019；实施时锁定实际 toolkit commit。

输出原始图像坐标下的逐图检测 TXT，类别回映射到 1..10，保留每张图的输出文件（空预测也需要）。
按官方工具处理 ignore regions、others、score=0 及匹配规则；
不能仅删除忽略 GT 后直接用普通 COCO API 并标为“官方 VisDrone AP”。
记录 IoU 阈值集合、每图检测数量限制、截断与排序、NMS/端到端推理规则。
COCO 导出 conf=0.001、max_det=300；VisDrone 候选导出 max_det=500、conf=0.001，
正式评价遵循锁定 toolkit 的取值/截断，两组一致。

采用官方 MATLAB 工具作基准；若用 Python 包/适配器，必须在合成边界样例和固定真实图集上与 MATLAB 数值对齐，
误差阈值预设 AP 绝对值 1e-6（0..1 口径），不能对齐就继续使用官方工具。
本机已有 MATLAB 可作为离线基准，服务器不必安装 MATLAB；数据/代码与预测跨机路径由运行参数提供。
运行前重新核实工具运行环境和许可；不要把普通 COCO AP 与 VisDrone 官方 AP 混表。

test-dev 在融合、aux、预算、seed 和模型都冻结后只运行最终评测，不据结果回调参数。
官方 val 用于开发与筛选，报告中明确它不是未见测试集。

### 7.3 缓存

ViT-S 的 train+val 共 7,019 图，三层 FP16 原始张量约 **24.10 GiB**；test-dev 另约 **5.53 GiB**。
初始不抽 test-dev，待 E7 再生成。每个 split 校验数量、唯一 ID、相对路径、标签与特征对应关系及 SHA256。
抽取时固定 rank 分区、batch 和尾批上下文；在线抽查按原 batch 重现 FP16 特征，不要求任意重组 batch 后逐元素一致。
缓存只依赖数据、Teacher 和预处理，不包含训练 seed/aux 超参，可供各训练 seed 复用。

### 7.4 实际验收结果与下一阶段参考

- 干净提交回归 **300 passed、10 skipped**；96 图训练/8 图验证完成保存和严格恢复，共 2 次更新。
- 完整 6,471/548 图短基准完成 3 轮、204 次更新，无 AMP 重试，九条 Adapter 分支及三个尺度的 Router/residual gain 已验证梯度或实际更新。
- 六张 A40、global batch96、每卡16、workers4/rank；第 2/3 轮中位 **17.9522 秒/轮**，峰值 allocated **4.67 GiB**、reserved **8.20 GiB**。
- 同配方早期外推：60 轮约 **17.95 分钟**，300 轮约 **89.76 分钟**；不含启动、Teacher 抽取和独立官方评分，不能用三轮结果保证完整训练时间。
- 完整 val 的第 3 轮 EMA checkpoint：官方 AP **3.0618%**、AP50 **8.6529%**、AP75 **1.5506%**；官方评分耗时 **22.0004 秒**。
- 同次内部监控 mAP50-95 **2.6650%**、mAP50 **7.5580%**，与官方协议分列；不混用两套数值来选参或报告提升。

详细参数、资源状态、全部指标、checkpoint 和预测哈希见 [E2 执行记录第 7/8 节](E2.md#7-回归与训练验收)。
这些结果只解除 E2 工程门禁，既不证明 P1 的 50% 成本收益，也不决定 E3 的 aux 最优配置或最终架构。
MATLAB 是当前锁定官方评测的实现，不是训练依赖或课题强制语言；替换为 Python 时须先证明同预测下的官方协议数值一致。

## 8. E3 P2 辅助损失扫描（仅 VisDrone，全部候选三 seed）

### 8.1 范围、固定设置与损失解释

**2026-09-09 用户确认：aux 消融仅在 VisDrone 完整 train/val 执行，不新增 COCO aux 消融或开关复核。**
保留 E1 的 COCO B/C 历史结果，但不将其算作 VisDrone 的运行，也不据此额外启动 COCO 多 seed aux 实验。
E4 两数据集的 Frozen/Scratch 正式对照仍保留，不能用该系统对照替代 COCO 的 aux 独立贡献证据。

**2026-09-10 更新：用户确定 BN64 为后续冻结模型架构基座，并因时间预算将 E3 从四 seed 减为 seed0/1/2。**
第一阶段从 36 次减为 27 次，第二阶段新增从 12 次减为 9 次，合计从 48 次减为 36 次，减少 25%；每次仍为 60/300 轮。所有正式运行尚未开始时完成该修订，不依据 E3 结果事后剔除种子。

BN64 只表示 P5 的 64 通道瓶颈：每个来源独立执行 1x1 Conv(384->64)+GN+SiLU，再执行 3x3 stride2 Conv(64->256)+GN+SiLU，不是 BatchNorm。P3 使用已验收的 separable_bilinear2x，模型 EMA 使用 foreach-v1，恢复保留 rank-local buffers 和 epoch-boundary-v1 温度推进。VisDrone 的 10 类头预计可训练参数为 1,340,259，启动前以构造模型核实。
COCO 已有 P5 对照支持 BN64 的精度与参数量取舍，但其可比无干扰耗时没有低于 BASE；选为基座不改变原成本门槛失败记录，不代表已满足 P1 的 GPUh 降低 50%。详见 [P5 对照报告](P5_FAST_RUN_20260909.md)。

固定 ViT-S、BN64、weighted_sum、batch96、`mixture_aux_budget=3.0` 和 300 轮学习率调度；
每次执行前 60 轮，不把余弦周期压缩为 60 轮，不因曲线提前见顶而改变筛选时点。
所有候选均使用 **seed 0、1、2**，不再采用“seed0 预筛，再只给少数候选补种子”的流程。
同一 seed 的各 aux 配置复用完全相同的下游初始 state dict，并记录 SHA256；不同 seed 分别生成初始化。
同 seed 的样本顺序、DDP 分区、随机性与 AMP/优化器规则相同；每次运行独立保存优化器、EMA 和恢复状态。
第一批启动前锁定同一 Teacher、Adapter、LatentMixture、Detect 和源码身份；不得把 P5/Neck 等结构修改混入 aux 扫描。
后续若采用新架构，E3 结论仍归属于原架构，不能将不同架构的候选混合排名。

当前实现先由各 LatentMixture 生成：

```text
raw_latent = sum_over_p3_p4_p5(c_balance * balance + c_z * z_loss)
scaled_latent = latent_aux_gain * raw_latent / clamp(EMA(raw_latent))
effective_aux = scaled_latent * detached_budget_scale
total_scalar = sum(native_detection_loss) + effective_aux
```

以上是 D1 只有 latent routed loss 时的结构示意，实际 EMA 更新、边界与 DDP 缩放以锁定代码为准。
共同放大 balance/z 系数可能被 EMA 抵消，预算限幅也可能压平候选差异；
因此不能把配置权重大小直接当作实际正则强度。

### 8.2 两阶段候选矩阵与运行数

| 阶段 | 固定内容 | 候选 | seed | 候选所需结果 | 实际新增训练 |
| --- | --- | --- | --- | ---: | ---: |
| 第一阶段：balance/z 网格 | gain=0.1，budget=3.0 | balance ∈ {0,0.01,0.1}；z ∈ {0,0.001,0.01}，完整 9 组笛卡尔积 | 0、1、2 | 9 × 3 = 27 | 27 |
| 第二阶段：全局 gain | 第一阶段选中的非零系数组合；budget=3.0 | gain ∈ {0,0.03,0.1,0.3} | 0、1、2 | 4 × 3 = 12 | 9；gain0.1 的三 seed 结果复用第一阶段 |

总计 **36 个独立有效训练运行**，即 27 + 12 - 3 = 36；每次 60 轮，总计 2,160 个训练 epoch。
两张阶段表可引用同一组 gain0.1 结果，但不得把它们重复计为新训练或重复累计成本。
不再另设原方案的“默认/候选 seed1、seed2 复核”第三阶段；三个 seed 已覆盖每个候选。
第一阶段全部 27 次有效结果齐全后才能选 balance/z；第二阶段需四个 gain 各三 seed 结果齐全后才能锁定 AUX_STAR。

第一阶段同时保留全零、仅 balance、仅 z 和两项同时开启的对照。
第二阶段选择第一阶段三 seed 平均 AP 最高的**非全零**系数组合；若全零组领先，仍如实报告，
并在最佳非零组合上完成包含 gain0 的强度扫描，以免全零系数使四个 gain 失去可比较的正则作用。
全零系数/gain0.1 与非零系数/gain0 的数值及诊断路径可能不同，不未经验证合并为同一个配置。
默认配置 balance0.01/z0.001/gain0.1 已包含在第一阶段，每个 seed 都有对照结果，不另加重复运行。

只有模型、源码行为、数据/缓存、初始化、seed、预算、AMP/优化器与评测身份均一致，才允许复用结果。
失败或中断的配置不得从排名中静默剔除，先按同身份恢复或补齐；重试及失败支出单列，不算新的候选配置。

### 8.3 三 seed 选参规则与统计边界

主指标为 VisDrone 官方评测口径的**固定第 60 轮 AP**，按 seed0/1/2 的算术均值排名；
保存原始 0..1 数值后计算，展示时才换算 AP 点，不用显示舍入值决定名次。
standard-best 和其他 epoch 的最高值只作补充，不替代固定第 60 轮。
每个配置报告三个原始 AP、均值、样本标准差（分母 n-1，n=3）、AP50/AP75、AR 与实际 GPU-hours。

第一阶段按三 seed 平均 AP 选出一个非零 balance/z 组合；第二阶段同样按三 seed 平均 AP 选 gain。
均值完全相同时依次比较 AP 标准差、更低且可比的平均 GPU-hours，仍相同按参数元组升序固定选择，完整披露并列。
不沿用 E1 的 0.5 AP 容差自动改选低成本配置；均值相近或配对差方向不一致时注明证据有限，不宣称统计显著。
这是一组预先限定候选和固定 gain 下的顺序搜索，不保证全局最优；不临时扩大到多个系数组合或完整三参数联合网格。

第二阶段逐 seed 计算每个非零 gain 相对同一 balance/z 下 gain0 的 AP 差，同时与默认配置配对比较。
报告配对差的均值、样本标准差与正/负差的 seed 数；若给出 95% t 区间，n=3、自由度 2，且注明小样本局限。
三个 seed 都参与了配置选择，相关区间仅为探索性描述，不能当作独立确认或已校正多重比较的显著性证据。
若所有非零 gain 都不如 gain0，允许关闭 aux 成为 AUX_STAR，并保留完整负结果。
若结果不能稳定区分，则明确报告“不足以证明 aux 有稳定收益”，不以“训练未报错”代替效果验证。

选择后锁定 `AUX_STAR`，用于 E4 两个数据集和 E5 两种底座；不按各组结果再分别调参。
本轮 aux 效果结论限定于 VisDrone、锁定架构和 60/300 轮窗口，不宣称已证明 COCO 上也有同等收益。
COCO 只保留 E1 既有开关对照和 E4 固定配方系统对照，不追加 COCO aux 消融或跨数据集 aux 复核。

### 8.4 机制证据与执行门禁

每 epoch 记录三个尺度各自的 raw balance/z、加权项、EMA 分母、预算缩放、effective_aux、
aux/native loss 比、Router 概率熵/负载、residual gain、检测梯度与 aux 梯度范数。
主损失为向量时单测必须证明 aux 只贡献一次；日志中的 raw aux 与最终加入总 loss 的值分别命名。
新鲜运行的 aux EMA 初始化相同；resume 保留其状态。监控补丁不得改变训练计算图。
若两个权重设置因归一化/限幅实际等价，应报告这一机制，不强行声称权重越大正则越强。

先通过 E2 的数据/官方评测对齐和真实 batch 工程门槛，再锁定矩阵、代码与配对初始化。
第一批是 9 组各三 seed，共 27 次；可以优先执行默认与全零配置，但不得只看 seed0 就淘汰其他候选。
第一批完成后报告三 seed 汇总和选中的组合，第二批新增 3 个 gain 各三 seed，共 9 次；结果复用按第 8.2 节校验。
2026-09-11 用户已确认第一阶段官方结果，并授权按原计划启动第二阶段新增9次实验。固定balance=0.1、z=0，扫描gain0/0.03/0.1/0.3、seed0/1/2，其中gain0.1三次复用；不自动扩展至E4。
每个运行按第 13 节的长任务约定执行；缩减矩阵或增加配置、seed、epoch 必须事先修订，不事后隐去负结果。
实施与状态说明见 [E3 执行说明](E3.md)。逐 epoch 机制探针在各 rank 的独立 FP32 模型副本及固定两张训练图上执行，保留并恢复 RNG；它是固定样本诊断，不冒充整轮数据的均值。三尺度原始量与实际 aux、梯度分开记录；全零系数时额外计算 detached 路由诊断，不能把核心模块短路返回的零误认作自然达到完美负载均衡。

每 5 轮保留严格重载的 checkpoint 和全量 548 图预测；GPU 导出与本机固定 MATLAB 工具评分分开执行。评分可以在训练后补齐，训练不依赖评分结果，所有采样点仍保留同一评测协议。官方评分未齐前只标记 awaiting_official，不用内部 AP 排名、不生成 standard-best 或 AUX_STAR；MATLAB/传输时间独立计入总墙钟，GPU 导出等待仍计入训练 GPUh。

重点验证“路由正则变化是否转化为检测收益”，不能用 E1 的 B 比 C 高 0.21 AP 直接选定 VisDrone 系数，
也不能从 E1 受干扰的总时长推断关闭 aux 更快。

### 8.5 后续正式对照边界

AUX_STAR 与最终冻结架构锁定、正式基线通过第 9 节确认后，再进入 E4 两数据集配对训练。
原 E1 80% 暂停门禁仍保留，但当前 E1 未触发；以下仅为该门禁触发后的独立系统确认，不是 COCO aux 消融，也不包含在 E3 的 36 次中：
最多增加一次采用固定融合和 AUX_STAR 的 COCO50、seed0 确认，保留 100 轮调度，与同身份 E1 Scratch50 比较。
仅当代码、初始化规则、数据和预算身份仍完全一致时复用原参照；否则先确认一对重跑预算。
仍低于 80% 则暂停 COCO 正式长训练；若与已有 E1 配置和身份完全相同则直接引用，不重复运行。
该门禁未触发时不启动这些额外任务；COCO 上任何新的 aux 参数扫描或开关复核均不在本方案范围内。

### 8.6 第一阶段官方结果与第二阶段锁定（2026-09-11）

27/27个固定第60轮checkpoint均完成548图官方MATLAB评分，证据绑定与原服务器汇总校验通过。详细九组表、逐seed结果、其他官方指标、成本及第二阶段复现见 [E3第8-9节](E3.md#8-第一阶段官方结果2026-09-11)，机器可读证据见 [官方汇总](manifests/e3-stage1-official-20260911.json)。

最优非全零组合为balance=0.1、z=0：AP **8.132 +/- 0.351点**；默认为8.085 +/- 0.116点，关闭aux为8.022 +/- 0.069点。最优相对关闭aux仅+0.110点，三个seed配对差为-0.245/+0.339/+0.235点；筛选胜出但尚无稳定收益证据。
固定该组合开展原计划gain扫描，不增加候选或改变60/300预算。新增gain0/0.03/0.3各三个seed共9次，gain0.1严格复用；从每个seed原始下游初始化新训，不续训第一阶段模型。
复用组实测平均18.8556分钟/次，预计新增训练约2小时50分钟，MATLAB与传输另计。第一阶段27组作业累计50.097549 GPUh，含中断尝试，不能当作相对Scratch的成本降幅。
上述为第二阶段启动时的安排；实际完成结果和AUX_STAR见第8.7节。未采用上一轮讨论中未获采纳的额外实验或评测变更。

### 8.7 第二阶段官方结果与 AUX_STAR（2026-09-11）

新增9次训练和548图官方MATLAB评分全部完成，与复用gain0.1的3次构成4个gain各3个seed。固定balance=0.1、z=0、budget=3.0，仍比较300轮调度的固定第60轮，不按best epoch或best seed选取。

| gain | seed0/1/2 AP（点） | AP均值 +/- 样本标准差 | 相对gain0平均AP差 | 正向seed数 |
| ---: | --- | ---: | ---: | ---: |
| 0 | 7.972 / 7.994 / 8.102 | 8.022 +/- 0.069 | 0 | 参照 |
| 0.03 | 8.047 / 8.144 / 8.156 | 8.115 +/- 0.060 | +0.093 | 3/3 |
| **0.1，复用** | **7.726 / 8.333 / 8.336** | **8.132 +/- 0.351** | **+0.110** | **2/3** |
| 0.3 | 7.874 / 8.220 / 8.150 | 8.081 +/- 0.183 | +0.059 | 2/3 |

按未四舍五入的官方AP均值优先规则，锁定 **AUX_STAR=(balance_loss_coeff=0.1, router_z_loss_coeff=0, latent_aux_gain=0.1, mixture_aux_budget=3.0)**。gain0.1比gain0.03仅高0.016558点；gain0.03波动更小，但本轮不事后改变选参规则。三个非零gain对gain0的配对95%探索性区间都跨0，n=3且同一批seed参与选参，**不足以证明aux有稳定或显著收益**。

gain0三个seed的最终548份预测TXT均与第一阶段全零系数组逐文件字节一致；这只证明本次最终预测一致，不是全部中间状态等价。新增9次成本仍计入账本。两阶段36次累计67.160097 GPUh，其中第二阶段新增17.062548 GPUh，不重复计入复用组；这不是与Scratch的成本降幅。第二阶段队列2小时50分45.916秒，本机MATLAB评分211.315秒，传输与独立准备另计。

详细指标、配对差、统计边界、成本和复现入口见 [E3第10节](E3.md#10-第二阶段官方结果与最终-aux-设置2026-09-11)，证据见 [第二阶段官方汇总](manifests/e3-stage2-official-20260911.json)。两阶段432份定期预测已保留，只完成36份第60轮主结果评分，不宣称完整standard-best曲线或300轮收敛。

本节完成了此前条目中的AUX_STAR待定门槛；第9节其余基线、完整配对合同、训练/恢复验收、资源与预算门槛仍保留。E4/E5使用同一锁定设置，不新增COCO aux扫参，不自动启动下一轮实验。

## 9. E4 P1 两个数据集的正式对照（基线待定）

**启动门槛：先确定最终冻结模型架构，再设计正式 Scratch 基线。** 当前 E1 S 不自动成为正式基线；下面的种子和训练上限保留为预算草案，不授权现在选型或启动。
正式对比前完成以下确认清单：

1. 锁定最终冻结模型的 Teacher ID/权重、Adapter、融合/LatentMixture、Neck（如有）和 Detect，记录结构配置、代码及 SHA256。
2. 确认“同参数量”指完整系统总参数还是可训练参数，记录导师/课题负责人的确认依据、匹配容差和是否要求同架构；不把当前 E1 的 1% 约定当成官方定义。
3. 对 COCO 80 类和 VisDrone 10 类分别实际构建模型，审计 Teacher、下游、完整系统的总/冻结/可训练参数，以及 640 输入下可覆盖的前向计算量。
4. 按确认口径选择合理的从零训练架构，公开初始化、有效层、参数偏差和配置；禁止闲置参数、故意低效结构或看过正式结果后更换有利基线。
5. 核对双方数据、优化器、更新/轮数、增强、评测、设备和计时边界。架构变化后重新做短基准并更新显存、batch 可行性和 ETA；不擅自照搬 E1 的 144-166 秒/轮或旧 Scratch 预算。
6. 将正式基线及配对合同单独提交，完成前向/loss/backward、checkpoint 和恢复测试，获得启动预算确认后才执行；现有 E1 另列为阶段性证据，不改写原始实验身份。

| 数据集 | Frozen | Scratch | seed | 完整预算 | 新运行数 |
| --- | --- | --- | --- | --- | --- |
| COCO 2017 | 最终锁定架构；当前 Teacher 计划为 ViT-S，固定融合与 AUX_STAR | 架构锁定和参数口径确认后选定的 RGB 检测器 | 0、1、2 | 暂拟各 100 epochs | 暂拟 6 |
| VisDrone2019-DET | 同一最终架构与 AUX_STAR，适配 10 类 | 按确认口径重新审计的 10 类基线 | 0、1、2 | 暂拟各 300 epochs | 暂拟 6 |

两条路径使用相同数据、seed、全局 batch、更新规则、优化器、调度、验证频率和实际硬件。
同预算在本表指相同曝光/更新上限；不同方法实际 GPU-hours 不强行设相同，否则无法观察速度差异。
另在成本曲线中比较相同 GPU-hours 下 AP，避免将“同轮数”误称“同计算量”。

主结果使用固定最后一轮的标准 AP；best 作为补充，按固定每 5 轮标准评测 AP 选择，并列选更早 checkpoint。
保留内部 Validator 的 best，但明确其选择口径不同。最终 last 与标准 best 均严格重载、独立评测。
两组都输出 AP、AP50、AP75、类别结果；COCO 额外输出 small/medium/large。
正式结论包含 3 seed 均值、样本标准差、逐 seed 明细和配对差值，不只报告最高 seed。

完整配对表至少包含：

```text
dataset, seed, model_variant, teacher_id, architecture_ref,
total_params, trainable_params, frozen_params, teacher_params, downstream_params,
parameter_match_scope, parameter_tolerance, parameter_delta, baseline_ref,
epochs, optimizer_updates, global_batch, gpu_count, imgsz,
AP_last, AP_best, AP50, precision, recall,
train_peak_allocated_bytes, train_peak_reserved_bytes, sampled_device_peak,
teacher_peak_bytes, train_val_seconds, final_eval_seconds,
extract_GPUh, train_val_GPUh, final_eval_GPUh,
cold_GPUh, amortized_GPUh, storage_bytes, retention, savings, status
```

训练峰值显存取各 rank 的最大值，不把 `nvidia-smi` 总占用当作 PyTorch allocated。
Teacher 抽取阶段峰值单列；宣称端到端峰值时取所有阶段峰值，不能隐藏大底座抽取显存。
同一个有效运行如同时满足 E4/E5 的 ViT-S 基线身份，可以引用一次，不重复训练或累加证据。

## 10. E5 P2 底座尺寸对照

本轮必做最小尺寸扫描：**ViT-S/16 对比 ViT-B/16，在 VisDrone 上完成**。
两者都选 LVD-1689M 版本；不把不同预训练数据来源混作纯尺寸变化。

| 项目 | ViT-S/16 | ViT-B/16 |
| --- | --- | --- |
| 模型 ID | facebook/dinov3-vits16-pretrain-lvd1689m | facebook/dinov3-vitb16-pretrain-lvd1689m |
| 通道 / block 数 | 384 / 12 | 768 / 12 |
| 取层 | block4/8/12 | block4/8/12 |
| 原特征 grid | 40×40 | 40×40 |
| train+val 三层 FP16 张量估算 | 24.10 GiB | 48.20 GiB |
| 数据与下游尺度 | 10 类，P3/P4/P5 通道 64/128/256 | 相同 |
| 训练 | 300 epochs，seed0/1/2，固定融合和 AUX_STAR | 相同 |

模型规格依据 [DINOv3 官方 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)
及 [官方骨干实现](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py)。
ViT-B 的本地权重可访问性尚未验证；实施前核对许可、下载来源、权重 SHA256、实际 config 和 Transformers 兼容性。
不得把“官方模型存在”写成“本机已经获得权重”。不获得合法可用权重则标记该项未完成，不静默换模型。

三次 ViT-S 正式结果可复用 E4 的 VisDrone 基线，ViT-B 新增 3 个 300 轮运行。
ViT-B 输入改为 768 通道会增加 Adapter 可训练参数，因此这是“底座规格及其必要适配”的系统对照，
不能把 AP 变化全归因于冻结 Teacher。必须同时报告 Teacher 与下游参数/计算/显存/成本。
若要声明纯底座容量效应，另补下游容量控制实验，不能用未参与前向的参数凑数。
本轮 P1 主结果固定 ViT-S；若改用 ViT-B 冲刺 P1，需要重新匹配 Scratch 并重新登记该对照。

不直接重建 ViT-B 的完整 COCO 缓存：原始张量预计约 846.54 GiB，
超过 2026-09-08 核对时该节点 NVMe 约 811 GiB 的剩余空间，且尚未算分片/转换临时副本。
后续是否扩展 COCO/ViT-L/SigLIP2，取决于本轮结果、存储和独立预算确认。

## 11. E6 成本口径与 50% 验证

### 11.1 精度与固定预算成本

```text
retention = AP_frozen / AP_scratch
AP_drop = AP_scratch - AP_frozen
saving = 1 - GPUh_frozen / GPUh_scratch
```

正式 P1 收益判定以第 9 节确认后的基线为参照。当前 E1 同可训练参数量的精度/成本曲线只支持其阶段性口径，不能自动证明课题页“同参数量”主对照已完成。
同参数量不保证同 FLOPs 或 GPU-hours，也不保证降低 50%；若正式口径下未达到目标，保留并报告负结果，不更换基线或遗漏 Teacher 成本来凑达标。

所有 AP 分母必须大于零，百分制与 0..1 不能混算。
逐 seed 计算后报告均值、标准差和所有原始值；`1-均值成本之比`可另列，但不与逐 seed saving 均值混用。
固定轮数下只有同时达到精度工作判据和 saving≥0.5，才记为该口径的收益达标。
若只快了 50% 但 AP 大幅下降，不能把它当作满足本课题的精度/成本收益。

### 11.2 冷启动、复用和墙钟

- `H_extract`：Teacher 从原图生成 train/val 缓存的实际 GPU-hours，含该阶段 GPU 参与的校验。
- `H_train_val`：训练及逐轮验证的 GPU-hours，包括数据等待；保留分阶段计时。
- `H_eval`：不包含在上一项的独立评测 GPU-hours。
- `H_cold = H_extract + H_train_val + H_eval`，这是主成本口径。
- `H_amortized(K) = H_extract/K + H_train_val + H_eval`，K 只能是实际复用该缓存的完成运行数。
- Scratch 同样记录预处理、训练、评测；不含 Teacher，但不能漏计图片预处理和等待。
- CPU 转换与磁盘搬运不虚构 GPU-hours，但必须计入端到端墙钟、CPU/RAM/磁盘成本。

GPU-hours 用每阶段实际分配 GPU 数乘时间积分，不能按利用率折算。
六卡训练时单卡验证但六卡均被该任务占用，则六卡等待也计费；
完全释放其余卡后的独立单卡评测按单卡计。不得重叠重复累计同一阶段。
数据解压、抽取、转换和迁移如有重叠，墙钟按实际起止/关键路径记录，不简单相加。

已有缓存并不让冷启动 `H_extract=0`；使用原抽取日志的实际成本。
若旧成本无法可靠恢复，冷启动项标记未知；先补足计时证据，不能宣称冷启动降低 50%。
试错、基准与失败重跑单列为总研发预算，不能从项目成本账本消失。

### 11.3 匹配精度的达标时间

固定轮数表不能把 Frozen 少跑的轮数与 Scratch 完整轮数直接相除。
先从 E4 标准评测曲线，按同一预注册算法比较达到共同 AP 目标的成本：

```text
AP_target(dataset) = 0.90 * mean(AP_last of the three E4 Scratch runs)
```

两种方法都查找首次**连续两个**标准评测点达到同一目标的位置，
两个点间隔 5 epochs；成本计到第二个点完成。不得仅挑 Frozen 的早期点而坚持让 Scratch 跑满。
未达到目标的运行标记 `target_not_reached`，不能输出“0 成本”或外推成功。
同时给出不插值的同 GPU-hours AP 曲线；无共同预算区间时不比较。

完整运行的前缀曲线只能给出潜在停止成本，不能改写实际整次运行的支出。
若至少一个数据集曲线显示有希望同时达到 retention≥90% 和冷启动 saving≥50%，
冻结 AP_target、配方和停止规则后，在该数据集启动独立 seed4/5/6 的 Frozen/Scratch 配对确认（避开 E3 已参与选参的 seed0/1/2，运行数不变）：
两组从头开始，每 5 轮同样评测，连续两个点达到目标就真正保存并退出；
未达到者运行到该数据集上限，不偷偷延长预算。

该确认阶段最多 6 次运行，必须另行确认预算。优先选曲线证据更充分、预计成本更低的数据集，
但必须披露候选选择来自 E4，而不是盲选。报告实际停止成本，而非估计少跑多少轮。
若两数据集均达不到目标，E6 输出负结果，不自动启动更多长任务追求阈值。

只有点估计达标但 3 seed 不稳定时，报告“点估计达到、统计证据不足”。
给出配对 AP 差与 saving 的均值/标准差及小样本 95% t 区间，明确 n=3 的局限；
不得将图片级 bootstrap 当成多训练 seed 的替代。

### 11.4 当前证据下的成本验证重点

E1 全部完成后，B/C/S 的 50 轮“训练加内置验证”耗时中位数为 **144.1166 / 144.7683 / 166.5368 秒**。
这是局部吞吐参考，不包含缓存抽取及全部独立评测；完整记录的训练加最终评测成本下降仅 **4.37% / 12.20%**，详见第 12.2 节，尚未达到 50%。
同样，吞吐提升倍数与耗时下降比例不是同一个数，不能混用。

早期收敛是更值得验证的成本来源：第 15 轮独立标准 AP，B 为 **27.5217**，S 为 **20.6465**。
S 现已完成，第 50 轮 AP 为 **22.7463**，标准 best 为第 30 轮的 **23.8814**；B/C 的同轮及固定末轮优势均仅限于本次阶段性参照。
待最终架构、正式基线及 E4 三 seed 结果确定后，仍使用第 11.3 节定义 AP_target，不将 E1 的 S 或临时精度直接指定为新的正式目标。
两种方法都按同一个 AP_target、连续两次达标、相同评测间隔记成本，严禁用 Frozen30 对比被强制跑满的 Scratch100 来宣称节省。
完整运行的前缀成本只标记为“潜在停止成本”，真正降低 50% 的确认仍需独立 seed 的实际停止实验。

B 在中途受到其他任务的 CPU 争用，实际支出照常计入账本；无争用窗口的单轮中位数只作为诊断和排期参考，不能直接替换已支付成本。
后续正式成本比较需先确认资源独占和 CPU 配额；如果仍有争用，保留证据并标注可比性限制，不直接归因于模型或 aux，也不擅自删除慢 epoch。

## 12. 资源、吞吐和时间预算

### 12.1 存储

缓存放显式指定的本地 NVMe 根目录，COCO 原缓存不移动、不重写。
本轮不默认访问 NFS，不删除数据原件、旧 checkpoint、原日志或其他用户目录。

**RGB 数据集也必须先复制到 NVMe，再开始后续训练。** 以运行参数 `NVME_ROOT` 指定已核实的本地 NVMe 根目录：

- COCO 副本使用 `${NVME_ROOT}/datasets/coco`，包含 train2017/val2017 图片、检测标签、所需官方标注和数据列表。
- VisDrone 使用 `${NVME_ROOT}/datasets/VisDrone`。已在目标 NVMe 上且校验通过的内容直接复用；原始标注和转换后的标签分别保存。test-dev 仍只在 E7 评测。
- 复制前核对挂载、实际剩余容量和源目录；已有目标先核验，不覆盖内容不一致的同名数据，不删除源件或调用破坏性同步。
- 复制支持断点续传与临时文件；完成后核对文件数量、相对路径、字节数及逐文件 SHA256。小型复制收据包含源/目标内容摘要，完整清单放外部工作区。
- 在新工作区生成指向 NVMe 副本的运行时数据配置和列表，不改写历史运行配置，不自动下载数据，也不重新随机划分。
- 不以目录名称含 `localssd` 代替挂载验证；解析真实路径与列表中的绝对路径，确认训练、验证和独立评测确实读取 NVMe 副本。预检未通过就停止启动。
- 迁移只改变存储位置，不解码重编码 JPEG、不调整尺寸、标签或样本顺序，不自动启用另一种图片缓存格式。

2026-09-08 实测 COCO train2017 的 118,287 张 JPEG 共 19,314,466,396 bytes，约 **17.99 GiB**；
该值不包含 val2017、标签、官方标注与临时文件。正式复制前统计完整待复制清单，将其实际空间需求加到下面的特征缓存预算中。
只记录执行约定不等于副本已经完成；本次文档变更没有启动数据复制或训练。

| 增量内容 | 不含文件头的 FP16 张量估算 |
| --- | ---: |
| VisDrone train+val，ViT-S | 24.10 GiB |
| VisDrone train+val，ViT-B | 48.20 GiB |
| VisDrone test-dev，ViT-S（最终阶段） | 5.53 GiB |

公式为 `N * 3 * C * 40 * 40 * 2 bytes`。NPY header、索引、Teacher 权重和训练产物另计。
保留源分片的转换可能临时占两份空间，按至少 **200 GiB 可用增量空间**预检；
任何完整抽取前重新检查磁盘、挂载类型、配额与其他任务，不以文档中的 811 GiB 快照代替实时检查。
若保留 ViT-B test-dev，则额外约 11.05 GiB，并单独登记。

### 12.2 时间估算

**2026-09-08 E1 最终实测：** 全部训练及最终评测已结束。训练作业列来自各组 `cost.json`；最终评测采用 `jobs/E1-*-final-*.json` 的外层时间，不以报告内部计时替代启动/加载成本。

| 组别 | 训练作业秒数 | 训练作业 GPUh | 两次最终评测秒数 | 训练加最终评测 GPUh | 相对 S 节省（训练 / 加最终评测） |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 7,894.34 | 13.1572 | 122.45 | 13.3613 | 11.78% / 11.44% |
| B | 8,551.05 | 14.2518 | 105.87 | 14.4282 | 4.44% / 4.37% |
| C | 7,842.26 | 13.0704 | 105.89 | 13.2469 | 12.36% / 12.20% |
| S | 8,948.05 | 14.9134 | 104.50 | 15.0876 | 参照 |

按本次六卡预留合同派生 GPUh：即使最终评测只使用一张卡计算，其余卡仍在该串行实验的预留范围内，不按利用率打折。
训练作业已含定期评测；加最终评测列仅额外加入 last 与 standard-best 两项，不重复累计定期评测。
四组训练加最终评测合计 **56.1240 GPUh**，对应六卡预留累计约 **9.3540 小时**；不含离线停顿、E0/短基准或 Teacher 抽取，不能当作冷启动总成本。
所有记录到的训练/恢复段和补评测段均保留；B 的外部 CPU 争用也保留，因此组间作业耗时差不能单独归因于 aux。
本次固定轮数对照没有达到 50% 成本降幅。后续排期可参考第 11.4 节的单轮中位数，但新架构、正式基线或 VisDrone 必须重新做共同短基准。

下面保留原方案制定时的历史粗估，供追溯，不作为新的实测结果或启动承诺。

已测参考：旧 ViT-S COCO100 的训练加逐轮验证为 3.947 小时；
旧 Scratch30 的 supervisor 时间为 1.396 小时，线性外推 100 轮约 4.654 小时。
二者计时口径并不完全一致，且新融合/独立评测/worker 条件可能改变耗时，只能用于排期：

| 工作 | 数量与预算 | 当前排期参考 |
| --- | --- | --- |
| E1 COCO 筛选 | 3 Frozen + 1 Scratch，各 50 轮 | 约 8.25 小时，须用 NVMe 上的新短基准更新 |
| E4 COCO 正式 | 3 Frozen + 3 Scratch，各 100 轮 | 约 25.80 小时，非实测承诺 |
| E3 VisDrone aux（2026-09-11 完成） | 27 + 9 = 36 次，各 60 轮，共 2,160 epochs | 累计67.160097 GPUh；第二阶段新增17.062548 GPUh，复用组不重复计费；MATLAB与传输另列，详见第8.7节 |
| E4 VisDrone 正式 | 6 次，各 300 轮 | 待基准 |
| E5 ViT-B | 3 次，各 300 轮，另加新抽取 | 待 ViT-B 权重和抽取/训练基准 |
| E6 停止确认 | 一个数据集最多 6 次，达到目标提前退出 | 由 E4 曲线给上界后单独确认 |

仅 COCO 两阶段的粗略排期已约 34.05 小时，不能承诺整个 P1/P2 一天完成。
E1 估算为 `3 * (3.947 * 50/100) + (1.396 * 50/30)` 小时；
Scratch 的旧 30 轮实测仅用作估时基准，不作为新 E1 的实验结果。
若触发 E3 后的 COCO50 门禁确认，额外 Frozen 运行按旧速度约 1.97 小时；
如 Scratch 参照也需重跑，另约 2.33 小时；均不包含在上述 34.05 小时内。
所有基准保留加载、读取、GPU 前向/反向、optimizer、验证各阶段时间；
至少覆盖完整 epoch 和真实盘读取，区分冷/热文件页缓存，不用驻留小 batch 微基准外推正式总时长。
Scratch 与 Frozen 都在同一 NVMe 存储条件下记录实际磁盘读量、吞吐、DataLoader 等待和 CPU 预处理时间；
复制、校验及预读造成的文件页缓存状态单列，不全局清缓存来制造冷启动条件。
JPEG 解码与特征读取本来就是两条方法的不同成本，不宣称换成同类存储后两者的数据处理开销完全相同。
ETA 包括标准评测与最终校验，按新实测计算；不并行跑多个对照争抢磁盘后比较速度。

## 13. 配置、证据与后台任务约定

### 13.1 可复现身份

每个新运行至少记录：

```text
experiment_id, code_commit, source_fingerprint, resolved_config_sha256,
dataset_protocol, split_lists_sha256, annotation_sha256,
teacher_id, teacher_weights_sha256, cache_contract_sha256, cache_content_sha256,
model_config_sha256, initial_state_sha256, seed,
world_size, per_gpu_batch, global_batch, workers, prefetch, sampler,
optimizer, effective_weight_decay, scheduler, warmup_updates, amp/scaler,
evaluation_toolkit_commit, evaluation_protocol_sha256,
storage_class, dataset_copy_receipt_sha256, storage_preflight, page_cache_state,
timing_by_phase, metrics, artifact_checksums, status, limitations
```

实验 ID 格式：`d1-{dataset}-{purpose}-{variant}-s{seed}-{UTC时间}`。
运行启动后合同不可变；resume 必须匹配模型、数据、Teacher、缓存、aux、seed、batch 和软件身份。
只给文件名加一个 seed 后缀不代表正确实现多 seed。
可重用缓存不应因无关文档 commit 失效，但实际训练源码指纹和版本必须可追溯。

### 13.2 后续新入口的接口约定

下表是待实现契约，不是当前已经存在的 CLI。实现完成后将其准确 `--help` 和实例命令写入执行记录。

| 子命令 | 输入 | 输出/约束 |
| --- | --- | --- |
| prepare | 新合同、模型、seed、数据/缓存/权重根目录、工作区 | resolved config、参数统计、数据与环境预检；不自动训练 |
| benchmark | 已通过预检的运行身份、真实数据 | 完整 epoch 分段性能和 ETA；独立基准运行，不并入正式指标 |
| train | 已通过的预检、独立 run ID、明确启动确认 | 有限状态训练、checkpoint、逐 rank 进度，异常安全停止 |
| evaluate | 固定 checkpoint、数据/评测协议、新输出目录 | 标准 AP、预测、ID 完整性、SHA256；不更新参数 |
| summarize | 明确列举的运行清单 | 配对统计、成本分解、目标状态；拒绝预算/协议不一致的混表 |

大文件统一在外部工作区：
`datasets/`、`weights/`、`feature_cache/`、`runs/`、`logs/`、`manifests/`。
Git 仅保存代码、配置、脱敏小型摘要和报告，不保存缓存、完整预测、checkpoint、访问令牌或机器私有路径。

### 13.3 超过 3 分钟的任务

预计超过 3 分钟的下载、抽取、训练、全量校验或评测，使用后台 supervisor/nohup。
主管必须写入 PID、进程启动时间、命令摘要、run ID、退出码及原子 `status.json`，
转发停止信号并在可恢复的边界保存 last checkpoint；失败不能只留下消失的 PID。
确认一次进程身份及首个有效 batch/已提交分片后结束会话，不持续轮询。
多 seed/扫描批次按用户确认的清单运行，不能从一个已批准运行自动扩大成全部矩阵。

后续 supervisor 必须支持以下状态查看方式（当前新任务尚未创建）：

```bash
: "${WORK_ROOT:?设置外部工作区}"
: "${RUN_ID:?设置已启动的实验 ID}"
cat "$WORK_ROOT/manifests/$RUN_ID/status.json"
tail -n 40 "$WORK_ROOT/logs/$RUN_ID.log"
```

以 `status.json` 的 `completed/failed/interrupted/running` 和进程启动身份联合判断。
`kill -0` 只能证明某个 PID 存在，不能证明训练成功。任何残留进程仅按本任务 run ID/PID/启动时间精准处理。
不施加内存压力回收，不全局清空其他任务页缓存，不删除源缓存换空间。

## 14. 最终验收与报告

| 项目 | 完成标准 |
| --- | --- |
| P1 基线确认 | 最终冻结架构锁定后确认总参数/可训练参数口径与容差；完整系统含 Teacher，正式 Scratch 配置和两数据集参数审计齐全 |
| P1 实验覆盖 | COCO、VisDrone 各有正式基线下的 Frozen/Scratch 三 seed 配对结果，满足已确认的参数口径/容差并统一评测；E1 阶段性对照单列 |
| P1 精度证据 | 每个数据集报告 AP 保留率和差值，不把口径差、训练轮数差或不同 seed 混算 |
| P1 成本目标 | 至少一个数据集在明示精度条件下实际降低≥50%；冷启动和摊销结论分开，不足则标记未达到 |
| P2 aux 工程 | 统一收集、scalar-once、有效梯度、DDP、AMP 恢复、EMA 和限幅日志均可验证 |
| P2 aux 研究 | 仅 VisDrone：9 组 balance/z 各三 seed，随后 4 个 gain 各三 seed，复用后 36 次；固定第 60 轮均值/标准差、gain0 配对差与路由机制分析齐全，包含负结果；不声称完成 COCO aux 独立贡献验证 |
| P2 尺寸 | ViT-S/B 在 VisDrone 的固定配方三 seed 对照、参数混杂说明及全阶段成本 |
| 复现 | 干净代码提交、配置/环境/数据/权重/cache/产物 SHA256、命令和已知局限齐全 |

最终交付新增 P1 对照报告、P2 消融报告、机器可读矩阵、精度/成本曲线和复现入口。
主表不得用未达标运行、推测 ETA、可选未来工作或部分测试替代实际结果。
若全部工程和实验执行完成但收益未达标，结论写为“实验完成，收益目标未达到”，不写“P1/P2 全部通过”。
