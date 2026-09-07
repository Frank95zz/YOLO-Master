# D1 P1/P2 后续实验实施方案

制定日期：2026-09-08。代码基线：`c44025eb36e63e4e858306b6e138a9d3c0f3aebd`。

**状态：E0 的 COCO 工程门槛已通过，E1 已获授权并进入完整 epoch 基准/启动准备；E2 及以后仍待实施。**
实际实现、验收、运行入口及结果边界见 [E0/E1 执行记录](E0_E1.md)。这不代表已经达到 P1/P2 收益目标。

已有成果见 [双周报告](BIWEEKLY_20260907.md)及其 [Issue #266](https://github.com/Tencent/YOLO-Master/issues/266)。
本方案在现有冻结特征检测闭环上继续实施，不恢复旧实验、不改写历史结果，不用内部阶段编号代替课题 P1/P2 验收。

## 1. 要求与交付对应关系

依据用户提供的《实战课题任务书（细化版，2026-08-22）》D1“目标分级”及课题页：

| 要求 | 本方案落实方式 | 必须提交的证据 |
| --- | --- | --- |
| P1：至少两个数据集的精度、显存、GPU-hours 三维对照 | 完整 COCO 2017、VisDrone2019-DET；每个数据集均设置冻结 ViT-S 与从零训练检测器 | 两个数据集的配方、参数统计、完整训练与独立评测、逐 seed 结果 |
| P1：课题页要求同参数量，细化任务书要求同预算 | 同数据集内匹配可训练参数、数据曝光量、增广、优化器、评测和硬件；另外报告同 GPU 预算曲线 | 参数匹配清单、resolved config、实际更新次数、时间轴 |
| P1：至少一个数据集训练成本降低至少 50% | 分别验证固定轮数和匹配精度下的成本，冷启动成本为主、缓存摊销另列 | 成本分解、精度保留率、达标成本及多 seed 统计 |
| P2：latent aux 注册并证明实际生效 | 复用已修复的统一收集与 scalar-once-v1，补充扫描配置的 loss/梯度验证 | 正向与反向测试、真实 batch 日志、EMA 和预算缩放记录 |
| P2：扫描 latent aux 权重 | balance/z-loss 网格、全局 gain 扫描、默认与最佳配置多 seed 复核 | 全部候选和负结果、路由/梯度分析、精度与成本表 |
| P2：扫描底座尺寸 | VisDrone 上比较 DINOv3 ViT-S/16 与 ViT-B/16，配方保持一致 | 权重与缓存合同、维度、冻结/可训练参数、抽取与训练成本 |
| 课题页可选扩展：DINOv3 对比 SigLIP2 | 不作为本轮必跑项；先完成 aux 和尺寸对照 | 如另行开展，新增 Teacher 协议与单独实验身份，不沿用 DINO 缓存 |

P1/P2 是项目内部分级，不替代官方最终评审。任务书允许有证据的负结果：
实验覆盖完成与“精度/成本目标达到”分别标记，不能因跑完实验就宣称降低 50%。

以下数值是本方案预注册判据，不是官方原文阈值：

- 可训练参数差异绝对值不超过 **1%**。
- “保留足够精度”的工作判据为 **AP 保留率至少 90%**；始终同时报告实际比例和 AP 差值。
- 正式结论使用 **seed 0、1、2**；成本停止确认使用独立 seed 3、4、5。
- 筛选实验只支持候选选择；不得将单 seed 的最佳结果当作正式多 seed 结论。

## 2. 已有基础与当前缺口

### 2.1 可以复用的内容

- `DINOv3Teacher(output_layers=(4,8,12))`：ViT-S 三层输出均为 `[B,384,40,40]`。
- `DINOFeaturePyramidAdapter`：九条独立分支；P3/P4/P5 输出通道为 64/128/256。
- `D1FoundationDetectionModel`：Adapter、三个 LatentMixture、YOLO26 Detect、CompositeCriterion。
- `value_fusion_mode` 已支持 `router_only` 和等权 `weighted_sum`；后者不是待新造的模型功能。
- 已完成的 COCO 2017 FP16 NPY 缓存、NVMe 读取、训练/评测、checkpoint 和 AMP 修复。
- 80 类 ViT-S 下游参数 3,542,567；当前 Scratch 参数 3,510,624，相差 -0.902%。
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

| 当前限制 | 后续必须完成的修改 |
| --- | --- |
| 旧正式入口固定 COCO 样本数、80 类、100 epochs、六卡 batch384 等条件 | 新建版本化 P1/P2 合同和入口，显式支持数据集、模型、seed、训练窗口和评测协议 |
| 旧 Scratch 入口固定 seed0、30 轮窗口、80 类和特定参数量 | 支持完整预算与多 seed，按实际 `nc` 重新构建并校验参数，不移除旧入口的保护 |
| `cache_features.py`、NPY 转换和部分校验存在 ViT-S 形状/COCO 假设 | 将新实验维度和 split 从新合同读取；保留旧缓存格式和旧测试兼容性 |
| `wp8-followup.yaml` 明确仅为准备配置 | 生成可执行的模型配置、完整训练合同与 run manifest，不能把它直接当训练配方 |
| VisDrone 目前只有原始数据准备 | 补标签转换、ignore 侧记录、评测适配和 ViT-S 缓存 |
| 现有 NPY 转换器为 `/root` 限定、验证后删除源分片的迁移工具 | 新增非破坏转换模式，支持显式批准的本地 NVMe 根目录；默认保留源件，不默默放宽旧入口 |
| 现有 `diagnose_wp8.py` 和部分报告偏向 COCO | 新入口复用推理逻辑，分派 COCO/VisDrone 评测器；不将 VisDrone ID 当 COCO ID |

E0 已新增独立的 COCO seed0 四组入口、NVMe 数据准备、恢复和评测，未移除旧入口保护。
上表涉及 VisDrone、多 seed、其他 Teacher 及新缓存转换模式的内容仍待对应阶段实施；
当前可运行命令以 [E0/E1 执行记录](E0_E1.md)为准，不应把后续接口设想当成已经实现。

## 3. 实验执行顺序

| 顺序 | 工作 | 输出与继续条件 |
| --- | --- | --- |
| E0 | 合同、数据接口、日志和训练入口准备 | 小样本闭环、配置拒绝规则、参数匹配和恢复测试通过 |
| E1 | 完整 COCO 的 A/B/C + Scratch，seed0，执行完整 100 轮预算的前 50%（50 轮） | 判断空间融合及 aux 开关作用，选择固定融合方式 |
| E2 | VisDrone 数据协议、评测和 ViT-S 缓存；短基准 | 10 类小样本闭环、官方评测对齐、资源与 ETA 记录 |
| E3 | VisDrone aux 系数/gain 扫描及多 seed 复核 | 锁定 aux 配置，完成 P2 辅助损失证据 |
| E4 | COCO 和 VisDrone 的正式 ViT-S/Scratch 多 seed 对照 | 每个数据集 6 个完整运行，形成 P1 三维表 |
| E5 | VisDrone ViT-B 与 ViT-S 尺寸对照 | ViT-B 新缓存和 3 个完整运行，形成 P2 尺寸表 |
| E6 | 汇总成本曲线；有希望的一个数据集做匹配精度停止确认 | 判断是否真正达到 50%，不足则明确报告 |
| E7 | 固定配方后评测 VisDrone test-dev，归档结果及复现说明 | test-dev 只作最终保留集，不反向调参 |

E2 的工程准备可先于 E1 的全部训练结束，但不能并行占用正式对照的 GPU 或制造磁盘竞争。
每个长任务单独报告预算并获得启动确认；上一任务完成不自动授权整个矩阵。

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
COCO 上选出的 gain 不直接宣称适合 VisDrone；本方案先在 VisDrone 扫描，固定最终配置再做跨数据集验证。

### 4.3 参数匹配

Scratch 复用 [当前结构](../../ultralytics/cfg/models/26/yolo26-d1-scratch-matched-n.yaml)：
Conv/C3k2 主干、SPPF/C2PSA、上采样/拼接和自底向上 neck、YOLO26 Detect；没有 Teacher、缓存输入或 LatentMixture。

每个数据集构建实际模型后计算：

```text
parameter_delta = (P_scratch_trainable - P_frozen_downstream_trainable)
                  / P_frozen_downstream_trainable
require abs(parameter_delta) <= 0.01
```

VisDrone 改为 10 类后必须重新统计，两边都不能沿用 COCO 的参数数字。
若超差，仅在训练前以深度、宽度和真实参与前向的通道做离散匹配，
选择误差最小的合法结构并提交配置；并列时选择较低 FLOPs。
禁止添加不参与计算的参数凑数，禁止看过精度后再挑 Scratch 结构。
额外记录 FLOPs、总参数、冻结 Teacher 参数、训练参数和推理时 Teacher 依赖。
同参不等于同 FLOPs，比较的是两种完整方法，不宣称只有一个网络因素不同。

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

完整 train2017/val2017，不重建现有 ViT-S 缓存；seed0，四组统一执行完整训练预算的 **50%**：
完整调度仍为 100 epochs，筛选在第 50 轮验证与保存完成后停止，不将余弦调度压缩成 50 轮。
本次比例调整只针对 E1；VisDrone 的 E3 aux 筛选仍为前 60/300 轮，E4 正式预算不变。

| ID | 模型/融合 | balance / z | latent_aux_gain | 唯一主要对照 |
| --- | --- | --- | --- | --- |
| COCO-A50 | ViT-S，router_only | 0.01 / 0.001 | 0.1 | 新代码基线 |
| COCO-B50 | ViT-S，weighted_sum，权重 [1,1,1] 归一化 | 0.01 / 0.001 | 0.1 | 与 A 比较空间融合 |
| COCO-C50 | 同 B | 0.01 / 0.001 | 0 | 与 B 比较 aux 开关 |
| COCO-S50 | 参数匹配 Scratch | 不适用 | 0 | 同预算从零训练参照 |

四组均新建运行。旧 A100 或旧 Scratch30 不直接混入此表。
固定推理语义：LatentMixture 训练和评测都使用全部 4 个专家，不启用 top-k 剪枝/稀疏回退。

筛选依据为第 50 轮独立 COCO AP；每 5 轮保留曲线，窗口内 best 仅作补充，不替代固定轮数的主筛选指标。
50 轮覆盖旧冻结运行最佳第 42 轮附近的观察区间，但不保证新配方也会在此前达到最佳值。
出现 NaN、数据错位、丢样本、静默跳更新则工程失败，
修复后所有受影响组重新开始，不能仅补跑赢家。
AP 差不超过 0.5 个百分点视为筛选并列，优先实际训练成本更低者；仍并列保留 A。
最佳 Frozen 若低于同轮 Scratch AP 的 80%，不立即投入整套 COCO 多 seed 长训练，
先完成 VisDrone/aux 筛选并分析空间梯度；若仍无改善，提交负结果及新方案，不无上限堆预算。
该 80% 是节省探索预算的触发线，不是 P1 成功标准。

A/B 的机制解释至少包括：block4/8/12 各分支的检测梯度、融合前后特征幅值、
各尺度 Router 概率、专家负载和 residual gain；仅“B 比 A 高几点”不作为完整机制结论。
不在同一个 A/B 对照里同时更改 aux、专家数、输入尺度、优化器或训练时长。

## 7. E2 VisDrone 数据和评测协议

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

## 8. E3 P2 辅助损失扫描

### 8.1 固定设置与损失解释

先在 VisDrone 完整 train/val 执行，固定 E1 选出的融合模式、ViT-S、batch96 和 300 轮调度；
筛选窗口为前 60 轮。每个候选 seed0，完全相同初始化；不改变 `mixture_aux_budget=3.0`。

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

### 8.2 候选矩阵

| 扫描 | 固定内容 | 候选 | 运行数 |
| --- | --- | --- | --- |
| balance/z 网格 | gain=0.1，budget=3.0 | balance ∈ {0,0.01,0.1}；z ∈ {0,0.001,0.01}，完整笛卡尔积 | 9 |
| 全局 gain | 网格中 AP 最高的非零系数组合；budget=3.0 | gain ∈ {0,0.03,0.1,0.3} | 新增 3，gain0.1 复用同身份网格结果 |
| 独立 seed 复核 | 最终候选与默认配置 | seed1、seed2，同为 60/300 轮 | 最多 4 |

总计最多 **16 个 60 轮 VisDrone 筛选/复核运行**。
全零系数与 gain0 的数值路径可能不同，不把它们未经验证合并为同一实验。
如果所有非零 aux 都劣于关闭 aux，允许 gain0 成为最终配置，仍报告完整扫描和负结果。
默认配置是 balance0.01/z0.001/gain0.1。候选并列规则同 E1；
若默认就是最终候选，不重复做相同配置、同 seed、同代码/初始化身份的运行。

每 epoch 记录三个尺度各自的 raw balance/z、加权项、EMA 分母、预算缩放、effective_aux、
aux/native loss 比、Router 概率熵/负载、residual gain、检测梯度与 aux 梯度范数。
主损失为向量时单测必须证明 aux 只贡献一次；日志中的 raw aux 与最终加入总 loss 的值分别命名。
新鲜运行的 aux EMA 初始化相同；resume 保留其状态。监控补丁不得改变训练计算图。
若两个权重设置因归一化/限幅实际等价，应报告这一机制，不强行声称权重越大正则越强。

选择配置后锁定为 `AUX_STAR`。该配置用于 E4 两个数据集和 E5 两种底座，不再为各组偷偷调参。
后续发现 COCO 不适配时，新配置需独立登记，不能将新的 COCO 设置与旧 VisDrone 设置拼成单配方结论。

若 E1 触发了 80% 暂停线，E4 的 COCO 部分保持待定：
最多增加一次采用固定融合和 `AUX_STAR` 的 COCO50、seed0 确认，同样保留 100 轮调度，
与 E1 Scratch50 比较；仅当代码、初始化规则、数据和预算身份仍完全一致时复用该 Scratch 参照。
如参照身份已改变，则先登记并确认一对重跑预算，不能混用结果。
仍低于 80% 就暂停 COCO 正式长训练并报告该覆盖项未完成；不能在没有解除条件时继续 E4。
若配置和身份与 E1 某组完全相同，直接引用该结果，不重复运行。

## 9. E4 P1 两个数据集的正式同参对照

| 数据集 | Frozen | Scratch | seed | 完整预算 | 新运行数 |
| --- | --- | --- | --- | --- | --- |
| COCO 2017 | ViT-S，固定融合与 AUX_STAR | 同参 RGB 检测器 | 0、1、2 | 各 100 epochs | 6 |
| VisDrone2019-DET | ViT-S，固定融合与 AUX_STAR，10 类 | 重新匹配参数的 10 类检测器 | 0、1、2 | 各 300 epochs | 6 |

两条路径使用相同数据、seed、全局 batch、更新规则、优化器、调度、验证频率和实际硬件。
同预算在本表指相同曝光/更新上限；不同方法实际 GPU-hours 不强行设相同，否则无法观察速度差异。
另在成本曲线中比较相同 GPU-hours 下 AP，避免将“同轮数”误称“同计算量”。

主结果使用固定最后一轮的标准 AP；best 作为补充，按固定每 5 轮标准评测 AP 选择，并列选更早 checkpoint。
保留内部 Validator 的 best，但明确其选择口径不同。最终 last 与标准 best 均严格重载、独立评测。
两组都输出 AP、AP50、AP75、类别结果；COCO 额外输出 small/medium/large。
正式结论包含 3 seed 均值、样本标准差、逐 seed 明细和配对差值，不只报告最高 seed。

完整配对表至少包含：

```text
dataset, seed, model_variant, teacher_id, trainable_params, frozen_params,
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
冻结 AP_target、配方和停止规则后，在该数据集启动独立 seed3/4/5 的 Frozen/Scratch 配对确认：
两组从头开始，每 5 轮同样评测，连续两个点达到目标就真正保存并退出；
未达到者运行到该数据集上限，不偷偷延长预算。

该确认阶段最多 6 次运行，必须另行确认预算。优先选曲线证据更充分、预计成本更低的数据集，
但必须披露候选选择来自 E4，而不是盲选。报告实际停止成本，而非估计少跑多少轮。
若两数据集均达不到目标，E6 输出负结果，不自动启动更多长任务追求阈值。

只有点估计达标但 3 seed 不稳定时，报告“点估计达到、统计证据不足”。
给出配对 AP 差与 saving 的均值/标准差及小样本 95% t 区间，明确 n=3 的局限；
不得将图片级 bootstrap 当成多训练 seed 的替代。

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

已测参考：旧 ViT-S COCO100 的训练加逐轮验证为 3.947 小时；
旧 Scratch30 的 supervisor 时间为 1.396 小时，线性外推 100 轮约 4.654 小时。
二者计时口径并不完全一致，且新融合/独立评测/worker 条件可能改变耗时，只能用于排期：

| 工作 | 数量与预算 | 当前排期参考 |
| --- | --- | --- |
| E1 COCO 筛选 | 3 Frozen + 1 Scratch，各 50 轮 | 约 8.25 小时，须用 NVMe 上的新短基准更新 |
| E4 COCO 正式 | 3 Frozen + 3 Scratch，各 100 轮 | 约 25.80 小时，非实测承诺 |
| E3 VisDrone aux | 最多 16 次，各 60 轮 | 待 VisDrone 完整 epoch 基准，不照搬 COCO 每轮时间 |
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
| P1 实验覆盖 | COCO、VisDrone 各有完整的 Frozen/Scratch 三 seed 配对结果，参数差≤1%，统一评测 |
| P1 精度证据 | 每个数据集报告 AP 保留率和差值，不把口径差、训练轮数差或不同 seed 混算 |
| P1 成本目标 | 至少一个数据集在明示精度条件下实际降低≥50%；冷启动和摊销结论分开，不足则标记未达到 |
| P2 aux 工程 | 统一收集、scalar-once、有效梯度、DDP、AMP 恢复、EMA 和限幅日志均可验证 |
| P2 aux 研究 | 完整网格/gain 扫描、默认/候选多 seed 复核与路由机制分析，包含负结果 |
| P2 尺寸 | ViT-S/B 在 VisDrone 的固定配方三 seed 对照、参数混杂说明及全阶段成本 |
| 复现 | 干净代码提交、配置/环境/数据/权重/cache/产物 SHA256、命令和已知局限齐全 |

最终交付新增 P1 对照报告、P2 消融报告、机器可读矩阵、精度/成本曲线和复现入口。
主表不得用未达标运行、推测 ETA、可选未来工作或部分测试替代实际结果。
若全部工程和实验执行完成但收益未达标，结论写为“实验完成，收益目标未达到”，不写“P1/P2 全部通过”。
