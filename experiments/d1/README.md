# D1：冻结 DINOv3 的缓存特征检测

研究问题：冻结 DINOv3，只训练多尺度适配器、LatentMixture 和检测头，能保留多少检测精度，能否降低训练成本？

本 README 是 D1 的统一入口，包含功能、复现步骤、研究结果、测试、局限与许可来源。`manifests/` 只保留程序直接读取的数据划分、Teacher 身份和预处理合同，不包含个人实验队列或阶段流水文档。

- [架构与边界](#架构与边界)
- [安装与输入](#安装与输入)
- [缓存](#缓存)
- [训练与独立评测](#训练与独立评测)
- [最终配对实验](#最终配对实验)
- [研究结果](#研究结果)
- [验收结论与局限](#验收结论与局限)
- [测试与复现身份](#测试与复现身份)
- [数据与模型许可](#数据与模型许可)

## 架构与边界

```text
RGB -> 固定 640 LetterBox -> 冻结 DINOv3 ViT-S/16
    -> block4 / block8 / block12：均为 [B,384,40,40]
    -> 三层分别通过 P3/P4/P5 Adapter，共九条分支
    -> 各尺度 LatentMixture -> YOLO26 Detect
```

- 三个 DINO block 本身均为 stride16，不是天然的 P3/P4/P5。
- Adapter 输出通道为 64/128/256，网格为 80/40/20。保留 BASE、DW、BN64 三个 P5 结构。
- BN64 是 384->64->256 的瓶颈适配，不是 BatchNorm。
- Teacher 默认仍输出 dense["p4"]；多层模式需显式 output_layers=(4,8,12)。
- Teacher 不进入 student optimizer、DDP、EMA 或 checkpoint。缓存特征仍须与原始图像和标签严格对应。
- 标量总损失为 detection.sum()+aux；latent 显式参与统一 aux 收集，raw/effective aux 单独记录。
- 默认库行为不变；示例入口显式启用 weighted_sum、可分离双线性 P3 和 foreach EMA。
- 固定缓存不支持任意颜色、几何、Mosaic 或随机多尺度增强。

## 安装与输入

D1 实测 Python 3.11；Foundation 可选依赖要求 Python >=3.10。缓存构建入口面向 Linux，单进程独占一个输出目录。

```bash
pip install -e ".[dev,foundation]"
# 仅 COCO 标准评分需要：
pip install faster-coco-eval==1.8.0
export D1_WORK=/path/to/external/d1-work
export COCO_ROOT="$D1_WORK/datasets/coco"
export TEACHER_DIR="$D1_WORK/weights/dinov3-vits16"
```

所有图片、权重、缓存和运行输出放在仓库外。不要把未信任来源的 pickle checkpoint 传给评测入口。

先使用现有下载工具准备输入，D1 不再维护自己的下载、解压和环境采集器：

- Teacher：从 [ModelScope 模型页](https://www.modelscope.cn/models/facebook/dinov3-vits16-pretrain-lvd1689m) 选择版本 2e601320d0545509ab03374e2f8707f303e1de7a，取得 config.json、model.safetensors、LICENSE.md、README.md，放入 TEACHER_DIR。每个文件的大小、SHA256 和来源版本记录在下方 Teacher manifest 中。
- COCO：下载 train2017.zip、val2017.zip、annotations_trainval2017.zip、coco2017labels.zip。每个压缩包的官方 URL、已验证镜像 URL、大小和 SHA256 均在 [数据划分 manifest](manifests/coco2017-splits.json) 的 archives 中。
- train/val 图片解压到 COCO_ROOT/images，官方标注解压到 COCO_ROOT/annotations；labels 压缩包包含 coco/labels，应解压到 D1_WORK/datasets。可保留压缩包供额外校验，但不是缓存读取的运行依赖。

下载完成应先核对 manifest 中的 SHA256 再解压，不以文件大小或“下载成功”代替完整性校验。

准备完毕后运行校验，生成两份完整、排序稳定的图片列表：

```bash
python -m scripts.d1.prepare_coco --coco-root "$COCO_ROOT" \
  --weights-dir "$TEACHER_DIR" --output "$D1_WORK/inputs"
```

它验证图片数量与列表 SHA256、train/val 互斥、标签数量与图片归属、官方标注文件存在性、Teacher 文件大小/SHA256/架构，并用本地权重构造 Teacher。只有提供 --archives-dir 时才验证源压缩包，报告会明确标记是否做过该项；不把文件存在性说成标注内容逐字节验证。生成结果写入外部 inputs，不覆盖仓库 manifest。

只需恢复列表时可以运行：

```bash
python -m scripts.d1.prepare_coco --coco-root "$COCO_ROOT" \
  --lists-only --output "$D1_WORK/inputs"
```

COCO 列表数量固定为118,287/5,000，不进行重新随机划分。重跑时输出文件必须一致，否则报错，不覆盖已有证据。旧 --download/--workspace 参数已移出当前入口。

来源与固定校验信息：[预处理合同](manifests/experiment-contract.json)、[Teacher](manifests/dinov3-vits16.json)、[数据划分](manifests/coco2017-splits.json)、[许可来源](#数据与模型许可)。三个 JSON 是准备和抽取脚本的校验输入，不以 Markdown 表格替代。COCO 固定输入配方见 [dinov3-vits16-coco2017.yaml](../../ultralytics/cfg/experiments/d1/dinov3-vits16-coco2017.yaml)。

公共文件按功能命名：`prepare_coco.py` 校验 COCO 和 Teacher 输入，`prepare_visdrone.py` 准备 VisDrone 标注，`cache_features.py` 提供缓存命令，`convert_npy.py` 实现无损 NPY 转换。合同 JSON 的历史 `schema_version` 保持不变，避免仅改名就改变已有校验依据；模型文件中的 P3/P4/P5 表示特征金字塔尺度，不是工作阶段。

## 缓存

```bash
python -m scripts.d1.cache_features build --data-root "$COCO_ROOT" --weights-dir "$TEACHER_DIR" \
  --samples-file "$D1_WORK/inputs/coco2017-train2017.txt" \
  --cache-dir "$D1_WORK/cache/train2017" --split train2017 --batch-size 16 --device 0
python -m scripts.d1.cache_features build --data-root "$COCO_ROOT" --weights-dir "$TEACHER_DIR" \
  --samples-file "$D1_WORK/inputs/coco2017-val2017.txt" \
  --cache-dir "$D1_WORK/cache/val2017" --split val2017 --batch-size 16 --device 0
python -m scripts.d1.cache_features verify --cache-dir "$D1_WORK/cache/train2017"

# 可选：无损转为一图一个 NPY，保留所有源分片：
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/train2017" --output "$D1_WORK/npy"
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/val2017" --output "$D1_WORK/npy"
```

工程小样本可在 build 中传 --limit，但对应训练 YAML 必须只列出这些已缓存图片；不能用 100 图缓存搭配完整 train2017 列表。读缓存支持 safetensors 和 NPY，不做有损量化。

图片列表必须按字典序排序、无重复，且每行是 images/SPLIT/ID.jpg。自定义列表代表显式子集，不自动等同于完整官方数据。build 固定 seed0、确定性算法和 TF32 关闭，记录图片摘要、batch、设备和依赖版本；同目录并发写入、运行身份变化、遗留 .part 或多进程 torchrun 启动都会报错。中断后按原 batch 重放，不把剩余图片重新组 batch；已提交成员的 FP16 特征必须逐元素一致。

同一代码版本下保留相同命令即可续跑该入口生成的缓存。抽取器源码 SHA256 属于 build.json 身份的一部分；本次入口命名迁移也会改变该摘要。已完成缓存仍可读取、校验和转换，但旧版本的未完成缓存应在原提交下续写，不修改 build.json 绕过身份校验。缺少 build.json 的更早缓存同样须用原提取器续写，不能静默改变 batch。独立的六卡调度器保留在研究归档，不进入本 PR。可选 --benchmark-read 会额外完整读取一次缓存，其吞吐不是训练吞吐。

VisDrone 使用已下载并解压的官方 DET train/val（6,471/548 张），源目录下应有 VisDrone2019-DET-train 与 VisDrone2019-DET-val，分别包含 images/annotations。转换标签时保留 ignore 原始信息，原始数据不删除：

```bash
python -m scripts.d1.prepare_visdrone \
  --source "$D1_WORK/visdrone-original" --output "$D1_WORK/visdrone-prepared"
python -m scripts.d1.cache_features build --data-root "$D1_WORK/visdrone-prepared" \
  --weights-dir "$TEACHER_DIR" --samples-file "$D1_WORK/visdrone-prepared/train.txt" \
  --split visdrone-train --cache-dir "$D1_WORK/cache/visdrone-train" --batch-size 16 --device 0
python -m scripts.d1.cache_features build --data-root "$D1_WORK/visdrone-prepared" \
  --weights-dir "$TEACHER_DIR" --samples-file "$D1_WORK/visdrone-prepared/val.txt" \
  --split visdrone-val --cache-dir "$D1_WORK/cache/visdrone-val" --batch-size 16 --device 0
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/visdrone-train" --output "$D1_WORK/npy"
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/visdrone-val" --output "$D1_WORK/npy"
```

prepare_visdrone 生成 dataset.yaml、train.txt、val.txt 和标注 sidecar；默认只要求 train/val。可选 --include-test-dev 准备额外保留集，但缓存训练入口不会将其混入 train/val。标签准备仍用硬链接保留原图，原始数据与 prepared 目录必须位于同一文件系统；缓存构建不再硬编码200 GiB空闲空间或六张GPU条件。请自行核对容量，训练与缓存优先使用本地高速存储。

## 训练与独立评测

新入口不依赖 E1/P5/E3 矩阵。默认示例配置为 [cached-detection.yaml](../../ultralytics/cfg/experiments/d1/cached-detection.yaml)，CLI 的 batch/epochs/workers/seed 明确覆盖运行规模。它是功能复现配方，不是对历史全部实验的逐位重放。

```bash
python -m scripts.d1.train inspect --variant BN64
python -m scripts.d1.train train --approved --variant BN64 \
  --data /path/to/coco-local.yaml \
  --train-cache "$D1_WORK/npy/train2017" --val-cache "$D1_WORK/npy/val2017" \
  --output "$D1_WORK/runs/example" --device 0 --batch 16 --epochs 100 --workers 4

python -m scripts.d1.train evaluate \
  --data /path/to/coco-local.yaml --val-cache "$D1_WORK/npy/val2017" \
  --checkpoint "$D1_WORK/runs/example/weights/best.pt" \
  --output "$D1_WORK/evaluations/example-best" --device 0 --batch 16 \
  --annotations /path/to/coco/annotations/instances_val2017.json
```

coco-local.yaml 按仓库常规检测 YAML 指定本地 path、train、val、80 类 names。VisDrone 使用 10 类 YAML 和 --dataset visdrone；checkpoint 的类别数须匹配。输出目录必须不存在，防止覆盖已有实验。

多卡必须由外部 torchrun 启动，例如 torchrun --standalone --nproc_per_node=6 -m scripts.d1.train train，并提供完整 --device 0,1,2,3,4,5 和可整除的全局 batch。此次 PR 验收没有运行真实六卡训练，启动正式实验前仍需独立门禁。

--ema scalar-v1/foreach-v1 和 --p3-upsample bilinear/separable_bilinear2x 控制实现；--fp32 用于 CPU 或精度对照。默认逐样本验证缓存，只有已经单独完成全量校验时才使用 --trusted-cache。

评测输出 evaluation.json 与 predictions.json，记录 checkpoint SHA256、epoch、数据合同、完整图像覆盖和内部指标。COCO 只有显式提供 --annotations 时才报告标准 AP，maxDets=[1,10,100]；预测导出最多 300 框，不改变标准 AP 的 maxDets=100。VisDrone 输出官方格式 TXT（最多500框）；裁剪或坐标舍入后宽/高为零的框被移除并计数，非有限值报错。官方 ignore 规则评分使用固定 MATLAB toolkit，不能用内部 COCO-style 指标代替。导出检查入口为 scripts.d1.evaluate_visdrone；本 PR 不包含原机器上的 MATLAB 任务调度器。

## 最终配对实验

最终候选固定为 ViT-S/16 + BN64 + weighted_sum，P5 的瓶颈宽度为64；balance=0.1、z=0、gain=0.1、budget=3.0。Scratch 从随机权重训练标准 YOLO26-L 拓扑的宽度匹配版：depth=1.0、width=0.9375、max_channels=512。两组均重新初始化，不续训候选筛选权重。

集中合同为 [paired-comparison.yaml](../../ultralytics/cfg/experiments/d1/paired-comparison.yaml)，运行入口为 [compare.py](../../scripts/d1/compare.py)。

| 数据集 | train / val | 两组各自预算 | seeds | 每卡 / 全局 batch | 总运行数 |
|---|---:|---:|---|---:|---:|
| COCO 2017 | 118,287 / 5,000 | 100 epochs | 0/1/2 | 64 / 384 | 6 |
| VisDrone2019-DET | 6,471 / 548 | 300 epochs | 0/1/2 | 16 / 96 | 6 |

每个运行独占六张A40，组间串行。两组输入几何固定640方形单次LetterBox，禁用增强、多尺度、矩形验证和RGB内存缓存；两组的RGB/NPY均放NVMe。优化器使用仓库共享AdamW参数分组，lr0=0.001、lrf=0.01、momentum=0.9、weight_decay=0.0005、cosine、warmup3。Router保留共享实现的半学习率分组，不另改优化器。nbs等于全局batch，每batch一次有效更新。AMP初始scale16、growth_interval=1000000，workers4/rank、prefetch1。Frozen使用foreach EMA和可分离P3；Scratch使用原生EMA，分别记录实现成本。

总参数统计包含21,596,544个冻结Teacher参数：COCO Frozen/Scratch为23,001,383/23,133,560；VisDrone为22,936,803/23,032,340。1%为项目工程匹配容差，不声称是官方规定。两个模型计算路径不同，本实验是系统对照而不是单因素同架构消融。

启动门禁先执行真实六卡连续两轮与一轮后恢复到两轮的状态比较，保留100/300轮调度；随后每种数据集/架构测五个完整epoch，使用warmup后的第4/5轮估时，并覆盖第5轮周期checkpoint。任何缺失样本、非有限loss/梯度、漏更新、恢复不一致或OOM都会停止，不能自动缩小单组batch。数据加载按明确epoch边界创建迭代器，避免无限预取跨轮改变续跑样本顺序；这是新正式运行的版本化执行合同，不声称逐位重放历史研究队列。普通训练仍可不启用该测量模式。

```bash
python -m scripts.d1.compare --approved --output "$D1_WORK/final-comparison" \
  --coco-root "$COCO_ROOT" --coco-cache "$D1_WORK/coco-npy" \
  --visdrone-root "$D1_WORK/datasets/visdrone-prepared" \
  --visdrone-cache "$D1_WORK/visdrone-npy" --device 0,1,2,3,4,5
```

Linux入口要求新的外部输出目录、干净代码提交和明确批准。完整数据及缓存不下载、不重新抽取、不删除。工作目录写入plan.json、status.json、各任务日志、门禁状态和eta.json；ETA按新基准更新，不能用旧小型Scratch的速度代替。实际数据根路径只写入外部运行身份，不进入Git。

单次测量训练使用train入口的--telemetry；--window只限制已执行轮数，不缩短学习率调度。--resume-snapshot指向同一运行的resume.pt，恢复FP32模型、EMA、优化器、scaler、scheduler、criterion及各rank状态；必须保持提交、模型、数据、batch、seed等身份一致。普通last.pt用于独立评测，不用其FP16序列化代替精确训练恢复。

主结果固定第100/300轮；每5轮保留checkpoint，用同一标准协议补评分选standard-best，平局取更早者。COCO采用maxDets100标准AP；VisDrone采用官方MATLAB DET评分，不能把内部指标当作官方结果。报告三个seed的逐项结果、均值、样本标准差和配对差值。GPU-hours包括分配GPU的等待；冷启动计入已有日志中的Teacher抽取，训练-only和实际复用摊销另列。若旧抽取成本无法可靠恢复则标记未知，不宣称冷启动降低50%。

正式实验结果尚待上述门禁和完整运行，不因启动队列就提前标记P1通过；不自动追加其他Teacher或新的消融。

2026-09-11 启动准备回归：D1、Foundation、LatentMixture、损失组合、checkpoint与DDP相关测试共623 passed、56 skipped、2 deselected，耗时68.17秒。两项deselected沿用下方已确认的上游问题排除记录。新增覆盖原生Scratch严格重载、有限epoch采样与恢复、统一FP32评测checkpoint、周期文件重复发布和残留临时文件恢复。静态检查、格式、编译和git diff --check通过；这仍是CPU/离线回归，不冒充真实六卡门禁或正式实验结果。

## 研究结果

### COCO 架构筛选

固定 COCO 2017 train2017/val2017（118,287/5,000 张）、seed0、50轮窗口，在相同研究配方下对比 P5 分支。下表是固定第50轮 checkpoint 的独立 COCO 标准评测，AP 按 0-100 点显示，不是内部 mAP，也不是每组最优 epoch。

| P5 结构 | COCO 下游参数 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE：3x3 stride2 Conv | 3,542,567 | 28.870 | 49.197 | 30.113 | 12.921 | 33.463 | 40.228 |
| DW：深度可分离分支 | 1,195,943 | 28.593 | 48.542 | 29.723 | 13.033 | 32.510 | 41.559 |
| BN64：64通道瓶颈 | 1,404,839 | 29.745 | 49.284 | 31.203 | 12.965 | 33.086 | 42.992 |

BN64 被选作后续基座，归一化仍为 GroupNorm。单 seed 结果仅支持本轮候选筛选，不构成统计等效或稳定提升证明。

完整配方、成本拆分、曲线、checkpoint 身份和无 FinsSim 干扰的正常 epoch 计时见固定版本[架构筛选报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/P5_FAST_RUN_20260909.md)与[机器可读汇总](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/p5-screen-20260909/suite-summary.json)。

### VisDrone Latent Aux 消融

VisDrone2019-DET 完整 train/val 为6,471/548张，使用 BN64、weighted_sum、三个独立种子0/1/2、固定300轮学习率调度的前60轮。主结果统一使用第60轮 checkpoint 和固定官方 MATLAB DET 工具，保留官方 ignore 语义。

第一阶段扫描 balance={0,0.01,0.1} 与 z={0,0.001,0.01}，gain=0.1，共27次训练；第二阶段固定 balance=0.1、z=0，扫描 gain，新增9次并复用3次。共36个独立运行，不把复用结果重复计数。

| latent_aux_gain | 官方 AP 均值（3 seeds） |
|---|---:|
| 0 | 8.022406397 |
| 0.03 | 8.115370260 |
| 0.1 | 8.131928488 |
| 0.3 | 8.081231746 |

按预先确定的均值优先规则，候选为 balance=0.1、z=0、gain=0.1、budget=3.0。gain=0.1 相对关闭aux平均仅高0.109522点，三 seed 配对差为 -0.245386/+0.339334/+0.234618点；探索性95%区间跨0。相同三个种子还参与了选参，未做独立确认，**不能宣称稳定或显著收益**。gain=0.03 是波动较小的备选。

每5轮导出的预测并未全部完成官方评分，因此这不是完整 standard-best 曲线，也不能证明300轮已经收敛。完整指标、配对差、工具和 checkpoint SHA256 见[消融报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/E3.md)、[第一阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage1-official-20260911.json)、[第二阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage2-official-20260911.json)。

## 验收结论与局限

| 项目 | 当前结论 |
|---|---|
| P0 可运行闭环 | 冻结 Teacher、缓存、Adapter、LatentMixture、Detect、训练和独立评测已实现；区分历史真实实验与本分支合成回归 |
| P1 对照 | 已覆盖 COCO 与 VisDrone 的冻结模型研究；最终同总参数量 scratch 对照尚未完成，不宣称 P1 完成 |
| P1 GPU-hours 至少下降50% | 尚无满足最终公平对照合同的证据，不能认定达标 |
| P2 辅助损失研究 | 已显式注册 latent aux，完成 balance/z/gain 的三 seed 消融；收益不稳定也是结果 |
| 其他 P2 扩展 | 尚未完成其他 Teacher 对比；不把已有缓存工程误称为新增精度收益 |

按当前约定，总参数统计包含冻结 Teacher：ViT-S/16 为21,596,544参数；加 BN64 后 COCO/VisDrone 总量为23,001,383/22,936,803。保留的 [scratch-total-l 配置](../../ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml)分别为23,133,560/23,032,340参数，差约+0.57%/+0.42%；这是待正式验证的对照配置，不是已训练的最终基线。

最终比较必须固定数据划分、分辨率、种子、评测与 checkpoint 选择，分别报告总参数/可训练参数、精度保留率、峰值显存、GPU-hours；同时给出含一次特征抽取与不含抽取的成本，声明缓存复用次数。不能把缓存读取吞吐或短窗口筛选直接作为最终降本结论。

P3 可分离双线性实现与 foreach EMA 为显式选项，保留原实现作数值对照；它们不改变模型结构。NPY 与 safetensors 存储相同 FP16 特征，不引入额外有损量化。实际吞吐依赖磁盘、缓存热度、CPU 和任务干扰，不能把更换存储介质的收益全部归为模型降本。

## 测试与复现身份

测试按功能组织，不再按个人工作阶段拆文件；公共 Teacher 测试统一放在已有的 `test_foundation_dinov3.py`。

| 测试文件 | 覆盖内容 |
|---|---|
| [test_d1_contracts.py](../../tests/test_d1_contracts.py) | 数据划分、预处理、Teacher 身份、scratch 总参数匹配 |
| [test_d1_cache.py](../../tests/test_d1_cache.py) | safetensors/NPY、校验、恢复、来源保留、特征读取 |
| [test_d1_cache_cli.py](../../tests/test_d1_cache_cli.py) | 统一抽取入口、确定性、原 batch 续跑、并发拒绝 |
| [test_d1_adapter.py](../../tests/test_d1_adapter.py) | 九条适配分支、梯度、P3 上采样数值与梯度等价 |
| [test_d1_model.py](../../tests/test_d1_model.py) | 检测模型、checkpoint、显式 latent aux 与标量损失 |
| [test_d1_pipeline.py](../../tests/test_d1_pipeline.py) | Dataset/Trainer/Validator、坐标还原、最终评测与各 rank 行为 |
| [test_d1_training.py](../../tests/test_d1_training.py) | 训练入口、合成闭环、独立评测与 EMA 等价 |
| [test_d1_visdrone.py](../../tests/test_d1_visdrone.py) | 标注转换、ignore 信息、划分隔离、官方导出协议 |
| [test_foundation_dinov3.py](../../tests/test_foundation_dinov3.py) | 默认及多层 Teacher 协议、冻结约束、可选真实权重 |

相关回归命令（Bash）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q tests/test_d1_*.py \
  tests/test_foundation_{dinov3,teacher_protocol,distill_model,checkpoint}.py \
  tests/test_foundation_{cache_training,config,losses,taps,weight_schedule,projectors}.py \
  tests/test_foundation_{mixture_interaction,routing_contract,offline}.py \
  tests/test_latent_mixture.py tests/test_mixture_loss_composition.py \
  tests/test_ddp_lifecycle_ema_nan.py tests/test_prevalidation_recovery.py \
  tests/test_optimizer_group_audit.py tests/test_checkpoint_compat.py \
  tests/test_ddp_checkpoint_coordination.py tests/test_default_config_integrity.py \
  tests/test_master_model_configs.py tests/test_engine.py::test_load_checkpoint_state_dict_rejected \
  --deselect tests/test_foundation_cache_training.py::test_response_kd_builds_pseudo_batch_from_cached_responses \
  --deselect tests/test_foundation_config.py::test_enabled_without_teacher_is_rejected
git diff --check
```

真实 Teacher/CUDA 测试按显式环境变量启用；普通 CI 不下载模型。新入口测试使用合成图像和合成特征，完成一轮 optimizer 更新、保存、严格重载和独立评测，不产生真实数据集精度结论。

相关回归的两项排除项已在同一上游提交复现，不计为通过；没有运行完整仓库测试集。真实集成使用 `D1_DINOV3_WEIGHTS`、`D1_WP2_CACHE`（数据管线测试还需 `D1_COCO_ROOT`）、`D1_NPY_CACHE` 指定外部输入；保留这些环境变量以兼容既有使用方式，启用 CUDA 验收时不要清空 `CUDA_VISIBLE_DEVICES`。

本分支基于 upstream af961b99b8ef80491e58cb5fd16e25ebaf3741eb；研究归档固定在 [f4d2bc2](https://github.com/Frank95zz/YOLO-Master/tree/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1)。核心实现与研究版本的语义对齐已通过去除位置属性的 Python AST 比较，不能据此声称所有文件逐字节相同。历史运行必须使用报告记录的执行提交。个人队列的每 rank RNG/buffer 快照、严格重试、阶段筛选和周期官方评测策略未迁入新入口；此 CLI 只启动新 run，不提供旧研究 run 的精确 resume。

后续正式实验继续使用获确认的研究合同；需记录执行提交，并核对与 PR 的核心实现一致，不能把默认示例参数冒充正式对照配方。

PR 不包含旧实验队列、内存回收、迁移删除脚本、巨型路径列表、权重、数据集、缓存、完整预测或逐阶段流水报告。研究分支保留全部原始工作与证据；这些内容没有被删除或覆写。

代码验收提交为 [0842deb](https://github.com/Frank95zz/YOLO-Master/commit/0842deb98404f46d0dd40f6aa21d6e90b8280c66)：**556 passed、56 skipped、2 deselected**，pytest 耗时53.05秒。命名迁移前后的612个有效用例逐项对应，无遗漏或额外重复；11个相关 Python 文件经命名归一化后语法树等价，合同 JSON 和 NPY 转换实现逐字节不变，YAML 参数和抽取器生成的缓存合同均与改名前一致。此次只调整文件名、导入、引用和说明，不修改模型结构、训练参数或数据协议。

11个相关 Python 文件通过本机 Ruff 检查、格式及编译检查；远端 py_compile、六组命令入口检查、相对链接、旧文件引用检查及 git diff --check 通过。服务器未安装 Ruff，未为此更改环境；本轮未执行 codespell。没有运行真实权重/CUDA 集成或正式实验。完整日志保存在外部工作区，SHA256 为 `8f59b0c9b7161d4cbd84acea0f2cd0e3fe04d6afd50e873e6b53d9a5b21c86cb`，不增加过程性 JSON 到 PR。

## 数据与模型许可

本节只记录上游许可来源，不替代上游条款，也不作额外法律判断。

### COCO 2017

- [官方 Terms of Use](https://cocodataset.org/#termsofuse)；[官方下载源](http://images.cocodataset.org/)。
- COCO 不为全部源图片提供统一的总括许可；图片仍受各自原始 Flickr 许可约束，应遵循官方条款和逐图片许可元数据。
- YOLO detection labels 来自 Ultralytics 的 COCO 2017 labels 压缩包，只是官方 COCO 标注的表示形式；来源与校验值保留在数据划分 JSON。

### DINOv3 ViT-S/16

- [ModelScope 模型来源](https://www.modelscope.cn/models/facebook/dinov3-vits16-pretrain-lvd1689m)、[上游项目](https://github.com/facebookresearch/dinov3)、[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。
- 模型许可副本随权重保存在外部工作区，其 SHA256 记录在 Teacher JSON；模型及其相关代码的使用以对应上游许可为准。
