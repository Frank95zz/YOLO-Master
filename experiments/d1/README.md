# D1：冻结 DINOv3 的缓存特征检测

冻结 DINOv3，只训练多尺度 Adapter、LatentMixture 和检测头，研究检测精度保留率与训练成本。

本 README 是唯一的 D1 文档入口。代码、配置和必要测试进入 PR；三个 manifest 是程序直接读取的校验输入。数据集、Teacher 权重、特征缓存、checkpoint、完整预测和日志均放在仓库外。

## 验收状态

截至 2026-09-12，正式实验执行提交固定为 [9004eac](https://github.com/Frank95zz/YOLO-Master/commit/9004eac438acd7de0023702e26029b14276069a6)。按所提供《历史成果、基线锁定与增量验收补充细则》，统一公共验收基线是 [acce839](https://github.com/Tencent/YOLO-Master/commit/acce839c7e895d6b179de7f7093fa879e237cc7b)，不是后续上游同步点 af961b9；版本角色、来源和可计入增量见下一节。

| 项目 | 已有证据 | 当前边界 |
|---|---|---|
| P0 训练与评测闭环 | Teacher 多层输出、缓存、Adapter、LatentMixture、Detect、训练、保存与严格重载；两个数据集、两个架构均完成真实六卡五轮短测和独立评测 | VisDrone 短测完成完整验证集推理与官方格式导出，官方评分使用 MATLAB |
| P1 公平对照 | 两数据集、同总参数量、三个 seed 的最终配对实验已启动 | 完整结果待补，不提前认定精度保留率或 GPU-hours 降低至少 50% |
| P2 latent aux | 复用既有 latent 收集通道，补齐 D1 指标与梯度验证；完成 balance/z/gain 的三 seed 消融 | 收益较小且种子间不一致，未比较其他 Teacher，不能宣称稳定或显著提升 |
| 其他扩展 | 保留 BASE、DW、BN64 三种 Adapter 候选及相关研究结果 | 未进行其他 Teacher 对照 |

五轮短测是工程验收，不是正式训练结果。历史研究与最终配对实验的提交、预算和 checkpoint 选择分别记录，不混合统计。

## 基线与新增贡献

### 固定引用与核验方式

补充细则将全部课题的代码验收基线锁定为 2026-08-21 23:59:59（UTC+8）的 Tencent/YOLO-Master main。该规则用于划分历史成果和本轮新增 diff，不等于要求把实验中的 Scratch 检测器替换成这个日期的某个官方权重。

| 引用 | 完整 SHA / 定位方式 | 用途 |
|---|---|---|
| BASE_REF：统一公共验收基线 | acce839c7e895d6b179de7f7093fa879e237cc7b | 所有新增成果按此固定起点审计，不随 main 移动 |
| 发布来源：YOLO-Master-v26.08 | 43d40117c30811204fb9347efeabddce15f11a62 | 仅说明发布版本来源，不代替 BASE_REF |
| UPSTREAM_REF：本 PR 整合采用的上游快照 | af961b99b8ef80491e58cb5fd16e25ebaf3741eb | 分离后续上游同步与 D1 自有改动；不是新的验收基线 |
| RUN_REF：正式配对实验执行提交 | 9004eac438acd7de0023702e26029b14276069a6 | 固定代码、运行合同、checkpoint 与评测身份 |
| FINAL_REF：提交给评审的代码快照 | 在 PR 正文锁定完整 40 位 SHA；检出该版本后用 git rev-parse HEAD 核对 | 后续结果补充若形成新提交，保留旧引用并重新锁定，不用可移动分支名代替 |

本 PR 面向 Tencent/YOLO-Master 的 main。整合分支包含上述上游快照，不回退上游，也不为对齐文案重写正在运行的 RUN_REF。已验证 BASE_REF 是 UPSTREAM_REF 和当前 PR 提交的祖先，当前 PR 提交与 UPSTREAM_REF 的 merge-base 为 af961b9。

在检出 PR 正文指定的 FINAL_REF 后执行：

~~~bash
BASE_REF=acce839c7e895d6b179de7f7093fa879e237cc7b
UPSTREAM_REF=af961b99b8ef80491e58cb5fd16e25ebaf3741eb
RUN_REF=9004eac438acd7de0023702e26029b14276069a6
FINAL_REF=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "$BASE_REF" "$FINAL_REF"
git merge-base "$UPSTREAM_REF" "$FINAL_REF"
git diff --stat "$BASE_REF" "$FINAL_REF"
git log --reverse --format=fuller "$BASE_REF..$FINAL_REF"

# 后续上游同步：保留原作者，不计为本 PR 的个人新增。
git diff "$BASE_REF" "$UPSTREAM_REF"
git log --reverse --format=fuller "$BASE_REF..$UPSTREAM_REF"

# D1 交付范围：与统一基线总 diff 一同提供给评审。
git diff --name-status "$UPSTREAM_REF" "$FINAL_REF"
git diff "$UPSTREAM_REF" "$FINAL_REF"
git log --reverse --format=fuller "$UPSTREAM_REF..$FINAL_REF"
~~~

公共基线到上游快照包含 104 个可达提交、172 个变更文件；D1 交付相对上游快照为 58 个文件（38 新增、20 修改）。两段有 8 个重叠文件，合并后的公共基线到 PR 总 diff 为 222 个文件，不能把这 222 个文件全部记为 D1 新增。重叠文件是 .gitignore、tests/test_ddp_lifecycle_ema_nan.py、tests/test_mixture_loss_composition.py、ultralytics/engine/extensions/recovery.py、ultralytics/engine/trainer.py、ultralytics/nn/foundation/__init__.py、ultralytics/nn/mixture_loss.py、ultralytics/nn/tasks.py；逐段 diff 核对归属，不按文件名整体认领。GitHub PR 的实际合并差异由届时目标 main 决定，不能用其动态变化替代固定 BASE_REF 审计。

### 已有能力与本轮增量

| 能力 | BASE_REF 已有内容，不重复认领 | 本轮交付与证据 |
|---|---|---|
| Foundation / Teacher | FoundationFeatures、DINOv3Teacher、预处理与冻结推理、默认 dense["p4"]，见[原始 Teacher](https://github.com/Tencent/YOLO-Master/blob/acce839c7e895d6b179de7f7093fa879e237cc7b/ultralytics/nn/foundation/teachers/dinov3.py) | 新增 output_layers 一基编号 API，公开 stage 选择、三层输出与严格形状/冻结验证；默认接口保持兼容 |
| LatentMixture | router_only、weighted_sum、value_fusion_weights、Router 与 aux 已存在，见[原始模块](https://github.com/Tencent/YOLO-Master/blob/acce839c7e895d6b179de7f7093fa879e237cc7b/ultralytics/nn/modules/latent_mixture.py) | 复用这些机制适配三个 DINO 来源；新增 BASE/DW/BN64 架构与同协议实验，不声称发明 weighted_sum |
| latent aux 收集 | collect_aux_loss 默认集合不含 latent，但 CompositeCriterion 的调用已经显式 include_kinds 包含 latent，见[原始损失组合](https://github.com/Tencent/YOLO-Master/blob/acce839c7e895d6b179de7f7093fa879e237cc7b/ultralytics/nn/mixture_loss.py) | D1 接入现有收集通道，增加 raw/effective 指标、标量/有限值/梯度验证和三 seed 扫描；不声称首次注册 latent |
| 检测与缓存 | 现有 YOLO 检测头、Trainer、Dataset、Foundation 蒸馏及其缓存能力 | 新增作为检测器输入的 D1 多层缓存合同、分片/NPY、九分支 Adapter、模型、Dataset/Trainer/Validator 和严格重载链路，不把已有蒸馏链路改称 D1 新实现 |
| 运行与性能 | 现有 DDP、EMA、checkpoint、优化器与恢复基础设施 | 新增 D1 预取边界、精确恢复与计时门禁、可分离 P3、foreach EMA、显式 Scratch FP32 Attention；存储收益与模型收益分开报告 |
| 研究结论 | 基线原有实验、报告及目录结构不作为本轮结论 | 锁定后完成的 P5 架构筛选、36 次独立 aux 运行、性能/恢复诊断与最终配对实验；改名、合并测试、重排文档不单独计为 P1/P2 |

后续上游已经在 [b69acc4](https://github.com/Tencent/YOLO-Master/commit/b69acc4f63e48742460a2d02c391e0491464b44d) 修复 native_loss 先求和、标量 aux 只加一次；该提交由 onion-hong 贡献并包含于 UPSTREAM_REF。本 PR 在此基础上增加显式输入验证、D1 指标报告与防回归测试，不把已经同步的修复重复记作本 PR 首创。

### 来源、归属与证据状态

D1 交付负责人为 [@Frank95zz](https://github.com/Frank95zz)，负责本 PR 中的多层 Teacher 扩展、缓存检测实现、运行验收及实验分析；关联进展为 [Issue #266](https://github.com/Tencent/YOLO-Master/issues/266)。公共模块、Teacher 模型、数据集及后续上游提交保留各自原作者署名。没有可核验的他人成员产物时，不补写协作者贡献。

本 PR 首次整合提交为 [f5bf7bc](https://github.com/Frank95zz/YOLO-Master/commit/f5bf7bc56d128e02d3485fe8df1e15301bee7556)，它将研究分支中需要交付的实现移植、精简到后续上游，不是逐提交原样 cherry-pick。原研究历史和证据保留在 [f4d2bc2 固定归档](https://github.com/Frank95zz/YOLO-Master/tree/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1)，不 force-push 改写，不将压缩后的提交日期当成全部工作的首次发生时间。

| 工作包 | 可追溯原始提交 | 最终交付定位 |
|---|---|---|
| Teacher 多层输出 | [3b05836](https://github.com/Frank95zz/YOLO-Master/commit/3b05836aca025bcb9dbacc596c1c049de214a87a) | teachers/dinov3.py 与 test_foundation_dinov3.py |
| 分片与 NPY 特征缓存 | [5cd566d](https://github.com/Frank95zz/YOLO-Master/commit/5cd566dcdb1e47a9227a81852296ade76d9ba20f)、[e4297e7](https://github.com/Frank95zz/YOLO-Master/commit/e4297e742f472c443ce55914e31144afcc67705d) | foundation/cache.py、npy_cache.py 与 scripts/d1/cache_features.py |
| 多尺度 Adapter 与检测闭环 | [e1c60c2](https://github.com/Frank95zz/YOLO-Master/commit/e1c60c27c2a57285a384048ce9e379035f2a046a)、[a1255c0](https://github.com/Frank95zz/YOLO-Master/commit/a1255c09b989d20f4f130ddf55e29614811f5fc2)、[81c98b3](https://github.com/Frank95zz/YOLO-Master/commit/81c98b334eb7a696561753164695fe30eb38531f) | foundation_adapter.py、foundation_detection_model.py 与缓存训练/验证实现 |
| 轻量 P5 与性能实现 | [3dfe0f6](https://github.com/Frank95zz/YOLO-Master/commit/3dfe0f6e72022b62a2f169e693b4996ac0361f7e)、[6f88a42](https://github.com/Frank95zz/YOLO-Master/commit/6f88a422f426b5fe9bd013aa25a7ef6a95570231) | 三种 P5 配置、Adapter 与性能测试；数值结论见下文固定证据 |
| 正式配对与稳定性 | [2920727](https://github.com/Frank95zz/YOLO-Master/commit/2920727209a5e1a6b7cd53080d37a1c5d83e3517) 至 [9004eac](https://github.com/Frank95zz/YOLO-Master/commit/9004eac438acd7de0023702e26029b14276069a6) | compare.py、runtime.py、scratch 数值策略和真实门禁 |

上述链接用于核验代码对象与报告的对应关系，不仅凭作者日期或后补 README 判断新增时间。本 PR 不追补锁定前个人成果登记，也不主张用未备案历史成果兑换新增分数。P0 以实际训练/评测闭环举证；P1 最终两数据集三 seed 成本与精度结论仍待完成；P2 提交可复核的机制/工程/实验增量，由评审按课题要求判断，不因脚本存在或一次最佳数字直接自认达标。

## 架构与实现

~~~text
RGB -> 固定 640 LetterBox -> 冻结 DINOv3 ViT-S/16
    -> block4 / block8 / block12：均为 [B,384,40,40]
    -> 每层分别适配到 P3/P4/P5，共九条独立分支
    -> 每个尺度的三路候选经 LatentMixture 融合
    -> YOLO26 Detect
~~~

- 三个 DINO block 都是 stride 16，不是天然的 P3/P4/P5。Adapter 输出通道为 64/128/256，网格为 80/40/20。
- 正式候选使用 BN64：P5 为 384->64->256 的瓶颈分支；BN64 不是 BatchNorm，归一化仍为 GroupNorm。
- Teacher 多层 API 为 DINOv3Teacher(output_layers=(4,8,12))。默认调用继续返回 dense["p4"]，不强制现有蒸馏消费者改用多层输出。
- Teacher 不进入 student optimizer、DDP、EMA 或 checkpoint。固定缓存仍须与原图、标签和预处理合同对应。
- 复用既有 CompositeCriterion 和显式包含 latent 的 include_kinds，标量目标为 detection.sum()+aux；新增 D1 检测损失、raw/effective aux 报告及有限值验证，不改变收集器默认集合。
- weighted_sum 融合、可分离双线性 P3 和 foreach EMA 通过 D1 配方或入口显式启用，保留原实现供数值对照。
- 通用改动集中在 Trainer 扩展点、checkpoint 运行状态清理及辅助损失报告；Attention 的 FP32 分支默认关闭，旧对象未设置该属性时仍走原路径。

主要代码位置：

| 功能 | 入口 |
|---|---|
| Teacher 多层输出 | [dinov3.py](../../ultralytics/nn/foundation/teachers/dinov3.py) |
| safetensors / NPY 缓存 | [cache.py](../../ultralytics/nn/foundation/cache.py)、[npy_cache.py](../../ultralytics/nn/foundation/npy_cache.py) |
| 多尺度适配与检测模型 | [foundation_adapter.py](../../ultralytics/nn/modules/foundation_adapter.py)、[foundation_detection_model.py](../../ultralytics/nn/foundation_detection_model.py) |
| Dataset / Trainer / Validator | [d1_cache.py](../../ultralytics/data/d1_cache.py)、[foundation_train.py](../../ultralytics/models/yolo/detect/foundation_train.py)、[foundation_val.py](../../ultralytics/models/yolo/detect/foundation_val.py) |
| 单次训练、评测及 scratch | [train.py](../../scripts/d1/train.py)、[rgb.py](../../scripts/d1/rgb.py) |
| 配对实验、计时与精确续训 | [compare.py](../../scripts/d1/compare.py)、[runtime.py](../../scripts/d1/runtime.py) |

## 安装与数据准备

D1 使用 Linux 和 Python >=3.10。已验证环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 5.15.1、Ultralytics 8.4.101、safetensors 0.8.0；真实多卡验收使用六张 A40。该记录是实测环境，不保证任意依赖组合均等价。

~~~bash
pip install -e ".[dev,foundation]"
pip install faster-coco-eval==1.8.0
export D1_WORK=/path/to/external/d1-work
export COCO_ROOT="$D1_WORK/datasets/coco"
export TEACHER_DIR="$D1_WORK/weights/dinov3-vits16"
~~~

Teacher 使用 [ModelScope ViT-S/16](https://www.modelscope.cn/models/facebook/dinov3-vits16-pretrain-lvd1689m)，固定来源版本 2e601320d0545509ab03374e2f8707f303e1de7a。将 config.json、model.safetensors、LICENSE.md、README.md 放入 TEACHER_DIR。文件大小、SHA256 与架构由 [Teacher manifest](manifests/dinov3-vits16.json) 校验。

COCO 下载 train2017.zip、val2017.zip、annotations_trainval2017.zip 和 coco2017labels.zip；URL、镜像来源、大小与 SHA256 见 [数据 manifest](manifests/coco2017-splits.json)。图片解压到 COCO_ROOT/images，官方标注到 COCO_ROOT/annotations；labels 压缩包内含 coco/labels，应解压到 D1_WORK/datasets。

~~~bash
python -m scripts.d1.prepare_coco --coco-root "$COCO_ROOT" \
  --weights-dir "$TEACHER_DIR" --output "$D1_WORK/inputs"
~~~

prepare_coco 验证 118,287/5,000 图片、稳定列表摘要、train/val 互斥、标签归属、官方标注文件存在性与 Teacher 权重，并从本地权重构造 Teacher。只有提供 --archives-dir 才检查源压缩包；不把标注文件存在性说成内容逐字节验证。仅恢复图片列表时可使用 --lists-only。重复执行必须得到相同输出，不重新随机划分或覆盖不同证据。

正式预处理见 [合同](manifests/experiment-contract.json) 和 [COCO 配方](../../ultralytics/cfg/experiments/d1/dinov3-vits16-coco2017.yaml)：固定 640 方形 LetterBox、RGB CHW、[0,1]、DINOv3 mean/std；Teacher 不再缩放或裁剪，block 4/8/12 均输出 40x40 网格。缓存采用 d1-cache-v1 和 FP16。

VisDrone 使用官方 DET train/val，共 6,471/548 张，原始目录包含 VisDrone2019-DET-train 与 VisDrone2019-DET-val，各自有 images/annotations：

~~~bash
python -m scripts.d1.prepare_visdrone \
  --source "$D1_WORK/visdrone-original" --output "$D1_WORK/visdrone-prepared"
~~~

输出 dataset.yaml、train.txt、val.txt 与保留 ignore 信息的 sidecar；原始数据不删除。当前图片准备采用硬链接，source 与 output 须位于同一文件系统。--include-test-dev 是可选保留集，不混入 train/val。

## 特征缓存

~~~bash
python -m scripts.d1.cache_features build --data-root "$COCO_ROOT" --weights-dir "$TEACHER_DIR" \
  --samples-file "$D1_WORK/inputs/coco2017-train2017.txt" \
  --cache-dir "$D1_WORK/cache/train2017" --split train2017 --batch-size 16 --device 0
python -m scripts.d1.cache_features build --data-root "$COCO_ROOT" --weights-dir "$TEACHER_DIR" \
  --samples-file "$D1_WORK/inputs/coco2017-val2017.txt" \
  --cache-dir "$D1_WORK/cache/val2017" --split val2017 --batch-size 16 --device 0
python -m scripts.d1.cache_features verify --cache-dir "$D1_WORK/cache/train2017"
python -m scripts.d1.cache_features verify --cache-dir "$D1_WORK/cache/val2017"

# 可选：一图一个 NPY，无损转换并保留源分片。
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/train2017" --output "$D1_WORK/coco-npy"
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/val2017" --output "$D1_WORK/coco-npy"
~~~

VisDrone 使用相同入口：

~~~bash
python -m scripts.d1.cache_features build --data-root "$D1_WORK/visdrone-prepared" \
  --weights-dir "$TEACHER_DIR" --samples-file "$D1_WORK/visdrone-prepared/train.txt" \
  --split visdrone-train --cache-dir "$D1_WORK/cache/visdrone-train" --batch-size 16 --device 0
python -m scripts.d1.cache_features build --data-root "$D1_WORK/visdrone-prepared" \
  --weights-dir "$TEACHER_DIR" --samples-file "$D1_WORK/visdrone-prepared/val.txt" \
  --split visdrone-val --cache-dir "$D1_WORK/cache/visdrone-val" --batch-size 16 --device 0
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/visdrone-train" --output "$D1_WORK/visdrone-npy"
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/visdrone-val" --output "$D1_WORK/visdrone-npy"
~~~

图片列表必须排序、无重复，每行形如 images/SPLIT/ID.jpg。build 使用 seed 0、确定性执行和关闭 TF32，记录图片、代码、batch、设备与依赖身份。工程小样本可使用 --limit，但训练 YAML 必须同步限制到已缓存图片。

单进程独占一个输出目录；并发写入、身份不一致或残留临时文件报错。中断后用相同代码和命令重放原 batch，已提交成员必须逐元素一致；不能编辑 build.json 绕过检查。旧代码的未完成缓存应在原提交续写，已完成缓存可继续校验、读取和转换。六卡抽取调度器不在本 PR 中。

safetensors 与 NPY 保存相同 FP16 特征，不引入额外有损量化。数据、缓存与输出建议放高速本地盘。--benchmark-read 测得的缓存读取吞吐不是训练吞吐。

## 训练、恢复与评测

默认功能配方为 [cached-detection.yaml](../../ultralytics/cfg/experiments/d1/cached-detection.yaml)，CLI 显式覆盖 batch、epochs、workers 和 seed。它不依赖个人研究队列。

~~~bash
python -m scripts.d1.train inspect --variant BN64
python -m scripts.d1.train train --approved --variant BN64 --dataset coco \
  --data /path/to/coco-local.yaml \
  --train-cache "$D1_WORK/coco-npy/train2017" --val-cache "$D1_WORK/coco-npy/val2017" \
  --output "$D1_WORK/runs/example" --device 0 --batch 16 --epochs 100 --workers 4 --seed 0

python -m scripts.d1.train evaluate --variant BN64 --dataset coco \
  --data /path/to/coco-local.yaml --val-cache "$D1_WORK/coco-npy/val2017" \
  --checkpoint "$D1_WORK/runs/example/weights/best.pt" \
  --output "$D1_WORK/evaluations/example-best" --device 0 --batch 16 \
  --annotations "$COCO_ROOT/annotations/instances_val2017.json"
~~~

coco-local.yaml 使用仓库常规检测 YAML，指定 path、train、val 与 80 类 names。VisDrone 使用 prepare_visdrone 生成的 10 类 dataset.yaml 和 --dataset visdrone。SCRATCH 使用 --variant SCRATCH，不传缓存参数。新运行输出目录必须不存在，避免覆盖。

多卡由外部 torchrun 启动，使用 --standalone --nproc_per_node=6 -m scripts.d1.train train，并传 --device 0,1,2,3,4,5 与可整除的全局 batch。--ema scalar-v1/foreach-v1、--p3-upsample bilinear/separable_bilinear2x 保留实现选择；--fp32 用于数值对照。只有提前完成全量缓存校验才使用 --trusted-cache。

精确恢复用于启用 --telemetry 的运行：重复原训练命令，并追加 --resume-snapshot 指向该运行的 resume.pt；提交、数据、模型、batch、seed 等身份必须一致。--window 仅限制已执行轮数，不缩短学习率调度。恢复 9004eac 的运行须使用 9004eac，不能以更新后的 PR 提交绕过身份校验；普通 last.pt 用于评测，不替代精确恢复快照。

评测严格重载 checkpoint，输出 evaluation.json、predictions.json 和 checkpoint SHA256。COCO 只有提供 --annotations 才报告标准 AP，使用 faster-coco-eval 1.8.0、完整 5,000 张验证图、maxDets=[1,10,100]；最多导出 300 个预测框不改变标准 AP 的 maxDets=100。

VisDrone 评测完整 548 张验证图并导出官方 TXT，每图最多 500 框；坐标还原到原图，序列化后宽或高不大于零的框被剔除并计数，非有限值报错。官方 ignore 语义和评分使用固定 MATLAB DET toolkit，不能用轮内 COCO-style 指标代替；该 MATLAB 调度器不在本 PR 中。

## 最终配对实验

冻结组固定 ViT-S/16 + BN64 + weighted_sum：balance=0.1、z=0、gain=0.1、budget=3.0。Scratch 使用随机初始化的标准 YOLO26-L 拓扑宽度匹配版：depth=1.0、width=0.9375、max_channels=512。两组重新训练，不续训筛选权重。

| 数据集 | train / val | 每组预算 | seeds | 每卡 / 全局 batch | 运行数 |
|---|---:|---:|---|---:|---:|
| COCO 2017 | 118,287 / 5,000 | 100 epochs | 0/1/2 | 64 / 384 | 6 |
| VisDrone2019-DET | 6,471 / 548 | 300 epochs | 0/1/2 | 16 / 96 | 6 |

| 参数口径 | COCO | VisDrone |
|---|---:|---:|
| 冻结 Teacher | 21,596,544 | 21,596,544 |
| BN64 下游可训练参数 | 1,404,839 | 1,340,259 |
| 冻结组总参数 | 23,001,383 | 22,936,803 |
| scratch 总参数 | 23,133,560 | 23,032,340 |
| scratch 相对总参数差 | +0.57% | +0.42% |

总参数包含 Teacher；1% 是本项目工程匹配容差，不声称为官方规定。这是不同计算路径的系统对照，不是同架构单因素消融。配置为 [paired-comparison.yaml](../../ultralytics/cfg/experiments/d1/paired-comparison.yaml) 和 [scratch-total-l](../../ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml)。

每次运行独占六张 A40，各组串行。固定 640 方形单次 LetterBox、无颜色/几何/翻转/Mosaic/MixUp/Copy-Paste/多尺度增强，两组 RGB/NPY 均使用 NVMe。AdamW 使用共享参数分组，lr0=0.001、lrf=0.01、momentum=0.9、weight_decay=0.0005、cosine、warmup=3；Router 沿用半学习率分组。nbs 等于全局 batch，每 batch 一次有效更新。

两组统一 AMP 初始 scale=0.0625、growth_interval=1000000、workers=4/rank、prefetch=1。Scratch 显式启用 fp32_attention：三个 Attention 的 QK、softmax 与加权 V 使用 FP32，其余继续 AMP；不改变参数量或拓扑。该策略经真实输入有限性校准，训练、评测和恢复采用同一配置，不静默升级旧 checkpoint 或跳过失败更新。Frozen 使用 foreach EMA 和可分离 P3，Scratch 使用原生 EMA，实际实现成本纳入比较。

~~~bash
python -m scripts.d1.compare --approved --output "$D1_WORK/final-comparison" \
  --coco-root "$COCO_ROOT" --coco-cache "$D1_WORK/coco-npy" \
  --visdrone-root "$D1_WORK/visdrone-prepared" --visdrone-cache "$D1_WORK/visdrone-npy" \
  --device 0,1,2,3,4,5
~~~

入口要求干净代码提交、新的外部输出目录及明确批准，不下载或删除数据。外部目录记录 plan.json、status.json、日志、运行身份和 ETA。正式主结果固定第 100/300 轮；每 5 轮保留周期 checkpoint，单组训练结束后依次独立评测周期快照、last.pt 与内部 best.pt。COCO 从周期快照选标准 AP 最优者并生成 standard-best.json，平局取更早轮；VisDrone 导出预测后标记等待 MATLAB 评分。

### 工程验收

执行提交 9004eac 已完成四种数据集/模型组合的真实六卡恢复一致性门禁：连续两轮与一轮后恢复到两轮的模型、EMA、优化器、scaler、scheduler、criterion 和各 rank 状态一致。每种组合还完成五个完整 epoch、周期 checkpoint 保存、完整验证集独立推理及严格重载。

DDP 在首次训练和恢复时先进行三次无 optimizer 更新的前反向以建立相同分桶，再恢复模型、损失调度和随机状态；一次性准备耗时单独记录。加载器按 epoch 重建并确定性设种子，避免无限预取跨轮改变恢复后的取样顺序。缺失样本、非有限值、漏更新、恢复不一致或 OOM 均停止，不自动缩小某一组的 batch。

以下时间为同次五轮短测第 4/5 轮的中位数，包含轮内验证与保存，不含训练后独立评测、Teacher 抽取和 MATLAB：

| 数据集 | BN64 秒/epoch | scratch 秒/epoch |
|---|---:|---:|
| COCO | 123.23 | 372.91 |
| VisDrone | 18.32 | 40.53 |

短测时间仅用于预算与工程验收，不作为最终 GPU-hours 降本结论。最终实验 AP、显存、GPU-hours 与三个 seed 的统计结果仍待完整运行和官方评分后填入。

### 最终结果口径

- AP 统一按 0-100 点报告；COCO 同时报 AP50/AP75/APs/APm/APl，VisDrone 报官方 AP/AP50/AP75/AR1/AR10/AR100/AR500。
- 主表比较相同固定末轮，不混用一组 best 与另一组 last。标准 best 作为单独补充结果。
- 报告每个 seed、均值、样本标准差及配对差值；精度保留率为冻结组 AP / scratch AP。
- GPU-hours 包含已分配 GPU 的数据等待，训练、独立评测和抽取分别计量；同时报告训练-only、含一次 Teacher 抽取的冷启动成本及实际复用次数下的摊销成本。
- 成本降低率为 1 - 冻结组成本 / scratch 成本。无法可靠恢复的旧抽取成本标记未知，不用 ETA 或缓存读取吞吐宣称达到 50%。

## 已完成的研究结果

### COCO Adapter 筛选

完整 COCO、seed 0、固定第 50 轮 checkpoint，独立标准 COCO 评测：

| P5 结构 | 下游参数 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE：3x3 stride2 Conv | 3,542,567 | 28.870 | 49.197 | 30.113 | 12.921 | 33.463 | 40.228 |
| DW：深度可分离分支 | 1,195,943 | 28.593 | 48.542 | 29.723 | 13.033 | 32.510 | 41.559 |
| BN64：64 通道瓶颈 | 1,404,839 | 29.745 | 49.284 | 31.203 | 12.965 | 33.086 | 42.992 |

BN64 被选作后续基座。单 seed 结果仅支持候选筛选，不构成统计等效或稳定提升证明。固定归档包含[完整报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/P5_FAST_RUN_20260909.md)与[机器可读汇总](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/p5-screen-20260909/suite-summary.json)，记录配方、计时与 checkpoint 身份。

### VisDrone Latent Aux 消融

完整 VisDrone train/val、BN64、weighted_sum、seeds 0/1/2，固定 300 轮调度的前 60 轮。统一评测第 60 轮，采用固定官方 MATLAB DET toolkit，保留 ignore 语义。

先扫描 balance={0,0.01,0.1} 与 z={0,0.001,0.01}、gain=0.1，共 27 次；再固定 balance=0.1、z=0 扫描 gain，新增 9 次并复用 3 次，共 36 个独立运行。

| latent_aux_gain | 官方 AP 均值（3 seeds） |
|---|---:|
| 0 | 8.022406397 |
| 0.03 | 8.115370260 |
| 0.1 | 8.131928488 |
| 0.3 | 8.081231746 |

按预定均值优先规则选择 balance=0.1、z=0、gain=0.1、budget=3.0。gain=0.1 相对关闭 aux 仅高 0.109522 点，三 seed 配对差为 -0.245386/+0.339334/+0.234618 点，探索性 95% 区间跨 0。同一批种子还用于选参，不能宣称稳定或显著收益；gain=0.03 为波动较小的备选。

这些是短周期筛选，不代表 300 轮已经收敛，也不是完整 standard-best 曲线。固定归档：[消融报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/E3.md)、[第一阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage1-official-20260911.json)、[第二阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage2-official-20260911.json)。

## 测试与兼容性

测试按功能组织：test_d1_contracts/cache/cache_cli/adapter/model/pipeline/training/visdrone，公共 Teacher 测试并入已有的 test_foundation_dinov3.py。覆盖非法输入、缓存校验与续写、九条分支梯度、aux 标量组合、Teacher 隔离、checkpoint 严格重载、默认 Attention 行为、AMP 有限值及恢复协议。

普通 CI 使用合成数据并跳过未配置的真实 Teacher/CUDA 测试，不自动下载模型。真实输入通过 D1_DINOV3_WEIGHTS、D1_WP2_CACHE、D1_COCO_ROOT、D1_NPY_CACHE 显式指定；保留这些环境变量以兼容已有使用方式，启用 CUDA 验收时不要清空 CUDA_VISIBLE_DEVICES。

~~~bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -q \
  tests/test_checkpoint_compat.py tests/test_d1_adapter.py \
  tests/test_d1_cache.py tests/test_d1_cache_cli.py \
  tests/test_d1_contracts.py tests/test_d1_model.py \
  tests/test_d1_pipeline.py tests/test_d1_training.py \
  tests/test_d1_visdrone.py tests/test_ddp_checkpoint_coordination.py \
  tests/test_ddp_lifecycle_ema_nan.py tests/test_foundation_cache_training.py \
  tests/test_foundation_checkpoint.py tests/test_foundation_config.py \
  tests/test_foundation_dinov3.py tests/test_foundation_distill_model.py \
  tests/test_foundation_f08_effect_gate.py tests/test_foundation_f09_foreground_effect_gate.py \
  tests/test_foundation_f15_benchmark.py tests/test_foundation_f15_effect_matrix.py \
  tests/test_foundation_f15_real_effect_gate.py tests/test_foundation_f15_release_audit.py \
  tests/test_foundation_losses.py tests/test_foundation_metrics_logging.py \
  tests/test_foundation_mixture_interaction.py tests/test_foundation_multirouter.py \
  tests/test_foundation_multitask.py tests/test_foundation_offline.py \
  tests/test_foundation_projectors.py tests/test_foundation_recipe_integrity.py \
  tests/test_foundation_routing_contract.py tests/test_foundation_sam3.py \
  tests/test_foundation_semantic.py tests/test_foundation_siglip2.py \
  tests/test_foundation_taps.py tests/test_foundation_teacher_protocol.py \
  tests/test_foundation_weight_schedule.py tests/test_latent_mixture.py \
  tests/test_mixture_loss_composition.py tests/test_yoloe_released_checkpoint_compat.py \
  tests/test_prevalidation_recovery.py tests/test_optimizer_group_audit.py \
  tests/test_default_config_integrity.py tests/test_master_model_configs.py \
  tests/test_engine.py::test_load_checkpoint_state_dict_rejected --deselect=tests/test_foundation_cache_training.py::test_response_kd_builds_pseudo_batch_from_cached_responses \
  --deselect=tests/test_foundation_config.py::test_enabled_without_teacher_is_rejected
~~~

训练执行提交 9004eac 的相关回归为 637 passed、56 skipped、2 deselected；PR 整理版本 5af743f 的扩展 CPU 回归为 675 passed、56 skipped、2 deselected，pytest 耗时 80.42 秒。两项排除用例在 UPSTREAM_REF=af961b9 和 PR 整理版本上均复现，不计为通过；这不表示已经在公共验收基线 acce839 上重跑同一套新增测试。没有运行完整仓库测试集、所有导出后端或 GitHub Actions。

仓库质量入口以下使用 UPSTREAM_REF 检查 D1 相对整合上游的质量增量，不替代上文 BASE_REF 的成果审计：

~~~bash
python scripts/check_changed_quality.py --base af961b99b8ef80491e58cb5fd16e25ebaf3741eb --no-untracked
git diff --check
~~~

本轮 45 个相关 Python 文件的 Ruff 格式和全部支持文件的 codespell 检查通过。完整变更质量命令仍返回非零：当前 66 条 Ruff 告警均可在 UPSTREAM_REF=af961b9 的同一文件、规则、消息和源行中匹配，本 PR 相对该快照新增告警为 0；不将该命令写成通过，也不全局关闭规则。必要的析构清理、第三方路由兼容和训练生命周期异常保留原处理方式，使用局部注释说明原因。

PR 整理仅涉及导入、格式、脚本权限、说明及无数值行为变化的清理。16 个训练入口、运行实现、配置与 manifest 文件相对 9004eac 逐字节不变；BN64/scratch 在 10/80 类、固定初始化和 640 输入下的 CPU 模型状态及前向输出摘要一致。这不是新的真实六卡训练，也不保证不同提交可绕过运行身份进行恢复。完整回归、质量输出和四组摘要保存在外部提交验证记录中。

## 局限与许可

固定离线特征不支持任意颜色、几何、Mosaic 或随机多尺度增强；更换 Teacher、输入尺寸、层选择或预处理须重新构建缓存。NPY 和 safetensors 的训练速度依赖存储、CPU、缓存热度及共享任务干扰，不能把存储迁移收益全部归为模型贡献。

公共入口面向可信本地 checkpoint，不接受未信任 pickle。旧研究运行需使用各自报告中的执行提交；新入口不保证逐位重放旧研究队列。研究归档固定在 [f4d2bc2](https://github.com/Frank95zz/YOLO-Master/tree/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1)，不随 PR 精简而删除。

本节只记录许可来源，不作额外法律判断：

- 代码沿用仓库 AGPL-3.0 许可，见根目录 [LICENSE](../../LICENSE)。
- COCO：[Terms of Use](https://cocodataset.org/#termsofuse)、[下载源](http://images.cocodataset.org/)。图片仍受各自原始 Flickr 许可约束；YOLO labels 仅是官方标注的表示形式。
- DINOv3：[ModelScope 来源](https://www.modelscope.cn/models/facebook/dinov3-vits16-pretrain-lvd1689m)、[上游项目](https://github.com/facebookresearch/dinov3)、[License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。许可副本随权重保存，SHA256 见 Teacher manifest。
- VisDrone 使用官方 DET 数据与评测工具；原始标注和 ignore 信息保留，数据及工具使用以各自上游条款为准。
