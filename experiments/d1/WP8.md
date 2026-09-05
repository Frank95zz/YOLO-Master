# D1 WP8：COCO 2017 正式训练、诊断与 P1 对照方案

## 1. 当前状态与最终资源决策

WP8 的完整 COCO 2017 特征缓存、缓存校验、训练组件、最小闭环和正式训练入口已经具备。P0 正式
运行 `wp8-p0-b384-s0-e36864d` 已完成 30 epochs，随后为排查精度增长缓慢问题安全停止，并已对
`best.pt` 完成独立的完整 COCO val2017 内部评测和官方 COCO 评测。

该 P0 运行没有完成原计划的 100 epochs，并包含第 8.1 节记录的辅助损失广播偏差，因此只能作为
如实保留的阶段性 P0 结果，不能表述为修复后最终结果。当前不自动恢复该运行。下一步先实施第 24
节的课题目标 P1 同参数量从零训练对照，以判断低精度主要来自训练配方还是冻结特征路径；对照训练
仍需单独获得用户启动确认。

第 3～23 节保留原始 P0 正式合同、启动决策和 as-run 记录。该运行的资源合同为：

```text
单个正式任务
world_size = 6
GPU = 0,1,2,3,4,5
每卡 batch = 64
global batch = 384
workers = 每 rank 4，共 24
梯度累积 = 1
seed = 0
epochs = 100
实际完成 epochs = 30
```

六张 NVIDIA A40 全部服务于同一个 DDP 训练任务，不再拆成三组双卡任务，也不并发运行其他训练或特征抽取作业。WP8 当前只登记一个 seed 0 正式主实验；seed 1/2 重复实验不属于本次启动合同，如后续需要统计方差，必须另建预注册运行身份并串行执行，不能与 seed 0 的结果混写。

当前状态：**P0 在绑定提交 `e36864dbea38e60bd7e5f0202bff7d1c5fb62f6f` 上完成 30 epochs 后
安全停止，checkpoint 和独立诊断均已完成；下一项待执行工作是第 24 节的 P1-COCO-30 对照准备，
不是恢复原 P0 任务。**

## 2. WP8 目标与边界

WP8 完成 D1 的正式 P0 实验：使用冻结 DINOv3 ViT-S/16 预先生成的多层特征，只训练 Adapter、三个 LatentMixture 和 YOLO Detect Head，在完整 COCO 2017 上得到可复现的训练结果、验证指标和训练成本。

WP8 必须回答：

1. 缓存的 block 4/8/12 特征能否稳定驱动完整 COCO 检测训练；
2. Adapter、三个 LatentMixture、Router 和 Detect 是否都获得有效梯度并发生参数更新；
3. 完整 COCO val2017 的 mAP50-95、mAP50、Precision 和 Recall 是多少；
4. 六卡 A40 下的实际吞吐、显存、数据等待、墙钟时间和 GPU-hours 是多少；
5. checkpoint 能否严格恢复，且不包含冻结 Teacher 参数；
6. 训练中断后能否在不改变实验身份的前提下从健康 `last.pt` 继续。

WP8 不负责：

- P1 在第二数据集上的正式复现；
- 基于单次 COCO 预实验直接宣称“GPU 时降低至少 50%”的最终结论；
- P2 的 DINOv3/SigLIP2 Teacher 对比；
- balance、z-loss、latent aux 系数的大规模扫描；
- 根据中途精度临时更换学习率、batch、增强或训练轮数；
- 部署、导出、蒸馏后推理速度或实时推理优化。

第 24 节新增 P1 的第一组 COCO 诊断性对照。该基线必须复用本文件锁定的 COCO split、640 输入、
global batch 384、优化器、已完成的 30 epochs、增强和评测口径，才能与当前 P0 checkpoint
比较。只有完成修正后的 P0 正式实验和至少第二个数据集，才能形成任务书意义上的 P1 最终结论。

## 3. 为什么选择六卡每卡 batch 64

### 3.1 已有六卡短基准

历史基准 `wp8-batch-worker-local-6ff002d-r1` 使用六张 A40 和本地特征缓存。以下数字均来自同一份 `summary.json`：

| 每卡 batch | global batch | workers/rank | 六卡吞吐 | data wait | 峰值显存 | 100 epochs + val + 15% 估计 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 96 | 4 | 462.42 images/s | 0.136% | 约 1.90 GiB | 8.52 h |
| 32 | 192 | 4 | 679.86 images/s | 0.128% | 约 3.63 GiB | 5.79 h |
| 64 | 384 | 4 | **848.12 images/s** | **0.108%** | **最高约 7.06 GiB** | **4.64 h** |
| 64 | 384 | 2 | 258.36 images/s | 42.17% | 最高约 7.06 GiB | 15.24 h |

在已成功的组合中，每卡 batch 64、每 rank 4 workers 吞吐最高，显存仍有很大余量，因此选作正式候选。workers 2 明显供数不足；workers 8/16 的早期失败与当时错误地把可回收文件页缓存当成匿名内存风险有关，但增加 worker 也会扩大进程数、打开 shard 数和页缓存活跃集合，所以正式方案不继续扩大，固定为 4。

### 3.2 短基准的限制

`848.12 images/s` 只统计 17 个稳态 batch，共 6,528 张图片和 7.697 秒计时区间。它没有覆盖完整 423.5 GiB 缓存，也没有经历容器页缓存饱和后的持续读取，更没有包含完整 val2017、checkpoint 和 100 次 epoch 边界开销。

因此：

- `4.64 h` 是已有短测得到的乐观估计，不是正式承诺；
- 正式预算暂按 **5～8 小时**；
- 启动前必须完成第 12 节 308-step 长基准；
- 最终 ETA 以长基准的稳态吞吐和最慢 rank 为准。

### 3.3 放弃的方案

- 三组双卡、每卡 batch 8：并发实测吞吐分别为 70.16、77.24 和 101.32 images/s，缓存竞争使保守 ETA 达到 38.87～56.14 小时；同时产生 24 个 worker 和三套随机 I/O，资源效率低。
- 单卡 batch 256：实测 89.98 images/s，data wait 29.45%，峰值分配显存约 29.47 GiB，保守 ETA 43.77 小时；不能发挥六张 A40 的总吞吐。
- 六卡每卡 batch 128/256：尚无可信成功结果，且没有必要在已经达到 848.12 images/s 后继续扩大显存和主机内存风险。

## 4. 正式实验身份

| 项目 | 固定值 |
| --- | --- |
| 课题 | D1：冻结 DINOv3 × LatentMixture |
| 阶段 | WP8 / P0 正式完整 COCO 运行 |
| 正式运行数 | 1 |
| seed | 0 |
| GPU | 6 × NVIDIA A40 |
| DDP world size | 6 |
| 物理设备 | `0,1,2,3,4,5` |
| 每卡 batch | 64 |
| global batch | 384 |
| `nbs` | 384 |
| 梯度累积 | 1 |
| 输入尺寸 | 640 × 640 |
| epochs | 100 |
| 训练 split | COCO 2017 train2017 |
| 验证 split | COCO 2017 val2017 |
| 运行标识 | `wp8-p0-b384-s0-<commit>` |

运行身份必须同时记录完整 Git commit、配置 SHA256、模型 YAML SHA256、train/val 缓存摘要、seed、world size、global/per-GPU batch、优化器参数、AMP 参数和环境信息。任一字段变化都必须创建新运行目录，不得在旧目录覆盖或续跑。

## 5. 数据与预处理合同

### 5.1 COCO 2017 官方 split

| Split | 图片数 | 用途 |
| --- | ---: | --- |
| train2017 | 118,287 | 完整训练 |
| val2017 | 5,000 | 每 epoch 验证和最终评测 |

不进行二次随机划分，不使用 COCO8、COCO-mini 或训练子集代替正式数据。输入预处理继续复用 WP0 合同：

```text
LetterBox(
    new_shape=(640, 640),
    auto=False,
    scale_fill=False,
    scaleup=True,
    center=True,
    stride=32,
    padding_value=114,
    interpolation=INTER_LINEAR,
)
RGB -> CHW -> [0,1]
DINOv3 mean=(0.485, 0.456, 0.406)
DINOv3 std=(0.229, 0.224, 0.225)
```

正式训练不在线运行 Teacher，也不重新进行 DINOv3 resize/crop。检测图像和缓存特征必须通过同一个 image path/sample ID 对齐。

### 5.2 完整 DINOv3 缓存

Teacher 固定为 `facebook/dinov3-vits16-pretrain-lvd1689m`。权重 SHA256：

```text
4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d
```

缓存合同 SHA256：

```text
6bfda0e13bde01001c3f3f2d72631a2401fb9a77b146d6fb2794303e379e47a7
```

| Split | 样本 | Tensor | Shard | Cache bytes | 内容摘要 |
| --- | ---: | ---: | ---: | ---: | --- |
| train2017 | 118,287 | 354,861 | 204 | 436,232,631,192 | `0224bbd302faf62b39e914be11f4d3469f7567791ce48994a9fc0f378e070cbe` |
| val2017 | 5,000 | 15,000 | 12 | 18,439,541,408 | `cd5840678d79550482aaf4110b9e68b8c22819f2f6b31e0af40f0e5b7476fa1b` |

每个样本包含：

```text
block4  [384,40,40] FP16
block8  [384,40,40] FP16
block12 [384,40,40] FP16
```

完整缓存共 123,287 个样本、369,861 个 tensor、216 个 shard，约 423.5 GiB。训练使用 `/root` 下的本地只读副本，`/data` 中保留外部工作区和校验证据。启动前比对索引、样本数、合同摘要、内容摘要、shard 文件集合和文件大小；已经有完整校验报告时不重复 SHA256 扫描全部 423.5 GiB。

## 6. 模型与可训练参数

模型配置：

```text
ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml
```

数据流：

```text
block4/block8/block12 缓存特征
  -> DINOFeaturePyramidAdapter 的九条独立分支
  -> 每尺度三个 P3/P4/P5 候选
  -> P3/P4/P5 三个 LatentMixture
  -> YOLO26 Detect
  -> detection loss + latent aux loss
```

正式模型共有 3,542,567 个可训练参数。训练期间：

- 不实例化 DINOv3 Teacher；
- optimizer、DDP、EMA 和 checkpoint 只包含下游模型；
- Adapter、LatentMixture、Router、residual gain 和 Detect 必须获得有限梯度；
- 缓存 tensor 不作为参数，不保留输入梯度；
- DDP 使用已经验证的静态图策略，`find_unused_parameters=False`、`static_graph=True`；
- checkpoint 中出现 `teacher` 或 `dinov3` 参数键即判定失败。

## 7. 固定训练超参数

| 参数 | 正式值 |
| --- | --- |
| seed | 0 |
| deterministic | true |
| epochs | 100 |
| batch | 384（全局） |
| per-GPU batch | 64 |
| nbs | 384 |
| workers | 每 rank 4，共 24 |
| DataLoader prefetch | 每 worker 1 |
| 每 worker 最大打开 shard | 4 |
| AMP | true |
| AMP init scale | 16 |
| AMP growth interval | 1,000,000 |
| optimizer | AdamW |
| lr0 | 0.001 |
| lrf | 0.01 |
| scheduler | cosine |
| momentum/beta1 | 0.9 |
| weight decay | 0.0005 |
| warmup | 3 epochs |
| patience | 100 |
| pretrained student | false |
| save period | 10 epochs |
| validation | 每 epoch 完整 val2017 |
| plots | false |
| compile | false |
| fraction | 1.0 |
| dataset RAM cache | false |

### 7.1 batch、nbs 和 optimizer step

Ultralytics 配置中的 `batch=384` 是六个 DDP rank 合计的 global batch：

```text
6 ranks × 64 images/rank = 384 images/optimizer step
nbs = global batch = 384
accumulate = max(round(384 / 384), 1) = 1
```

因此每个 DDP step 后更新一次参数，不进行额外梯度累积。COCO train2017 每 epoch 预计约 309 个 DDP step，100 epochs 约 30,900 个 step。DistributedSampler 为保证各 rank 长度一致时可能补齐极少量样本，正式报告必须记录框架实际 seen 数和 optimizer step，而不能只写理论值。

global batch 384 与先前 global batch 16/48 是不同优化合同。它会显著减少 optimizer update 数并改变梯度噪声，不能把不同 batch 的精度直接解释为同一配方的纯系统加速。

### 7.2 学习率选择

正式 `lr0` 保持 AdamW 的 `0.001`，不按 batch 从 48 到 384 直接线性放大到 `0.008`。理由是：

1. WP7 和现有 D1 闭环已经验证 AdamW `0.001` 能稳定反向传播；
2. global batch 改变已经是一次较大的优化变化；
3. 未经门禁直接把 AdamW 学习率提高 8 倍会引入 NaN、Router 饱和和检测头不稳定风险；
4. 本次 P0 优先获得完整、稳定、可复现的正式结果。

前三个 epoch 属于预先登记的 warmup/健康观察窗口，但不得在同一个 run 中根据曲线临时改学习率。如果 loss 非有限或完全不下降，停止并将该运行记为失败；任何 `lr0=0.002/0.004/0.008` 试验必须使用新的运行身份，不能覆盖主实验。

### 7.3 数据增强

为保持 WP0 的冻结特征与图像几何对齐，以下增强全部关闭：

```text
hsv_h = hsv_s = hsv_v = 0
degrees = translate = scale = shear = perspective = 0
flipud = fliplr = 0
mosaic = mixup = copy_paste = erasing = 0
multi_scale = false
```

不得在训练图像上执行会改变几何或颜色、但缓存特征没有同步变化的增强。

## 8. Latent aux 固定合同

| 参数 | 值 |
| --- | ---: |
| `balance_loss_coeff` | 0.01 |
| `router_z_loss_coeff` | 0.001 |
| `latent_aux_gain` | 0.1 |
| `mixture_aux_budget` | 3.0 |

训练总损失必须通过 `collect_aux_loss(include_kinds=...)` 显式收集 latent aux。每个 epoch 记录：

- P3/P4/P5 balance loss；
- P3/P4/P5 router z-loss；
- mixture aux 总值和加权后贡献；
- Router 平均概率、熵和专家使用率；
- residual gain；
- 三个尺度 Router 与 residual gain 相对初始化的参数变化。

正式运行期间不调整上述系数。P2 的系数扫描属于后续实验。

### 8.1 运行中确认的辅助损失广播偏差

当前正式运行绑定代码提交 `e36864dbea38e60bd7e5f0202bff7d1c5fb62f6f`。运行进行至第 29 个 epoch
时复核损失执行链，确认 `CompositeCriterion` 的“只加入一次模型级 routed aux”设计意图与 D1
实际反向传播公式存在偏差。

`E2ELoss` 返回形状为 `(3,)` 的检测损失向量：

```text
native_loss = [L_box, L_cls, L_reg]
```

`mixture_aux_loss` 是标量。当前实现执行 `total = native_loss + aux`，PyTorch 会把标量广播到
三个分量；Trainer 随后执行 `loss.sum()`。因此：

```text
设计意图：L_total = L_box + L_cls + L_reg + 1 * L_mixture
当前实现：L_total = L_box + L_cls + L_reg + 3 * L_mixture
```

`results.csv` 只记录一份 `train/mixture_aux_loss`，不能直接反映三次计入。还需注意，原生检测
loss 的反向传播向量已经乘以每 rank 的本地 batch，而 CSV 中的 box/class/regression 是 detached
显示项；因此不能把 CSV 四列直接相加后解释为辅助损失占比。

截至已完成的 epoch 28，`train/mixture_aux_loss=0.01215`，对应进入 `loss.sum()` 的辅助标量总量
为 `0.03645`。所有检测与 aux 指标仍有限，未观察到该偏差导致数值爆炸，但它违反“辅助项只加入
一次”的严格实验合同。本运行保持 as-run 状态，不在中途热修改代码；最终报告必须披露该偏差，
不得把本运行表述为修正后正式结果。

后续修复与验证要求：

1. 对向量原生损失使用 `native_loss + aux / native_loss.numel()`，或先将原生损失求和后再加入一次
   `aux`；
2. 新增向量 criterion 测试，断言 `total.sum() == native_loss.sum() + aux`；
3. 单独明确 aux 是否应随本地 batch 缩放，避免每卡 batch 改变时有效正则强度隐式变化；
4. 修复后使用新的代码提交、preflight 和 run identity 重新执行正式实验，不能覆盖当前曲线。

## 9. 训练工作量与时间预算

训练样本暴露量：

```text
118,287 images/epoch × 100 epochs = 11,828,700 training images
5,000 val images/epoch × 100 epochs = 500,000 validation images
总处理量约 12,328,700 images
```

按短基准 `848.12 images/s`：

```text
纯训练约 3.87 h
训练 + 每轮验证约 4.04 h
增加 15% 启动、checkpoint 和波动余量约 4.64 h
```

由于短基准没有跨越完整缓存，正式资源窗口按 **5～8 小时**预留。对应六卡 GPU-hours 约 **30～48 GPU-hours**。第 12 节长基准若低于 500 images/s，则保守总时间会超过约 7.9 小时，必须停止在启动门禁并重新评估 I/O，不自动进入正式训练。

## 10. 缓存读取、主机内存与 GPU 约束

服务器容器内存上限约 176 GiB，无法同时容纳 423.5 GiB 完整特征缓存。Linux 会用可回收 file cache 加速最近访问的 shard，并在接近上限时回收旧页面。这是正常行为，但必须避免多任务随机读取造成页面抖动。

正式规则：

- 六个 rank 只运行一个任务，避免三套随机采样流互相竞争；
- 24 个 worker 固定，不动态增加；
- `max_open_shards_per_worker=4`、`prefetch_factor=1` 固定；
- 缓存目录只读，不在训练期间校验全部 SHA256；
- 不在训练中执行 `drop_caches`、内存压力程序或手动清理页缓存；
- 监控必须分开记录 anonymous RSS 和 file cache；
- file cache 接近 cgroup 上限本身不判失败，持续 data wait、major fault 激增、OOM/failcnt 增长才是问题；
- GPU keeper 在正式任务开始前暂停，在任务结束或失败退出时恢复；
- 所有训练进程结束后再定向释放 train/val 特征文件页。

每卡 batch 64 的短测峰值显存最高约 7.06 GiB。正式训练包含 EMA、验证和 checkpoint，启动门禁要求每张 A40 始终至少保留 8 GiB 显存余量；任何 rank 峰值超过 40 GiB 或出现 CUDA OOM 都必须失败关闭。

## 11. 正式训练代码合同

`wp8-formal-coco2017.yaml` 和 `run_wp8_train.py` 必须共同执行以下合同；YAML、代码门禁、identity 和测试必须在同一个干净提交中保持一致，不得通过命令行绕过：

1. 将合同 schema 升级为 `d1-wp8-train-v2`，避免旧 batch 48 identity 被误恢复；
2. `hardware.world_size=6`、`devices="0,1,2,3,4,5"` 保持不变；
3. 将 `train.batch` 和 `train.nbs` 同时改为 384；
4. 在配置或解析报告中明确 `per_gpu_batch=64`、`gradient_accumulation=1`；
5. 将 `run_wp8_train.py` 的硬门禁改为 batch/nbs 384，并校验可整除 world size；
6. identity 增加 per-GPU batch、实际 accumulation、optimizer 全参数、aux 系数和 DDP policy；
7. preflight 验证六张可见 GPU 都是 A40、无其他计算进程、每卡显存余量满足要求；
8. preflight 验证 train/val 缓存证据、COCO 列表数量、配置和模型 SHA256；
9. 恢复逻辑拒绝 batch 48、不同 schema、不同 commit、不同缓存或不同 seed 的 checkpoint；
10. 汇总器记录实际 optimizer step、样本 seen、AMP scale、data wait、显存和 Router 变化；
11. 增加正式任务 supervisor，负责 PID、状态、GPU 遥测、keeper 暂停/恢复和训练后缓存释放；
12. 更新 `tests/test_d1_wp8_formal_training.py`，覆盖 batch 拆分、identity、恢复拒绝和汇总门禁；
13. 运行 WP0-WP8、checkpoint、缓存、recovery 回归测试及 `git diff --check`；
14. 在干净 commit 上重新生成 preflight，训练期间不得修改代码或配置。

改造只改变 WP8 正式运行合同和工程门禁，不修改 Teacher、Adapter、LatentMixture、Detect 或损失定义。

## 12. 正式训练前的 308-step 长基准

### 12.1 目的

正式训练前使用最终代码 commit、六卡、每卡 batch 64、workers 4 和正式本地缓存运行一次接近完整 train2017 epoch 的基准：

```text
308 steps × global batch 384 = 118,272 images
```

该规模只比 train2017 少 15 张图片，读取量约 423 GiB，足以跨越 176 GiB 容器页缓存上限，暴露短基准无法发现的持续磁盘读取和页面回收问题。前 20 step 作为 warmup，后 288 step 用于计算稳态吞吐。

### 12.2 目标命令

```bash
export D1_REPO=/root/yolo-master/repo
export D1_WORKSPACE=/data/yingxi/yolo-master-d1
export D1_CACHE_ROOT=/root/yolo-master/datasets/d1_feature_cache
export D1_COMMIT=$(git -C "$D1_REPO" rev-parse --short HEAD)
export D1_BENCH="wp8-b384-six-gpu-long-$D1_COMMIT"

cd "$D1_REPO"

nohup /root/yolo-master/.conda/d1/bin/python \
  scripts/d1/benchmark_wp8_training.py \
  --workspace "$D1_WORKSPACE" \
  --data-root "$D1_WORKSPACE/datasets/coco" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --run-id "$D1_BENCH" \
  --world-size 6 --seed 0 --sample-offset 0 \
  all --per-gpu-batches 64 --worker-candidates 4 \
  --steps 308 --warmup-steps 20 \
  --candidate-timeout-seconds 3600 --memory-headroom-gib 8 \
  >"$D1_WORKSPACE/logs/$D1_BENCH.log" 2>&1 &

echo $! >"$D1_WORKSPACE/logs/$D1_BENCH.pid"
```

实际启动必须由 supervisor 暂停 keeper、采集每秒 GPU 指标并在退出时恢复 keeper。由于任务预计超过 3 分钟，只确认启动成功一次后退出会话，不持续轮询。

### 12.3 长基准通过条件

1. 六个 rank 均完成 308 step；
2. loss、梯度、AMP scale 和所有计时值有限；
3. 无 CUDA OOM、NCCL、DataLoader、safetensors、文件描述符或 cgroup OOM 错误；
4. 每卡实际 batch 为 64，global batch 为 384，accumulation 为 1；
5. 每张 GPU 峰值显存不超过 40 GiB；
6. cgroup anonymous RSS 保留至少 8 GiB 余量，OOM/failcnt 不增长；
7. 稳态 aggregate throughput 不低于 500 images/s；
8. 平均 data wait ratio 不高于 10%，且后半程没有持续恶化；
9. 按 train + 每 epoch val + 15% 余量估计总时长不超过 8 小时；
10. 退出后无残留 rank/worker，keeper 已恢复，file cache 已定向释放。

长基准完成后必须先报告吞吐、每 rank step time、data wait、GPU 利用率、功耗、显存、主机 RSS/file cache、major fault 和新 ETA，并等待用户明确确认。基准不得自动串联正式训练。

### 12.4 本次启动决策记录

2026-09-05，用户在查看六卡每卡 batch 64 的历史实测、双卡并发实测和单卡 batch 256 实测后，明确选择六卡 global batch 384，并明确要求启动正式训练。该指令视为本次正式运行的最终人工批准。

本次运行复用已经成功的同硬件、同 batch、同 workers、同 AMP 和同本地缓存路径六卡短基准作为启动证据，不再额外重复 308-step 独立长基准。此例外必须如实保留：不能把 308-step 基准写成已经完成。正式训练的第一个完整 epoch 用于验证跨越完整 train2017 缓存后的实际吞吐、data wait、显存和 cgroup 行为；如出现 OOM、非有限 loss、持续 I/O 停滞或 rank 退出，则正式任务失败关闭，不在原运行中调整 batch 或 workers。
## 13. 正式 preflight

长基准通过并得到确认后，正式 preflight 必须满足：

1. Git 工作区干净，HEAD 与计划登记 commit 完全一致；
2. 配置 schema、配置 SHA256 和模型 YAML SHA256 已锁定；
3. 六张 GPU 均为空闲 A40，GPU 配置为 `0,1,2,3,4,5`；
4. 指定 master port 未被占用；
5. 没有其他 WP8、训练、缓存构建或内存压力进程；
6. train/val COCO 数量分别为 118,287/5,000；
7. train/val 缓存样本数、合同摘要、内容摘要、shard 集合和文件大小正确；
8. 缓存目录无 `.part` 文件，且以只读方式使用；
9. 外部工作区具有足够空间保存 checkpoint、日志、遥测和报告；
10. 运行目录不存在，或已存在目录的 identity 与本次完全一致；
11. global batch 384、per-GPU batch 64、nbs 384、accumulation 1 被解析报告明确确认；
12. optimizer、scheduler、augmentation 和 aux 参数与第 7～8 节一致；
13. preflight 报告明确写入 `approval_required_before_training=true`；
14. 配置、日志和 identity 不包含密码、令牌或个人凭据。

## 14. 正式目录与命名

```text
/data/yingxi/yolo-master-d1/
  runs/wp8-p0-b384-s0-<commit>/
    inputs/
    weights/best.pt
    weights/last.pt
    results.csv
  manifests/wp8-p0-b384-s0-<commit>/
    preflight.json
    epochs/
    validation/
    summary.json
    final-eval.json
  logs/
    wp8-p0-b384-s0-<commit>.log
    wp8-p0-b384-s0-<commit>.pid
    wp8-p0-b384-s0-<commit>.status
    wp8-p0-b384-s0-<commit>-gpu.csv
```

缓存和 checkpoint 不进入 Git。Git 只提交脱敏后的配置、摘要、指标 CSV/JSON、测试结果和文档。

## 15. 目标执行流程

以下命令以第 11 节改造完成后的最终接口为准。正式运行 ID 必须绑定最终干净 commit。

### 15.1 公共变量

```bash
export D1_REPO=/root/yolo-master/repo
export D1_WORKSPACE=/data/yingxi/yolo-master-d1
export D1_PYTHON=/root/yolo-master/.conda/d1/bin/python
export D1_CACHE_ROOT=/root/yolo-master/datasets/d1_feature_cache
export D1_COMMIT=$(git -C "$D1_REPO" rev-parse --short HEAD)
export D1_RUN="wp8-p0-b384-s0-$D1_COMMIT"
export D1_RUN_ROOT="$D1_WORKSPACE/runs/$D1_RUN"
export D1_REPORT_DIR="$D1_WORKSPACE/manifests/$D1_RUN"
```

### 15.2 生成 preflight

```bash
cd "$D1_REPO"

"$D1_PYTHON" scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --data-root "$D1_WORKSPACE/datasets/coco" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  prepare
```

检查 `preflight.json` 后再次确认：

```text
world_size=6
per_gpu_batch=64
global_batch=384
nbs=384
gradient_accumulation=1
seed=0
epochs=100
```

### 15.3 后台启动正式训练

正式启动必须使用 supervisor 封装以下核心命令，并负责状态、遥测、keeper 和退出清理：

```bash
cd "$D1_REPO"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
"$D1_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=6 \
  --master_port=29518 \
  scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --data-root "$D1_WORKSPACE/datasets/coco" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  train
```

正式任务预计超过 3 分钟。启动时只检查一次：六个 rank 存活、GPU 0～5 均进入训练、状态为 `RUNNING`、日志没有立即错误；随后退出会话，由用户通过第 16 节命令检查。

## 16. 状态检查

```bash
cat "$D1_WORKSPACE/logs/$D1_RUN.status"

pid=$(cat "$D1_WORKSPACE/logs/$D1_RUN.pid")
kill -0 "$pid" 2>/dev/null && echo RUNNING || echo FINISHED_OR_FAILED

tail -n 30 "$D1_WORKSPACE/logs/$D1_RUN.log"
tail -n 5 "$D1_RUN_ROOT/results.csv"

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader
```

状态文件至少包含 PID、运行 ID、commit、当前 epoch、最后更新时间和最终退出码。父进程退出不等于实验通过，最终以 `summary.json` 和 `final-eval.json` 为准。

## 17. Checkpoint 与恢复

### 17.1 保存策略

- `last.pt`：每个 epoch 更新，用于中断恢复；
- `best.pt`：按框架固定 fitness 选择，用于最终评测；
- 每 10 epochs 保存一个周期 checkpoint；
- 保存 `results.csv`、epoch 遥测和 validation 报告；
- checkpoint 写入失败、大小为零或无法严格加载时立即失败。

### 17.2 恢复前检查

恢复前必须确认：

1. 原训练进程及其 DataLoader worker 已全部退出；
2. `last.pt` 可读取，epoch 合法，模型类型为 `D1FoundationDetectionModel`；
3. checkpoint 不含 Teacher 参数；
4. commit、schema、配置 SHA256、模型 SHA256、缓存摘要、seed 和 batch identity 完全一致；
5. 六张 A40 可用，仍使用 world size 6；
6. 不从不同 seed、不同 batch 或不同代码版本 checkpoint 恢复。

### 17.3 恢复命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
"$D1_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=6 \
  --master_port=29518 \
  scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --data-root "$D1_WORKSPACE/datasets/coco" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  train --resume "$D1_RUN_ROOT/weights/last.pt"
```

恢复只能补完同一个 100-epoch 合同，不能修改剩余 epoch、batch、学习率或增强。不得从多个恢复片段中择优拼接曲线。

## 18. 每 epoch 遥测与异常判定

每个 epoch 至少记录：

- rank、epoch、batch_count 和 optimizer_steps；
- train step time、data wait time、images/s；
- 每 rank 峰值显存和 AMP scale；
- box、class、DFL、balance、z-loss 和 latent aux；
- val2017 seen 数及检测指标；
- Router 概率、熵、专家使用率和 residual gain；
- checkpoint 保存结果；
- cgroup RSS、file cache、major fault、failcnt；
- GPU 利用率、功耗和温度的分钟级摘要。

立即失败条件：

- 任意 loss、梯度或指标出现 NaN/Inf；
- AMP scale 持续下降到无法有效更新；
- 任意 rank OOM、NCCL 超时、退出或失去同步；
- 缓存 sample ID/shape/dtype/checksum 不匹配；
- Teacher 参数进入 optimizer/checkpoint；
- checkpoint 无法保存或严格加载；
- 运行 identity 在训练期间变化。

性能下降但结果仍有限时，不在原 run 中自动调参。记录原因并由用户决定继续或停止。

## 19. 最终评测与汇总

训练完成后：

1. 确认 `results.csv` 恰好包含 100 个 epoch；
2. 确认最后一次常规验证处理 5,000 张 val2017 图片；
3. 对 `best.pt` 做健康检查和严格 state-dict 重载；
4. 使用相同 val2017、640 输入、无增强设置重新执行一次最终评测；
5. 对 `last.pt` 也执行严格加载检查，但主精度报告使用 `best.pt`；
6. 计算 checkpoint SHA256；
7. 比较初始和最终 Router/residual gain，证明参数发生有限非零更新；
8. 汇总总墙钟时间、训练时间、验证时间和六卡 GPU-hours；
9. 训练和评测进程全部退出后，统一定向释放 train/val 特征文件页；
10. 生成脱敏 Git 证据，不提交大 checkpoint。

目标汇总命令：

```bash
"$D1_PYTHON" scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --data-root "$D1_WORKSPACE/datasets/coco" \
  --train-cache "$D1_CACHE_ROOT/coco2017-train2017-d1-cache-v1" \
  --val-cache "$D1_CACHE_ROOT/coco2017-val2017-d1-cache-v1" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  summarize
```

## 20. 验收标准

WP8 通过必须同时满足：

1. identity 为 world size 6、每卡 batch 64、global batch/nbs 384、seed 0；
2. 完成 100 epochs，训练记录连续且无重复/缺失 epoch；
3. 所有检测和 latent aux loss 有限；
4. 最终 val2017 评测覆盖全部 5,000 张图片；
5. 报告 mAP50-95、mAP50、Precision 和 Recall，不隐去低精度结果；
6. Adapter、三个 LatentMixture、Router、residual gain 和 Detect 均发生有限非零更新；
7. `best.pt`、`last.pt` 可严格加载，Teacher 参数计数为 0；
8. train/val 缓存和代码/config identity 在全程保持一致；
9. 六个 rank 无 OOM、NCCL 或未处理异常；
10. 日志、results.csv、epoch 遥测、最终评测、checkpoint 摘要和异常记录齐全；
11. 训练结束后无残留训练/worker 进程，keeper 恢复，缓存页完成定向清理；
12. Git 只包含脱敏小型证据，不包含数据集、缓存或 checkpoint。

P0 任务书没有规定绝对 mAP 门槛，因此工程验收不使用事后选择的精度阈值。精度无论高低都必须如实报告；若精度不理想，作为后续优化问题处理，不能通过删除结果或改变同一运行配置来“修正”。

## 21. 交付物

正式训练完成后提交：

- 更新后的 `WP8.md` 完成记录；
- 最终正式 YAML 和 SHA256；
- 脱敏 `preflight.json` 和 run identity；
- 100 epochs `results.csv`；
- 最终 COCO 指标 JSON；
- loss、mAP、吞吐、显存、data wait 和 Router 行为汇总；
- `best.pt`、`last.pt` 的文件大小、SHA256 和严格加载证据；
- 训练/验证墙钟时间和 GPU-hours；
- 中断、恢复和异常记录；
- 测试命令、测试数量和 `git diff --check` 结果；
- checkpoint 的外部存储位置说明，但不把 checkpoint 提交 Git。

## 22. 代码与证据基线

既有关键提交：

- `6e9eb22`：WP8 正式训练准备；
- `ead6b59`：缓存训练性能优化；
- `7b4fc75`：训练后文件页缓存释放；
- `440d868`：可配置 GPU 数量的训练基准；
- `164d460`：D1 DDP 静态图和未使用参数扫描优化；
- `8f6c458`：并发基准样本流隔离。

主要文件：

- [`run_wp8.py`](../../scripts/d1/run_wp8.py)：六卡缓存构建、校验和合并；
- [`benchmark_wp8_training.py`](../../scripts/d1/benchmark_wp8_training.py)：训练吞吐与 batch/worker 基准；
- [`run_wp8_train.py`](../../scripts/d1/run_wp8_train.py)：正式 preflight、训练、恢复、遥测和汇总；
- [`wp8-formal-coco2017.yaml`](../../ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml)：六卡、每卡 batch 64、global batch 384 的正式训练合同；
- [`test_d1_wp8_formal_training.py`](../../tests/test_d1_wp8_formal_training.py)：正式合同和恢复测试；
- [`manifests/wp8-full-cache.json`](manifests/wp8-full-cache.json)：完整缓存 Git 证据。

本文件描述的是待执行合同，不得把已有短基准写成正式训练完成。

## 23. 最终审核清单

正式训练启动前需要用户再次明确确认：

1. 接受六张 A40 全部用于单个 DDP 任务；
2. 接受每卡 batch 64、global batch/nbs 384、accumulation 1；
3. 接受 WP8 当前只执行 seed 0 主实验；
4. 接受 AdamW `lr0=0.001` 不做未经验证的线性放大；
5. 接受 100 epochs、每 epoch 完整 val2017 和无数据增强合同；
6. 接受固定 latent aux 系数；
7. 接受先完成 308-step 长基准，基准不会自动启动正式训练；
8. 接受正式时间暂按 5～8 小时预留，以长基准为准；
9. 接受正式运行中不调参，只允许同 identity checkpoint 恢复；
10. 接受 P0 不设事后绝对 mAP 门槛，最终精度如实报告。

上述清单是 P0 原始启动门禁，现作为历史合同保留。P0 已按第 12.4 节例外启动并完成 30 epochs，
目前处于安全停止状态；在第 8.1 节问题修复前不得把它按原身份恢复为修正后正式实验。

## 24. 课题目标 P1：同参数量从零训练检测器对照

### 24.1 名称与实验问题

本节的“P1”指任务书中的课题目标等级 P1，不是检测金字塔中的 P1/P2/P3 层。

第一阶段只执行 `P1-COCO-30` 诊断性对照，回答：

1. 在相同 COCO split、输入尺寸、数据暴露量、optimizer step 和随机种子下，一个与 D1 下游模型
   可训练参数量相差不超过 1% 的 RGB 检测器从零训练 30 epochs 后能达到什么精度；
2. 当前冻结 DINOv3 特征方案相对于从零训练方案保留了多少精度；
3. 两者的训练墙钟时间、GPU-hours、峰值显存、吞吐和存储成本有何差异；
4. 当前 P0 的低精度更接近冻结特征信息瓶颈，还是同训练配方下普遍优化不足。

这是一组单数据集、单 seed、30 epochs 的先导实验，不用于直接宣称 P1 已完成，也不用于事后修改
P0 或 scratch 配方。

### 24.2 当前冻结方案的固定参照

对照只允许引用以下 checkpoint，不得从中途 checkpoint 事后挑选更有利结果：

| 项目 | P0 阶段性参照 |
| --- | --- |
| run ID | `wp8-p0-b384-s0-e36864d` |
| code commit | `e36864dbea38e60bd7e5f0202bff7d1c5fb62f6f` |
| checkpoint | 第 30 epoch 的 `best.pt`，与 `last.pt` 内容一致 |
| checkpoint SHA256 | `8391dbbb9794bff91184fb2e1ae34b8ecf3c92c761ef7f248d8e3806d79e096d` |
| 可训练参数 | 3,542,567 |
| 冻结 Teacher | DINOv3 ViT-S/16，约 21.6M 参数，不进入 optimizer/checkpoint |
| 完成 epochs | 30 |
| 理论 optimizer steps | 309/epoch，共 9,270 |
| `results.csv` 累计训练时间 | 31,182.8 秒，约 8.66 小时 |
| 六卡原始 GPU-hours | 约 51.97，不含缓存准备 |

独立完整 val2017 评测固定为：

| 指标 | 数值 |
| --- | ---: |
| Ultralytics Precision | 0.33974 |
| Ultralytics Recall | 0.25353 |
| Ultralytics mAP50 | 0.22301 |
| Ultralytics mAP50-95 | 0.11082 |
| COCO 官方 AP50 | 0.230556 |
| COCO 官方 AP50-95 | 0.116156 |
| COCO 官方 AP-small | 0.052458 |
| COCO 官方 AP-medium | 0.123415 |
| COCO 官方 AP-large | 0.161929 |

上述累计训练时间是框架 `results.csv` 的 as-run 时间，正式对照表还应优先使用 supervisor
起止时间和 GPU 遥测交叉验证。P0 含第 8.1 节的 aux 三次计入偏差，数值影响初步判断很小，但它
使本次比较只能标记为 preliminary/as-run。

### 24.3 同参数量 scratch 模型

从零训练基线派生自标准
[`yolo26.yaml`](../../ultralytics/cfg/models/26/yolo26.yaml)，不使用
`yolo26-master-n.yaml`、预训练权重、DINOv3、特征缓存、LatentMixture 或 Teacher。

预注册模型配置为：

```yaml
nc: 80
end2end: true
reg_max: 1
scales:
  n: [0.75, 0.26, 1024]
```

远端构造探针得到该模型有 3,510,624 个可训练参数，与 D1 下游模型相差 31,943 个，即
`-0.902%`，满足预先规定的 `abs(delta) <= 1%`。参数量比较按全部
`requires_grad=True` 参数进行，不通过注册但不参与前向的无用参数凑数。

现成模型不作为主对照的原因：

| 候选 | 可训练参数 | 相对 D1 |
| --- | ---: | ---: |
| 标准 YOLO26n | 2,572,280 | -27.41% |
| 预注册 matched YOLO26 | 3,510,624 | **-0.90%** |
| YOLO26-P6n | 4,063,872 | +14.72% |
| YOLO26-Master-n | 5,115,336 | +44.40% |

已新增的模型文件为
`ultralytics/cfg/models/26/yolo26-d1-scratch-matched-n.yaml`。实现后必须重新构造模型、
记录精确参数量和模型 YAML SHA256；如果实际参数差超过 1%，preflight 必须失败，不得启动训练。

### 24.4 公平性合同

除模型输入和架构外，两边固定：

| 项目 | 冻结 DINOv3 P0 | scratch P1 |
| --- | --- | --- |
| 数据 | COCO 2017 train2017/val2017 | 完全相同 |
| train/val 数量 | 118,287 / 5,000 | 完全相同 |
| 输入尺寸 | 640×640 | 640×640 |
| seed | 0 | 0 |
| epochs | 已完成 30 | 固定 30 |
| global batch / nbs | 384 / 384 | 384 / 384 |
| GPU | 6×A40 | 同一台服务器的 6×A40 |
| optimizer | AdamW | AdamW |
| lr0 / lrf | 0.001 / 0.01 | 0.001 / 0.01 |
| scheduler | 100 epoch cosine 计划的前 30 epoch | 同样保留 100 epoch 调度跨度 |
| warmup | 3 epochs | 3 epochs |
| weight decay | 0.0005 | 0.0005 |
| AMP | 开启 | 开启 |
| 数据增强 | 全部关闭 | 全部关闭 |
| 每轮验证 | 完整 val2017 | 完整 val2017 |
| checkpoint 选择 | 前 30 epochs 的 best 和 epoch30 | 同口径 |
| 官方评测 | COCO evaluator | 同一 evaluator 和参数 |

scratch 使用标准 RGB LetterBox 和 `[0,1]` 输入，不使用 DINO 的 ImageNet normalization。
这是模型输入协议的必要差异，不是额外数据增强。两边均不允许 mosaic、mixup、copy-paste、翻转、
仿射、颜色扰动、多尺度或额外训练数据。

**调度纠正（2026-09-05）：** 框架实际传入 `epochs=100`，通过回调在第 30 epoch 的完整验证、
checkpoint 保存完成后停止，不能传入 `epochs=30`。除了 cosine 学习率，`E2ELoss` 的
one-to-many / one-to-one 权重衰减也读取 `args.epochs`，因此两项均须保留 100 epoch 跨度。
第 30 epoch（零基 29）的普通参数组学习率约为 `0.000808389`。scratch 没有 Router 或 Expert
参数组，不复制 P0 特有的 Router / Expert 学习率倍率；这里只匹配普通参数组的基础优化合同。
warmup bias LR 为 `0.1`，warmup momentum 为 `0.8`，均已核对 P0 实际 `args.yaml`。
scratch 使用原始 RGB 图像直接 LetterBox，不走默认加载器的预先缩放，也不使用矩形验证；
图像与检测框共同变换，反投影使用实际 gain/padding。

相同 epochs 和 global batch 对应相同的理论数据暴露量：

```text
118,287 × 30 = 3,548,610 原始样本暴露
ceil(118,287 / 384) = 309 optimizer steps/epoch
309 × 30 = 9,270 optimizer steps
```

DDP sampler 为对齐 rank 产生的少量补齐样本必须在两边以相同规则处理，并报告实际 seen 数。

### 24.5 scratch 固定训练配置

已新增
`ultralytics/cfg/experiments/d1/wp8-p1-scratch-coco2017.yaml`，固定：

```text
schema                  d1-wp8-p1-scratch-v1
model                   yolo26-d1-scratch-matched-n.yaml
pretrained              false
resume                  false（首次运行）
world_size              6
per_gpu_batch           64
global_batch / nbs      384 / 384
gradient_accumulation   1
train.epochs            100（学习率与 E2E loss 调度跨度）
window_epochs           30（第 30 epoch 保存后停止）
workers                 由启动前 4/8 短基准锁定
AMP                     true
optimizer               AdamW
lr0 / lrf               0.001 / 0.01
warmup                   3 epochs
cosine scheduler        true
patience                100，不允许提前停止
save_period             10
validation              每 epoch 完整 val2017
compile                 false
dataset RAM cache       false
augmentation            全关闭
```

tracked YAML 不写服务器绝对路径。COCO 根目录、run root、report root 和设备通过运行参数注入。
scratch 只从随机初始化开始；出现任何非空 pretrained checkpoint、自动 optimizer 选择或参数冻结
都必须失败关闭。

### 24.6 实施与启动门禁

当前已新增配置、准备/受门禁约束的训练入口及离线测试；六卡 benchmark、独立官方 evaluate
与自动 summarize 尚待后续实现和实测，不属于本次已完成项：

- `scripts/d1/run_wp8_p1_control.py`：当前支持 prepare 和 train；train 默认不获批准；
- `tests/test_d1_wp8_p1_control.py`：参数匹配、无预训练、配置合同、预处理、identity、训练门禁与严格重载；
- 模型 YAML 与训练合同 YAML；
- 外部工作区中的日志、checkpoint、官方预测 JSON 和完整遥测；
- Git 中脱敏后的 P1 摘要，正式结果完成后再更新本节。

执行顺序固定：

1. 构造 scratch 模型并断言参数差不超过 1%，检查 Detect 为 P3/P4/P5、`nc=80`、
   `reg_max=1`、`end2end=true`；
2. 验证没有载入预训练权重、没有 DINO/Teacher/LatentMixture 参数和特征缓存依赖；
3. 将 COCO 图片放在本地只读存储或使用已经校验的本地副本，列表 SHA256 必须与 WP0 一致；
4. 六卡分别以每卡 batch 64 做 20 个 warmup 后的短基准，workers/rank 只比较 4 和 8；
5. 在不 OOM 且数据等待稳定的组合中选择吞吐更高者，并把选择写入 immutable identity；
6. 执行一个 batch 的 loss、backward、optimizer step、AMP、checkpoint 严格重载和完整
   val dataloader smoke；
7. 生成 preflight，记录代码/config/model/data 摘要、参数量、GPU、batch、worker 和 seed；
8. 向用户报告基准吞吐、显存和 30 epochs ETA，等待明确启动确认；
9. 获得确认后通过 supervisor 后台启动，只检查六个 rank 正常、GPU 已进入训练且首个 loss 有限；
10. 任务预计超过 3 分钟，正常挂起后退出会话，不持续轮询。

短基准只允许选择 I/O worker 数，不能据此改变 batch、模型、学习率或 epoch。若每卡 batch 64
OOM，则停止并报告；只有通过新的预注册方案使用梯度累积保持 global batch 384，不能在原 identity
中临时降 batch。

### 24.7 运行身份与目录

运行标识：

```text
wp8-p1-coco30-scratch-b384-s0-<commit>
```

外部目录：

```text
/data/yingxi/yolo-master-d1/
  runs/<run-id>/
    weights/best.pt
    weights/last.pt
    results.csv
  manifests/<run-id>/
    preflight.json
    benchmark.json
    identity.json
    official-coco-best.json
    official-coco-epoch30.json
    summary.json
  logs/
    <run-id>.log
    <run-id>.pid
    <run-id>.status
    <run-id>-gpu.csv
```

checkpoint、预测明细和日志不进入 Git。Git 只提交配置、代码、测试和脱敏摘要。

### 24.8 精度评测与解释规则

scratch 完成后对 `best.pt` 和 epoch30 checkpoint 分别严格重载，完整评测 5,000 张
val2017。主表同时报告：

- Precision、Recall、Ultralytics mAP50 和 mAP50-95；
- COCO 官方 AP、AP50、AP75、AP-small、AP-medium、AP-large；
- 最佳 epoch、epoch30 指标和两者差值；
- 训练集固定 5,000 图诊断指标，但不得把它表述为完整 train AP；
- checkpoint SHA256、预测数和实际评测图片数。

主要比较采用“同窗口 best 对 best”；同时提供“epoch30 对 epoch30”，避免不同收敛速度被
checkpoint 选择掩盖。预注册计算：

```text
AP 保留率(%) = AP_P0_frozen / AP_P1_scratch × 100
AP50 保留率(%) = AP50_P0_frozen / AP50_P1_scratch × 100
绝对 AP 差 = AP_P0_frozen - AP_P1_scratch
```

解释规则：

- scratch train/val AP 都明显高于 P0：优先判断当前冻结特征/Adapter/融合结构是瓶颈；
- 两者 train/val AP 都低：优先判断 global batch、学习率、更新数或无增强配方导致共同欠拟合；
- scratch train AP 高而 val AP 低：优先判断从零训练泛化不足；
- P0 的 AP-small 差距远大于 AP-medium/large：进一步支持 stride-8 信息不足假设；
- 单 seed 的小差异不作显著性结论，不根据结果反向改变本次实验定义。

### 24.9 训练成本与显存口径

必须分别报告以下三种成本，不能只选择对冻结方案最有利的一种：

1. **重复训练成本**：只计下游 30 epochs 训练和逐 epoch 验证；
2. **首次端到端成本**：冻结方案额外计入 DINOv3 train/val 特征抽取与最终校验；
3. **摊销成本**：缓存被 `N` 次实验复用时，按 `cache_cost/N` 计入每次冻结实验。

当前完整缓存证据为 123,287 样本、约 423.5 GiB。首次准备实测：

```text
train cache worker wall = 2,178.90 s
train final verification = 1,326.90 s
val cache worker wall = 79.92 s
val final verification = 28.58 s
总墙钟近似 = 3,614.30 s，约 1.00 h
```

GPU-hours 只对实际使用 GPU 的抽取和训练阶段求和；纯 CPU/NVMe 校验不伪计为 GPU-hours。
成本公式为：

```text
训练 GPU 时降低率 = 1 - GPUh_P0_train / GPUh_scratch_train
首次 GPU 时降低率 = 1 - (GPUh_cache_extract + GPUh_P0_train) / GPUh_scratch_train
N 次摊销降低率 = 1 - (GPUh_cache_extract/N + GPUh_P0_train) / GPUh_scratch_train
墙钟降低率使用相同公式替换为 wall time
```

同时记录：

- 每个 rank 的峰值 allocated/reserved 显存和 `nvidia-smi` 峰值；
- aggregate images/s、step time、data wait、验证时间和 checkpoint 时间；
- 六卡利用率、功耗采样和总 GPU-hours；
- 主机匿名内存、file cache、major fault；
- 冻结方案额外 423.5 GiB 缓存空间，scratch 不产生该特征缓存。

如果降低率为负数，必须如实报告为成本增加。任务书的“GPU 时降低至少 50%”使用 GPU-hours，
不能用磁盘读取吞吐或单步延迟替代。

### 24.10 验收、结论边界与后续正式 P1

`P1-COCO-30` 工程通过要求：

1. scratch 参数量与 P0 下游模型差不超过 1%；
2. 确认随机初始化、无预训练、无 Teacher 和无特征缓存输入；
3. 完成连续 30 epochs 和理论 9,270 optimizer steps；
4. loss、梯度、AMP scale 和指标全部有限；
5. 完整 val2017 评测覆盖 5,000 张图片；
6. `best.pt` 和 epoch30 checkpoint 可严格重载；
7. 六卡无 OOM、NCCL 或未处理 DataLoader 异常；
8. 参数、精度、墙钟、GPU-hours、显存和吞吐证据完整；
9. 使用第 24.8～24.9 节公式生成自动摘要，不手工改写数字；
10. Git 只提交脱敏的小型证据。

无论 scratch 精度高低，工程完成都不等同于 P1 科学验收完成。任务书层面的 P1 还需要：

1. 修复第 8.1 节 aux 合同后，以新 identity 完成可作为正式结论的冻结 P0；
2. 在同一最终代码基线上完成正式 matched scratch 对照；
3. 至少增加一个第二数据集，建议使用课题指定的 VisDrone；
4. 至少在一个数据集明确给出精度保留比例，并验证 GPU-hours 是否降低至少 50%；
5. 对关键结论增加重复 seed 或置信区间，避免单 seed 偶然性。

本节新增后只批准方案准备和短基准，不构成正式训练启动许可。完成实现、测试与短基准后，必须先
向用户报告实际 ETA，再等待明确的“开始 P1 对照训练”指令。

### 24.11 scratch 准备落实记录（2026-09-05）

本次仅准备代码与离线验收，不启动正式训练，不运行六卡 benchmark，不读取大特征缓存。
COCO 复制已在本轮期间完成：245,525 个文件、41,458,555,664 bytes 全部完成 SHA256 比对，
原件保留，正式目标目录已发布。train2017 的 118,287 张图片和 117,266 份标签、val2017 的
5,000 张图片和 4,952 份标签已与 WP0 manifest 对齐；少于图片数的标签文件是原数据中无检测
标注的图片，不应据此错误补造标注。
外部 `manifests/wp8-p1-scratch-preparation/preparation.json` 当前为
`data_ready_runtime_pending`，运行列表与 data YAML 已生成，`formal_training_approved=false`。
该记录绑定当前工作区摘要，当前尚未提交；正式实验需先锁定干净 commit 再生成新的运行记录。

| 文件 | 本次落实内容 |
| --- | --- |
| [模型 YAML](../../ultralytics/cfg/models/26/yolo26-d1-scratch-matched-n.yaml) | 标准 YOLO26 的参数匹配版本，随机初始化，无 Teacher/LatentMixture |
| [实验合同](../../ultralytics/cfg/experiments/d1/wp8-p1-scratch-coco2017.yaml) | 六卡、每卡 64、global batch/nbs=384；30 epoch 窗口与 100 epoch 调度分离 |
| [准备与训练入口](../../scripts/d1/run_wp8_p1_control.py) | 模型/数据检查、固定几何 RGB dataset、ScratchTrainer、有限值遥测、启动门禁 |
| [离线测试](../../tests/test_d1_wp8_p1_control.py) | 参数量、真实三尺度、标签/图像几何、调度、CPU loss/backward、checkpoint 与失败关闭 |

CPU 实测模型参数为 **3,510,624**，与 D1 的 3,542,567 相差 **-0.90169%**。
对 `[1,3,640,640]` 输入，送入 Detect 的特征为：

| 层 | 特征尺寸 | 来源 |
| --- | --- | --- |
| P3 | `[1,72,80,80]` | 原生 stride-8 特征经过 neck 融合 |
| P4 | `[1,136,40,40]` | 原生 stride-16 特征经过 neck 融合 |
| P5 | `[1,272,20,20]` | 原生 stride-32 特征经过 neck 融合 |

离线测试已验证：使用标准三项检测损失和 E2ELoss，不引入 latent aux；损失和梯度有限；
随机初始化参数可以更新；checkpoint 可以严格重载。CPU backward 使用 2 张合成 96×96 输入，
不能冒充真实 COCO、640 输入、每卡 batch 64 的六卡实测。

准备命令（在仓库根目录、D1 Python 环境中；下列变量由实际环境设置）：

```bash
# WORKSPACE 是外部实验工作区，DATA_ROOT 是复制后的 COCO 根目录。
# COPY_RECEIPT 指向复制任务的 .status.json。模型检查不需要读取数据集。
REPORT_DIR="$WORKSPACE/manifests/wp8-p1-scratch-preparation"
RUN_ROOT="$WORKSPACE/runs/wp8-p1-coco30-scratch-b384-s0-$(git rev-parse --short HEAD)"
python -m scripts.d1.run_wp8_p1_control prepare --model-only \
  --output-dir "$REPORT_DIR" --run-root "$RUN_ROOT"

# 仅在复制状态为 COMPLETED 后执行；4 是待基准核定的候选值。
python -m scripts.d1.run_wp8_p1_control prepare \
  --output-dir "$REPORT_DIR" --run-root "$RUN_ROOT" \
  --data-root "$DATA_ROOT" --copy-receipt "$COPY_RECEIPT" --workers 4
```

`preparation.json` 的 `model_ready_runtime_pending` 或 `data_ready_runtime_pending`
均不表示获准训练。正式启动必须具备干净代码 commit、相同文件摘要、对应 preparation 摘要、
六卡 worker 基准、真实 batch backward、checkpoint 严格重载、评测就绪记录和用户明确批准。
`--approved` 不能跳过这些门禁。本次不生成虚假的通过记录。

训练入口保留未剥离 optimizer 的 best/last checkpoint，在第 30 epoch 完整验证和保存后停止。
当前不开放自动 resume；已有 run 会被拒绝，异常恢复须另行审核并验证 optimizer/scaler/E2E
调度状态，不能用重新随机初始化覆盖已有实验。

待办顺序：复制完成检查 → 六卡 workers 4/8 基准与真实 batch 验收 → 独立官方评测入口就绪 →
锁定 worker、实测 ETA 和运行身份 → 用户确认 → 正式启动。任何预计超过 3 分钟的任务均后台
运行并保存 PID、状态和日志，确认正常启动一次后退出会话，不持续轮询。

本次相关回归：`70 passed`（包含 45 项 scratch 测试及 WP8 正式配置、诊断、
默认配置和 Master 模型回归）。服务器缺少 `ruff`、`codespell`，对应工具检查未能运行，
不能记为通过；另执行语法检查和 `git diff --check`。

### 24.12 启动授权与后台流水线（2026-09-05）

用户已明确授权“准备好可以正式启动训练”。该授权允许真实门禁通过后自动进入正式 scratch 训练，
不允许跳过门禁、修改 global batch 384、加载测速权重或恢复原 P0 任务。
第 24.11 节保留为上一阶段记录；以下为本次新增的实际执行路径。

新增 [launch_wp8_p1.py](../../scripts/d1/launch_wp8_p1.py)，提供 `all / worker / evaluate`，
并增加 [启动测试](../../tests/test_d1_wp8_p1_launch.py)。底层仍调用第 24.11 节的
ScratchTrainer，不修改共享训练引擎。当前相关离线回归为 **88 passed**；`py_compile` 与
命令行入口检查通过。ruff/codespell 仍缺失，不将缺失工具计为通过。

后台 `all` 的顺序与硬门禁：

1. 检查干净代码版本、已校验 COCO 副本、六张 A40 和官方评测依赖。
2. 使用原程序接口临时停止已有 GPU keeper；退出时恢复原来的 10% 保活配置。
3. 对 workers/rank=4、8 分别运行六卡真实 COCO 前缀训练：每卡 64，40 个 batch，前 20 个预热，
   后 20 个计时；每个 rank 必须完成 40 次成功参数更新，loss/梯度/参数变化有限，AMP 保持开启。
4. 前缀共 15,360 张训练图；每个候选均完整验证 5,000 张 val2017，并严格重载 best/last。
   为匹配正式实验，测速 warmup 的有效长度保持 `3×309=927` 个 batch，而不是按前缀缩短。
5. 两个候选必须通过，按慢速 rank 对应的 aggregate images/s 选择 worker 数；不同候选使用
   同一排序前缀且从随机初始化开始，测速 checkpoint 不用于正式训练。
6. 用选定候选的 checkpoint 独立运行全部 val2017 官方评测，验证预测 JSON、类别映射、
   图像数量与评测接口。空预测也须输出有效零精度结果，不能被当作接口失败或虚假成功。
7. 生成 benchmark 和 runtime-gate，绑定代码、preparation、checkpoint 与评测摘要。
8. 重新随机初始化正式模型，六卡 global batch 384，100 epoch 调度跨度内执行前 30 epoch。
9. 完成后独立评测正式 best/last，并生成 preliminary/as-run 对照摘要。P0 的 aux 广播偏差
   继续明确保留，不将本轮单 seed COCO 对照表述为完整 P1 已完成。

ETA 使用实测平均 batch 时间 × 309 batch/epoch，再加完整验证与 checkpoint 实测耗时，
乘以 30 epochs，并提供 15%～35% 余量区间。它仍是本地热数据前缀估计，完整训练实际数据分布、
I/O 与验证成本可能不同；最终两次独立官方评测另计。测速、门禁和正式训练计时分开记录。

运行方式（变量含义同第 24.11 节）：

```bash
RUN_ID="wp8-p1-coco30-scratch-b384-s0-$(git rev-parse --short HEAD)"
python -m scripts.d1.launch_wp8_p1 all \
  --workspace "$WORKSPACE" --data-root "$DATA_ROOT" --copy-receipt "$COPY_RECEIPT" \
  --run-id "$RUN_ID" --approved
```

实际服务器启动使用后台 nohup，并显式传入既有 keeper 脚本。状态写入
`logs/<run-id>.status` 和 `manifests/<run-id>/status.json`；阶段日志为
`logs/<run-id>-probe-w4.log`、`-probe-w8.log`、`-probe-evaluate.log`、`-train.log`、
`-evaluate-best.log`、`-evaluate-last.log`。总日志及各阶段 PID 同目录保存。
`PROBE-W4 / PROBE-W8 / PROBE-EVALUATE` 不是正式训练，只有 `TRAIN` 表示正式训练已启动；
`COMPLETED` 表示训练及最后两次评测完成，`FAILED / STOPPED` 表示流水线不再继续。

任何阶段失败或收到停止信号，只终止本流水线创建的独立进程组并恢复 keeper，不扫描杀死其他
训练。匿名内存逼近容器上限或出现 OOM 时失败关闭，不用文件缓存高占用本身判为训练失败。
正式启动前已通过原有 fadvise 工具释放约 79.9 GiB 不用于 scratch 的特征文件缓存页；
未删除缓存、权重或数据集文件，也未进行大块匿名内存压力分配。

### 24.13 首次六卡门禁失败与 AMP 同批重试

提交 `24bb1e8` 的首次 `probe-w4` 在反向传播时发现非有限梯度，门禁正确停止，
正式训练尚未开始，日志与失败状态保留在该 run ID 下。实际 AMP 初始 scale 为 16；
没有 CUDA OOM，不能把该失败归因于 batch 64 放不下显存。

后续修复只作用于 ScratchTrainer 的数值恢复：保留初始 scale=16 和 growth_interval=1000000，
保留 global batch、学习率、调度与数据顺序。若某个 rank 溢出，则所有 rank 将 scale 同步减半，
在不更新参数的前提下重算同一批；恢复该批前的模型 buffers（包括 BatchNorm 运行统计）和
CPU/CUDA RNG，避免重试导致 BatchNorm 多累积或随机数偏移。每批最多回退 8 次，仍失败就退出。

每个接受的 batch 仍必须恰好完成一次有限梯度的 optimizer 更新，不通过跳过 batch 或减少
样本达到“通过”。scale 回退属于数值执行记录，不是学习率衰减；实际最终 scale 和累计
`amp_same_batch_retries` 写入各 rank 的测速及 epoch 证据。
新增测试覆盖 BatchNorm/RNG 恢复、一次参数更新与有界失败，相关回归 **90 passed**。
原先“非有限梯度立即退出”改为“有限次数同批重算仍失败才退出”；不放宽最终有限值验收。


## 25. P1 结果后的受限修复与消融准备（2026-09-05）

### 25.1 本次授权边界

用户已允许修复、配置准备、小规模检查和已有 checkpoint 评测，但明确禁止重新开始大规模实验。
不运行任何新的 epoch 训练，不生成完整缓存，不覆盖旧模型配置、checkpoint、CSV 或诊断报告。
本节的 A/B/C 配置不是已完成实验，也没有取得新的检测精度结论。

现有第 30 epoch 官方 COCO 验证集结果：冻结方案 AP=11.6156、AP50=23.0556；
scratch AP=24.0808、AP50=36.2924。CSV 累计耗时分别为 31182.8 / 4906.91 秒，
同为六卡，均包含逐 epoch 验证，不含缓存抽取。原冻结结果保留为带 aux 偏差的 as-run 参照。

### 25.2 aux 修复合同

共享的 `CompositeCriterion` 和 `compose_native_result` 使用相同加法：

```text
loss_vector = native_loss + aux / native_loss.numel()
loss_vector.sum() = native_loss.sum() + aux
```

保留原生损失形状及 detached 日志，不再因三分量广播把 aux 计入三次。标量原生损失行为不变。
本次不修改 EMA 归一化、aux budget、原生检测损失本地 batch 缩放或 Trainer 的 DDP 乘数。
aux 仍是模型级标量，不额外乘本地 batch；这不是 batch 不变性保证。
后续对照继续锁定本地 batch=64、world_size=6。若另行改变 aux batch 归一化，必须单独建实验。
该修复影响共享 routed loss 的向量输出路径；旧训练不能直接按原身份恢复到修复后的目标函数。

### 25.3 独立 A/B/C 配置

配置矩阵：`ultralytics/cfg/experiments/d1/wp8-followup.yaml`。
历史 `yolo26-d1-dinov3-latent-n.yaml` 不修改。受限工具生成各组独立模型 YAML 到外部工作区。

| 组别 | 主特征融合 | 应用的 latent aux gain | 目的 |
|---|---|---:|---|
| A | router_only | 0.1 | 仅修复重复计入，建立修正版参照 |
| B | weighted_sum，三层各 1/3 | 0.1 | 检查 block8/block12 空间内容直接进入检测的影响 |
| C | 与 B 相同 | 0.0 | 检查施加辅助约束的影响 |

三组均为 3,542,567 个可训练参数；不修改 Teacher、缓存、检测头或九分支结构。
C 仍计算和记录原始 aux 诊断，但不将其施加到总损失，不能把它当成省去 aux 计算的性能优化。
原 router_only 的主内容来自 block4，其余层经全局池化后控制专家权重。
weighted_sum 使用现有公开接口，不改模块默认行为。均匀专家权重实验暂不实现。

### 25.4 受限工具与检查

`python -m scripts.d1.inspect_wp8_followup` 仅支持以下命令，不提供 train/all 或 epoch 参数：

- `prepare`：解析 A/B/C、核对参数量、生成独立配置及来源摘要。
- `check`：在完整 train/val 中各均匀抽取 8 图，验证源图片和选中 tensor SHA256，
  比较正式 Teacher 在线 FP16 特征，使用原 `rtol=atol=1e-3`；不重校验全部 423.5 GiB。
- `check` 同时对 A/B/C 各执行 batch=2、2 次预热加 5 次计时的驻 GPU forward/backward。
  不创建 optimizer，不更新参数，不保存训练 checkpoint。记录各阶段 forward GPU 时间和梯度。
  这是局部算子诊断，不是冷盘 I/O、DDP 或正式吞吐基准，不能据此推算完整训练时间。
- `scratch-train-eval`：对原 scratch checkpoint 评测与冻结诊断严格相同的 5,000 张训练图，
  校验列表摘要，使用 train annotations；不是完整训练集 AP，也不是独立验证集结果。

输出目录必须是仓库外的新目录，拒绝覆盖已有目录。数据、权重、缓存路径由参数传入。
报告绑定代码 commit、dirty 状态和源码 SHA256；通过检查不代表授权启动训练。

### 25.5 后续正式实验仍待授权

先审查真实输入检查和训练集诊断结果，再决定 A/B/C 的短训练窗口。
必要的持续 I/O 与六卡 DDP 性能定位尚未执行；它们必须另行规划并遵守超过 3 分钟后台挂载的约定。
不得仅根据短时热缓存速度宣称达到课题 GPU-hours 降低 50% 的目标。


### 25.6 已完成检查与新发现

- 修复与初始工具提交：`1f0698a`；独立零更新 probe 提交：`19ecf4b`。
- 共享损失、Foundation、LatentMixture、D1 WP0-WP8 与配置回归：246 passed、5 skipped；
  新增独立 probe 后定向回归：29 passed。未安装 Ruff/codespell，未声称这些 lint 检查通过。
- A/B/C 真实 FP16 缓存 forward/backward 均有限，optimizer_steps=0，参数逐项检查未变化。
  B/C 九条 Adapter 分支均有有限非零梯度。
  A 的 block8/block12 分支在本次 residual_gain=0、零初始化 Router 的冷启动状态下梯度为零，
  不能据此声称它们在原 30 轮训练全程都没有梯度。
- 单卡驻 GPU batch=2、5 次计时的全前向/反向均值：A 65.30 ms、B 68.65 ms、C 66.21 ms。
  这是局部算子检查，不足以决定正式吞吐、workers 或六卡 ETA。

**缓存 batch 依赖性必须保留披露：**
train/val 各均匀抽取 8 图，图片及选中 tensor SHA256 全部匹配。
单图在线提取时 14/16 图三层逐元素一致；最后两个样本超出原 rtol=atol=1e-3。
首轮严格检查保持 failed，没有放宽容差，也没有重新生成缓存。
按原六卡 index%6 分区、batch16 对应末批复现：

| 样本 | rank | 末批实际大小 | 原上下文复现 |
|---|---:|---:|---|
| train2017/000000581929 | 2 | 3 | block4/8/12 全部逐元素一致，最大误差 0 |
| val2017/000000581781 | 1 | 2 | block4/8/12 全部逐元素一致，最大误差 0 |

证据支持两张图存在 FP16 batch 上下文依赖，而非已检测到的缓存文件损坏。
这不证明全部缓存都已重新验证，也不满足“任意 batch 单图结果一致”的承诺。
独立 `probe` 只检查已缓存下游计算，不会将该失败改写为通过，且永远不授权训练。
后续完整在线对齐应明确原抽取分区、batch 大小及尾批规则；若要求 batch 不变的特征，
需另外评估数值合同和重建成本，本次不执行。

### 25.7 补充训练集诊断

原 scratch 第 30 epoch checkpoint 严格重载，对与冻结诊断相同的 5,000 张训练图评测。
列表 SHA256=`0c8c27127b5feef0f3a82133b5cf235ee24d78aca97963faff4345e65952e6e9`。

| 官方指标（百分制） | 冻结原方案 | Scratch epoch30 |
|---|---:|---:|
| train5000 AP | 15.78 | 43.02 |
| train5000 AP50 | 28.33 | 58.98 |
| val2017 AP（原已完成评测） | 11.62 | 24.08 |

该结果进一步支持原冻结方案存在拟合、空间内容适配或优化限制；不能单凭此表确立因果。
Scratch 本身也存在训练/验证差距，不能据此声称已达到充分训练或最优泛化。
本次只是已有 checkpoint 的评测，没有增加训练轮次，评测约 74.48 秒。

### 25.8 复现与证据

先在服务器中设置环境变量：`PYTHON` 为项目 Python，`WORK_ROOT` 为外部工作区，
`DATA_ROOT` 为 COCO 根目录，`TRAIN_CACHE`/`VAL_CACHE` 为完整缓存目录，
`TEACHER_DIR` 为本地 ViT-S/16 权重目录。在仓库根目录执行，输出目录必须尚不存在：

```bash
"$PYTHON" -m scripts.d1.inspect_wp8_followup prepare \
  --output-dir "$WORK_ROOT/manifests/followup-prepare-new"

"$PYTHON" -m scripts.d1.inspect_wp8_followup check \
  --data-root "$DATA_ROOT" --train-cache "$TRAIN_CACHE" --val-cache "$VAL_CACHE" \
  --weights-dir "$TEACHER_DIR" --output-dir "$WORK_ROOT/manifests/followup-check-new"

"$PYTHON" -m scripts.d1.inspect_wp8_followup probe \
  --data-root "$DATA_ROOT" --train-cache "$TRAIN_CACHE" \
  --output-dir "$WORK_ROOT/manifests/followup-probe-new"

"$PYTHON" -m scripts.d1.inspect_wp8_followup scratch-train-eval \
  --data-root "$DATA_ROOT" \
  --checkpoint "$WORK_ROOT/runs/wp8-p1-coco30-scratch-b384-s0-44c53e6/weights/last.pt" \
  --p0-report "$WORK_ROOT/manifests/wp8-diagnosis-epoch030-e36864d/train2017/report.json" \
  --output-dir "$WORK_ROOT/manifests/followup-scratch-train5000-new"
```

当前单图 `check` 会在上述尾批差异处失败，这是保留的已知约束，不能改日志冒充通过。

Git 小型汇总：[wp8-followup-preparation.json](manifests/wp8-followup-preparation.json)。
原始证据分别保存在外部工作区 `manifests/` 下：
`wp8-followup-1f0698a-preparation`、`wp8-followup-1f0698a-checks`、
`wp8-followup-19ecf4b-probe`、`wp8-followup-19ecf4b-scratch-train5000`。
对应日志在 `logs/`；完整预测 JSON 不进入 Git。旧冻结及 scratch 的 checkpoint、CSV 未修改。
