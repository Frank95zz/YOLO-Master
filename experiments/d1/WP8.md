# D1 WP8：六卡全量 COCO 2017 正式训练方案（待审核）

## 1. 当前状态与最终资源决策

WP8 的完整 COCO 2017 特征缓存、缓存校验、训练组件、最小闭环和正式训练入口已经具备。正式 100 epochs 训练和最终 COCO val2017 评测尚未启动。

本文件固定 WP8 的正式资源方案为：

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
```

六张 NVIDIA A40 全部服务于同一个 DDP 训练任务，不再拆成三组双卡任务，也不并发运行其他训练或特征抽取作业。WP8 当前只登记一个 seed 0 正式主实验；seed 1/2 重复实验不属于本次启动合同，如后续需要统计方差，必须另建预注册运行身份并串行执行，不能与 seed 0 的结果混写。

当前状态：**方案已确定，代码仍保留旧的六卡 global batch 48 门禁。完成第 11 节的配置和测试改造、通过第 12 节长基准并再次得到用户确认前，不得启动正式训练。**

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

- P1 的同参数量从零训练检测器；
- 第二数据集实验和“GPU 时降低至少 50%”的最终对照结论；
- P2 的 DINOv3/SigLIP2 Teacher 对比；
- balance、z-loss、latent aux 系数的大规模扫描；
- 根据中途精度临时更换学习率、batch、增强或训练轮数；
- 部署、导出、蒸馏后推理速度或实时推理优化。

P1 基线以后必须复用本文件锁定的 COCO split、640 输入、global batch 384、优化器、训练轮数、增强和评测口径，才能进行公平成本与精度比较。

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

## 11. 正式启动前的代码改造

当前 `wp8-formal-coco2017.yaml` 和 `run_wp8_train.py` 已固定六卡，但仍校验 `batch=48`、`nbs=48`。不得绕过校验直接传命令行覆盖。正式启动前必须完成以下改造并形成独立提交：

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
- [`wp8-formal-coco2017.yaml`](../../ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml)：正式训练合同，启动前需升级到 batch 384；
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

只有第 11 节代码改造和测试完成、第 12～13 节门禁通过、长基准结果完成汇报，并再次收到明确启动确认后，才能运行正式 100 epochs 训练。
