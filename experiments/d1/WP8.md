# D1 WP8：完整 COCO 2017 正式训练方案（待审核）

## 1. 当前状态

WP8 的完整缓存和训练前工程准备已经完成，正式训练与正式评测尚未启动。本文是启动前实验合同；参数、数据、代码提交和验收口径经审核确认后才允许运行。

当前状态：**等待审核确认，不得启动 `torchrun`。**

已完成：

- 完整 COCO 2017 train2017/val2017 DINOv3 特征缓存；
- 六卡缓存构建、完整 SHA256 校验和统一索引；
- 正式训练配置 `wp8-formal-coco2017.yaml`；
- 六进程 DDP 训练、恢复、epoch 级遥测和最终汇总入口 `run_wp8_train.py`；
- 缓存身份、数据规模、GPU 型号、代码提交和运行目录门禁；
- 离线测试和既有 D1 回归测试。

未完成：

- 六卡 DDP 真实启动门禁；
- 100 epochs 完整训练；
- COCO val2017 最终评测；
- 精度、训练成本、显存和 Router 行为报告。

## 2. 本阶段目标

WP8 完成 D1 的正式 P0 实验：使用冻结 DINOv3 ViT-S/16 已生成的离线特征，只训练 Adapter、三个 LatentMixture 和 Detect Head，并在完整 COCO val2017 上报告检测精度和训练成本。

WP8 不负责 P1 的同预算从零训练基线、第二数据集对照和“GPU 时降低至少 50%”结论，也不进行 P2 的 teacher 尺寸或 aux 参数扫描。正式运行期间不根据中间精度继续调参。

## 3. 数据与缓存合同

### 3.1 COCO 2017

| Split | 图片数 | 用途 |
| --- | ---: | --- |
| train2017 | 118,287 | 完整训练 |
| val2017 | 5,000 | 每 epoch 验证与最终评测 |

使用官方 split，不二次随机划分。输入几何沿用 WP0：640×640、确定性居中 LetterBox、无二次 resize/crop。

### 3.2 DINOv3 缓存

Teacher 固定为 `facebook/dinov3-vits16-pretrain-lvd1689m`，权重 SHA256：

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

合计 123,287 个样本、369,861 个 tensor、216 个 shard，约 423.5 GiB。每个样本包含：

```text
block4  [384,40,40] FP16
block8  [384,40,40] FP16
block12 [384,40,40] FP16
```

完整证据见 [`manifests/wp8-full-cache.json`](manifests/wp8-full-cache.json)。训练 preflight 只比对已完成的完整校验报告、索引、shard 集合和文件大小，不在每次训练前重复读取并哈希 423.5 GiB。

## 4. 模型与训练边界

模型配置：`ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml`。

训练数据流：

```text
三个缓存 block 特征
  -> DINOFeaturePyramidAdapter
  -> 每尺度三个 P3/P4/P5 候选
  -> P3/P4/P5 LatentMixture
  -> YOLO26 Detect
  -> detection loss + latent aux loss
```

模型共有 3,542,567 个参数，全部属于下游 Adapter、LatentMixture 和 Detect。训练过程不实例化 DINOv3 Teacher，checkpoint 不得包含 `teacher` 或 `dinov3` 参数。

固定 aux 配置：

| 参数 | 值 |
| --- | ---: |
| `balance_loss_coeff` | 0.01 |
| `router_z_loss_coeff` | 0.001 |
| `latent_aux_gain` | 0.1 |
| `mixture_aux_budget` | 3.0 |

## 5. 正式训练配置

配置文件：[`wp8-formal-coco2017.yaml`](../../ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml)。

| 项目 | 固定值 |
| --- | --- |
| GPU | 6 × NVIDIA A40，单机 DDP |
| seed | 0 |
| deterministic | true |
| epochs | 100 |
| 全局 batch | 48 |
| 每卡 batch | 8 |
| `nbs` | 48，不做额外梯度累积 |
| workers | 每 rank 4，共 24 |
| precision | AMP |
| optimizer | AdamW |
| `lr0` / `lrf` | 0.001 / 0.01 |
| scheduler | cosine |
| weight decay | 0.0005 |
| warmup | 3 epochs |
| patience | 100 |
| checkpoint | `last.pt`、`best.pt`，每 10 epochs 额外保存 |
| validation | 每 epoch 完整 val2017，结束后严格加载最佳健康 checkpoint 再评测 |
| augmentation | 所有颜色、仿射、翻转、mosaic、mixup、copy-paste、erasing 和 multi-scale 均关闭 |
| pretrained student | false |

缓存抽取基准中的 batch 16 是每张 GPU 执行 DINOv3 前向时的 batch，只用于生成缓存。正式下游训练使用全局 batch 48，即每卡 batch 8；这是因为 DDP 全局 batch 必须能被 6 整除，且 WP7 已真实验证每卡 batch 8 的 loss、AMP、checkpoint 和过拟合闭环。二者不能混为同一个参数。

AdamW、余弦衰减、3 epoch warmup 和 weight decay 是本次正式 P0 的工程起始配方，不是课题书强制值。它们在本文确认后锁定，正式运行中不修改。

## 6. 启动前门禁

正式启动前执行 `prepare`，必须同时满足：

1. Git 工作区干净，训练代码与 preflight 记录同一 commit；
2. 六张可见 GPU 均为 NVIDIA A40；
3. train/val 缓存样本数、合同摘要、内容摘要、shard 集合和文件大小与完整校验报告一致；
4. COCO 官方路径列表分别为 118,287 和 5,000；
5. 数据、缓存、配置和运行目录身份写入外部 `inputs/identity.json`；
6. 已有运行目录只能在身份完全一致时复用，否则拒绝覆盖；
7. 配置不包含服务器绝对路径，运行路径由 CLI 注入。

`prepare` 只做读取和小型 manifest 写入，不训练模型。

## 7. 审核通过后的执行顺序

### 7.1 生成最终 preflight

```bash
export D1_REPO=/path/to/YOLO-Master
export D1_WORKSPACE=/path/to/yolo-master-d1
export D1_COMMIT=$(git -C "$D1_REPO" rev-parse --short HEAD)
export D1_RUN_ROOT="$D1_WORKSPACE/runs/wp8-formal-$D1_COMMIT"
export D1_REPORT_DIR="$D1_WORKSPACE/manifests/wp8-formal-$D1_COMMIT"

cd "$D1_REPO"
python scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  prepare
```

### 7.2 六卡正式训练

审核确认后才运行以下命令。该任务预计远超 3 分钟，必须使用后台方式并记录 PID、状态和日志；启动后只确认六个 rank 正常占用 GPU 一次，然后退出会话。

核心 DDP 命令：

```bash
python -m torch.distributed.run \
  --nproc_per_node 6 \
  --master_port 29518 \
  scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  train
```

如果中断且 `last.pt` 健康，使用同一 commit、配置、缓存和运行目录恢复：

```bash
python -m torch.distributed.run \
  --nproc_per_node 6 \
  --master_port 29518 \
  scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  train --resume "$D1_RUN_ROOT/weights/last.pt"
```

不得在恢复时改变 seed、batch、学习率、数据或模型配置。

### 7.3 训练完成后的汇总

```bash
python scripts/d1/run_wp8_train.py \
  --workspace "$D1_WORKSPACE" \
  --run-root "$D1_RUN_ROOT" \
  --report-dir "$D1_REPORT_DIR" \
  summarize
```

汇总只有在 100 个 epoch 记录完整、所有数值有限、最终验证覆盖 5,000 张图片、checkpoint 严格重载且不包含 Teacher 参数时才通过。

## 8. 运行状态与恢复检查

后台启动时将统一状态写入：

```text
$D1_WORKSPACE/logs/wp8-formal-<commit>.status
$D1_WORKSPACE/logs/wp8-formal-<commit>.pid
$D1_WORKSPACE/logs/wp8-formal-<commit>.log
```

状态检查：

```bash
cat "$D1_WORKSPACE/logs/wp8-formal-$D1_COMMIT.status"
pid=$(cat "$D1_WORKSPACE/logs/wp8-formal-$D1_COMMIT.pid")
kill -0 "$pid" 2>/dev/null && echo RUNNING || echo FINISHED_OR_FAILED
```

训练进度：

```bash
tail -n 5 "$D1_RUN_ROOT/results.csv"
tail -n 50 "$D1_WORKSPACE/logs/wp8-formal-$D1_COMMIT.log"
nvidia-smi
```

每个 epoch 结束后，六个 rank 分别保存数据等待时间、训练 step 时间、峰值显存和 Router 状态；rank 0 另存验证覆盖数量与指标。恢复运行会按 rank/epoch 原子覆盖同名报告，不生成逐 batch 大日志。

## 9. 评测指标与验收

正式报告必须包含：

- COCO `mAP50-95`、`mAP50`、Precision、Recall；
- box、class、DFL 和四项 latent/mixture loss 曲线；
- 训练墙钟时间、每 epoch 时间、六卡峰值显存；
- 数据等待比例和缓存读取瓶颈；
- P3/P4/P5 Router 参数变化、平均概率、熵、balance、z-loss 和 residual gain；
- 最佳/最终 checkpoint SHA256、严格重载结果和 Teacher 参数计数；
- 训练代码 commit、配置 SHA256、两个缓存内容摘要和环境信息。

P0 任务书没有给出绝对 mAP 阈值，因此 WP8 不用事后选择的精度门槛判定工程失败。验收门槛是：完整训练和评测闭环成功、指标有限、5,000 张 val 全覆盖、checkpoint 可恢复且 Teacher 隔离正确。实际精度必须原样报告，不因结果高低修改本次配方。

## 10. 时间预估与交互约定

完整训练共约 246,500 个六卡同步 step。按 WP7 每卡 batch 8 的实测和完整缓存位于 NFS 的条件，启动前保守估计为 34 至 48 小时；第一完整 epoch 会提供更准确的剩余时间。每 epoch 的完整 val2017 也计入总时长。

训练预计远超 3 分钟。审核通过并启动后，本会话只检查一次 PID、六个 rank、GPU 占用和早期错误，随后退出，并提供状态命令。用户确认任务结束后再继续执行汇总、证据提交和结果解释。

## 11. 代码与测试证据

训练准备实现提交：`6e9eb22`。

主要文件：

- [`run_wp8.py`](../../scripts/d1/run_wp8.py)：六卡缓存构建与合并；
- [`run_wp8_train.py`](../../scripts/d1/run_wp8_train.py)：正式 preflight、DDP worker、恢复、遥测和汇总；
- [`wp8-formal-coco2017.yaml`](../../ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml)：正式训练合同；
- [`test_d1_wp8_multi_gpu_cache.py`](../../tests/test_d1_wp8_multi_gpu_cache.py)：六卡缓存测试；
- [`test_d1_wp8_formal_training.py`](../../tests/test_d1_wp8_formal_training.py)：训练配置、缓存身份和入口测试；
- [`manifests/wp8-full-cache.json`](manifests/wp8-full-cache.json)：完整缓存脱敏证据。

本次准备阶段回归结果：

```text
151 passed, 5 skipped
```

另有 WP8 专项结果 `17 passed`。Python 编译检查和 `git diff --check` 通过；服务器未安装可选 Ruff。当前未运行六卡 DDP 训练，因此不能把离线测试结果写成正式训练通过。

## 12. 审核项

启动前请重点确认：

1. 是否接受全局 batch 48（每卡 8）；
2. 是否接受 AdamW、`lr0=0.001`、余弦衰减、3 epoch warmup 和 `weight_decay=0.0005`；
3. 是否接受 100 epochs 且每 epoch 完整验证；
4. 是否接受 P0 不设事后绝对 mAP 门槛，只如实报告精度；
5. 是否接受保守 34 至 48 小时训练窗口；
6. 是否确认正式运行期间不调参，异常时只做同身份 checkpoint 恢复。

只有收到明确确认后，才执行 preflight 和六卡正式训练。