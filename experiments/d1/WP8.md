# D1 WP8：六卡缓存准备与启动门禁

## 1. 阶段状态

六卡缓存能力和 1,200 图真实写入基准已经完成，但启动门禁未通过。完整 COCO 2017 缓存没有启动，WP8 正式训练也尚未开始。

门禁失败不是显存不足：六张 A40 均能运行 batch 8/16/32。瓶颈出现在特征落盘链路，实际缓存吞吐低于 30 images/s，按实测结果推算的完整缓存总时间超过 2 小时上限。

## 2. 已完成实现

- `FeatureCacheWriter` 新增可选 `shard_prefix`，默认值保持原有 WP2 单卡命名和行为不变。
- 六个 worker 按排序后索引 `index % 6 == rank` 确定性分区，分别使用 GPU 0 至 GPU 5。
- 每个 rank 使用独立目录、日志、PID、报告和 `train2017-rXX-XXXXX.safetensors` 文件名，可只续跑失败 rank。
- `finalize` 拒绝缺失 rank、报告身份不一致、合同不一致、重复或遗漏样本、额外 shard 及残留 `.part`。
- rank shard 通过同文件系统硬链接汇入最终目录，不复制第二份缓存数据；统一 `index.json` 只在全部 rank 验证后生成。
- 正式入口会检查 COCO `train2017=118,287`、`val2017=5,000`，并检查两个 split 的样本 ID 不重叠。
- 调度器记录单卡显存、GPU 利用率、模型加载耗时、抽取吞吐、真实缓存吞吐和最终校验耗时。

代码提交：`651ffd8b0ae9ad3aeda15a31c4f5ce5c5f825343`。

## 3. 基准设置

| 项目 | 固定值 |
| --- | --- |
| Teacher | DINOv3 ViT-S/16 |
| 输入 | 640×640，WP0 确定性预处理 |
| 输出 | block 4/8/12，各 `[384,40,40]` |
| 缓存 | FP16 safetensors，`d1-cache-v1` |
| GPU | 6 × NVIDIA A40 |
| 计算候选 batch | 8 / 16 / 32 |
| 真实写入样本 | train2017 排序后的前 1,200 张 |
| 基准缓存 | 独立外部目录，不与正式缓存混用 |

## 4. 实测结果

### 4.1 纯计算基准

| 每卡 batch | 六卡聚合吞吐 | 端到端吞吐 | 每卡峰值显存 |
| ---: | ---: | ---: | ---: |
| 8 | 368.207 images/s | 74.561 images/s | 346,383,360 bytes |
| 16 | **460.096 images/s** | **121.657 images/s** | 614,523,904 bytes |
| 32 | 386.387 images/s | 108.683 images/s | 1,144,103,936 bytes |

batch 16 在无 OOM 的候选中吞吐最高，因此用于真实缓存写入基准。

### 4.2 真实缓存写入

| 指标 | 实测值 |
| --- | ---: |
| 样本 / tensor | 1,200 / 3,600 |
| 缓存大小 | 4,425,498,192 bytes |
| 六卡缓存吞吐 | 16.048 images/s |
| 端到端吞吐 | 14.498 images/s |
| 有效缓存写入速度 | 56.441 MiB/s |
| 写入阶段墙钟时间 | 82.772 s |
| 最终完整校验 | 6.667 s |
| 每卡峰值显存 | 613,721,088 bytes |
| 每卡平均 GPU 利用率 | 8.91% 至 9.63% |
| 每卡最高 GPU 利用率 | 20% 至 33% |

缓存内容摘要为 `ebfd139e8b82da54c1b9261599e4828b143907e0a62975197cb1c9d85ffc9e4a`。低 GPU 利用率、远高于写入吞吐的纯计算吞吐，以及约 56 MiB/s 的有效落盘速度共同表明，当前主要限制在 CPU 校验、GPU 到 CPU 搬运和 NFS 写入链路，而不是 DINOv3 前向计算。这是依据本次指标作出的工程推断。

## 5. 完整缓存估算与门禁结论

完整 train2017 和 val2017 共 123,287 张图片，预计缓存数据为 454,485,196,800 bytes，约 423.3 GiB。按真实写入基准线性推算：

- 抽取、校验并写入约 7,682 秒，即 2 小时 8 分；
- 最终完整校验约 685 秒，即 11 分 25 秒；
- 合计约 8,367 秒，即 **2 小时 19 分**；
- 基准时可用空间约 1.67 TiB，容量满足 1.4 倍安全余量。

门禁要求聚合缓存吞吐至少 30 images/s 且预计总时间不超过 2 小时。本次两项均未满足，因此结果为 `failed`，按合同停止，没有创建正式 train2017/val2017 完整缓存。

## 6. 复现命令

```bash
export D1_REPO=/path/to/YOLO-Master
export D1_WORKSPACE=/path/to/external/d1-workspace

cd "$D1_REPO"
python scripts/d1/run_wp8.py benchmark-cache \
  --workspace "$D1_WORKSPACE" \
  --cache-dir "$D1_WORKSPACE/feature_cache/wp8-benchmark-651ffd8" \
  --report "$D1_WORKSPACE/manifests/wp8-cache-benchmark-651ffd8.json" \
  --devices 0,1,2,3,4,5 \
  --split train2017 \
  --limit 1200 \
  --batch-candidates 8 16 32
```

完整缓存只能在门禁重新通过或明确接受当前实测时间后，使用 `build-cache` 分别构建 train2017 和 val2017；本阶段未执行该命令。

## 7. 证据与验证

Git 内脱敏摘要：[`manifests/wp8-cache-benchmark.json`](manifests/wp8-cache-benchmark.json)。完整 rank 报告、日志、PID 和 1,200 图基准 shard 只保存在外部工作区。

代码提交前回归结果为 `191 passed, 5 skipped`；补充直接执行入口测试后，WP8 专项结果为 `11 passed`。服务器未安装可选 Ruff 命令，已运行 Python 编译检查和 `git diff --check`。

## 8. 阶段边界

本报告证明六卡确定性分区、并发写入、失败 rank 续跑和统一索引流程可用，也给出了真实 I/O 条件下的时间与容量依据。它不代表完整 COCO 缓存已经构建，不包含正式训练、COCO val2017 精度、训练成本对照或 P0/P1/P2 结论。