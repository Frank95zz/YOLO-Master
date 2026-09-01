# D1 WP2：可复现特征缓存

## 1. 阶段结论

WP2 已将 WP1 的 DINOv3 block 4/8/12 输出固化为可校验、可恢复、可重建索引的离线特征缓存。训练阶段可以直接读取冻结 Teacher 特征，避免每轮重复执行 DINOv3 前向，并可在服务器中断后从已提交分片继续构建。

两次相互独立的 100 图缓存构建已经完成。两次缓存的样本顺序、tensor 数量、逐 tensor 内容摘要和整体内容摘要完全一致，证明 `d1-cache-v1` 在当前实验合同下可重复生成。

## 2. WP2 解决的问题

WP2 将昂贵且不参与训练的 Teacher 前向从检测训练循环中移出，形成以下数据流：

```text
COCO 图像
  -> WP0 固定预处理
  -> WP1 DINOv3Teacher(block 4/8/12)
  -> FP16 safetensors 分片
  -> 后续 Adapter / LatentMixture / Detect 训练读取
```

该阶段同时处理三类工程风险：

1. **训练成本**：同一图像的 DINOv3 特征只抽取一次。
2. **可复现性**：cache key、合同哈希、样本列表和 tensor 内容均可验证。
3. **中断恢复**：正式分片采用原子提交；异常只留下 `.part`，已完成分片可以恢复并重建索引。

## 3. 已完成实现

### 3.1 缓存协议与读写器

[`ultralytics/nn/foundation/cache.py`](../../ultralytics/nn/foundation/cache.py) 实现了 `d1-cache-v1`：

- 缓存合同绑定协议版本、模型 ID、模型文件哈希、预处理、输出层、dtype 和期望形状；
- 单样本 cache key 绑定图像内容 SHA256、Teacher 权重 SHA256、预处理 SHA256、输出层、dtype 和协议版本；相对路径作为可移植记录保存；
- 每个样本写入 `block4`、`block8`、`block12` 三个 FP16 BCHW tensor；
- 使用 safetensors 分片，目标分片大小为 2 GiB；
- 分片先写入 `.part`，完成校验后再原子改名为正式文件；
- 分片头内嵌样本记录，因此索引丢失时可从正式分片重建；
- 支持已完成样本跳过、重复 key 拒绝、损坏分片拒绝和逐 tensor SHA256 校验；
- Writer 在写入前核对 shape、dtype 和有限值；`verify` 核对分片大小与 SHA256，以及逐 tensor 的 shape、dtype、字节数和可选 SHA256。

### 3.2 命令行工具

[`scripts/d1/cache_features.py`](../../scripts/d1/cache_features.py) 提供三个子命令：

| 子命令 | 作用 |
| --- | --- |
| `build` | 按 WP0 路径列表和预处理合同构建缓存，支持断点恢复 |
| `verify` | 不运行 Teacher，只校验合同、索引、分片和 tensor 内容 |
| `compare` | 比较两份独立缓存，验证样本集合和内容摘要一致 |

`build` 直接调用 WP1 的 `DINOv3Teacher(output_layers=(4, 8, 12))`，不通过 smoke 配置，也不在仓库配置中写入服务器绝对路径。

## 4. 缓存结构与可恢复性

每份缓存目录包含：

```text
cache-dir/
├── index.json
├── samples.jsonl
├── train2017-00000.safetensors
└── ...
```

- `index.json` 保存协议、合同、样本数、tensor 数、分片和整体内容摘要。
- `samples.jsonl` 保存每个样本的相对路径、cache key、tensor 名称、shape、dtype 和 SHA256。
- `<split>-NNNNN.safetensors` 是正式分片，直接位于缓存根目录；大缓存属于外部实验工作区，不进入 Git。

缓存目录可放在任意有足够容量的工作区。本文统一用 `$D1_WORKSPACE` 表示外部路径，不依赖当前服务器的固定挂载位置。

## 5. 100 图双构建结果

两次构建使用相同的前 100 个 `train2017` 样本，但写入两个独立缓存目录。结果如下：

| 指标 | Build A | Build B |
| --- | ---: | ---: |
| 样本数 | 100 | 100 |
| tensor 数 | 300 | 300 |
| tensor 数据量 | 368,640,000 bytes | 368,640,000 bytes |
| 缓存占用 | 368,791,336 bytes，约 352 MiB | 368,791,336 bytes，约 352 MiB |
| 特征抽取时间 | 12.755 s | 13.821 s |
| 特征抽取速度 | 7.840 images/s | 7.236 images/s |
| 峰值显存 | 320,947,200 bytes | 320,947,200 bytes |
| 热缓存读取速度 | 2778.509 MiB/s | 2759.697 MiB/s |

关键一致性摘要：

| 项目 | SHA256 |
| --- | --- |
| 选中样本路径列表 | `d2c16d9021e923f4435c14af706048784e558ca57c2addb676d67a46352bb080` |
| 缓存合同 | `6bfda0e13bde01001c3f3f2d72631a2401fb9a77b146d6fb2794303e379e47a7` |
| 两次构建的内容摘要 | `0102f2e707b369ca8bfb3d996d54fa0d8ba9eba79787501bd5fe6542f90c1a8d` |

`compare` 对重新加载后的 FP16 tensor 逐项比较，两份缓存内容完全一致。

## 6. Git 证据

仓库只提交小型、可审查的证据文件：

| 文件 | 内容 |
| --- | --- |
| [`wp2-cache-100-samples.jsonl`](manifests/wp2-cache-100-samples.jsonl) | 100 个输入样本及其顺序 |
| [`wp2-cache-100-index-a.json`](manifests/wp2-cache-100-index-a.json) | Build A 缓存索引 |
| [`wp2-cache-100-index-b.json`](manifests/wp2-cache-100-index-b.json) | Build B 缓存索引 |
| [`wp2-cache-100-reproducibility.json`](manifests/wp2-cache-100-reproducibility.json) | 双构建耗时、吞吐、显存和一致性汇总 |

两份约 352 MiB 的完整缓存保存在 Git 之外的实验工作区。Git 证据足以核对输入、协议、规模和摘要，但不能替代大文件备份；迁移服务器时应将缓存目录与 COCO、Teacher 权重一起备份到外部存储。

## 7. 复现命令

先设置外部工作区，并确保已经完成 [WP0](WP0.md) 的数据和权重准备：

```bash
export D1_WORKSPACE=/path/to/yolo-master-d1
```

独立构建两份 100 图缓存：

```bash
python scripts/d1/cache_features.py build \
  --workspace "$D1_WORKSPACE" \
  --cache-dir "$D1_WORKSPACE/feature_cache/wp2-train100-a" \
  --split train2017 --limit 100 --batch-size 8 --device 0 \
  --report "$D1_WORKSPACE/manifests/wp2-train100-a.json"

python scripts/d1/cache_features.py build \
  --workspace "$D1_WORKSPACE" \
  --cache-dir "$D1_WORKSPACE/feature_cache/wp2-train100-b" \
  --split train2017 --limit 100 --batch-size 8 --device 0 \
  --report "$D1_WORKSPACE/manifests/wp2-train100-b.json"
```

只校验缓存，不运行 Teacher：

```bash
python scripts/d1/cache_features.py verify \
  --cache-dir "$D1_WORKSPACE/feature_cache/wp2-train100-a"
```

比较两份独立构建：

```bash
python scripts/d1/cache_features.py compare \
  --cache-dir "$D1_WORKSPACE/feature_cache/wp2-train100-a" \
  --other-cache-dir "$D1_WORKSPACE/feature_cache/wp2-train100-b" \
  --first-report "$D1_WORKSPACE/manifests/wp2-train100-a.json" \
  --second-report "$D1_WORKSPACE/manifests/wp2-train100-b.json"
```

相关自动化测试：

```bash
python -m pytest -q \
  tests/test_d1_wp2_feature_cache.py \
  tests/test_d1_wp2_cache_cli.py \
  tests/test_d1_wp0_contract.py \
  tests/test_foundation_dinov3.py
```

WP2 交付时的相关测试汇总为 `121 passed, 1 skipped`。

## 8. 提交与阶段边界

- 实现提交：`5cd566d`（`feat(d1): add sharded feature cache`）
- 证据提交：`af433a8`（`docs(d1): record WP2 cache evidence`）

WP2 尚未完成以下工作：

- 未构建完整 COCO 2017 缓存；按当前张量规模估算约需 423 GiB；
- 未实现多尺度 Adapter、LatentMixture 或 Detect Head 训练；
- 热缓存顺序读取吞吐只说明存储读取能力，不代表后续正式训练吞吐；
- 100 图结果是缓存链路验收，不是检测精度结果。

下一阶段从 [主方案](README.md) 的 WP3 开始，将同为 stride 16 的 block 4/8/12 特征转换为检测所需的多尺度特征。
