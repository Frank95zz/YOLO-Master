# D1 WP0：实验合同、正式数据与 Teacher 准备

## 1. 阶段结论

WP0 已完成 D1 正式实验的输入、数据划分、Teacher、预处理、随机性和缓存协议锁定。完整 COCO 2017 数据及 DINOv3 ViT-S/16 权重已经下载、校验并放在 Git 之外的工作区；仓库中保存了可复现配置、路径列表、哈希、环境和许可证记录。

WP0 解决的是“后续实验使用什么数据、模型和预处理”的问题，不代表 Adapter、LatentMixture、检测头或完整训练已经完成。

## 2. 固定实验合同

| 项目 | WP0 固定值 |
| --- | --- |
| 数据集 | COCO 2017 detection |
| 训练集 | 官方 `train2017`，118,287 张 |
| 验证集 | 官方 `val2017`，5,000 张 |
| 划分策略 | 按文件名排序，使用官方划分，不二次随机切分 |
| Teacher | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| Teacher 架构 | DINOv3 ViT-S/16，384 维、12 blocks、6 heads |
| 输入 | RGB、确定性 letterbox 到 `640×640` |
| Teacher 归一化 | ImageNet mean/std |
| 目标层 | block 4/8/12，实现索引 3/7/11 |
| 缓存 | `d1-cache-v1`、FP16、safetensors、目标 2 GiB 分片 |
| 随机性 | seed 0、deterministic、关闭随机空间增强 |

完整字段以 [p0-experiment-contract.json](manifests/p0-experiment-contract.json) 为唯一证据。

## 3. 已完成工作

### 3.1 正式配置

新增 [p0-dinov3-vits16-coco2017.yaml](../../ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml)，固定：

- `imgsz=640`、`nc=80`、seed 0 和确定性执行；
- 禁用 mosaic、mixup、copy-paste、翻转、仿射和多尺度训练；
- Teacher 为 DINOv3 ViT-S/16，缓存 dtype 为 FP16；
- 数据、权重和输出路径由运行参数注入，不写死服务器目录。

### 3.2 COCO 2017 数据与划分

官方 `train2017` 和 `val2017` 已完整准备，路径列表排序后提交 Git：

| 列表 | 数量 | SHA256 |
| --- | ---: | --- |
| [coco2017-train2017.txt](manifests/coco2017-train2017.txt) | 118,287 | `cdb18d3c86093bf6f9259f2a74b15c1aea22ba3a400bfe3bf77042bebfb750d1` |
| [coco2017-val2017.txt](manifests/coco2017-val2017.txt) | 5,000 | `70d1b9c55663f6820a30e4ccc02b4c3174e9aafb6512b500f5ce7869f5d1b3d6` |

两个列表无重复、无交集，不进行 COCO-mini 或其他二次抽样。压缩包来源、大小和 SHA256 记录在 [coco2017-splits.json](manifests/coco2017-splits.json)。

### 3.3 DINOv3 Teacher

正式 Teacher 从 ModelScope 固定 revision 下载，Transformers 本地目录包含配置、权重、许可证和模型说明。

| 文件 | 大小 | SHA256 |
| --- | ---: | --- |
| `model.safetensors` | 86,406,384 bytes | `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d` |
| `config.json` | 743 bytes | `9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b` |
| `LICENSE.md` | 7,503 bytes | `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e` |

模型来源和结构元数据见 [dinov3-vits16.json](manifests/dinov3-vits16.json)。

### 3.4 准备与校验工具

新增 [prepare_d1_wp0.py](../../scripts/prepare_d1_wp0.py)，支持：

- 断点下载和 `.part` 临时文件；
- 文件大小、ZIP CRC 和 SHA256 校验；
- 安全解压和重复执行；
- 生成稳定的 train/val 路径列表；
- 验证本地 Transformers 权重可正常构造；
- 记录 Python、PyTorch、CUDA、cuDNN、Transformers、GPU 和系统环境。

## 4. 交付证据

| 证据 | 作用 |
| --- | --- |
| [p0-experiment-contract.json](manifests/p0-experiment-contract.json) | 输入、Teacher、层号、缓存和随机性合同 |
| [coco2017-splits.json](manifests/coco2017-splits.json) | 数据来源、数量和压缩包哈希 |
| [dinov3-vits16.json](manifests/dinov3-vits16.json) | Teacher 来源、结构和文件哈希 |
| [environment.json](manifests/environment.json) | 代码 commit 与软硬件环境 |
| [licenses.md](manifests/licenses.md) | COCO 与 DINOv3 许可来源记录 |

环境记录对应 6 张 NVIDIA A40、Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4 和 Transformers 5.15.1。

## 5. 复现与校验

在仓库根目录执行，工作区由使用者自行指定：

```bash
export D1_WORKSPACE=/path/to/yolo-master-d1

# 首次下载、解压、校验并生成 manifest
python scripts/prepare_d1_wp0.py \
  --workspace "$D1_WORKSPACE" \
  --repo "$PWD" \
  --download

# 已有数据与权重时只做校验
python scripts/prepare_d1_wp0.py \
  --workspace "$D1_WORKSPACE" \
  --repo "$PWD" \
  --verify-only

python -m pytest -q tests/test_d1_wp0_contract.py
```

数据集、Teacher 权重和下载临时文件不进入 Git。长期迁移时应备份源压缩包、Teacher 权重和 manifest；解压数据可由源文件重新生成。

## 6. 提交与边界

- `c13dc18 feat(d1): define P0 experiment contract`
- `0bc6d66 docs(d1): record WP0 provenance`

WP0 尚未实现多层 Teacher API、特征缓存、P3/P4/P5 Adapter、检测模型或训练评测。多层输出由 [WP1](WP1.md) 完成，缓存由 [WP2](WP2.md) 完成。
