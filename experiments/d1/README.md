# D1：冻结 DINOv3 的缓存特征检测

本目录提供可独立使用的功能实现与复现入口，不包含个人实验队列、资源清理脚本和阶段流水文档。主要研究结果与限制见 [REPORT.md](REPORT.md)。

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

准备完毕后运行校验，生成两份完整、排序稳定的图片列表：

```bash
python -m scripts.d1.prepare_wp0 --coco-root "$COCO_ROOT" \
  --weights-dir "$TEACHER_DIR" --output "$D1_WORK/inputs"
```

它验证图片数量与列表 SHA256、train/val 互斥、标签数量与图片归属、官方标注文件存在性、Teacher 文件大小/SHA256/架构，并用本地权重构造 Teacher。只有提供 --archives-dir 时才验证源压缩包，报告会明确标记是否做过该项；不把文件存在性说成标注内容逐字节验证。生成结果写入外部 inputs，不覆盖仓库 manifest。

只需恢复列表时可以运行：

```bash
python -m scripts.d1.prepare_wp0 --coco-root "$COCO_ROOT" \
  --lists-only --output "$D1_WORK/inputs"
```

COCO 列表数量固定为118,287/5,000，不进行重新随机划分。重跑时输出文件必须一致，否则报错，不覆盖已有证据。旧 --download/--workspace 参数已移出当前入口。

来源与固定校验信息：[预处理合同](manifests/p0-experiment-contract.json)、[Teacher](manifests/dinov3-vits16.json)、[数据划分](manifests/coco2017-splits.json)、[许可来源](manifests/licenses.md)。

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

保留相同命令即可续跑新入口生成的缓存。旧缓存缺少 build.json 身份时仍支持读取、校验和转换，但须用原提取器续写，不能静默改变 batch。独立的六卡调度器保留在研究归档，不进入本 PR。可选 --benchmark-read 会额外完整读取一次缓存，其吞吐不是训练吞吐。

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

## 测试与复现身份

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q tests/test_d1_*.py \
  tests/test_foundation_dinov3.py tests/test_foundation_teacher_protocol.py \
  tests/test_latent_mixture.py tests/test_mixture_loss_composition.py
git diff --check
```

真实 Teacher/CUDA 测试按显式环境变量启用；普通 CI 不下载模型。新入口测试使用合成图像和合成特征，完成一轮 optimizer 更新、保存、严格重载和独立评测，不产生真实数据集精度结论。

研究归档固定在 [f4d2bc2](https://github.com/Frank95zz/YOLO-Master/tree/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1)。历史运行必须使用报告记录的执行提交。个人队列的每 rank RNG/buffer 快照、严格重试、阶段筛选和周期官方评测策略未迁入新入口；此 CLI 只启动新 run，不提供旧研究 run 的精确 resume。

后续正式实验继续使用获确认的研究合同；需记录执行提交，并核对与 PR 的核心实现一致，不能把默认示例参数冒充正式对照配方。

上一轮精简的过程证据保留在 [历史提交](https://github.com/Frank95zz/YOLO-Master/blob/d6fe25ef0011294cf12014bbf6c4629291e7ab07/experiments/d1/manifests/pr-verification.json)，不再放入当前 PR 文件树。本轮验证结果见 REPORT；未启动真实训练或缓存抽取。scratch 总参数匹配配置及测试继续保留。
