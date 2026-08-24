# D1 8.24 准入检查

本检查对应课题要求：100 张图特征缓存可复现；估算磁盘、I/O 与显存；完成接口维度表。

## 验收口径

1. 从 COCO128 固定排序后选取前 100 张不同图片，并记录源文件 SHA256。
2. 冻结 Foundation backbone，缓存第 3/6/9/12 个 block 的空间特征和最终 CLS 特征。
3. 独立执行两次完整缓存，逐张比较源图 SHA256 和特征张量 SHA256，要求 100/100 一致。
4. 实测缓存体积、编码吞吐、缓存读取吞吐、CUDA 峰值显存和剩余磁盘。
5. 在同一 commit 上运行 D1 单元测试、LatentMixture 1 epoch smoke 和 checkpoint predict。

## 模型降级声明

DINOv3 ViT-S/16 权重受 gated license 限制，需要单独申请。依据 D1 任务书的风险降级条款，本次 8.24
准入缓存使用许可明确的 Meta DINOv2 ViT-S/14 官方预训练权重，来源为：

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

该降级只用于验证冻结、多层特征、缓存、维度和资源边界，不声称完成 DINOv3 精度实验。DINOv3
适配器的冻结、预处理和输出协议由 `tests/test_foundation_dinov3.py` 覆盖。

## 一键复现

```bash
cd /root/yolo-master/repo
bash smoke/d1/run_admission.sh
```

脚本只接受干净工作树，自动下载 COCO128、锁定当前 commit、执行两次缓存、比较哈希，并重跑测试、训练和预测。

## 证据位置

最近一次准入日志目录记录在：

```text
/root/yolo-master/logs/latest-d1-admission.txt
```

最近一次主缓存目录记录在：

```text
/root/yolo-master/feature_cache/latest-d1-admission.txt
```

主缓存目录包含：

- `features/*.pt`：100 个逐图缓存文件。
- `manifest.jsonl`：源图、缓存文件和特征张量哈希及维度。
- `summary.json`：模型、commit、数据、资源和验证汇总。
- `dimension_table.md`：Foundation 输出和 LatentMixture 目标接口维度。
- `resource_report.md`：磁盘、I/O、吞吐和显存实测。

日志目录包含 `repeatability-report.json`，其中 `result=PASS`、`matching_tensor_hashes=100` 才表示缓存复现通过。

## 边界说明

8.24 检查验证的是 Foundation 特征缓存边界。当前 LatentMixture train/predict smoke 与 Foundation 特征缓存均已
独立验证；从 `B x 384 x 16 x 16` 特征到 P3/P4/P5 的 resize/channel projection 是后续 D1 P0 必须完成的
接线，不在本准入结果中冒充已完成。
