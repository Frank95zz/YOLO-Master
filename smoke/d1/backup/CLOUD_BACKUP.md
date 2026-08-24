# D1 云盘备份说明

## 1. 备份范围

云盘目录 `D1-backup-20260824/` 保存普通 Git 不适合承载的大文件：

- COCO test2017 mini100 与 `coco8` 数据集；
- DINOv2 ViT-S/14 与 YOLO 基础权重；
- 原始准入运行 `511b2675` 的 checkpoint 和两次特征缓存；
- 可移植复现运行 `aca4ee6b` 的 checkpoint 和两次特征缓存。

服务器待上传目录：

```text
/root/yolo-master/d1-cloud-backup-20260824
```

该目录当前只是服务器上的待上传副本，不算异地备份。上传云盘并完成校验后，在此补充长期有效的云盘地址、访问权限和上传日期。

## 2. 校验方法

下载后进入备份根目录执行：

```bash
sha256sum -c SHA256SUMS.txt
```

必须全部显示 `OK`。文件大小见 [`FILES.txt`](FILES.txt)，校验值见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。

## 3. 恢复原则

1. 从 Git 克隆 `feat/topic-d1-fengyanqi` 分支。
2. 根据新服务器 GPU 驱动和 CUDA 条件重新创建 Python 环境，不复制旧 `.conda` 目录。
3. 从云盘下载大文件并执行 SHA256 校验。
4. 数据、权重和缓存路径可通过 `D1_ROOT` 等环境变量重新指定。
5. 运行 `bash smoke/d1/run_admission.sh` 验证迁移后的环境与链路。

## 4. 上传状态

| 项目 | 状态 |
| --- | --- |
| 服务器备份包生成 | 已完成 |
| SHA256 清单生成 | 已完成 |
| 云盘上传 | 待完成 |
| 云盘回下载校验 | 待完成 |
| 云盘地址登记 | 待完成 |
