# D1 环境参考

本目录记录 8.24 准入服务器的实际环境，用于审计和迁移时对照，不用于整体复制旧环境。

迁移到新服务器时，应根据新机器的 GPU 驱动和 CUDA 兼容条件重新创建 Python 环境，再安装项目依赖并运行准入脚本。`pip-freeze.txt` 和 `conda-explicit.txt` 是本次运行快照，不保证可以跨操作系统、驱动或 CUDA 版本直接安装。

| 文件 | 用途 |
| --- | --- |
| `python-version.txt` | Python 版本 |
| `pip-freeze.txt` | Python 包快照 |
| `conda-explicit.txt` | 原服务器 Conda 精确包记录，平台相关 |
| `torch-cuda-check.txt` | PyTorch 与 CUDA 可用性检查 |
| `gpu-summary.csv` | GPU 型号和摘要 |
| `nvidia-smi-q.txt` | GPU 驱动与设备详细信息 |
| `yolo-version.txt` | Ultralytics/YOLO 环境信息 |
