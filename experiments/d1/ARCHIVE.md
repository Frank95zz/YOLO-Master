# D1 历史资料归档

当前目录保留可维护的实现入口、主要研究结果与必要机器可读证据。旧开发过程资料从当前文件树移除，不代表实验被取消或证据被改写。

## 固定版本

清理前提交：`373b8a4a563bd885c13b78652a75ae4dee7edfdd`。

- [历史阶段文档目录](https://github.com/Frank95zz/YOLO-Master/tree/373b8a4a563bd885c13b78652a75ae4dee7edfdd/experiments/d1)：WP0-WP8、HANDOFF、E0_E1、EMA_BATCHING_RESULTS、P3_UPSAMPLE_RESULTS、RESUME_DIAGNOSTICS。
- [历史准入资料](https://github.com/Frank95zz/YOLO-Master/tree/373b8a4a563bd885c13b78652a75ae4dee7edfdd/smoke/d1)：早期 DINOv2 smoke、重复准入日志、环境输出与备份说明，不作为最终 DINOv3 实验入口。
- [历史诊断脚本](https://github.com/Frank95zz/YOLO-Master/tree/373b8a4a563bd885c13b78652a75ae4dee7edfdd/scripts/d1)：benchmark_cache_layout、benchmark_cache_layout_gpu、check_ema_runtime、diagnose_resume、inspect_wp8_followup 及其专属测试和配置从当前树移除。正式训练依赖的公共函数与恢复逻辑仍保留。
- [历史完整 COCO 列表](https://github.com/Frank95zz/YOLO-Master/tree/373b8a4a563bd885c13b78652a75ae4dee7edfdd/experiments/d1/manifests)：两份路径列表不再受 Git 跟踪，数量和 SHA256 仍由 coco2017-splits.json 固定；按 README 命令生成。

历史报告中的命令、文件哈希和执行提交属于当时版本；重放旧实验应检出对应执行提交，不能将旧证据解释为新版本验收。

## 清理安全性

清理前另存了包含工作区未提交内容的 282 文件归档，并逐文件校验。归档 SHA256：`7a81db49fe33288ddd3f9778c5b327ae63f60ee62fb2fe16f3df247abc64a7bd`。归档保存在仓库外，不把冗余内容换个目录重新提交。

本次不删除外部数据集、特征缓存、权重、checkpoint、预测结果或实验日志；不修改历史 JSON 证据。双周报告和最终 P5/E3 研究报告继续保留。
