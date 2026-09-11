# D1 最终 PR 代码范围

## 1. 比较基准与结论

对比 Tencent/YOLO-Master main 的固定提交 `af961b99b8ef80491e58cb5fd16e25ebaf3741eb`，不是把 fork 的所有文件都当作新增贡献。该上游版本已合入本分支。

清理前分支为 `373b8a4a563bd885c13b78652a75ae4dee7edfdd`；原 PR 差异为 279 文件、175,356 行新增。另有此前尚未提交的总参数 Scratch 配置、测试和说明，本轮一并保留。

核心代码冻结提交：`cad5a82481a63a0a681a57ad119d4ae598cac4db`。本报告及验收证据随后单独提交，不反向改写旧实验的执行版本。

当前交付含本报告与小型验收 JSON，共 177 个差异文件：158 个新增、19 个修改，不删除原仓库文件。新增行数约 4 万，较清理前减少约 77%。完整逐文件清单见第 6 节。

| 分类 | 新增 | 修改 | 保留理由 |
| --- | ---: | ---: | --- |
| 核心实现 | 7 | 12 | Teacher、缓存、Adapter、检测模型、Trainer/Validator 及必要扩展接口 |
| 训练/模型配置 | 13 | 0 | 实验合同和可复现架构，包含最新总参数对照基线 |
| 脚本 | 32 | 0 | 数据准备、缓存、训练、评测、消融与实际被入口导入的共享逻辑 |
| 测试 | 33 | 5 | 功能、数值、梯度、恢复、证据及配置回归 |
| 文档与小型证据 | 73 | 0 | README、研究结果、历史证据和本次 PR 验收 |
| 仓库设置 | 0 | 2 | 可选依赖及生成列表忽略规则 |

## 2. 已删除或移出 Git

- 删除 107 个已备份冗余文件：整个旧 smoke/d1、14 份阶段/交接/诊断文档、5 个一次性诊断脚本、4 个专属测试和 1 份废弃准备配置。
- 两份共 123,287 行的 COCO 路径列表取消跟踪，服务器原文件仍在；新 clone 从数据集重建并严格验证已发布数量和 SHA256。
- README 改为功能入口、最小复现、主要结果与边界；保留文档指向已归档文件的链接改为固定历史提交。
- [ARCHIVE.md](ARCHIVE.md) 提供历史版本与备份校验值。双周报告、P5/E3 结果、小型 JSON 证据保留。
- 没有删除外部图片、特征、模型、checkpoint、预测文件、日志或旧实验工作区；没有把多余文件改名后重新塞入 PR。

## 3. 为什么不再按数量继续删

七个新增核心文件已经按职责分离，不再拆出一套重复框架。32 个脚本中部分名字包含 WP，但仍被当前入口导入，不能仅因名称像旧阶段而删除。例如 run_wp8_p1_control 提供共用 ScratchTrainer，run_p5_ablation 被 E3 复用，run_wp7 仍服务完整训练入口。

测试与配置不是多余代码。BASE/DW/BN64、历史训练参数匹配对照和最新总参数匹配对照分别对应不同证据，不用新配置覆盖旧实验。历史对照不得冒充最终总参数 P1 基线。

本轮不重构已验收的 Adapter、LatentMixture、缓存读取和 EMA 算法，不以“更少文件”为由合并职责、移除失败检查或降低验收标准。

## 4. 上游兼容与行为变化

- 保留上游 optimizer 参数审计和预评测恢复重放。
- 采用上游的标量 loss 契约：`native_loss.sum() + aux`；保留 D1 的输入检查及 raw/effective aux 日志。检测项与 aux 的梯度权重仍各计一次，日志项保持独立。
- 保留 D1 的缓存输入 checkpoint smoke、AMP 检查、DDP 策略和 resume 初始步进接口。
- model-only 准备不再读取生成列表，报告明确 `split_lists_verified=false`；正式数据准备仍严格验证列表与文件。
- 辅助损失返回形状从 D1 历史向量变为上游标量，恢复路径也对齐了上游。不能把新代码的运行冒充旧实验逐位复现；新正式实验应使用新 run 和新门禁，不覆盖旧 checkpoint。

## 5. 验收与正式实验边界

验收命令、固定代码提交、测试排除项和结果见 [pr-cleanup-20260911.json](manifests/pr-cleanup-20260911.json)。

最终干净检出结果：**979 passed、59 skipped、2 deselected**，耗时 75.90 秒；59 项跳过由环境或可选集成条件决定，2 项排除均已在纯上游复现。不是无条件全仓库测试通过。

- 在干净检出、无 COCO 生成列表、CPU 隔离条件下运行所有 D1 和 Foundation 测试，以及 LatentMixture、CompositeCriterion、DDP/checkpoint、预验证恢复、optimizer 审计和原模型配置测试。
- 两个纯上游既有失败单独复现并记录，不修改它们以制造“全绿”：Foundation response-KD 测试中的 feature/stride 数量不匹配；foundation_teacher 缺省错误消息断言过时。
- 对保留的 Python 代码按仓库 Ruff 配置静态检查；检查 Python 编译、Markdown 相对链接、diff 空白错误及原历史证据哈希。
- 真实 NVMe COCO 的列表可重建，原 manifest 不变；数据内容完整性仍以既有 WP0/缓存校验流程为准。
- 本轮没有重新跑 GPU 长训练或完整官方评测，不能把 CPU 回归当作新配方的速度、显存或精度结论。
- 尚未提交上游 PR，也未启动新正式训练。P1 仍需两个数据集的总参数匹配配对实验，分别报告精度保留率、显存、GPU-hours 及一次性 Teacher 抽取成本；E3 消融结果不代替 P1 达标证明。

正式实验前：确认本范围和总参数基线；绑定最终执行 commit；完成真实批次 backward、checkpoint 重载、评测导出及资源检查；再启动获确认的配对实验。PR 标题使用 `[犀牛鸟-D1]` 前缀，研究结论按实际完成状态提交。

## 6. 逐文件清单

A 表示相对上游新增，M 表示修改。清单不把上游自身近期合入的文件算作 D1 新增。

### 核心实现

| 状态 | 文件 |
| --- | --- |
| M | [ultralytics/data/__init__.py](../../ultralytics/data/__init__.py) |
| M | [ultralytics/data/build.py](../../ultralytics/data/build.py) |
| A | [ultralytics/data/d1_cache.py](../../ultralytics/data/d1_cache.py) |
| M | [ultralytics/engine/extensions/recovery.py](../../ultralytics/engine/extensions/recovery.py) |
| M | [ultralytics/engine/trainer.py](../../ultralytics/engine/trainer.py) |
| M | [ultralytics/models/yolo/detect/__init__.py](../../ultralytics/models/yolo/detect/__init__.py) |
| A | [ultralytics/models/yolo/detect/foundation_train.py](../../ultralytics/models/yolo/detect/foundation_train.py) |
| A | [ultralytics/models/yolo/detect/foundation_val.py](../../ultralytics/models/yolo/detect/foundation_val.py) |
| M | [ultralytics/nn/__init__.py](../../ultralytics/nn/__init__.py) |
| M | [ultralytics/nn/foundation/__init__.py](../../ultralytics/nn/foundation/__init__.py) |
| A | [ultralytics/nn/foundation/cache.py](../../ultralytics/nn/foundation/cache.py) |
| A | [ultralytics/nn/foundation/npy_cache.py](../../ultralytics/nn/foundation/npy_cache.py) |
| M | [ultralytics/nn/foundation/teachers/dinov3.py](../../ultralytics/nn/foundation/teachers/dinov3.py) |
| A | [ultralytics/nn/foundation_detection_model.py](../../ultralytics/nn/foundation_detection_model.py) |
| M | [ultralytics/nn/mixture_loss.py](../../ultralytics/nn/mixture_loss.py) |
| M | [ultralytics/nn/modules/__init__.py](../../ultralytics/nn/modules/__init__.py) |
| A | [ultralytics/nn/modules/foundation_adapter.py](../../ultralytics/nn/modules/foundation_adapter.py) |
| M | [ultralytics/nn/tasks.py](../../ultralytics/nn/tasks.py) |
| M | [ultralytics/utils/torch_utils.py](../../ultralytics/utils/torch_utils.py) |

### 配置

| 状态 | 文件 |
| --- | --- |
| A | [ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml](../../ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml) |
| A | [ultralytics/cfg/experiments/d1/p1p2/e1-coco2017.yaml](../../ultralytics/cfg/experiments/d1/p1p2/e1-coco2017.yaml) |
| A | [ultralytics/cfg/experiments/d1/p1p2/e2-visdrone.yaml](../../ultralytics/cfg/experiments/d1/p1p2/e2-visdrone.yaml) |
| A | [ultralytics/cfg/experiments/d1/p1p2/e3-visdrone.yaml](../../ultralytics/cfg/experiments/d1/p1p2/e3-visdrone.yaml) |
| A | [ultralytics/cfg/experiments/d1/p1p2/p5-coco2017.yaml](../../ultralytics/cfg/experiments/d1/p1p2/p5-coco2017.yaml) |
| A | [ultralytics/cfg/experiments/d1/wp7-minimal-tests.yaml](../../ultralytics/cfg/experiments/d1/wp7-minimal-tests.yaml) |
| A | [ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml](../../ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml) |
| A | [ultralytics/cfg/experiments/d1/wp8-p1-scratch-coco2017.yaml](../../ultralytics/cfg/experiments/d1/wp8-p1-scratch-coco2017.yaml) |
| A | [ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml) |
| A | [ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml) |
| A | [ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-dw-n.yaml](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-dw-n.yaml) |
| A | [ultralytics/cfg/models/26/yolo26-d1-scratch-matched-n.yaml](../../ultralytics/cfg/models/26/yolo26-d1-scratch-matched-n.yaml) |
| A | [ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml](../../ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml) |

### 复现与实验脚本

| 状态 | 文件 |
| --- | --- |
| A | [scripts/d1/__init__.py](../../scripts/d1/__init__.py) |
| A | [scripts/d1/benchmark_wp8_training.py](../../scripts/d1/benchmark_wp8_training.py) |
| A | [scripts/d1/cache_features.py](../../scripts/d1/cache_features.py) |
| A | [scripts/d1/cache_visdrone.py](../../scripts/d1/cache_visdrone.py) |
| A | [scripts/d1/convert_cache_to_npy.py](../../scripts/d1/convert_cache_to_npy.py) |
| A | [scripts/d1/diagnose_e3_gradients.py](../../scripts/d1/diagnose_e3_gradients.py) |
| A | [scripts/d1/diagnose_wp8.py](../../scripts/d1/diagnose_wp8.py) |
| A | [scripts/d1/e3_mechanism.py](../../scripts/d1/e3_mechanism.py) |
| A | [scripts/d1/e3_probe_precision.py](../../scripts/d1/e3_probe_precision.py) |
| A | [scripts/d1/ema.py](../../scripts/d1/ema.py) |
| A | [scripts/d1/evaluate_visdrone.py](../../scripts/d1/evaluate_visdrone.py) |
| A | [scripts/d1/evaluate_visdrone_official.m](../../scripts/d1/evaluate_visdrone_official.m) |
| A | [scripts/d1/launch_wp8_npy.py](../../scripts/d1/launch_wp8_npy.py) |
| A | [scripts/d1/launch_wp8_p1.py](../../scripts/d1/launch_wp8_p1.py) |
| A | [scripts/d1/p1p2_data.py](../../scripts/d1/p1p2_data.py) |
| A | [scripts/d1/p1p2_runtime.py](../../scripts/d1/p1p2_runtime.py) |
| A | [scripts/d1/prepare_visdrone.py](../../scripts/d1/prepare_visdrone.py) |
| A | [scripts/d1/prepare_wp0.py](../../scripts/d1/prepare_wp0.py) |
| A | [scripts/d1/run_e2.py](../../scripts/d1/run_e2.py) |
| A | [scripts/d1/run_e3.py](../../scripts/d1/run_e3.py) |
| A | [scripts/d1/run_e3_stage2.py](../../scripts/d1/run_e3_stage2.py) |
| A | [scripts/d1/run_p1p2.py](../../scripts/d1/run_p1p2.py) |
| A | [scripts/d1/run_p5_ablation.py](../../scripts/d1/run_p5_ablation.py) |
| A | [scripts/d1/run_p5_suite.py](../../scripts/d1/run_p5_suite.py) |
| A | [scripts/d1/run_with_cache_cleanup.py](../../scripts/d1/run_with_cache_cleanup.py) |
| A | [scripts/d1/run_wp7.py](../../scripts/d1/run_wp7.py) |
| A | [scripts/d1/run_wp8.py](../../scripts/d1/run_wp8.py) |
| A | [scripts/d1/run_wp8_p1_control.py](../../scripts/d1/run_wp8_p1_control.py) |
| A | [scripts/d1/run_wp8_train.py](../../scripts/d1/run_wp8_train.py) |
| A | [scripts/d1/summarize_e3_stage2.py](../../scripts/d1/summarize_e3_stage2.py) |
| A | [scripts/d1/test_visdrone_official.m](../../scripts/d1/test_visdrone_official.m) |
| A | [scripts/d1/visdrone_matlab_compat/mean2.m](../../scripts/d1/visdrone_matlab_compat/mean2.m) |

### 回归测试

| 状态 | 文件 |
| --- | --- |
| A | [tests/test_d1_cache_cleanup_runner.py](../../tests/test_d1_cache_cleanup_runner.py) |
| A | [tests/test_d1_e2_engineering.py](../../tests/test_d1_e2_engineering.py) |
| A | [tests/test_d1_e2_evidence.py](../../tests/test_d1_e2_evidence.py) |
| A | [tests/test_d1_e3_aux_sweep.py](../../tests/test_d1_e3_aux_sweep.py) |
| A | [tests/test_d1_e3_gradient_precision.py](../../tests/test_d1_e3_gradient_precision.py) |
| A | [tests/test_d1_e3_stage2.py](../../tests/test_d1_e3_stage2.py) |
| A | [tests/test_d1_e3_stage2_results.py](../../tests/test_d1_e3_stage2_results.py) |
| A | [tests/test_d1_ema.py](../../tests/test_d1_ema.py) |
| A | [tests/test_d1_final_eval.py](../../tests/test_d1_final_eval.py) |
| A | [tests/test_d1_npy_migration.py](../../tests/test_d1_npy_migration.py) |
| A | [tests/test_d1_npy_training.py](../../tests/test_d1_npy_training.py) |
| A | [tests/test_d1_p1p2_contract.py](../../tests/test_d1_p1p2_contract.py) |
| A | [tests/test_d1_p1p2_data.py](../../tests/test_d1_p1p2_data.py) |
| A | [tests/test_d1_p3_upsample.py](../../tests/test_d1_p3_upsample.py) |
| A | [tests/test_d1_p5_ablation.py](../../tests/test_d1_p5_ablation.py) |
| A | [tests/test_d1_p5_evidence.py](../../tests/test_d1_p5_evidence.py) |
| A | [tests/test_d1_scratch_total_contract.py](../../tests/test_d1_scratch_total_contract.py) |
| A | [tests/test_d1_visdrone_protocol.py](../../tests/test_d1_visdrone_protocol.py) |
| A | [tests/test_d1_wp0_contract.py](../../tests/test_d1_wp0_contract.py) |
| A | [tests/test_d1_wp1_dinov3.py](../../tests/test_d1_wp1_dinov3.py) |
| A | [tests/test_d1_wp2_cache_cli.py](../../tests/test_d1_wp2_cache_cli.py) |
| A | [tests/test_d1_wp2_feature_cache.py](../../tests/test_d1_wp2_feature_cache.py) |
| A | [tests/test_d1_wp3_foundation_adapter.py](../../tests/test_d1_wp3_foundation_adapter.py) |
| A | [tests/test_d1_wp4_foundation_detection_model.py](../../tests/test_d1_wp4_foundation_detection_model.py) |
| A | [tests/test_d1_wp5_cached_pipeline.py](../../tests/test_d1_wp5_cached_pipeline.py) |
| A | [tests/test_d1_wp6_latent_aux.py](../../tests/test_d1_wp6_latent_aux.py) |
| A | [tests/test_d1_wp7_acceptance.py](../../tests/test_d1_wp7_acceptance.py) |
| A | [tests/test_d1_wp8_diagnostics.py](../../tests/test_d1_wp8_diagnostics.py) |
| A | [tests/test_d1_wp8_formal_training.py](../../tests/test_d1_wp8_formal_training.py) |
| A | [tests/test_d1_wp8_multi_gpu_cache.py](../../tests/test_d1_wp8_multi_gpu_cache.py) |
| A | [tests/test_d1_wp8_p1_control.py](../../tests/test_d1_wp8_p1_control.py) |
| A | [tests/test_d1_wp8_p1_launch.py](../../tests/test_d1_wp8_p1_launch.py) |
| A | [tests/test_d1_wp8_training_benchmark.py](../../tests/test_d1_wp8_training_benchmark.py) |
| M | [tests/test_ddp_lifecycle_ema_nan.py](../../tests/test_ddp_lifecycle_ema_nan.py) |
| M | [tests/test_foundation_dinov3.py](../../tests/test_foundation_dinov3.py) |
| M | [tests/test_foundation_distill_model.py](../../tests/test_foundation_distill_model.py) |
| M | [tests/test_latent_mixture.py](../../tests/test_latent_mixture.py) |
| M | [tests/test_mixture_loss_composition.py](../../tests/test_mixture_loss_composition.py) |

### 结果与证据

| 状态 | 文件 |
| --- | --- |
| A | [experiments/d1/ARCHIVE.md](ARCHIVE.md) |
| A | [experiments/d1/BIWEEKLY_20260907.md](BIWEEKLY_20260907.md) |
| A | [experiments/d1/E2.md](E2.md) |
| A | [experiments/d1/E3.md](E3.md) |
| A | [experiments/d1/E3_GRADIENT_DIAGNOSTIC_20260911.md](E3_GRADIENT_DIAGNOSTIC_20260911.md) |
| A | [experiments/d1/P1_P2_EXPERIMENT_PLAN.md](P1_P2_EXPERIMENT_PLAN.md) |
| A | [experiments/d1/P5_FAST_RUN_20260909.md](P5_FAST_RUN_20260909.md) |
| A | [experiments/d1/README.md](README.md) |
| A | [experiments/d1/SCRATCH_TOTAL_PARAMETER_BASELINE.md](SCRATCH_TOTAL_PARAMETER_BASELINE.md) |
| A | [experiments/d1/manifests/coco2017-splits.json](manifests/coco2017-splits.json) |
| A | [experiments/d1/manifests/dinov3-vits16.json](manifests/dinov3-vits16.json) |
| A | [experiments/d1/manifests/e0-acceptance-20260908.json](manifests/e0-acceptance-20260908.json) |
| A | [experiments/d1/manifests/e01-environment-20260908.json](manifests/e01-environment-20260908.json) |
| A | [experiments/d1/manifests/e01-static-checks-20260908.json](manifests/e01-static-checks-20260908.json) |
| A | [experiments/d1/manifests/e1-benchmark-20260908.json](manifests/e1-benchmark-20260908.json) |
| A | [experiments/d1/manifests/e1-completion-20260908.json](manifests/e1-completion-20260908.json) |
| A | [experiments/d1/manifests/e1-diagnostic-repair-20260908.json](manifests/e1-diagnostic-repair-20260908.json) |
| A | [experiments/d1/manifests/e1-launch-20260908.json](manifests/e1-launch-20260908.json) |
| A | [experiments/d1/manifests/e2-acceptance.json](manifests/e2-acceptance.json) |
| A | [experiments/d1/manifests/e2-benchmark.json](manifests/e2-benchmark.json) |
| A | [experiments/d1/manifests/e2-closure-validation.json](manifests/e2-closure-validation.json) |
| A | [experiments/d1/manifests/e2-engineering-regression.json](manifests/e2-engineering-regression.json) |
| A | [experiments/d1/manifests/e2-engineering-results.json](manifests/e2-engineering-results.json) |
| A | [experiments/d1/manifests/e2-official-evaluation.json](manifests/e2-official-evaluation.json) |
| A | [experiments/d1/manifests/e2-regression.json](manifests/e2-regression.json) |
| A | [experiments/d1/manifests/e2-visdrone-cache.json](manifests/e2-visdrone-cache.json) |
| A | [experiments/d1/manifests/e2-visdrone-data.json](manifests/e2-visdrone-data.json) |
| A | [experiments/d1/manifests/e2-visdrone-matlab-tests.json](manifests/e2-visdrone-matlab-tests.json) |
| A | [experiments/d1/manifests/e3-gradient-diagnostic-20260911.json](manifests/e3-gradient-diagnostic-20260911.json) |
| A | [experiments/d1/manifests/e3-stage1-official-20260911.json](manifests/e3-stage1-official-20260911.json) |
| A | [experiments/d1/manifests/e3-stage2-official-20260911.json](manifests/e3-stage2-official-20260911.json) |
| A | [experiments/d1/manifests/e3-stage2-plan-20260911.json](manifests/e3-stage2-plan-20260911.json) |
| A | [experiments/d1/manifests/e3-stage2-startup-20260911.json](manifests/e3-stage2-startup-20260911.json) |
| A | [experiments/d1/manifests/ema-integration-20260909/loop-summary.json](manifests/ema-integration-20260909/loop-summary.json) |
| A | [experiments/d1/manifests/ema-integration-20260909/validation.json](manifests/ema-integration-20260909/validation.json) |
| A | [experiments/d1/manifests/environment.json](manifests/environment.json) |
| A | [experiments/d1/manifests/final-evaluation-20260907.json](manifests/final-evaluation-20260907.json) |
| A | [experiments/d1/manifests/licenses.md](manifests/licenses.md) |
| A | [experiments/d1/manifests/p0-experiment-contract.json](manifests/p0-experiment-contract.json) |
| A | [experiments/d1/manifests/p3-deterministic-probe-20260909/comparison-launch.json](manifests/p3-deterministic-probe-20260909/comparison-launch.json) |
| A | [experiments/d1/manifests/p3-deterministic-probe-20260909/comparison-status.json](manifests/p3-deterministic-probe-20260909/comparison-status.json) |
| A | [experiments/d1/manifests/p3-deterministic-probe-20260909/comparison-summary.json](manifests/p3-deterministic-probe-20260909/comparison-summary.json) |
| A | [experiments/d1/manifests/p3-deterministic-probe-20260909/deterministic-micro.json](manifests/p3-deterministic-probe-20260909/deterministic-micro.json) |
| A | [experiments/d1/manifests/p3-deterministic-probe-20260909/layer-parity.json](manifests/p3-deterministic-probe-20260909/layer-parity.json) |
| A | [experiments/d1/manifests/p3-deterministic-probe-20260909/numerical-status.json](manifests/p3-deterministic-probe-20260909/numerical-status.json) |
| A | [experiments/d1/manifests/p3-deterministic-probe-20260909/regression.log](manifests/p3-deterministic-probe-20260909/regression.log) |
| A | [experiments/d1/manifests/p5-screen-20260909/BASE-results.csv](manifests/p5-screen-20260909/BASE-results.csv) |
| A | [experiments/d1/manifests/p5-screen-20260909/BN64-results.csv](manifests/p5-screen-20260909/BN64-results.csv) |
| A | [experiments/d1/manifests/p5-screen-20260909/DW-results.csv](manifests/p5-screen-20260909/DW-results.csv) |
| A | [experiments/d1/manifests/p5-screen-20260909/archive-index.json](manifests/p5-screen-20260909/archive-index.json) |
| A | [experiments/d1/manifests/p5-screen-20260909/matrix-summary.json](manifests/p5-screen-20260909/matrix-summary.json) |
| A | [experiments/d1/manifests/p5-screen-20260909/nonoverlap-timing.json](manifests/p5-screen-20260909/nonoverlap-timing.json) |
| A | [experiments/d1/manifests/p5-screen-20260909/resource-overlap.json](manifests/p5-screen-20260909/resource-overlap.json) |
| A | [experiments/d1/manifests/p5-screen-20260909/suite-launch.json](manifests/p5-screen-20260909/suite-launch.json) |
| A | [experiments/d1/manifests/p5-screen-20260909/suite-summary.json](manifests/p5-screen-20260909/suite-summary.json) |
| A | [experiments/d1/manifests/p5-screen-20260909/training-integrity.json](manifests/p5-screen-20260909/training-integrity.json) |
| A | [experiments/d1/manifests/publication-20260907.json](manifests/publication-20260907.json) |
| A | [experiments/d1/manifests/resume-diagnostics-20260909/summary.json](manifests/resume-diagnostics-20260909/summary.json) |
| A | [experiments/d1/manifests/resume-diagnostics-20260909/validation.json](manifests/resume-diagnostics-20260909/validation.json) |
| A | [experiments/d1/manifests/visdrone-toolkit.json](manifests/visdrone-toolkit.json) |
| A | [experiments/d1/manifests/wp2-cache-100-index-a.json](manifests/wp2-cache-100-index-a.json) |
| A | [experiments/d1/manifests/wp2-cache-100-index-b.json](manifests/wp2-cache-100-index-b.json) |
| A | [experiments/d1/manifests/wp2-cache-100-reproducibility.json](manifests/wp2-cache-100-reproducibility.json) |
| A | [experiments/d1/manifests/wp2-cache-100-samples.jsonl](manifests/wp2-cache-100-samples.jsonl) |
| A | [experiments/d1/manifests/wp7-coco8.json](manifests/wp7-coco8.json) |
| A | [experiments/d1/manifests/wp7-overfit32.json](manifests/wp7-overfit32.json) |
| A | [experiments/d1/manifests/wp7-parity.json](manifests/wp7-parity.json) |
| A | [experiments/d1/manifests/wp7-summary.json](manifests/wp7-summary.json) |
| A | [experiments/d1/manifests/wp8-cache-benchmark.json](manifests/wp8-cache-benchmark.json) |
| A | [experiments/d1/manifests/wp8-followup-preparation.json](manifests/wp8-followup-preparation.json) |
| A | [experiments/d1/manifests/wp8-full-cache.json](manifests/wp8-full-cache.json) |

### 仓库设置

| 状态 | 文件 |
| --- | --- |
| M | [.gitignore](../../.gitignore) |
| M | [pyproject.toml](../../pyproject.toml) |

本次清理额外新增本文件 [PR_SCOPE.md](PR_SCOPE.md) 与 [验收摘要](manifests/pr-cleanup-20260911.json)，包含在第 1 节统计中。
