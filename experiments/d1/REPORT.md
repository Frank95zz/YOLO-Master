# D1：冻结基础模型检测实现与研究结果

## 1. 问题与实现

研究问题：冻结 DINOv3，不更新基础模型，只训练多尺度适配器、LatentMixture 和检测头，能保留多少检测精度，能否降低训练成本？

本实现固定 DINOv3 ViT-S/16、640 输入、block 4/8/12、FP16 特征缓存。三个中间层均为 stride16，经九条独立 Adapter 分支形成 P3/P4/P5 候选，再由各尺度的 LatentMixture 融合并送入 YOLO26 Detect。Teacher 不进入下游优化器或 checkpoint。

交付包括：多层 Teacher API、严格缓存合同与校验、可恢复 safetensors 写入、保留源文件的 NPY 转换、缓存 Dataset/Trainer/Validator、三种 P5 Adapter、显式 latent aux 汇总、checkpoint 与回归测试，以及不依赖个人实验队列的训练/评测入口。使用步骤见 [README](README.md)。

P3 可分离双线性实现与 foreach EMA 为显式选项，保留原实现作数值对照；它们不改变模型结构。NPY 与 safetensors 存储相同 FP16 特征，不引入额外有损量化。实际吞吐依赖磁盘、缓存热度、CPU 和任务干扰，不能把更换存储介质的收益全部归为模型降本。

## 2. 架构筛选

固定 COCO 2017 train2017/val2017（118,287/5,000 张）、seed0、50轮窗口，在相同研究配方下对比 P5 分支。下表是固定第50轮 checkpoint 的独立 COCO 标准评测，AP 按 0-100 点显示，不是内部 mAP，也不是每组最优 epoch。

| P5 结构 | COCO 下游参数 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE：3x3 stride2 Conv | 3,542,567 | 28.870 | 49.197 | 30.113 | 12.921 | 33.463 | 40.228 |
| DW：深度可分离分支 | 1,195,943 | 28.593 | 48.542 | 29.723 | 13.033 | 32.510 | 41.559 |
| BN64：64通道瓶颈 | 1,404,839 | 29.745 | 49.284 | 31.203 | 12.965 | 33.086 | 42.992 |

BN64 被选作后续基座。此处 BN64 指瓶颈宽度64，不是 BatchNorm；归一化仍使用 GroupNorm。单 seed 结果仅支持本轮候选筛选，不构成统计等效或稳定提升证明。

完整配方、成本拆分、曲线、checkpoint 身份和无 FinsSim 干扰的正常 epoch 计时见固定版本 [架构筛选报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/P5_FAST_RUN_20260909.md) 与 [机器可读汇总](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/p5-screen-20260909/suite-summary.json)。

## 3. Latent Aux 消融

VisDrone2019-DET 完整 train/val 为6,471/548张，使用 BN64、weighted_sum、三个独立种子0/1/2、固定300轮学习率调度的前60轮。主结果统一使用第60轮 checkpoint 和固定官方 MATLAB DET 工具，保留官方 ignore 语义。

第一阶段扫描 balance={0,0.01,0.1} 与 z={0,0.001,0.01}，gain=0.1，共27次训练；第二阶段固定 balance=0.1、z=0，扫描 gain，新增9次并复用3次。共36个独立运行，不把复用结果重复计数。

| latent_aux_gain | 官方 AP 均值（3 seeds） |
|---|---:|
| 0 | 8.022406397 |
| 0.03 | 8.115370260 |
| 0.1 | 8.131928488 |
| 0.3 | 8.081231746 |

按预先确定的均值优先规则，候选为 balance=0.1、z=0、gain=0.1、budget=3.0。gain=0.1 相对关闭aux平均仅高0.109522点，三 seed 配对差为 -0.245386/+0.339334/+0.234618点；探索性95%区间跨0。相同三个种子还参与了选参，未做独立确认，**不能宣称稳定或显著收益**。gain=0.03 是波动较小的备选。

每5轮导出的预测并未全部完成官方评分，因此这不是完整 standard-best 曲线，也不能证明300轮已经收敛。完整指标、配对差、工具和 checkpoint SHA256 见 [消融报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/E3.md)、[第一阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage1-official-20260911.json)、[第二阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage2-official-20260911.json)。

## 4. 验收结论与局限

| 项目 | 当前结论 |
|---|---|
| P0 可运行闭环 | 冻结 Teacher、缓存、Adapter、LatentMixture、Detect、训练和独立评测已实现；报告中区分历史真实实验与本分支合成回归 |
| P1 对照 | 已覆盖 COCO 与 VisDrone 的冻结模型研究；最终同总参数量 scratch 对照尚未完成，不宣称 P1 完成 |
| P1 GPU-hours 至少下降50% | 尚无满足最终公平对照合同的证据，不能认定达标 |
| P2 辅助损失研究 | 已显式注册 latent aux，完成 balance/z/gain 的三 seed 消融；收益不稳定也是结果 |
| 其他 P2 扩展 | 尚未完成其他 Teacher 对比；不把已有缓存工程误称为新增精度收益 |

按当前约定，总参数统计包含冻结 Teacher：ViT-S/16 为21,596,544参数；加 BN64 后 COCO/VisDrone 总量为23,001,383/22,936,803。保留的 scratch-total-l 配置分别为23,133,560/23,032,340参数，差约+0.57%/+0.42%；这是待正式验证的对照配置，不是已训练的最终基线。

最终比较必须固定数据划分、分辨率、种子、评测与 checkpoint 选择，分别报告总参数/可训练参数、精度保留率、峰值显存、GPU-hours；同时给出含一次特征抽取与不含抽取的成本，声明缓存复用次数。不能把缓存读取吞吐或短窗口筛选直接作为最终降本结论。

## 5. PR 范围与可复现性

本分支直接基于 upstream af961b99b8ef80491e58cb5fd16e25ebaf3741eb，研究归档固定为 f4d2bc268bb6339f6545fc3ebe6a247c238cd883。19个核心源码文件与 EMA 实现保留研究语义，部分仅格式化；验证使用去除位置属性的 Python AST 比较，不把它说成全部文件字节相同。

新入口从个人队列中解耦，新增独立训练、严格重载、COCO/VisDrone 预测导出；通用示例不提供历史队列的逐rank恢复、重试、筛选和周期官方评分策略。历史训练逐位重放仍以各报告记录的研究执行提交为准。正式新实验需另锁定完整合同并通过启动门禁。

PR 不包含旧实验队列、内存回收、迁移删除脚本、巨型路径列表、权重、数据集、缓存、完整预测或逐阶段流水报告。研究分支保留全部原始工作与证据；这些内容没有被删除或覆写。

上一轮干净代码提交 f5bf7bc56d128e02d3485fe8df1e15301bee7556 的回归合计540通过、56跳过，另单列2项已在相同上游提交复现的既有失败。该数字只对应历史提交，不作为本轮改动后的测试结果。

过程性验证 JSON 已从当前 PR 文件树移出，完整记录仍可从 [历史提交](https://github.com/Frank95zz/YOLO-Master/blob/d6fe25ef0011294cf12014bbf6c4629291e7ab07/experiments/d1/manifests/pr-verification.json) 查看。本轮保留 scratch 总参数匹配配置及测试，收缩数据准备器为现有输入校验，并将 COCO/VisDrone 缓存抽取统一为显式图片列表入口。下载解压、多卡抽取调度及其专用测试仅保留于历史版本，不再作为当前公共 API。

通用文件写入与校验方法独立到 artifacts，NPY 转换和评测不再依赖 VisDrone 数据准备模块。缓存格式、模型结构及训练损失未变；补充断点续跑的原 batch 上下文检查。实际模型权重/CUDA 验收仍需显式启用，本轮不下载模型、不运行真实训练或完整缓存生成。

本轮在干净提交 a6d4b295fd7c7f14179c66eae124805d4e162da3 上完成相关回归：**556 passed、56 skipped、2 deselected**，测试耗时51.78秒。两项 deselected 仍是此前在 upstream af961b9 复现的既有失败，不列为通过；未运行完整仓库测试集。覆盖 D1（含 scratch）、Foundation、LatentMixture、损失汇总、EMA、恢复、checkpoint 和配置；其中68项专项验证覆盖输入校验、原 batch 续跑、索引重建、并发拒绝、两种数据集及合成训练/评测。

48个候选 Python 文件通过 Ruff、格式和编译检查，入口 --help、相对文档链接及 git diff --check 通过；codespell 未安装，未执行。19个核心文件、EMA 实现及 scratch 配置/测试均与上一版逐字节一致。脚本总行数由2,240降至1,549，净减691行；候选变更文件由63变为62，新增公共 I/O 模块用于消除交叉依赖，不合并测试凑文件数。

完整日志与机器可读运行记录保存在外部工作区，不进入 PR。日志 SHA256 为 9e974b7a2bf85df54e2cfcaf602908bffef88dbd4403142f5ff27f0615767735；该记录对应上述代码提交，不包含真实数据集训练结果。

建议 PR 标题：`[犀牛鸟-D1]：冻结 DINOv3 多层特征检测与缓存训练闭环`。
