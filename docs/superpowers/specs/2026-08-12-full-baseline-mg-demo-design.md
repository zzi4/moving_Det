# 全量 Baseline + MG-VTOD 训练与初版 Demo 设计

## 1. 目标

在冻结的 VRUD 序列划分上，用完整训练集分别完成单帧 Baseline 和五帧
MG-VTOD-OBB 的正式训练，通过独立验证集选择 checkpoint 和置信度阈值，最后在一次性
冻结的测试序列连续窗口上生成可播放的 OBB 检测 Demo。

“完成整个训练集的训练”在本设计中有明确含义：一个正式 epoch 必须消费
`runs/vrud-pilot/manifest/train.jsonl` 中全部 13,998 条记录；训练持续到验证集早停，或
达到配置上限 80 epoch。只遍历一轮、只跑固定 step 或继续使用 64 条 overfit
checkpoint，都不算正式训练完成。

初版 Demo 的范围是视频级 VRU 检测和四分类，不展示未经实现和评测的预测 Track ID。
完整 OBB 轨迹关联、轨迹补全和轨迹级分类属于下一阶段。

## 2. 已有证据和它能说明什么

固定 64 条过拟合样本上的同帧配对诊断显示：

| 指标 | Baseline | MG-VTOD |
| --- | ---: | ---: |
| TP | 138 | 183 |
| FP | 39 | 14 |
| FN | 130 | 85 |
| Precision | 77.97% | 92.89% |
| Recall | 51.49% | 68.28% |

MG-VTOD 补回 50 个 Baseline 漏检目标，同时退化 5 个。短边 16–24 px 的 Recall 从
42.66% 提升到 60.14%，短边 24–32 px 的 Recall 从 54.26% 提升到 73.40%。这证明
五帧运动增强分支能够学习到有价值的时序信号，也说明旧的“相对 loss 必须继续下降
50%”过拟合 gate 不适合从已训练 Baseline 初始化的二阶段模型。

这些数字来自参与训练的 64 条记录，只是可学习性证据，不能作为泛化性能结论。正式
结论必须来自序列隔离的 validation/test 数据。

## 3. 冻结数据和公平性

使用现有不可变 manifest：

- train：13,998 条 tile，含 11,199 个正样本和 2,799 个背景样本，覆盖 6 个序列；
- validation：16,575 条记录，覆盖 3 个独立序列；
- test：60,900 条记录，覆盖另外 3 个独立序列。

两个模型必须共享配置、训练样本、增强、有效 batch size 16、优化器、后处理、类别
映射和 seed `20260806`。Baseline 从公开 `yolo11m-obb.pt` 初始化；MG-VTOD 只能从
本次全量 Baseline 的最优 checkpoint 初始化。64 条实验 checkpoint 不进入正式训练。

test 在 Baseline 和 MG-VTOD 都完成、validation 阈值都冻结之前不用于调参或模型
选择。Demo 窗口在第一次打开模型预测前由固定 seed 从 test 连续窗口中选定。

## 4. 方案比较

### 方案 A：可恢复的顺序正式训练（采用）

先训练全量 Baseline，再以其最佳 checkpoint 初始化全量 MG-VTOD。每个 epoch 做独立
validation，按照 mAP50 保存 best 并早停。训练交给独立的用户级服务，checkpoint 和
provenance 支持异常恢复。优点是公平、可解释、资源峰值可控；代价是两阶段总用时较长。

### 方案 B：Baseline 与 MG-VTOD 各占一张 GPU 并行训练

墙钟时间看似更短，但 MG 必须等待正式 Baseline checkpoint，无法真正同时开始；单卡
也会失去已经验证过的双 GPU 吞吐，并增加两个长任务相互争抢 NAS 和 CPU 的风险，不采用。

### 方案 C：直接扩训 64 条 MG checkpoint

启动快，但 Baseline 和 MG 的初始化数据不同，过拟合偏置会污染正式比较，也无法证明
全量训练的增益来源，不采用。

## 5. 实施架构

完整链路如下：

```text
冻结输入与指纹审计
  → 全量 Baseline 双 GPU 训练
  → Baseline validation + 阈值冻结
  → 从 Baseline best 初始化 MG-VTOD
  → 全量 MG-VTOD 双 GPU 训练
  → MG validation + 阈值冻结
  → 两模型公平比较与通过标准判定
  → 一次性 test 评测
  → 冻结连续窗口视频 Demo 与网页报告
```

新增一个只负责阶段编排和状态记录的正式训练控制器。它不修改模型内部训练语义，只做
输入校验、阶段转换、日志、恢复和输出检查。状态文件使用原子写入，并明确记录
`pending/running/completed/failed`、当前模型、epoch、optimizer step、checkpoint 指纹、
manifest 指纹和 alignment cache 指纹。

训练进程通过用户级 systemd service 托管。服务异常退出时不自动跳过阶段；控制器先
验证 `last.pt` 与 `best.pt` 配对、optimizer/scheduler/scaler/RNG 状态和缓存指纹，再从
最后合法 epoch 恢复。连续三次在同一 epoch 失败则停止并保留诊断，不进入下一模型。

## 6. 训练和吞吐策略

Baseline 和 MG-VTOD 都使用两个 RTX A6000、DDP、AMP、1024×1024 tile 和有效 batch
size 16。正式训练不使用过拟合 loss gate，模型选择只依据独立 validation 的 mAP50。

启动正式 Baseline 后，首个完整 epoch 同时作为真实吞吐基线，记录：

- 训练阶段和 validation 阶段各自耗时；
- 两张 GPU 的中位利用率和显存峰值；
- 数据等待时间、AMP overflow skip 和每个 optimizer step 的平均时间；
- 预计早停下限和 80 epoch 上限时间。

若 GPU 中位利用率低于 70%，先定位数据加载、NMS、validation 或同步瓶颈。只允许做有
输出等价性测试的吞吐优化，例如提高只读 DataLoader worker/prefetch，或把独立样本的
validation 前向批处理；不得降低分辨率、缩短五帧 clip、改变训练样本或用 validation
子集替代每 epoch 的完整验证。优化后必须从 epoch 边界 checkpoint 恢复。

## 7. MG-VTOD 稳定化

保留现有残差式运动融合和五帧偏移 `[-4, -2, 0, 2, 4]`。正式 MG 训练从全量 Baseline
best 初始化检测器参数，新增运动分支使用已有的近恒等初始化，使训练开始时输出接近
Baseline。首轮不引入新的检测损失或类别权重，避免同时改变多个变量。

若独立 validation 显示总体 Recall 或小目标 Recall 退化，再执行一个受控修订实验：前
3 个 epoch 冻结 detector，仅训练运动分支和融合 gate；之后以 detector 学习率为运动
分支的 0.1 倍联合微调。该修订必须写入独立 run，不能覆盖首次正式结果。

## 8. 正式验收标准

训练完成的工程条件：

1. run 状态为 `completed`，退出原因为早停或达到 80 epoch；
2. 每个完成 epoch 都遍历 13,998 条训练记录且两个 DDP rank 无重复/遗漏；
3. best/last checkpoint、optimizer、scheduler、scaler 和 RNG 状态有限且指纹匹配；
4. validation 每 epoch 完整运行，无 validation/test 数据进入训练；
5. 两张 GPU 在任务退出后释放显存。

MG-VTOD 的性能通过标准沿用冻结设计，并增加精度保护：

1. 短边不超过 24 px 的 `Recall@rIoU 0.25` 比 Baseline 提高至少 5 个百分点；
2. 全体 VRU 的 `Recall@rIoU 0.25` 提高至少 3 个百分点；
3. mAP50 相比 Baseline 下降不超过 1 个百分点；
4. Precision 不低于 Baseline 超过 1 个百分点；
5. 短暂停止阶段 Recall 不显著低于 Baseline；
6. 配对诊断中 `rescued > regressed`；
7. 类别映射错误为 0，四类和两个站点都单独报告。

未达到性能标准仍算一次完整、有效的正式训练，但 Demo 和报告必须标为“实验结果未通过
MG 性能 gate”，不能宣称多帧模型泛化优于 Baseline。

## 9. 评测与初版 Demo

每个模型先在 validation 上以每帧 FP 不超过 5 为约束，选择最大化
`F1@rIoU 0.25` 的全局置信度阈值。两个阈值分别冻结后，只运行一次 test 评测，报告：

- Precision、Recall、mAP50 和每帧 FP；
- pedestrian、bicycle、tricycle、motorcycle 分类别指标；
- 短边 `<16`、`16–24`、`24–32`、`>=32` px 分层指标；
- 速度、站点、轨迹覆盖、最长连续漏检和短暂停止 Recall；
- Baseline→MG 的逐目标 rescued/regressed 统计。

Demo 使用固定 test 连续 300 帧，保持 3840×2160 原始画面和 30 FPS 输出。页面提供原图、
Baseline 和 MG-VTOD 三种视图，绘制旋转框、四分类名称和置信度；MG 视图额外提供运动
强度 inset。Demo 同时输出 MP4、逐帧 JSONL 和网页播放器。视频上的类别是帧级结果，
不绘制 Track ID 或不存在的轨迹线。

## 10. 错误处理和安全边界

- checkpoint、manifest 或 alignment cache 指纹不匹配：拒绝启动或恢复；
- NaN/Inf、DDP rank 不一致、AMP scale 分叉：当前 run 失败，保留最后合法 checkpoint；
- NAS 图像、JSON 或 cache entry 缺失：记录精确样本身份并停止，不用零帧掩盖损坏；
- validation 指标或阈值非有限：拒绝进入 test；
- test 输出已存在：不覆盖，只有显式的新 run 目录可重新评测；
- Demo 编码失败：保留逐帧 JSONL 和已经完成的 PNG staging，原子发布失败时不替换旧 Demo；
- 用户级服务、网页服务和训练服务使用独立端口/单元，不终止任何无关进程。

## 11. 测试和交付物

实现遵循 TDD。新增测试覆盖阶段状态机、恢复检查、全量 epoch 计数、失败不越级、阈值
冻结、test 单次消费、连续窗口冻结、OBB 视频渲染和原子发布。训练前运行完整测试套件；
每个正式阶段结束后验证 artifact schema、指纹、有限数值和 JPEG/MP4 可解码性。

最终交付：

- `runs/vrud-pilot/baseline-full/`：Baseline 正式 run 和 checkpoint；
- `runs/vrud-pilot/mg-vtod-full/`：MG-VTOD 正式 run 和 checkpoint；
- 两个 validation run、两个冻结阈值和两个一次性 test run；
- `runs/vrud-pilot/baseline-mg-demo/`：300 帧 MP4、JSONL、代表帧和网页；
- 局域网报告：训练曲线、分层指标、配对案例、限制和下一阶段轨迹计划。

LSTFE、完整 MOT ID 关联、轨迹补全和轨迹级重分类不属于本轮交付。
