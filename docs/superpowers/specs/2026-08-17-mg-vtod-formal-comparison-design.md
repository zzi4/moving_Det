# MG-VTOD 正式对照实验设计

## 1. 目标与结论边界

本阶段只回答一个问题：在相同 Universal-P2 初始化、相同训练数据、相同 OBB 检测器、
相同训练预算和相同评测协议下，加入五帧运动分支的 MG-VTOD，能否比单帧 Baseline
更稳定地检出航拍视频中的小尺寸 VRU，并减少连续漏检，同时不显著损害精度、静止目标
和原有类别能力。

本阶段采用已经确认的方案 A：依次完成正式 Baseline、MG-VTOD Full、必要消融和一次性
人工基准评测。LSTFE、Track ID 预测、轨迹关联、轨迹补全和轨迹级分类全部延后；只有
MG-VTOD 的检测结论明确后，才进入追踪阶段。

人工基准中的视频可能与 Universal 模型的历史训练来源存在重叠。因此允许形成的结论是
“在当前目标域和相同 Universal 初始化下，MG-VTOD 相对 Baseline 的增量效果”，不能
宣称它证明了对完全未知站点或未知视频的无偏泛化能力。

## 2. 已冻结基础

正式实验必须直接消费以下现有产物，不重新生成、不修改内容：

- 人工基准：`runs/vrud-pilot/human-benchmark-20260816/`；
- 人工基准规模：873 帧、78,335 条源标注、53,735 个 VRU 真值、334 个边缘 ignore；
- 人工基准指纹：
  `90c00eadb50d38cc3be0ffd8e30399041855f8be81804e83288304160178b851`；
- Universal-P2 初始化：
  `runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt`；
- 原始 Universal 权重 SHA-256：
  `114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7`；
- 冻结 P2 产物 SHA-256：
  `d474b9cc8aa113e72de0352bfe4e45aea6b0b7c7a28f67de889214d495428948`；
- P2 迁移合同：427 个张量成功迁移到 859 个目标张量，全部有限；
- 冻结数据划分：`runs/vrud-pilot/manifest/`，包含 6 个 train、3 个 validation、
  3 个 test 序列；
- 冻结时序配准：`runs/vrud-pilot/alignment-cache/`；
- 训练配置：`configs/vrud-temporal-obb.yaml`，随机种子 `20260806`、30 FPS、
  1024×1024 tile、256 px overlap、有效 batch size 16、最多 80 epoch、早停耐心 15。

真实输入的 ZIP、源帧、manifest、alignment cache、P2、checkpoint 或阈值文件任一指纹
不符，当前阶段必须停止，不能自动重建或静默替换。

## 3. 方案选择

采用顺序正式对照：先由冻结 Universal-P2 训练单帧 Baseline，再由该次 Baseline 的
正式 `best.pt` 初始化 MG-VTOD。这样两者拥有可审计的共同起点，MG 的差异主要来自运动
分支。训练前增加快速 CUDA 和输入预检，但不另外运行会改变正式结论的短周期模型。

未采用“先跑 3–5 epoch 再正式训练”，因为它会增加一轮计算且不能替代正式结论；未采用
“直接训练 MG”，因为缺少同条件 Baseline 后无法把增益归因于运动信息；未采用“同时
训练 Baseline 与 MG”，因为 MG 必须等待本次正式 Baseline 的最佳 checkpoint。

## 4. 执行架构与阶段边界

正式流程严格按以下依赖顺序执行：

1. 只读预检冻结输入、代码版本、双 GPU、磁盘和输出目录；
2. 双 GPU 完成正式 Baseline；
3. 在 validation 上评估 Baseline 并冻结阈值；
4. 从正式 Baseline `best.pt` 初始化并双 GPU 训练 MG-VTOD Full；
5. 在 validation 上评估 MG Full 并冻结阈值；
6. 运行 MG Motion-Off，并在资源允许时完成 MG Frozen 消融；
7. 确认所有阈值已经冻结后，对人工 873 帧进行一次性评测；
8. 生成配对指标、视频、代表案例和局域网页；
9. 根据预先冻结的 gate 得出通过或未通过结论。

每一步只能读取上一步已经原子发布并通过严格加载的产物。test 结果不能反向修改模型、
阈值、后处理参数、类别映射、筛选帧或案例选择规则。

## 5. 训练公平性

Baseline 与 MG Full 共享以下内容：

- train/validation manifest、tile 和中心帧全集；
- 四类 VRU schema、OBB 几何、数据增强和随机种子；
- P2-P5 检测器结构、检测损失、优化器、学习率计划和训练预算；
- validation 全集、模型选择指标、NMS 和后处理规则；
- 双 RTX A6000、DDP、AMP 和有效 batch size 16。

两者唯一的主要结构差异是 MG Full 使用五帧偏移 `[-4, -2, 0, 2, 4]`、冻结的 ECC
配准和运动强度分支。首个正式 MG run 不增加类别权重、不修改检测损失，也不同时引入
其他结构变化。

正式 Baseline 必须直接从冻结 P2 启动并不间断完成。由于当前严格血缘合同拒绝由 resume
产生的 Baseline checkpoint 作为时序初始化，如果 Baseline 中断，应保留现场用于诊断，
然后从同一 P2 在新的、从未使用过的输出目录完整重启。禁止覆盖失败目录，也禁止把
`baseline-resumed-only` 的结果交给 MG。

MG Full 只能从本次正式 Baseline 的 `best.pt` 初始化。MG 自身中断后可以用同一 run 的
`last.pt` 严格恢复，但恢复 checkpoint 不能重新当作 `--baseline-init`。

## 6. 输出目录与不可覆盖规则

实施计划应在启动时解析一个唯一的正式实验 ID，并在预检后冻结。默认使用日期和递增
序号，例如 `20260817-01`。目录结构为：

- `runs/vrud-pilot/formal-20260817-01/preflight/`；
- `runs/vrud-pilot/formal-20260817-01/baseline/`；
- `runs/vrud-pilot/formal-20260817-01/baseline-validation/`；
- `runs/vrud-pilot/formal-20260817-01/mg-vtod-full/`；
- `runs/vrud-pilot/formal-20260817-01/mg-validation/`；
- `runs/vrud-pilot/formal-20260817-01/mg-motion-off/`；
- `runs/vrud-pilot/formal-20260817-01/mg-frozen/`；
- `runs/vrud-pilot/formal-20260817-01/human-test/`；
- `runs/vrud-pilot/formal-20260817-01/demo/`；
- `runs/vrud-pilot/formal-20260817-01/report/`。

所有正式命令使用创建型写入；目标目录非空时直接拒绝。run 状态、命令、环境、Git
提交、GPU、输入指纹、checkpoint 指纹、阈值来源和退出原因都必须写入 provenance。

## 7. 预检与启动门槛

启动 Baseline 前必须全部满足：

1. Git 提交固定且 tracked worktree 无修改；
2. 人工 benchmark 严格加载并匹配 873/78,335/53,735/334；
3. Universal-P2 严格加载并匹配批准的 Universal SHA、427/859 和有限张量；
4. manifest 与 alignment cache 指纹匹配，train/validation/test 无重叠；
5. 真实 CUDA smoke 同时通过 Baseline、MG 和 Motion-Off 路径；
6. 两张 GPU 可见、无非本实验计算进程、显存满足训练要求；
7. 磁盘空间能够容纳两个 80-epoch run、预测分片、视频和安全余量；
8. 正式输出目录不存在，持久化进程与日志路径可写；
9. 完整 CPU 测试套件通过；已知的 multiprocessing 清理 warning 可以记录，但不能有
   测试失败。

预检只允许读取 test 人工基准的结构、数量和指纹，不允许运行模型预测或查看性能指标。

## 8. 正式训练与监控

Baseline 和 MG Full 均最多训练 80 epoch，以完整 validation mAP50 选择 `best.pt`，
连续 15 个 epoch 无严格改善则早停。每个完成 epoch 必须证明恰好覆盖 13,998 条 train
记录，两个 DDP rank 无重复、无遗漏。checkpoint 中的模型、优化器、scheduler、scaler、
RNG、epoch、history、manifest 和加载来源均须有限且相互一致。

训练由持久化用户级进程运行，日志和进度状态落盘。监控只观察资源、loss、validation、
数据覆盖和错误，不人工挑选 checkpoint。第一个完整 epoch 后记录实际吞吐和剩余时间，
再给出可信工期；启动前不以历史短样本速度替代实测。

以下情况立即停止当前 run：非有限 loss/gradient/weight、AMP 连续溢出且无 optimizer
step、DDP rank 缺失、epoch 覆盖错误、validation 不完整、输入指纹变化、checkpoint
写入不完整或 GPU 进程异常退出。停止后保留全部日志和最后合法产物，不覆盖现场。

## 9. Validation、消融与阈值冻结

Baseline 和 MG Full 分别在同一 validation 全集生成完整预测和 PR 曲线，并各自冻结一个
运行阈值。阈值文件必须记录 checkpoint SHA、manifest SHA、选择规则和生成时间。人工
test 只能读取这些冻结阈值，不能重新搜索阈值。

MG Motion-Off 不额外训练：加载 MG Full `best.pt`，在推理时明确禁止消费 motion
tensor，用于判断运动分支在当前 checkpoint 上的即时贡献。MG Frozen 与 MG Full 共享
Baseline 初始化和 motion seed，冻结 RGB detector，仅训练运动分支与融合 gate；它是
归因消融，不替代 MG Full 主结果。如果当前代码尚不能形成完整、可验证的 Frozen run，
应先补齐最小实现与测试，不能用手工改 checkpoint 冒充。

除各自冻结阈值的运行点外，还输出相同每帧 FP 预算下的 Recall 对比，防止把单纯降低
置信度阈值误判为运动模型增益。

## 10. 一次性人工评测

只有 Baseline、MG Full、Motion-Off 及计划内 Frozen 的 validation 阈值全部发布后，
才允许开启人工 benchmark。所有模型消费完全相同的 873 帧、53,735 个可评估 VRU
真值和 334 个边缘 ignore，使用 class-aware rotated NMS 和冻结的 rIoU 规则。

报告至少包含：mAP50、mAP50-95、Precision、Recall、F1、每帧 FP、
`Recall@rIoU 0.25/0.5`，以及按四类、三个场景、短边 `<16`、`16–24`、`24–40`、
`>40` px 和 static/slow/moving 三个速度层的结果。

时序连续性使用人工 GT identity，不要求预测 Track ID。每个 GT identity 统计轨迹帧
覆盖率、最长和平均连续漏检、首次检测延迟、TP/FN 切换次数和完全未检出轨迹数。逐个
GT-frame identity 生成 `rescued`、`regressed`、`stable TP` 和 `stable FN` 四态配对。

## 11. MG-VTOD 结论门槛

MG Full 只有同时满足以下条件，才能标记为“通过 MG 优化 gate”：

1. 短边不超过 24 px 的 Recall 至少提高 5 个百分点；
2. 全体四类 VRU Recall 至少提高 3 个百分点；
3. moving VRU Recall 至少提高 5 个百分点；
4. `rescued` 数量严格大于 `regressed`；
5. 每条轨迹最长连续漏检的中位数至少降低 20%；
6. mAP50 相比 Baseline 下降不超过 1 个百分点；
7. Precision 相比 Baseline 下降不超过 1 个百分点；
8. static VRU Recall 下降不超过 2 个百分点；
9. 类别映射、帧配对、OBB 几何、全集覆盖和 provenance 错误均为零。

若运动召回提高但精度、静止目标或连续性保护失败，应报告“运动召回提升但综合 gate
未通过”。即使正式训练未通过性能 gate，它仍是有效实验结果；禁止根据 test 调参后
覆盖原结果或改写结论。

## 12. 可视化、网页与交付物

三个场景分别生成 30 FPS 对比 MP4。画面包含原始帧、人工 GT、Baseline、MG-VTOD 和
运动强度图，并为小目标提供同位置放大 crop。代表案例至少覆盖 rescued、regressed、
stable FN 和新增 FP，并显示支持帧、GT `track_id`、类别、短边、速度、两个模型置信度
及 291 帧检测时间轴。

局域网页展示训练曲线、冻结阈值、全局与分层指标、gate 每一项的通过状态、配对案例、
模型和数据指纹、运行限制以及可复现命令。网页不得只展示有利案例，案例选择规则必须
在读取预测前冻结。

正式交付至少包括 Baseline、MG Full、Motion-Off、可选 Frozen 的 run 与 checkpoint，
validation 阈值，人工 test 预测和指标，配对 transition 表，三段 MP4、代表帧、时间轴、
motion heatmap、静态报告和局域网访问入口。

## 13. 测试与完成定义

实现阶段遵循测试先行。新增或修改的正式编排、冻结训练、阈值绑定、配对比较、可视化
和网页功能必须有聚焦测试；启动前运行完整测试套件和真实 CUDA smoke。

本阶段完成需同时满足：正式 Baseline 和 MG Full 正常早停或达到 80 epoch；所有完整
epoch 覆盖正确；validation 阈值在 test 前冻结；人工 873 帧仅评测一次；所有指标、
配对案例和 gate 可从冻结产物重建；局域网页可访问；GPU 进程退出后显存释放；结果无论
通过或未通过均按预设口径发布。

如果 MG 通过 gate，下一项目进入统一 OBB tracker、轨迹补全和轨迹级评测。如果未通过，
下一项目只针对本次冻结结果做误差归因，再决定修改 MG 或启动 LSTFE；不得在本项目中
临时扩展算法范围。
