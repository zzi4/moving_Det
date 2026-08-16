# Universal 初始化的 MG-VTOD 人工视频基准设计

## 1. 目标与结论边界

本阶段只回答一个核心问题：在现有航拍 VRU 检测能力之上，引入五帧运动证据后，
MG-VTOD 是否能够稳定补回小目标、减少连续漏检，同时不显著增加误检或损害静止目标。

主实验使用同一份 Universal 权重初始化单帧 Baseline，再用该 Baseline 的最佳内部
checkpoint 初始化 MG-VTOD。两者共享训练数据、类别、P2 OBB 检测器、增强、损失、
后处理和评测帧，主要差别是 MG-VTOD 具有多帧运动分支。最终只在人工校核的 873 个
中心帧上做配对评价。

Universal 的历史训练数据可能包含这些测试视频的抽帧。因此，本阶段能够支持的主结论是
“在相同 Universal 初始化和当前目标域上，MG-VTOD 带来的增量优化”，而不是“模型对
完全未见场景的无污染泛化提升”。原始 Universal 仅作为工程参考，不能代替公平的
Universal-P2 Baseline。

LSTFE、预测 Track ID 的 HOTA/IDF1 评测以及完整轨迹关联不属于本阶段。MG 检测通过性能
门槛后，再把同一追踪器应用于 Baseline 和 MG 输出，进入下一阶段。

## 2. 冻结人工基准

人工标注输入固定为：

`/home/stu1/Projects/moving_Det/label_data/videolabel_annotated_291frames_20260816.zip`

压缩包包含三段序列，每段 291 个连续中心帧：

| 场景 | 序列 | 帧范围 | 帧数 |
| --- | --- | ---: | ---: |
| site19 白天 | `DJI_20240919093341_0002_V` | 002926–003216 | 291 |
| site22 白天 | `DJI_20240719183036_0006_V` | 003331–003621 | 291 |
| site22 夜间 | `DJI_20240719224127_0006_V` | 001865–002155 | 291 |

审计确认 873 张 JPEG 与 NAS 中对应源帧逐字节一致，873 个 LabelMe JSON 与图片完全
配对，分辨率均为 3840×2160。标注包含 78,335 个 `shape_type=rotation` 的四点 OBB，
不存在空帧、零面积框、非凸框、重复 ZIP 成员或损坏 JSON。类别计数为 car 23,975、
bus 291、truck 291、motorcycle 30,779、pedestrian 18,472、bicycle 1,787、
tricycle 2,740。

人工校核相对原伪标签净增加 10,279 个框，其中 pedestrian 从 12,808 增至 18,472，
说明这批数据不再只是 Universal 输出的复制品。所有人工帧都属于现有冻结 manifest 的
test 三序列；这三个序列在当前 train 六序列和 validation 三序列中出现次数均为零。

基准生成器必须：

1. 将原 ZIP、每个 JSON、中心源图、配置和 manifest 写入 SHA-256 provenance；
2. 保持 ZIP 与 NAS 源目录只读，将派生文件写到新的 benchmark run；
3. 以 `(site, sequence, frame)` 作为帧主键；
4. 以 `(site, sequence, group_id)` 作为 GT 轨迹主键；
5. 直接采用人工 JSON 的类别，不再使用旧伪标签或 VRUD CSV 覆盖类别；
6. 只把 pedestrian、bicycle、tricycle、motorcycle 纳入主指标，车辆类保留在审计
   artifact 中但不作为四类模型的漏检；
7. 对每个中心帧验证 `t-4,t-2,t,t+2,t+4` 支持图可从 NAS 连续序列读取。支持帧可以
   位于人工标注区间之外，但不能作为额外 GT 中心帧。

人工标注中共有 293 条轨迹，同一帧内没有重复 group_id，同一轨迹没有类别漂移。两个
轨迹存在内部不可见间隔。轨迹覆盖率使用完整轨迹主键汇总；连续漏检、首次检测延迟和
闪烁只在连续可见 span 内计算，不把人工标注的不可见间隔算作模型漏检。

## 3. 边缘目标与 ignore 规则

共有 334 个 OBB 的至少一个点落在图像边界外，占全部框约 0.43%。这些实例标记为
`edge_ignore`，不进入召回率、AP 或轨迹连续性分母。评测时先把 ignore 多边形裁剪到
图像范围；若同类别预测框与裁剪后 ignore 区域的交集面积占预测框面积至少 50%，该预测
被抑制，不计 TP，也不计 FP。其余预测继续参与正常匹配。

正常 GT 与预测按类别做一对一 rotated IoU 匹配。主运行指标使用 `rIoU >= 0.25`，用于
减小 20×40 px 量级小框对少量像素偏移的敏感性；同时报告 `rIoU >= 0.5` 和标准 AP，
避免只得到宽松定位下的召回提升。

## 4. Universal 初始化与模型结构

Universal 输入 checkpoint 固定为：

`/home/stu1/Projects/moving_Det/models/best_vru_universal.pt`

其 SHA-256 为
`114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7`。
该模型是 8 类标准 YOLO11m-OBB，类别为 car、truck、bus、motorcycle、pedestrian、
bicycle、tricycle、engineering_vehicle，检测步长为 8、16、32。

主 Baseline 仍采用项目现有四类 YOLO11m-P2-OBB，检测步长为 4、8、16、32。初始化时
只迁移名称和形状都兼容的 Universal 张量；当前冻结代码和权重组合应迁移 427/859 个
目标 state tensors。新增 P2 neck、四类 OBB 头以及所有不兼容张量按固定 seed
`20260806` 初始化。

初始化步骤输出 `transfer_report.json`，至少记录源/目标架构、源权重 SHA-256、已加载
张量名称与形状、未加载原因、加载前后参数摘要和总计数。Baseline 与 MG 不能分别随机
执行一次迁移；先生成并冻结一个 Universal-P2 初始化 artifact，Baseline 从它训练，
MG 再从本次 Baseline best 完整加载检测器。若代码变化造成迁移集合或数量不再等于冻结
报告，必须停止而不是静默训练。

模型定义如下：

- `Universal-P2 Baseline`：单帧 RGB、P2–P5、四类 OBB，是主对照；
- `MG-VTOD Full`：从 Baseline best 初始化检测器，用
  `[-4,-2,0,2,4]` 五帧的配准软运动强度，经两级 stride-2 motion stem 编码并以保守
  gate 融入 backbone layer 2，检测器和时序分支联合优化；
- `MG-VTOD Frozen`：同样从 Baseline best 初始化，冻结整个 RGB 检测器，只训练
  motion stem 和 fusion gate，并使用相同 validation 早停规则；
- `MG-VTOD Motion-Off`：不额外训练，加载 Full checkpoint 后在推理时将 motion tensor
  置零，使融合残差为零，用于验证 Full 增益是否依赖运动输入；
- `Universal Raw Reference`：原始 8 类、P3–P5 checkpoint 的直接推理结果，仅作工程
  参考，并显式标注潜在训练图像重叠。

## 5. 训练与评测数据流

完整顺序固定为：

```text
冻结输入与指纹审计
  → Universal-P2 初始化 artifact
  → Baseline 双 GPU 正式训练
  → Baseline validation 选择 best 与阈值
  → 从 Baseline best 初始化 MG-VTOD Full
  → MG Full 双 GPU正式训练
  → MG Full validation 选择 best 与阈值
  → MG Frozen 消融训练与 validation
  → 冻结所有阈值和 test run identity
  → 人工 873 帧一次性配对评测
  → Motion-Off 与 Universal reference
  → 指标、视频、案例和网页报告
```

Baseline 和 MG Full 使用相同的六个 train 序列、三个 validation 序列、四类 schema、
1024×1024 tile、256 px 重叠、增强、OBB loss、effective batch size 16、AdamW、seed、
双 RTX A6000 DDP 和 AMP。每个正式 epoch 必须覆盖全部 13,998 条 train manifest 记录，
并由 rank 合并后的 identity digest 证明无重复和遗漏。模型依据完整 validation mAP50
保存 best，以 patience 15 早停或最多训练 80 epoch。

MG Full 从本次 Baseline best 初始化，而不是从 64 样本 overfit checkpoint 初始化。
MG Frozen 与 Full 共享 Baseline best、motion 分支随机 seed 和输入数据。Frozen 仅用于
解释运动分支的增量能力，不取代 Full 主结果。

每个模型在 validation 上选择自己的运行阈值：在每帧 FP 不超过 5 的约束下最大化
`F1@rIoU 0.25`。冻结 checkpoint、阈值、后处理和 evaluation run identity 后，才允许
读取 test 模型预测。test 不重新选阈值。除了各自运行阈值，还在每帧 1、2、5 个 FP
的共同预算下比较 Recall，避免把低阈值误当作 MG 优化。

全图评测保持 3840×2160，使用 1024×1024 tile 和 256 px overlap，跨 tile 采用
class-aware rotated NMS，IoU 为 0.5。Baseline 与所有 MG 变体必须消费完全相同的 873
个中心帧、人工 GT、tile 网格和后处理实现。

## 6. 指标与速度分层

阈值无关指标包括 mAP50、mAP50-95 和完整 PR 曲线。冻结运行阈值指标包括 Precision、
Recall、F1、每帧 FP 和 `Recall@rIoU 0.25/0.5`。

按 OBB 短边分为 `<16`、`16–24`、`24–40` 和 `>40` px。`16–24` 是主小目标层，
`<16` 作为探索性结果；没有 GT 的层不生成伪造的零 AP，而报告 `null` 和样本数零。

运动速度由 GT OBB 中心计算。中心帧局部速度优先使用同一连续可见 span 内
`||c(t+2)-c(t-2)||/4`；缺少一侧时使用可获得的最远相邻中心并除以真实帧差。分层固定为：

- static：不超过 0.25 px/frame；
- slow：大于 0.25 且不超过 1.0 px/frame；
- moving：大于 1.0 px/frame。

分别报告四类、三个场景、四个尺寸层和三个速度层。时序连续性不要求预测 ID，而是根据
每个 GT identity 在每帧是否得到同类匹配，计算轨迹帧覆盖率、最长和平均连续漏检、
首次检测延迟、TP/FN 切换次数、完全未检出的轨迹数。

逐 GT-frame identity 生成四态配对：

- rescued：Baseline FN、MG TP；
- regressed：Baseline TP、MG FN；
- stable TP：两者均 TP；
- stable FN：两者均 FN。

## 7. MG-VTOD 通过门槛

MG Full 只有同时满足精度保护和主要增益条件，才能标为“通过 MG 优化 gate”：

1. 短边不超过 24 px 的 Recall 至少提高 5 个百分点；
2. 全体四类 VRU Recall 至少提高 3 个百分点；
3. moving VRU Recall 至少提高 5 个百分点；
4. rescued 数量严格大于 regressed；
5. 每条轨迹最长连续漏检的中位数至少降低 20%；
6. mAP50 相比 Baseline 下降不超过 1 个百分点；
7. Precision 相比 Baseline 下降不超过 1 个百分点；
8. static VRU Recall 下降不超过 2 个百分点；
9. 类别映射、输入配对、OBB 几何和评测全集错误均为零。

若 Recall 提高但 FP 或静止目标退化超限，结果标为“运动召回提升但综合 gate 未通过”。
完整训练即使未通过性能 gate 仍是有效实验，禁止修改 test 阈值或筛选帧来制造通过结果。

## 8. 可视化和交付物

三个场景分别输出 30 FPS 对比 MP4 和网页。主画面组织为原始图、人工 GT、Baseline、
MG-VTOD 和运动强度图；4K 总览之外，代表目标使用同一空间 crop 放大，避免小 OBB 在
缩略图中不可见。

每个代表案例显示当前帧、支持帧、GT `track_id`、类别、短边、速度、Baseline/MG
置信度、运动热图和该目标在 291 帧内的检测时间轴。案例至少覆盖 rescued、regressed、
stable FN 和新增 FP，并优先覆盖不同类别、速度、尺寸和昼夜场景。时间轴中绿色表示 TP、
红色表示 FN、灰色表示 GT 不可见或 edge_ignore。

交付目录至少包含：

- `benchmark/manifest.json`、审计摘要和全部输入指纹；
- `universal-p2-init/transfer_report.json` 与冻结初始化 artifact；
- Baseline、MG Full、MG Frozen 的正式 run、best/last checkpoint 和 validation 阈值；
- 各模型 test `predictions.jsonl`、`metrics.json`、分层 CSV 和配对 transition 表；
- 三段对比 MP4、代表帧、轨迹时间轴和 motion heatmap；
- 可在局域网访问的报告页面，明确展示通过/未通过条件和数据重叠限制。

## 9. 错误处理与不可变性

以下情况必须拒绝启动或停止当前阶段：

- ZIP、源帧、JSON、Universal 权重、manifest、alignment cache 或 checkpoint 指纹不符；
- 873 个中心帧不完整、帧号不连续、支持帧缺失、图片与 JSON 错配；
- 非法类别、同帧重复 track_id、轨迹类别漂移、非有限或退化 OBB；
- Universal 兼容迁移集合偏离冻结的 427 张量且没有新的已批准设计；
- DDP epoch 出现 train sample 重复、遗漏或 rank coverage 不一致；
- NaN/Inf、checkpoint 配对不完整、validation 阈值不可复现；
- 任一模型消费了不同的 test 帧、GT、tile 或后处理配置；
- test 输出目录已存在而调用方试图覆盖。

run 状态与派生 artifact 采用同目录临时文件或 staging 目录原子发布。中断只允许从验证
通过的 last/best checkpoint 对恢复，不能跳过失败阶段进入 test。任何训练和评测代码都
不得写回人工 ZIP、NAS 图片或源 JSON。

## 10. 测试策略

实施遵循 TDD。测试至少覆盖：

1. 人工 ZIP 索引、873 对配对、三段帧范围和源图字节一致性；
2. 直接类别映射、复合 track key、连续可见 span 与两条 gap 轨迹；
3. edge_ignore 生成、图像裁剪和预测 IoP 抑制；
4. Universal 权重指纹、兼容张量集合、冻结初始化的确定性；
5. Baseline/MG 公平 manifest、tile、GT、NMS 和阈值来源；
6. 尺寸/速度分层、rIoU 匹配、rescued/regressed 和连续漏检的合成已知答案；
7. Motion-Off 的运动残差严格为零，RGB detector 路径保持不变；
8. 三段各少量帧的端到端 smoke，包括 JPEG、JSONL、CSV、MP4 和网页可读取性；
9. 正式运行前完整测试套件、环境和双 GPU 检查；正式运行后 873 帧 artifact schema、
   指纹、有限数值和媒体解码检查。

本阶段不以“训练进程退出 0”作为完成条件。只有输入审计通过、正式训练 provenance
完整、人工 873 帧评测完成、所有 gate 被逐项报告且可视化可追溯时，才形成 MG-VTOD
初版优化结论。
