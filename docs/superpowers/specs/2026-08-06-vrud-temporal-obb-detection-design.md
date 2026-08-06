# VRUD 多帧小目标 OBB 检测论证设计

日期：2026-08-06  
状态：设计已口头确认，等待书面规格审阅  
替代方向：本设计替代“先用二值运动前景生成候选，再拟合 OBB”的主路线。原运动证据
PoC 保留为失败对照和运动图基础组件，不再作为最终检测器。

## 1. 目标

本项目面向悬停或已经稳定的 3840×2160、30 FPS 航拍视频，识别尺寸约为
20×40 像素的道路弱势交通参与者（VRU），并为后续多目标跟踪输出逐帧旋转目标框。

本轮实现和比较：

1. 单帧 YOLO11m-OBB 小目标基线。
2. MG-VTOD-OBB：利用连续帧运动强度增强当前帧的 OBB 检测。
3. LSTFE-OBB：利用短期特征对齐和长期上下文聚合增强当前帧的 OBB 检测。

检测类别固定为：

```text
0 pedestrian
1 bicycle
2 tricycle
3 motorcycle
```

本轮的主要研究问题是：在基础检测器、训练数据、损失和推理条件一致时，多帧运动引导
或长短期特征融合能否显著提高小尺度移动 VRU 的 OBB 召回率和跨帧检测连续性。

## 2. 范围与非目标

### 2.1 本轮范围

- 建立只读 VRUD 数据索引并恢复正确类别。
- 冻结序列级训练、验证和测试清单。
- 建立 4K 同位置多帧切片数据集。
- 实现单帧、MG-VTOD-OBB、LSTFE-OBB 三个公平对照模型。
- 完成小规模训练、旋转框评测、轨迹覆盖评测和对比可视化。
- 为后续跟踪器输出统一的逐帧 OBB 预测格式。

### 2.2 本轮非目标

- 不在本轮实现完整的 ID 关联、轨迹补全或离线全局跟踪。
- 不把 MG-VTOD-OBB 或 LSTFE-OBB 宣称为原论文逐行复现；它们是共享现代 OBB
  骨干后的 VRUD 适配版本。
- 不使用全部 4 小时数据进行首次论证。
- 不以实时推理为目标。
- 不修改 NAS 上的 JPG、JSON 或 CSV。
- 不用运动掩码硬过滤目标；运动只能作为可学习的软特征。

当两种时序检测器稳定后，另行设计一个共享 OBB 跟踪器。三个检测器必须接入同一个
跟踪器，避免跟踪差异掩盖检测差异。

## 3. 文献依据与适配原则

MG-VTOD 使用配准后的连续帧立方体计算单通道运动强度，并将运动响应和外观卷积响应
拼接后输入 YOLOv5。其关键价值是用运动注意力增强外观弱小目标，而不是把运动前景
直接当成检测框。本项目保留“配准帧立方体、单通道运动强度、双路卷积响应融合”三个
核心机制，将水平框 YOLOv5 改为带 P2 小目标层的 YOLO11m-OBB。

LSTFE-Net 使用短期帧的可变形卷积对齐、最低相似度长期帧选择和两阶段长短期特征聚合。
论文消融显示一个级联对齐块优于零个或多个对齐块。本项目保留这三个机制，但将
Faster R-CNN 的 RoI/Proposal 级聚合改为适合单阶段 OBB 检测器的 P2/P3 稠密特征
聚合。

参考文献：

- Yang et al., “Video Tiny-Object Detection Guided by the Spatial-Temporal Motion
  Information,” CVPR Workshops 2023.
- Xiao et al., “LSTFE-Net: Long Short-Term Feature Enhancement Network for Video
  Small Object Detection,” CVPR 2023.
- LSTFE-Net 官方实现：<https://github.com/WHU-xjs/LSTFE-Net>
- VRUD：<https://arxiv.org/abs/2604.01134>

## 4. 数据来源与已验证事实

图像和原始 OBB JSON：

```text
/mnt/nas/Processing_data/site19_22_sequence/
├── site19_sequence/
└── site22_sequence/
```

轨迹和类别元数据：

```text
/mnt/nas/Processing_data/VRUD/
├── site19/output/ADS_KHR_19/
└── site22/output/ADS_WZY_22/
```

本地数据包含 64 个序列、400,740 组完整 JPG/JSON 对，全部抽查图像均为
3840×2160。元数据类号为：

```text
0 car
1 truck
2 bus
3 pedestrian
4 bicycle
5 tricycle
6 motorcycle
```

64 个 `STD_TRK_META.csv` 中共有：

| 类别 | 全部轨迹 | `meanVelocity >= 0.1 m/s` |
|---|---:|---:|
| pedestrian | 6,079 | 4,276 |
| bicycle | 269 | 69 |
| tricycle | 598 | 373 |
| motorcycle | 5,260 | 2,966 |

704 个分层抽样帧包含 50,894 个 rotation shape，其中：

- 46,709 个可以通过序列名和 `group_id` 关联元数据，抽样关联率为 91.78%。
- 4,185 个无法关联，不能可靠恢复类别。
- 225 个 OBB 存在越界顶点；全部属于画面边缘的部分可见目标。
- 未发现非法 `group_id` 或同帧重复 Track ID。
- 已关联的 pedestrian、bicycle、tricycle、motorcycle 在原 JSON 中均写成
  `car`，所以原 JSON 类别不能直接用于 VRU 训练。

抽样 OBB 尺寸表明 pedestrian、bicycle 和 motorcycle 的典型短边约为
16–29 像素，典型长边约为 26–62 像素，与项目目标一致。

## 5. 类别恢复与样本资格

### 5.1 索引键

类别元数据使用以下唯一键：

```text
(site, sequence_name, group_id)
```

`group_id` 只能在所属序列内解释，不能跨序列建立全局 ID。CSV 的 frame 从 0 开始，
图像文件名从 `000001` 开始；需要关联帧范围时，使用 `image_frame = csv_frame + 1`。

### 5.2 类别转换

训练类号转换为：

```text
VRUD class 3 -> training class 0 pedestrian
VRUD class 4 -> training class 1 bicycle
VRUD class 5 -> training class 2 tricycle
VRUD class 6 -> training class 3 motorcycle
```

每个保留标注同时记录 `raw_json_label` 和 `corrected_class_name`，使类别修正可审计。
原始 JSON 中的 `car` 不参与训练类号判断。

### 5.3 移动 VRU 定义

当 `STD_TRK_META.csv` 中：

```text
class in {3, 4, 5, 6}
and meanVelocity >= 0.1 m/s
```

整条轨迹被定义为移动 VRU。该轨迹的所有有效标注帧均被保留，包括等候、短暂停止和
再次起步的帧。这样模型在 VRU 暂停时丢失目标仍会受到惩罚。

### 5.4 排除和忽略

- 找不到元数据键：从训练和评测中排除，写入 `unmatched_metadata` 审计表。
- 元数据类号不是 3–6：不进入本轮 VRU 数据。
- OBB 任一顶点越出图像：保留在审计表，标记为 `edge_clipped` ignore；不裁点、
  不拟合新的矩形，也不作为正负样本。
- 四点不有限、不构成非退化矩形或 `group_id` 非整数：标记为数据错误并终止该
  manifest 的生成。
- 原始数据永远只读；所有决定写入项目 `runs/` 下的新 manifest。

## 6. 序列划分

### 6.1 首轮固定 12 序列

首轮论证使用下列 12 个序列。选择覆盖两个站点，并保证三个集合都包含四类移动 VRU。

训练集（6）：

```text
site19/DJI_20240919154443_0005_V
site19/DJI_20240919162906_0003_V
site22/DJI_20240719181132_0001_V
site22/DJI_20240719091331_0001_V
site22/DJI_20240719181521_0002_V
site22/DJI_20240719085001_0003_V
```

验证集（3）：

```text
site19/DJI_20240919150818_0004_V
site22/DJI_20240719171610_0003_V
site22/DJI_20240719085350_0004_V
```

测试集（3）：

```text
site19/DJI_20240919093341_0002_V
site22/DJI_20240719224127_0006_V
site22/DJI_20240719183036_0006_V
```

对应的移动 VRU 轨迹数为：

| 集合 | pedestrian | bicycle | tricycle | motorcycle |
|---|---:|---:|---:|---:|
| train | 735 | 37 | 40 | 331 |
| validation | 207 | 12 | 12 | 153 |
| test | 349 | 9 | 10 | 192 |

manifest 写出后计算并验证序列集合、轨迹键集合和图像路径集合三者的交集均为空。
测试 manifest 在任何模型训练前冻结。

### 6.2 抽样

训练中心帧以原视频每 5 帧一个候选，即约 6 FPS。正样本按轨迹均匀抽取，单条轨迹
最多从 stride-5 候选中等时间间隔选择 32 个中心帧，不能因为持续时间长而无限主导
训练。每类最多生成 5,000 个首轮正样本 clip；不足的类别通过重复采样和独立增强
补齐，不复制源文件。背景负样本数量为正样本总数的 25%。

验证和测试检测指标在每 15 帧一个中心帧上运行 4K 全图切片推理。视频连续性指标另外
在每个测试序列中选择移动 VRU 有效标注总数最多的三个互不重叠连续 300 帧窗口，以
原始 30 FPS 推理；总数相同时选择起始帧更早的窗口。窗口选择只依赖 GT，不依赖任何
模型预测，并写入冻结 manifest。

## 7. 多帧同位置切片

训练和推理的空间单元是 1024×1024 tile。全图推理时相邻 tile 重叠 256 像素。

- 中心帧和所有支持帧使用完全相同的全图坐标裁剪。
- 训练正 tile 必须完整包含目标 OBB；不能因运动响应弱而删除。
- 落入重叠区的 GT 只分配给目标中心距离 tile 中心最近的 tile，避免重复监督。
- 推理结果映射回 4K 坐标后使用 rotated NMS 合并；不能使用水平框 NMS。
- 所有空间增强对 clip 内全部帧共享同一参数。
- 亮度、对比度和轻微噪声增强允许逐帧独立，以模拟航拍曝光变化。
- 不使用会把 20×40 像素目标进一步缩小到不可见的强 Mosaic 作为默认增强。

序列边界处不复制首尾帧。缺失某个支持帧时返回支持帧有效掩码，由模型忽略无效位置。

## 8. 统一 OBB 基线

三个模型共享 YOLO11m-OBB 主干、P2-P5 金字塔和四类 OBB Head。P2 的 stride 为 4，
用于保留 20 像素级目标的空间细节。

内部 OBB 继续使用项目约定：

```text
width >= height
theta in [-pi/2, pi/2)
```

数据适配器负责与检测框架的 `xywhr` 约定互相转换。训练监督、rotated NMS 和评测都
使用旋转框，不退化为水平框。

单帧基线只输入中心帧。它既提供时序模型初始化权重，也提供唯一的主对照。三个模型的
OBB Head、类别、损失、训练样本、训练轮数和推理参数必须一致。

## 9. MG-VTOD-OBB

### 9.1 输入窗口

```text
frames = {t-4, t-2, t, t+2, t+4}
```

在 30 FPS 下窗口覆盖约 0.27 秒。原论文只使用已经过去的相邻帧；本项目允许离线处理，
因此使用对称窗口，减少目标只在差分一侧出现的偏差。

### 9.2 残余配准

视频已经稳定，但仍在 960×540 灰度预览上估计每个支持帧到中心帧的全局欧氏变换。
变换只含平移和旋转，再按分辨率比例应用到 tile。

ECC 不收敛、相关系数低于 0.8、平移超过 20 个全分辨率像素或旋转超过 2 度时，记录
失败并使用单位变换。配准参数按 `(site, sequence, center_frame, support_frame)`
缓存为小型 JSON/NPZ，不缓存全量 4K 运动图。

### 9.3 运动强度

将配准后的帧转换为灰度，计算：

```text
D_k = abs(Y_t - warp(Y_(t+k))), k in {-4, -2, 2, 4}
D   = max(D_-4, D_-2, D_+2, D_+4)
```

`D` 使用 3×3 高斯低通抑制单像素噪声，然后用每个 tile 的中位数和 MAD 做鲁棒归一化，
截断到 `[0, 1]` 得到单通道运动强度 `M_t`。运动图没有二值阈值，也不生成连通域。

### 9.4 融合

```text
RGB tile -> RGB stem ------> F_rgb --┐
motion M_t -> motion stem -> F_m ----+-> gated residual fusion -> backbone/neck
```

Motion stem 使用两个小卷积层，将单通道运动图编码到与 P2 相同的通道数。融合为：

```text
F = F_rgb + sigmoid(G([F_rgb, F_m])) * F_m
```

门控偏置初始化为负值，使初始模型接近已训练的单帧基线，再逐步学习使用运动。运动为
软特征；即使 `M_t` 接近零，RGB 路径仍完整存在。

## 10. LSTFE-OBB

### 10.1 输入窗口

```text
current: t
short-term: {t-2, t+2}
long-term candidates: {t-30, t-15, t+15, t+30}
```

支持帧有效掩码处理序列边界。长期候选全部缺失时跳过长期增强，但短期和中心帧路径
仍正常工作。

### 10.2 共享特征提取

全部帧通过同一组 backbone 权重。时序增强只作用于 P2/P3，中心帧的 P4/P5 保持原始
路径，控制显存并避免高层语义被不相关长期帧破坏。

### 10.3 短期对齐与像素级聚合

每个短期 P2/P3 特征和中心特征拼接后，用一个 3×3 卷积预测偏移，再通过一个
deformable convolution block 对齐。只使用一个 block，与原论文最佳消融设置一致。

对齐后构造：

```text
[F_t - F_s, F_s - F_t, F_t, F_s]
```

两个卷积层和 softmax 生成短期帧自适应权重，将两个对齐短期特征以残差方式聚合到
中心特征。

### 10.4 长期帧选择

四个长期候选的 P3 特征经过自适应池化、降维和 max pooling 得到全局 embedding。
计算每个有效候选和中心帧 embedding 的余弦相似度，选择相似度最低的一帧，以补充
差异最大的真实背景上下文。选择过程不使用类别标签或模型预测。

### 10.5 长短期聚合

原论文使用 Proposal-Level RoI 特征；本项目在单阶段检测器中使用 P2/P3 稠密特征
等价实现：

1. 将通道分为四组。
2. 先用分组注意力把选中长期上下文聚合到短期特征。
3. 再把增强后的短期特征聚合到中心特征。
4. 相对特征网格坐标作为位置编码。
5. 聚合结果通过残差连接进入中心 P2/P3。

这一适配保留“长期选择、长期到短期、短期到中心”的两阶段结构，但不宣称复现原
Faster R-CNN RoI 实现。

## 11. 训练策略

建立独立 Python 3.11 深度学习环境，不修改当前传统 PoC 的 `.venv`。PyTorch、
TorchVision、Ultralytics、CUDA 相关依赖和版本写入可重建的环境文件。训练机器为两张
48 GB RTX A6000。

固定设置：

```text
random seed: 20260806
input tile: 1024x1024
optimizer: AdamW
initial learning rate: 2e-4
weight decay: 1e-2
schedule: 3 epoch warmup + cosine decay
pilot epochs: 80
early stopping patience: 15 epochs
effective batch size: 16 clips
AMP: enabled
```

若单卡显存不能容纳 LSTFE clip，通过梯度累积保持有效 batch size 16，不降低输入尺寸。
早停和最佳 checkpoint 选择都使用验证集 OBB `mAP50`，不得为三个模型使用不同的
选择指标。

训练顺序：

1. 从公开 YOLO11m-OBB 权重初始化单帧基线。
2. 在 VRUD 首轮训练集训练单帧基线。
3. MG-VTOD-OBB 和 LSTFE-OBB 都从同一个单帧 checkpoint 初始化。
4. 新增时序层单独初始化，原有 backbone、neck 和 head 权重完全相同。

自行车和三轮车通过轨迹级均衡采样提高出现频率。损失函数本身保持四类一致，不为不同
模型设置不同类别权重。

## 12. 分阶段运行

### 12.1 数据 smoke

读取一个 site19 和一个 site22 序列，生成至少各类一个正样本以及背景负样本，验证
类别修正、支持帧、切片、OBB 转换和增强后的可视化。

### 12.2 64 样本过拟合

固定 64 个正 tile，三个模型分别最多运行 300 个优化 step。要求：

- 损失相对初始值下降至少 50%。
- 在这 64 个样本上 `Recall@rIoU 0.25 >= 80%`。
- 前向和反向不存在 NaN/Inf。

未通过时不得启动 80 epoch 训练。

### 12.3 首轮论证

按第 6 节固定序列运行单帧、MG-VTOD-OBB 和 LSTFE-OBB。三个模型完成后统一生成
指标、预测文件和相同帧对比图。

## 13. 评测

### 13.1 检测指标

- OBB `mAP50`
- OBB `mAP50:95`
- `Recall@rotated-IoU 0.25`
- `Recall@rotated-IoU 0.50`
- 每类 AP、召回率和每帧误检数

AP 使用完整置信度排序。需要固定阈值的视频指标在验证集上选择一个全局置信度阈值：
在每帧误检不超过 5 个的约束下最大化 `F1@rIoU 0.25`，然后为该模型冻结并应用到
测试集。rotated NMS IoU 固定为 0.5。

### 13.2 分层指标

OBB 短边：

```text
< 16 px
16–24 px
24–32 px
>= 32 px
```

速度：

```text
low: meanVelocity < 1 m/s
medium: 1 <= meanVelocity < 4 m/s
high: meanVelocity >= 4 m/s
```

同时报告四个类别和两个站点的结果。bicycle 和 tricycle 样本较少，因此首轮不对单类
设置硬性通过门槛，但必须报告样本量和以 GT 轨迹为单位、固定种子 20260806、
1,000 次重采样得到的 95% bootstrap 置信区间。

### 13.3 视频连续性指标

在冻结的连续 300 帧窗口上报告：

- 每条 GT 轨迹的有效帧检测覆盖率。
- 最长连续漏检帧数。
- 短暂停止帧的召回率。
- 相邻帧 OBB 中心、长短边和 π 周期角度抖动。

短暂停止片段定义为 `STD_TRK.csv` 中速度绝对值低于 0.1 m/s 且连续至少 15 帧的
片段。该定义只用于分层评测，不会把所属移动轨迹从训练集中删除。

这些指标只检查检测输出的时间稳定性，不分配预测 Track ID，也不冒充 MOT 指标。

### 13.4 通过标准

至少一种时序模型必须同时满足：

1. 短边不超过 24 像素的 VRU，其 `Recall@rIoU 0.25` 比单帧基线提高至少
   5 个百分点。
2. 全体 VRU 的 `Recall@rIoU 0.25` 提高至少 3 个百分点。
3. `mAP50` 相比单帧基线下降不超过 1 个百分点。
4. 短暂停止阶段召回率不显著低于单帧基线。
5. 所有训练正样本都能成功关联 VRUD 元数据，类别映射错误数为 0。

不通过仍是有效实验结果，但不能直接扩大到全部 4 小时训练。应先根据分尺寸、分速度、
类别和站点指标定位问题。

## 14. VRUD 真值限制

VRUD 发布轨迹由 YOLO11x-OBB、ByteTrack 和后处理辅助生成，不是完全独立的逐帧人工
检测真值。因此首轮可以把它作为统一参考，但不能仅凭该测试宣称新模型超过标签生成器
或达到真实世界最终精度。

在正式性能结论前，从冻结测试集选取约 20 个短 clip，覆盖四类、两个站点、白天和
夜间，人工检查漏标、错类、OBB 偏移和 Track ID。人工检查结果作为独立审计层，不在
查看模型预测后修改测试 GT。

## 15. 错误处理

1. JPG/JSON 不配对：manifest 生成失败并报告序列与文件名。
2. 元数据 CSV 缺失、ID 重复或类号非法：manifest 生成失败。
3. 未匹配元数据：排除并计数，不猜测类别。
4. 边缘 OBB：标记 ignore，不裁点或重新拟合。
5. 支持帧不足：使用有效掩码，不复制帧。
6. ECC 失败：单位变换回退并记录原因和比例。
7. 空目标 tile：只作为受控负样本。
8. 模型出现 NaN/Inf：当前 run 失败，保存最后合法 step 和诊断输入。
9. checkpoint 与 manifest 指纹不匹配：拒绝评测。
10. 测试 manifest 已冻结后发生变化：指纹变化并拒绝与历史结果直接比较。

## 16. 测试策略

### 16.1 数据单元测试

- 原 JSON `car` 经真实或合成元数据映射为四类 VRU。
- `(site, sequence, group_id)` 键隔离不同序列中的相同 ID。
- CSV 0 基帧范围正确转换为 1 基图像文件名。
- 未匹配元数据、非 VRU 类别和 edge-clipped OBB 按规则处理。
- 序列、轨迹和图像路径在 train/validation/test 之间没有交集。
- 同一 clip 的所有帧使用相同 tile 坐标和空间增强。

### 16.2 几何和运动单元测试

- OBB 内部约定与检测框架 `xywhr` 往返转换。
- tile 到 4K 坐标回映和 rotated NMS。
- 合成全局平移背景的 ECC 变换恢复。
- 合成移动小矩形产生局部高运动强度，静态背景保持低响应。
- 序列边界支持帧有效掩码正确。

### 16.3 模型单元测试

- 三个模型输出相同 OBB Head 结构。
- MG 双 stem 融合形状、门控范围和梯度有效。
- LSTFE 短期对齐、长期选择、边界掩码和分组聚合形状正确。
- 每个新增模块在合成 batch 上可反向传播且梯度有限。

### 16.4 集成测试

- 一个小 clip 完成 `index -> dataset -> model -> loss -> backward`。
- 一个 4K 帧完成切片推理、全图回映、rotated NMS、指标和可视化。
- 64 样本过拟合 gate。
- 固定小型 checkpoint 和 manifest 能重复产生相同结构的预测 artifact。

## 17. 输出与复现

每次训练和评测写入：

```text
runs/<experiment-name>/
├── config.yaml
├── environment.json
├── manifest/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── exclusions.csv
│   └── class-audit.json
├── checkpoints/
├── metrics.json
├── per_class.csv
├── per_size.csv
├── per_speed.csv
├── per_track.csv
├── predictions.jsonl
├── overlays/
└── run.json
```

`run.json` 记录 Git commit、dirty 状态、输入根目录、manifest SHA-256、随机种子、依赖
版本、GPU、训练时长和峰值显存。

可视化至少包含：

- 原始 `t-1/t/t+1` 或完整模型支持窗口。
- MG 运动强度图。
- LSTFE 选中的长期帧和短期对齐响应。
- GT OBB、预测 OBB、类别、置信度和匹配状态。
- 单帧、MG-VTOD-OBB、LSTFE-OBB 同帧并排比较。

## 18. 实施顺序

1. VRUD 类别索引、审计和固定 manifest。
2. 多帧同位置切片与 OBB 坐标适配。
3. 单帧 P2 YOLO11m-OBB 基线。
4. MG 运动强度和 MG-VTOD-OBB。
5. LSTFE 短期对齐、长期选择和 LSTFE-OBB。
6. 统一训练、评测和可视化。
7. 数据 smoke 与 64 样本过拟合。
8. 12 序列首轮论证。
9. 报告更新和后续跟踪设计决策。

每一步必须先有失败测试，再写最小实现并使测试通过。不得在类别索引和 manifest 审计
完成前启动模型训练，也不得在 64 样本过拟合失败时启动完整首轮训练。
