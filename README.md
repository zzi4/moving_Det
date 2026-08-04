# Moving Det：航拍运动目标 OBB 论证工具

本项目实现一个无需训练的第一阶段论证：先利用稳定航拍视频的相邻帧运动证据发现交通参与者，再把持续运动区域连接成带旋转框（OBB）的短时轨迹候选。它用于比较单帧差分、MOG2、时间中值、多尺度运动证据和多尺度 tubelet，不包含最终分类器或学习式跟踪器。

## 环境安装

要求 Python 3.12。推荐在项目根目录创建独立环境：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

默认配置为 `configs/poc.yaml`。源数据位于 `/mnt/nas/Processing_data/mot_sequence`，程序只读取该目录；运行结果写入调用者指定的 `runs/` 子目录。不要在源数据目录生成缓存、修复标注或跳过非法标注。

## 常用流程

检查标定序列和评估序列的文件、尺寸及标注：

```bash
.venv/bin/moving-det inspect-data --config configs/poc.yaml
```

运行一个方法。`frame-start` 与 `frame-end` 必须同时给出：

```bash
.venv/bin/moving-det run \
  --config configs/poc.yaml \
  --sequence calibration \
  --method multiscale_tubelet \
  --scale 1.0 \
  --threshold 4 \
  --frame-start 16 \
  --frame-end 25 \
  --output runs/smoke
```

只在标定序列搜索固定候选阈值：

```bash
.venv/bin/moving-det calibrate \
  --config configs/poc.yaml \
  --output runs/poc-calibration
```

用已经冻结的 `calibration.json` 评估另一个序列。评估阶段不会重新选阈值：

```bash
.venv/bin/moving-det evaluate \
  --config configs/poc.yaml \
  --calibration runs/poc-calibration/calibration.json \
  --output runs/poc-evaluation
```

把评估指标整理成 Markdown：

```bash
.venv/bin/moving-det report \
  --metrics runs/poc-evaluation/metrics.json \
  --output docs/experiments/poc-results.md
```

为已有单方法 run 生成三张逐帧 PNG 和一张竖向三帧对比图：

```bash
.venv/bin/moving-det visualize \
  --run runs/smoke \
  --frames 20,21,22
```

输出写入 `runs/smoke/overlays/000020.png`、`000021.png`、`000022.png` 和 `comparison.png`。可视化不会覆盖已有 `overlays/`，也不会修改 run 内的源 artifact。

## Artifact 含义

每个单方法 run 目录包含：

- `config.yaml`：完整解析后的配置，以及序列、方法、处理尺度和阈值。
- `metrics.json`：总体、分层、边界帧、阈值候选和 gate 字段。
- `per_frame.csv`：每帧移动 GT、TP、FP、召回和 mask 覆盖。
- `per_track.csv`：首次检出、移动帧覆盖和 tubelet 碎片数。
- `proposals.jsonl`：每个候选的帧号、规范 OBB、运动分数和 tubelet ID。
- `frames/<frame>.npz`：最大 960×540 的 `uint8 preview_score` 和 `preview_mask`，只用于诊断显示。
- `run.json`：Git 提交、UTC 时间、依赖版本、输入路径、帧范围和随机种子。
- `overlays/`：GT、候选、忽略区、运动 inset 和竖向三帧对比图。

标定目录额外包含 `calibration.json`。其中保存全部五种方法、两个尺度、完整固定候选、选中值和可审计配置指纹。评估目录的组合 `metrics.json` 会列出每个子 gate 的实测值和布尔结果。

## 图例与 OBB 约定

可视化中 GT OBB 为青色，已匹配候选为橙色，未匹配候选为红色，忽略区域为黄色虚线。标签分别使用 `GT #<track_id>` 和 `P #<tubelet_id>`。右下角 inset 显示运动分数，并用青色边界标出二值 mask。

OBB 使用唯一的长边约定：

```text
width >= height
theta ∈ [-π/2, π/2)
```

角度表示车辆长轴，按 π 周期比较。运动连通域的朝向不是车辆最终航向；车头/车尾方向也不能仅由 OBB 推断。

## 当前 10 帧论证结果

以下结果来自标定序列第 16–25 帧、原始 3840×2160 分辨率、
`multiscale_tubelet`、scale 1.0、阈值 4。完整运行耗时
39 分 58.53 秒，峰值 RSS 2,009,648 KiB；产生 137,749 个候选。由于
10 帧窗口全部属于评估代码定义的边界区，这里引用 `metrics.json` 的
`all_*` 字段：

- 移动 GT 共 1,320 个，rIoU 0.25 匹配 14 个，召回率约 1.06%。
- rIoU 0.5 匹配 0 个，召回率为 0。
- 未匹配候选 137,735 个，每 100 个移动 GT 约有 10,434 个误报。
- GT 内 mask 平均覆盖率约 96.81%。

可视化显示，大量未匹配红框来自道路纹理、树木和建筑边缘。这说明当前
运动证据对目标像素的覆盖较高，但阈值 4 与现有连通/持久化规则缺乏选择性，
不能把这次结果当作可用检测器或正向精度结论。`runs/` 是本地忽略目录，
不随 Git 提交。

## 数据边界

当前两个序列主要是车辆，规模约 460 帧，目标尺寸和场景多样性都不足以代表最终的 4 小时小目标数据，因此这里只能回答“多帧运动证据是否值得继续”的工程问题，不能作为最终小目标 benchmark。

当前评估序列 `motorway_sequence2` 含有不满足严格矩形约束的四点 OBB。
严格读取在
`motorway_sequence2/000001.json: shape[2] label='car'` 处报告
`OBB points must form a non-degenerate rectangle` 并以退出码 2 停止。依照
既定数据策略，程序不会拟合、修复或跳过该标注。在源标注修正或数据策略
明确改变前，不能声称该序列完成了冻结评估。
