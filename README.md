# Moving Det：VRUD 多帧小目标 OBB 检测

本项目当前主线是在只读 VRUD 航拍序列上公平比较三个逐帧旋转框检测器：

- `baseline`：带 stride-4 P2 检测层的单帧 YOLO11m-OBB。
- `mg_vtod`：用 `t-4, t-2, t, t+2, t+4` 的配准软运动强度增强 P2。
- `lstfe`：对齐 `t-2, t+2`，并从 `t-30, t-15, t+15, t+30`
  选择长期上下文增强 P2/P3。

训练类别固定为：

```text
0 pedestrian
1 bicycle
2 tricycle
3 motorcycle
```

类别只通过 `(site, sequence_name, group_id)` 与 VRUD 元数据类号 3–6
恢复。原 Labelme JSON 中写成 `car` 的标签不决定训练类别。完整设计见
[`docs/superpowers/specs/2026-08-06-vrud-temporal-obb-detection-design.md`](docs/superpowers/specs/2026-08-06-vrud-temporal-obb-detection-design.md)。

## 两个独立环境

传统运动证据 PoC 仍使用 CPU `.venv`：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

VRUD 模型使用单独的 Python 3.11 / CUDA 环境，不向 `.venv` 引入
Torch、TorchVision 或 Ultralytics：

```bash
conda env create -f environment/temporal-obb.yml
conda run -n moving-det-vru python -c \
  "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__)"
```

固定版本为 PyTorch 2.5.1、TorchVision 0.20.1、Ultralytics 8.4.115。
`moving-det-vru --help` 和 CPU 面板渲染采用延迟导入，不会加载上述模型依赖。

## VRUD 完整工作流

源 JPG/JSON 和 CSV 永远只读；manifest、缓存、checkpoint、预测、指标和面板均写到
`runs/`。

先冻结 6/3/3 序列 manifest：

```bash
conda run -n moving-det-vru moving-det-vru build-manifest \
  --config configs/vrud-temporal-obb.yaml \
  --output runs/vrud-pilot/manifest
```

为 MG 和 LSTFE 预计算支持帧到中心帧的全局 ECC 变换。默认位置是配置
`output_root/alignment-cache`：

```bash
conda run -n moving-det-vru moving-det-vru cache-alignments \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest
```

训练单帧基线：

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline \
  --weights yolo11m-obb.pt
```

MG 和 LSTFE 从同一个内部基线 checkpoint 初始化。`--baseline-init` 是明确写法；
为兼容冻结的 Task 13 命令，时序模型的 `--weights <internal-best.pt>` 具有相同含义：

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model mg_vtod \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/mg_vtod \
  --baseline-init runs/vrud-pilot/baseline/checkpoints/best.pt

conda run -n moving-det-vru moving-det-vru train \
  --model lstfe \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/lstfe \
  --baseline-init runs/vrud-pilot/baseline/checkpoints/best.pt
```

64 样本 gate 必须同时给出固定样本数和最大 step。CLI 从原 manifest
确定性地派生恰好 64 行的新 manifest，不修改源 artifact：

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline-overfit \
  --weights yolo11m-obb.pt \
  --overfit-samples 64 \
  --max-steps 300
```

训练输出的主 checkpoint 是 `<output>/checkpoints/best.pt`。恢复同一 run
使用 `--resume <output>/checkpoints/last.pt`。非默认缓存可通过
`--alignment-cache /safe/path/alignment-cache` 显式指定。

评测必须先在 validation 选择并冻结阈值，再应用到 test：

```bash
conda run -n moving-det-vru moving-det-vru evaluate \
  --model baseline \
  --checkpoint runs/vrud-pilot/baseline/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest \
  --split validation \
  --output runs/vrud-pilot/baseline-validation

conda run -n moving-det-vru moving-det-vru evaluate \
  --model baseline \
  --checkpoint runs/vrud-pilot/baseline/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest \
  --split test \
  --threshold runs/vrud-pilot/baseline-validation/threshold.json \
  --output runs/vrud-pilot/baseline-eval
```

三个 test run 的 manifest、split、类别 schema 和逐帧评测全集必须完全兼容，
否则拒绝比较：

```bash
conda run -n moving-det-vru moving-det-vru compare \
  --runs runs/vrud-pilot/baseline-eval \
         runs/vrud-pilot/mg_vtod-eval \
         runs/vrud-pilot/lstfe-eval \
  --output runs/vrud-pilot/comparison
```

数据 smoke 与独立 GT 审计：

```bash
conda run -n moving-det-vru moving-det-vru visualize \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/data-smoke

conda run -n moving-det-vru moving-det-vru visualize \
  --manifest runs/vrud-pilot/manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --output runs/vrud-pilot/temporal-data-smoke

conda run -n moving-det-vru moving-det-vru audit-sample \
  --manifest runs/vrud-pilot/manifest \
  --count 20 \
  --output runs/vrud-pilot/manual-audit
```

`audit-sample` 只读取冻结 manifest、原始 GT 和类别元数据，固定种子
`20260806`，在可用时覆盖四类和两个站点；它不打开 prediction 或 checkpoint。
不带 `--alignment-cache` 的 `visualize` 是
`pre-cache-current-frame-geometry-smoke`：它从 train manifest 确定性选择一个
site19 与一个 site22 序列，覆盖四个修正类别、至少一个背景 tile 和一个边缘锚定
tile，只验证当前帧的 dataset 几何、局部/全图 OBB、类别修正和一次固定增强。面板中
的时序支持条是 `manual-display-only`，不能作为支持帧有效掩码或缓存仿射已被模型
dataset 消费的证据。

显式传入 `--alignment-cache` 后，模式变为
`post-cache-temporal-dataset-smoke`。命令先核对 cache summary、manifest 和
immutable snapshot 指纹，再把同一个 snapshot 注入真实 MG-VTOD 与 LSTFE
`TemporalClipDataset`。`index.json` 分模型记录配置 offsets、有效掩码、dataset
实际返回的局部仿射矩阵、支持路径、中心身份、帧 tensor 形状和 cache SHA-256。

## VRUD artifact 含义

- `manifest/{train,validation,test}.jsonl`：冻结的同位置 tile/中心帧全集。
- `manifest/manifest.json`：子文件 SHA-256 和固定种子。
- `alignment-cache/index.json` 与 `*.npz`：按
  `(site, sequence, center_frame, support_frame)` 索引的严格仿射变换。ECC
  接收原分辨率帧并在内部下采样，缓存矩阵仍使用原图像素坐标；训练与评测冻结同一
  内容指纹。
- `<train-output>/run.json`：训练环境、manifest/cache 指纹和
  `load_provenance`；公开权重的实际绝对路径与内容 SHA-256 和 checkpoint
  使用同一份 provenance。
- `<train-output>/checkpoints/{best,last}.pt`：包含 manifest 指纹、模型名、
  优化器、恢复状态和 `load_provenance` 的内部实验 checkpoint；从公开 YOLO
  权重初始化时，后者记录实际已加载本地文件的绝对路径与 SHA-256。内部
  init/resume 与公开权重字段严格分离；checkpoint 不能当作公开 YOLO 权重读取。
- `<evaluation>/run.json`：模型、split、manifest/checkpoint SHA-256、类别 schema、
  冻结逐帧全集、阈值和缓存来源。
- `<evaluation>/predictions.jsonl` 与 `ground-truth.jsonl`：带站点、序列、帧号的
  逐目标 OBB；test prediction 与指标使用同一个冻结阈值，不导出低于阈值的候选。
- `<evaluation>/metrics.json`：mAP、rIoU 召回、每帧误检、分层和连续性指标。
- `<evaluation>/per_{class,size,speed,track}.csv`：可审计的展开表。
- `<validation>/threshold.json`：validation 上在每帧误检不超过 5 时最大化
  `F1@rIoU 0.25` 的全局阈值；test 只加载，不重新选择。
- `<comparison>/metrics.json`：基线、MG、LSTFE 指标和两个五条件 gate。
- `<comparison>/overlays/*.jpg`：同一帧三模型 OBB、MG 软运动图和 LSTFE
  长期帧选择及归一化的 P2 学习可变形偏移幅值诊断面板（偏移值位于 P2 特征图
  采样坐标，不是原图像素位移）。支持帧、GT、预测与热图统一裁到同一个代表性
  `diagnostic_tile_xywh`，避免把 tile 诊断图拉伸为整张 4K。

所有 JSON 禁止 NaN/Infinity；文件写入采用同目录临时文件或完整 staging 目录后
原子替换。输出路径不得覆盖 manifest、run 或 NAS 源数据，也不得通过符号链接逃逸。

## OBB、NMS 与阈值约定

内部旋转框唯一约定为：

```text
width >= height
theta ∈ [-π/2, π/2)
```

tile 映射、监督、匹配、NMS 和面板均使用旋转多边形，不能退化为水平框。全图 tile
为 1024×1024、重叠 256 像素，跨 tile 合并采用 rotated NMS，IoU 固定为 0.5。
检测 AP 使用完整置信度排序；test 视频指标只能使用对应模型冻结的 validation 阈值。
训练期 validation 与 64 样本 gate 也通过 Task-11 解码和 rotated NMS，推理置信度
从 0.0 开始，避免在计算 mAP/召回前提前丢弃低分候选。

## 2026-08-03 传统运动 PoC：保留的负面结果历史

旧 `moving-det` 命令仍保留，用于复现无训练的差分、MOG2、时间中值、多尺度运动证据
和 tubelet 对照。它不是当前主检测器，也不再作为“二值运动前景生成候选后拟合 OBB”
的主路线。

当时在标定序列第 16–25 帧、3840×2160、`multiscale_tubelet`、scale 1.0、
阈值 4 上得到：

- 移动 GT 1,320 个，`rIoU 0.25` 匹配 14 个，召回约 1.06%。
- `rIoU 0.5` 匹配 0 个。
- 未匹配候选 137,735 个，每 100 个移动 GT 约 10,434 个误报。
- GT 内 mask 平均覆盖约 96.81%。

这证明运动响应能覆盖目标像素，但不能提供可用检测选择性。旧 run 来自开发期 dirty
工作区，不能宣称由记录的 `e062040` 干净 checkout 完整复现。相关 CPU 命令和测试
作为日期明确的负面对照继续保留；“当前数据只有车辆、约 460 帧”的限制只描述该旧
PoC，不描述当前覆盖两个站点、四类 VRU、固定 6/3/3 序列的 benchmark。
