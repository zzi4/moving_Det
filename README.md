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

### 人工视频 benchmark 与 Universal-P2 初始参数

正式实验固定使用同一份人工校核 benchmark 和同一份 Universal→P2 初始参数。
先从原始标注 ZIP 与 NAS 图像冻结 873 帧 benchmark：

```bash
conda run -n moving-det-vru moving-det-vru build-human-benchmark \
  --zip /home/stu1/Projects/moving_Det/label_data/videolabel_annotated_291frames_20260816.zip \
  --image-root /mnt/nas/Processing_data/site19_22_sequence_7class \
  --output runs/vrud-pilot/human-benchmark-20260816
```

再把已批准的 Universal 权重一次性转换并冻结为四类、P2–P5 的初始化文件：

```bash
conda run -n moving-det-vru moving-det-vru freeze-p2-init \
  --weights /home/stu1/Projects/moving_Det/models/best_vru_universal.pt \
  --output runs/vrud-pilot/universal-p2-init-20260816
```

正式 Baseline 从该冻结文件开始训练，不再直接读取原始 Universal
checkpoint。完整训练命令和中断处理见下文，这里不重复启动同一输出目录。

人工 benchmark 只能用于 `test`。待下文的正式 Baseline 和 validation 阈值冻结后，
再运行此评测。置信度阈值必须先在原 validation 集冻结，然后原样用于人工
test；禁止在这 873 帧上选择或调整阈值：

```bash
conda run -n moving-det-vru moving-det-vru evaluate \
  --model baseline \
  --checkpoint runs/vrud-pilot/baseline/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest \
  --split test \
  --threshold runs/vrud-pilot/baseline-validation/threshold.json \
  --human-benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --output runs/vrud-pilot/baseline-human-test
```

在正式训练前，可用真实冻结输入执行一次 GPU 前向自检：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru python \
  scripts/smoke_human_foundation.py \
  --benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --p2-init runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt
```

这批人工序列大部分来自 Universal 模型曾用于生成伪标注的目标视频，因此并非与
Universal 完全独立的数据。即使后续 MG 优于同初始化的 Baseline，可信结论也只能是
“在当前目标域和固定人工 test 上的增量改进”，不能表述为对未见场景的通用泛化提升。

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

训练单帧基线。下列 `runs/vrud-pilot/baseline` 专指直接从冻结 P2 启动、
从未使用 `--resume` 并一次不间断完成的正式 Baseline run（分支 A）：

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline \
  --weights runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt
```

若 Baseline 中断，分支 B 只用于继续 Baseline 训练或单独评测 Baseline。将恢复结果
写入明确命名的新目录，且不再传 `--weights`：

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline-resumed-only \
  --resume runs/vrud-pilot/baseline/checkpoints/last.pt
```

分支 B 生成的 `best.pt` 和 `last.pt` 都带 resume provenance；它们只能再恢复
Baseline 或用 `--model baseline` 评测，永远不能作为正式 MG/LSTFE 的
`--baseline-init` 或等价 `--weights`。

若中断后仍必须满足当前严格证据合同，不要恢复该 run，也不要覆盖或复用已中断的
`runs/vrud-pilot/baseline`。必须从同一冻结 P2 在全新输出目录重启，并不间断完成：

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline-formal-restart-01 \
  --weights runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt
```

重启后，只有该新目录中不间断完成的 `best.pt` 可以作为正式时序初始化。下文的字面
`runs/vrud-pilot/baseline` 路径假定分支 A 成功不间断完成；如果采用上述严格重启，
MG、LSTFE 和正式 Baseline 评测中的所有 checkpoint 路径都必须一致改为
`runs/vrud-pilot/baseline-formal-restart-01`，绝不能改为 `baseline-resumed-only`。

MG 和 LSTFE 只能从同一个不间断完成的正式 Baseline `best.pt` 初始化。该
checkpoint 必须直接由上面的冻结 Universal-P2 文件开始训练，不能带时序
alignment 指纹，也不能是内部 init 或 resume 的产物；其 P2 文件、SHA-256、
Universal 来源 SHA-256 和 427/859 迁移计数都会在加载检测器参数前重新严格验证。
`--baseline-init` 是明确写法；为兼容冻结的 Task 13 命令，时序模型的
`--weights <baseline-best.pt>` 具有相同含义：

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

`--baseline-init` 会拒绝 MG/LSTFE checkpoint、Baseline `last.pt`、由 resume/internal
init 产生的任何 Baseline checkpoint（包括恢复后的 `best.pt`），以及不来自正式
冻结 P2 artifact 的 checkpoint。
时序训练中断后必须使用 `--resume <temporal-output>/checkpoints/last.pt` 恢复，不能把
该时序 checkpoint 再作为 `--baseline-init`。

64 样本 gate 必须同时给出固定样本数和最大 step。CLI 从原 manifest
确定性地派生恰好 64 行的新 manifest，不修改源 artifact：

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline-overfit \
  --weights runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt \
  --overfit-samples 64 \
  --max-steps 300
```

训练输出的主 checkpoint 是 `<output>/checkpoints/best.pt`，模型内部恢复来源是
`<source-output>/checkpoints/last.pt`。Baseline 的 resume 仅按上述分支 B 处理，
不改变正式时序初始化合同。非默认缓存可通过
`--alignment-cache /safe/path/alignment-cache` 显式指定。

正式三模型比较中的 Baseline 评测必须使用与 MG/LSTFE 初始化相同的分支 A
`best.pt`（或上述严格重启后的新正式 `best.pt`）。评测必须先在 validation 选择并冻结
阈值，再应用到 test：

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

如果只评测分支 B，可将上述 checkpoint 指向
`runs/vrud-pilot/baseline-resumed-only/checkpoints/best.pt`，但 validation、test
和阈值必须写入独立的 `baseline-resumed-only-*` 输出。这些是 Baseline-only
评测产物，不属于下面的正式三模型比较。

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
  使用同一份 provenance。完成的 run 还用 `checkpoint_artifacts` 分别声明
  `best.pt`/`last.pt` 的角色、SHA-256、epoch、模型、manifest 与加载来源。
- `<train-output>/checkpoints/{best,last}.pt`：包含 manifest 指纹、模型名、
  明确的 `checkpoint_role`、优化器、恢复状态和 `load_provenance` 的内部实验
  checkpoint；从公开 YOLO 权重初始化时，后者记录实际已加载本地文件的绝对路径
  与 SHA-256。内部
  init/resume 与公开权重字段严格分离；checkpoint 不能当作公开 YOLO 权重读取。
  正式时序 checkpoint 还会完整保留 Baseline best 的路径/SHA-256/epoch/manifest，
  以及冻结 P2 的路径/SHA-256、Universal 来源 SHA-256 和 427/859 计数。
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
