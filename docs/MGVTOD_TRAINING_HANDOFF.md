# MG-VTOD 八类航拍小目标 OBB 训练交接手册

最后核对：2026-09-01（Asia/Shanghai）

代码仓库：<https://github.com/zzi4/moving_Det>

当前主线是 `main` 上的 `mg_vtod_8class`。新数据的继续训练应以本文为准；
`README.md` 中的四类 Baseline/MG-VTOD/LSTFE 内容保留用于历史实验复盘，
不是当前八类模型的直接训练入口。

## 1. 接手后先记住的结论

1. 当前模型识别全部交通参与者，不只识别 VRU，也不只识别运动目标。
2. 运动信息是 RGB 检测器的可学习辅助特征，不是候选区域硬筛选。静止目标仍可由 RGB 分支检出。
3. 每个样本使用 `t-4, t-2, t, t+2, t+4` 共 5 帧；30 fps 下约覆盖中心帧前后
   133 ms。
4. 全部框为旋转框 OBB，训练类别固定为 8 类，不得在新数据中自行改变类别编号。
5. 目前最可用的 checkpoint 是经过图像与标注重新配对后训练的 `best.pt`，
   对应人类所说的第 2 个 epoch（内部 `epoch=1`）。不要默认使用 `last.pt`。
6. 新数据必须先核对 JPG/JSON 是同一帧，再重算 ECC 配准缓存，最后才能启动训练。

## 2. 代码与产物边界

仓库只保存代码、测试、配置模板、设计文档和本交接文档。以下内容不进 GitHub：

- 原始视频、JPG、LabelMe JSON 数据集和 ZIP 包；
- `best.pt`/`last.pt`/Universal 初始权重；
- `runs/`、ECC `alignment-cache/`、TensorBoard 和 W&B 输出；
- 任何服务器凭据、GitHub token 或私钥。

核心文件：

| 路径 | 用途 |
| --- | --- |
| `src/moving_det/ml/models/mg_vtod_8class.py` | 八类 MG-VTOD 网络、Universal 权重继承、Motion Stem 和融合 |
| `src/moving_det/ml/motion_proposals.py` | 从对齐的 5 帧计算软运动强度图 |
| `src/moving_det/vrud/alignment.py` | ECC 配准缓存的写入、指纹和读取 |
| `src/moving_det/ml/dataset.py` | 5 帧时序 tile 加载及标注转换 |
| `src/moving_det/ml/training.py` | 单/双 GPU 训练、AMP、checkpoint 和恢复合同 |
| `src/moving_det/vru_cli.py` | `build-manifest`/`cache-alignments`/`train`/`evaluate` 入口 |
| `src/moving_det/vrud/expanded_dataset.py` | 追加新人工校正序列并保持 validation 冻结 |
| `configs/mg-vtod-8class-finetune.yaml` | 当前八类 10-epoch 训练配置模板 |
| `configs/models/yolo11m-obb-8class.yaml` | 八类 YOLO11m-OBB P3–P5 图结构 |
| `tests/ml/test_mg_vtod_8class.py` | 八类模型和权重迁移合同 |
| `tests/vrud/test_expanded_dataset.py` | 新序列接入、边界框忽略和图像配对合同 |

## 3. 模型原理

### 3.1 数据流

```text
5 帧 RGB tile
  │
  ├── ECC 支持帧→中心帧配准
  │      └── 时序中值背景、局部噪声归一化、多帧投票
  │             └── [B,1,H,W] 软运动强度图
  │                         └── Early Motion Stem
  │
  └── 中心 RGB 帧→YOLO 第 0 层特征
                             │
               concat(RGB, motion)→1×1 Conv→残差加回 RGB
                             │
                    YOLO11m-OBB P3/P4/P5 检测头
                             │
                     8 类置信度 + OBB
```

ECC 先把支持帧对齐到中心帧，尽可能消除飞行器或画面的全局抖动。
运动强度不是简单的两帧相减：它还会修正亮度变化，用其他支持帧的中值估计背景，
通过局部噪声和边缘惩罚压制建筑、道路边缘的大范围伪运动，再用多帧一致投票得到
`[0,1]` 软分数。当前八类模型只把软分数送给网络，不使用二值 proposal mask 硬切图。

### 3.2 为什么不会把静止车辆直接删掉

`EarlyMotionStem` 将单通道运动图编码成与 YOLO 第 0 层同尺寸、同通道数的特征。
融合式为：

```text
fused = rgb + Conv1x1(concat(rgb, motion))
```

`Conv1x1` 的权重和偏置初始化为 0。因此刚创建模型时 `fused == rgb`，输出从
Universal RGB 检测器的行为开始；训练只会在损失认为运动信息有帮助时学习残差。
运动为 0 时 RGB 路径仍然完整，所以静止车辆、静止 VRU 仍可被检测。

### 3.3 八类固定编号

| ID | 类别 |
| ---: | --- |
| 0 | `car` |
| 1 | `truck` |
| 2 | `bus` |
| 3 | `motorcycle` |
| 4 | `pedestrian` |
| 5 | `bicycle` |
| 6 | `tricycle` |
| 7 | `engineering_vehicle` |

Universal 权重必须与当前 8 类检测器的 691 个张量严格匹配。当前代码批准的初始权重为：

```text
models/best_vru_universal.pt
SHA-256 114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7
```

权重不在 GitHub 中，需从项目负责人处获取。拿到后必须先校验：

```bash
sha256sum models/best_vru_universal.pt
```

不应为了跳过校验而修改代码中的 approved SHA；更换初始模型属于新的可复现实验路线。

## 4. 环境建立

当前成功训练环境：

- Python 3.11.15
- PyTorch 2.5.1
- TorchVision 0.20.1
- CUDA 12.4
- Ultralytics 8.4.115
- NumPy 2.4.6
- 两张 NVIDIA RTX A6000，DDP 后端为 NCCL

新机器建议：

```bash
git clone https://github.com/zzi4/moving_Det.git
cd moving_Det
conda env create -f environment/temporal-obb.yml
conda run -n moving-det-vru moving-det-vru --help
```

当前服务器已有环境：

```text
/home/stu1/anaconda3/envs/moving-det-vru
```

如果以前在 worktree 中训练，发布后不再需要强制指向旧 worktree 的 `PYTHONPATH`。
应在仓库根目录重新执行一次可编辑安装：

```bash
/home/stu1/anaconda3/envs/moving-det-vru/bin/python -m pip install -e '.[dev,ml]'
```

## 5. 数据合同

每个序列应有连续的六位帧号：

```text
<image_root>/<site>_sequence/<sequence>/000001.jpg
<image_root>/<site>_sequence/<sequence>/000001.json
...
```

LabelMe JSON 必须满足：

- `imageWidth`/`imageHeight` 与 JPG 实际尺寸一致；
- `imagePath` 对应同帧 JPG，不能只因文件名相同就假定内容相同；
- `shape_type == "rotation"`，`points` 含 4 个有限坐标点；
- `group_id` 是整数 track ID，同一帧内不重复；
- `label` 只能是上述 8 类之一；
- `imageData` 在训练副本中会清空，避免 JSON 重复嵌入大图；
- 超出图像边界或压在边界上的旋转框会被忽略；
- 同一原始序列不能同时进入 train 和 validation/test。

全图为 3840×2160，训练 tile 为 1024×1024，重叠 256 px。典型小目标约 20×40 px，
所以不要再把 tile 大幅缩小后训练。

## 6. 新人工校正序列接入

### 6.1 先做只读核验

对每段新序列至少输出以下审计：

- JPG 数、JSON 数、同 stem 配对数、缺失帧号；
- 图像尺寸与 JSON 尺寸一致性；
- 类别计数、`group_id` 类型/重复、shape type、非有限坐标；
- 超边界框数；
- 随机及首/中/尾帧标注叠加图；
- 相邻帧是否真正连续，JPG 是否与 JSON 中的对象对应。

最重要的历史教训是：两组文件的帧号一样，不代表它们就是同一张图。
如果标注包内同时包含 JSON 和 JPG，应以包内 JPG 为权威图像。

### 6.2 构建新训练集

`build_expanded_training_dataset` 会原子地创建一个新目录，保留基础数据的冻结
validation/test，把新序列只追加到 train。输出目录必须不存在，不允许覆盖旧 run。

```bash
conda run -n moving-det-vru python - <<'PY'
from pathlib import Path
from moving_det.vrud.expanded_dataset import (
    ExpandedSequenceSource,
    build_expanded_training_dataset,
)

summary = build_expanded_training_dataset(
    base_run=Path("/path/to/approved-base-run"),
    output_run=Path("/path/to/new-expanded-run"),
    sources=(
        ExpandedSequenceSource(
            zip_path=Path("/path/to/corrected-sequence.zip"),
            site="site22",
            sequence="DJI_sequence_name",
            image_root=Path("/path/to/source-image-root"),
        ),
    ),
    tile_size=1024,
    overlap=256,
    support_offsets=(-4, -2, 0, 2, 4),
)
print(summary)
PY
```

ZIP 内必须能解析到 `<sequence>/<frame>.json` 和同 stem JPG。构建完成后核对：

```text
<new-run>/data-build-audit.json
<new-run>/manifest/manifest.json
<new-run>/manifest/class-audit.json
<new-run>/config.yaml
```

### 6.3 2026-09-01 新 600 帧的当前状态

只读盘点结果：

- `/mnt/nas/Processing_data/mot_sequence/DJI_20240919162517_0002_V_backup_static_ID1005_1011_20260830_220802`：
  300 个 JSON，该目录内 0 个 JPG；在找到与这 300 帧严格对应的图像前不能接入训练。
- `/mnt/nas/Processing_data/mot_sequence/DJI_20240919164000_0006_V_final_backup_20260831_215645`：
  300 个 JSON 和 300 个 JPG；还需完成尺寸、类别、框合法性及可视化核验才能接入。

这 600 帧未被本文声明为“已可训练”，也尚未进入当前最佳 checkpoint。

## 7. 重算 ECC 配准缓存

每次改变图像、manifest、tile 或时序 offset 都必须使用新目录重算缓存：

```bash
conda run -n moving-det-vru moving-det-vru cache-alignments \
  --config /path/to/new-run/config-mg-cache.yaml \
  --manifest /path/to/new-run/manifest \
  --output /path/to/new-run/alignment-cache
```

为了只计算 MG 所需的变换，`config-mg-cache.yaml` 中保持：

```yaml
mg_offsets: [-4, -2, 0, 2, 4]
lstfe_offsets: [-4, -2, 0, 2, 4]
```

成功后检查 `alignment-cache/summary.json`：

- `fallback_count` 应尽量为 0；
- 应只有 `-4,-2,+2,+4` 支持帧变换，中心帧不需缓存；
- `manifest_sha256` 必须与即将训练的 manifest 相同；
- 不能复制旧序列的变换冒充新缓存。

## 8. 训练命令

### 8.1 最小 smoke

smoke 不用来比较精度，只证明数据、缓存、权重、前向、损失和 checkpoint 能走通。
输出目录必须是新路径。

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru train \
  --model mg_vtod_8class \
  --config /path/to/new-run/config.yaml \
  --manifest /path/to/new-run/manifest \
  --weights /path/to/models/best_vru_universal.pt \
  --alignment-cache /path/to/new-run/alignment-cache \
  --devices 1 \
  --train-scope full \
  --overfit-samples 64 \
  --max-steps 2 \
  --output /path/to/new-run/smoke-YYYYMMDD
```

检查：`run.json` 不是 failed、loss 有限、`last.pt` 可读、manifest/cache/weights SHA 正确。

### 8.2 双 GPU 完整微调

当前工程路线采用全网解冻，即 RGB 主干、检测头、Motion Stem 和融合层全部训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n moving-det-vru \
  moving-det-vru train \
  --model mg_vtod_8class \
  --config /path/to/new-run/config.yaml \
  --manifest /path/to/new-run/manifest \
  --weights /path/to/models/best_vru_universal.pt \
  --alignment-cache /path/to/new-run/alignment-cache \
  --devices 2 \
  --train-scope full \
  --output /path/to/new-run/training-10epochs-YYYYMMDD
```

`pilot_epochs` 由 config 控制。目前模板为 10 epoch、AdamW、学习率 `2e-5`、
warmup 1 epoch、effective batch size 16。

### 8.3 训练中断和恢复

前台训练首选 `Ctrl-C`。不要首先使用 `kill -9`，因为它不给进程清理 DDP 和写入状态的机会。
确认进程可用：

```bash
pgrep -af 'moving_det.distributed_train|moving-det-vru train|torch.distributed.run'
nvidia-smi
```

恢复时使用上一个 run 的 `last.pt`，但必须写入一个新输出目录；配置、manifest、
alignment cache、双 GPU world size 和 `train_scope` 必须与原 run 兼容：

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n moving-det-vru \
  moving-det-vru train \
  --model mg_vtod_8class \
  --config /path/to/new-run/config.yaml \
  --manifest /path/to/new-run/manifest \
  --resume /path/to/interrupted-run/checkpoints/last.pt \
  --alignment-cache /path/to/new-run/alignment-cache \
  --devices 2 \
  --train-scope full \
  --output /path/to/new-run/training-resume-YYYYMMDD
```

正常 checkpoint 目录必须包含：

```text
best.pt
last.pt
history.json
run.json
```

`best.pt` 是 validation mAP50 最高时的权重；`last.pt` 是最后完成 epoch 的恢复点。
最终部署/对比不能因文件名更新就自动选 `last.pt`。

## 9. 评价与可视化

先在 validation 上评价：

```bash
conda run -n moving-det-vru moving-det-vru evaluate \
  --model mg_vtod_8class \
  --config /path/to/new-run/config.yaml \
  --checkpoint /path/to/training/checkpoints/best.pt \
  --manifest /path/to/new-run/manifest \
  --alignment-cache /path/to/new-run/alignment-cache \
  --split validation \
  --output /path/to/new-run/evaluation-best-YYYYMMDD
```

评价时至少同时看：

- mAP50：整体检测排序和定位/分类质量；
- recall@rIoU 0.25：对小 OBB 更宽容的找回能力；
- precision/FP：是否通过变得过于保守来减少误检；
- 每类 AP/recall，特别是 `pedestrian`/`bicycle`/`tricycle`；
- RGB+GT、运动热力图、Universal 预测、MG-VTOD 预测的同帧对比。

可视化组件位于：

- `src/moving_det/ml/best_checkpoint_visualization.py`：GT/TP/FP/FN/类别错误及运动图；
- `src/moving_det/ml/qualitative_comparison.py`：8 类对比面板；
- `src/moving_det/ml/inference.py`：分 tile 推理与旋转 NMS。

图片挑选只能用来诊断，不能取代完整 validation/test 指标。

## 10. 当前可用训练记录

### 10.1 最新正确数据 run

当前服务器路径：

```text
/home/stu1/Projects/moving_Det/runs/vrud-pilot/
  human-mgvtod-8class-expanded-1473-corrected-images-20260828/
```

数据与缓存：

- 总帧数：1473；validation 为独立序列的 64 个 tile；当时 `test.jsonl` 为空；
- taxonomy：`full-traffic-8class-v1`；
- manifest SHA-256：`b229a340ed413f752192a8e37bde461f9170423083ff5698b1c58653ee0908a7`；
- ECC center 数：1246，任务数：4960，fallback：0，worker：16；
- alignment-cache SHA-256：`dd82a176215ab67f96ccab02780519b037e822b1c6c5868bdba23581f0e451b1`。

成功训练目录：

```text
training-10epochs-dual-fresh-20260828/checkpoints/
```

结果：

| 项目 | 结果 |
| --- | ---: |
| 状态 | completed |
| GPU | 2× RTX A6000, NCCL DDP |
| 总时间 | 4148.3 s，约 69 分 8 秒 |
| 最佳 epoch | 内部 1，即第 2 轮 |
| 最佳 mAP50 | 0.6172705961 |
| 最佳 recall@rIoU0.25 | 0.9578794481 |
| 第 10 轮 mAP50 | 0.5942125924 |
| 第 10 轮 recall@rIoU0.25 | 0.9360929557 |
| train loss | 4.4432 → 1.9108 |
| AMP overflow skips | 1 |

checkpoint 指纹：

```text
best.pt  8e9fa04b4709f888cccfafe707d871a96bc80a22c2fb3b35260759cf06ba8035
last.pt  aef74c194b4f83a689fbf70a7e63d02b5374a296ecdf534d982e5244ef13274b
```

训练 loss 持续降低，但 validation 在第 2 轮达到最高，后面没有形成稳定改善；
这是数据量/场景多样性不足下的早期平台和过拟合信号。

### 10.2 Universal 与当前 MG-VTOD 的粗对比

在挑选的 6 个 validation tile、置信度 0.25、rIoU 0.25 下：

| 模型 | TP | FN | FP | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Universal RGB | 113 | 21 | 71 | 0.6141 | 0.8433 |
| MG-VTOD best | 96 | 38 | 34 | 0.7385 | 0.7164 |

MG-VTOD 的 FP 约少 52%，但 TP 约少 15%，明显更保守。该小样本上 MG 并没有全面超过
Universal；当前更合理的结论是“运动分支改变了 precision/recall 平衡，但需要更多样化、
严格配对的人工序列才能判断是否真正提升”。

这只是固定阈值的挑选帧诊断，不是全 validation 正式对比。

### 10.3 历史实验摘要

| 实验 | 状态 | 最佳结果 | 意义 |
| --- | --- | --- | --- |
| Baseline 64 样本 overfit | completed | mAP50 0.6599，recall 0.9254 | 证明基线训练链能拟合小集合，不代表泛化 |
| 四类 MG-VTOD 64 样本 overfit | completed | mAP50 0.8921，recall 0.9291 | 证明旧时序分支和 DDP 可学习 |
| 四类 temporal-only 6 epoch | completed 但路线失败 | mAP50 约 0.000007 | 只训新分支不足以修复分类/检测头不匹配 |
| 四类 full 10 epoch | completed | mAP50 0.06095，recall 0.6727 | 恢复了部分找回，但四类路线不再是当前主线 |
| 八类 smoke-r2 | completed | mAP50 0.3973，recall 0.9095 | 修复 CUDA validator 后的首次八类端到端通路 |
| 八类单 GPU 早期 run | 人工中断 | 第 3 轮 mAP50 0.5871，recall 0.9506 | 证明八类方向比四类微调更可靠 |
| 扩展 1473 帧旧 run | **数据无效** | 所有指标作废 | 600 帧中 JPG/JSON 配对错误，审计为 0/600 匹配 |
| 修正图像后的 1473 帧双 GPU run | completed | mAP50 0.6173，recall 0.9579 | 当前最可用的初版 checkpoint |

早期 AMP 梯度非有限、GPU 争用、DDP smoke 失败和 validator 隐式 CUDA 设备错误均保留在
`runs/` 的 `run.json` 中，它们属于工程诊断记录，不应与模型有效性结论混在一起。

## 11. 绝对禁止使用的 run

以下目录内的 checkpoint 和指标全部作废，不得恢复或作为初始权重：

```text
/home/stu1/Projects/moving_Det/runs/vrud-pilot/
  human-mgvtod-8class-expanded-1473-20260828/
```

原因：两个人工修正 ZIP 中的 600 个标注，被错误配对到另一批同名 JPG；
SHA-256 审计为 0/600 图像匹配。失败 run 的 mAP50 0.3559/0.4194 没有解释价值。

## 12. 监控与验收清单

训练启动后：

- `nvidia-smi` 中两张卡都有 DDP worker，利用率在 batch 期间波动为正常现象；
- 第一个 epoch 后 `best.pt`/`last.pt`/`history.json`/`run.json` 齐全；
- `run.json` 中 `model_name == mg_vtod_8class`、`train_scope == full`、`world_size == 2`；
- manifest、alignment cache、Universal 权重 SHA 与本次实验记录一致；
- loss 和梯度有限，AMP overflow 如发生应被记录并跳过，不能静默污染权重；
- 每轮同时记录 mAP50 和 recall，发现连续下降时优先保留早期 best；
- 不在训练进行中修改数据、manifest、缓存、学习率或 checkpoint。

正式交付一个新模型前，需保存：

1. Git commit/tag；
2. 配置 YAML；
3. manifest 与 alignment-cache SHA；
4. 初始权重 SHA；
5. `run.json`/`history.json`；
6. best/last checkpoint SHA；
7. 完整 validation 指标和若干固定样本可视化。

## 13. 下一阶段建议

1. 先补齐并严格核验新 600 帧中缺失的 300 帧 JPG。
2. 扩大场景、速度、目标尺寸、昼夜和遮挡多样性，不要只追加高度相似的连续帧。
3. 以序列为单位重做 train/validation/test 划分，建立一个完全不参与调参的测试集。
4. 在相同数据、相同初始权重、相同训练时长下比较 Universal/RGB 微调和 MG-VTOD，
   并增加 Motion ON/OFF 消融。
5. 当前 MG 偏保守，下一轮重点看分类别 recall、置信度校准和运动残差强度，
   而不是只看总 mAP50。

## 14. 一页式接手检查

- [ ] 从 GitHub `main` 克隆，记录 commit/tag。
- [ ] 按 `environment/temporal-obb.yml` 建立环境，`moving-det-vru --help` 正常。
- [ ] 获取 Universal 权重并校验 SHA-256。
- [ ] 对新 JPG/JSON 做数量、内容、类别和 OBB 可视化审计。
- [ ] 使用 `expanded_dataset.py` 创建新 run，不覆盖旧 run。
- [ ] 完成 manifest/class audit，确认语义和 validation 冻结。
- [ ] 重算 ECC cache，核对 manifest/cache SHA 和 fallback。
- [ ] 先跑单卡 2-step smoke，确认 checkpoint 产出。
- [ ] 用新输出目录启动双 GPU、full-unfreeze 训练。
- [ ] 保存 best/last/history/run 及全部 SHA。
- [ ] 在完整 validation/test 上评价，固定样本图只做诊断。
