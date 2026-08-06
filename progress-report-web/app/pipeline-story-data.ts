export type PipelineLayer = {
  id: string;
  label: string;
  src: string;
  src1x: string;
  alt: string;
  caption: string;
};

export type PipelineStage = {
  number: string;
  title: string;
  status: "real" | "planned";
  question: string;
  answer: string;
  inputs: string;
  process: string;
  output: string;
  value: readonly { label: string; value: string }[];
  judgement: string;
  negative?: boolean;
  visual:
    | { kind: "evidence"; layers: readonly PipelineLayer[] }
    | { kind: "classifier-plan" }
    | { kind: "lifecycle-plan" };
};

const alignmentLayers: readonly PipelineLayer[] = [
  {
    id: "before",
    label: "对齐前",
    src: "/evidence/pipeline/alignment-before.webp",
    src1x: "/evidence/pipeline/alignment-before-1x.webp",
    alt: "第19帧与第20帧对齐前的真实差分热图",
    caption: "同一色阶下的原始帧差；车辆轮廓和静态细边缘都会产生响应。",
  },
  {
    id: "after",
    label: "ECC 对齐后",
    src: "/evidence/pipeline/alignment-after.webp",
    src1x: "/evidence/pipeline/alignment-after-1x.webp",
    alt: "第19帧经过ECC对齐后与第20帧的真实差分热图",
    caption:
      "本帧相关系数很高，但该局部区域的平均残差反而增加，说明残余稳像必须门控。",
  },
];

const motionLayers: readonly PipelineLayer[] = [
  {
    id: "overlay",
    label: "叠加原图",
    src: "/evidence/pipeline/motion-overlay.webp",
    src1x: "/evidence/pipeline/motion-overlay-1x.webp",
    alt: "真实道路局部区域叠加帧差运动热图和青色车辆标注",
    caption:
      "青色是真实车辆 OBB；暖色运动响应覆盖了大量车辆，也照亮了树影和道路细边缘。",
  },
  {
    id: "heatmap",
    label: "仅运动热图",
    src: "/evidence/pipeline/motion-heatmap.webp",
    src1x: "/evidence/pipeline/motion-heatmap-1x.webp",
    alt: "frame_diff 0.7倍尺度Z等于6的真实运动热图",
    caption: "蓝色为低响应，黄红色为高响应；这里没有加入外观分类结果。",
  },
];

const proposalLayers: readonly PipelineLayer[] = [
  {
    id: "proposals",
    label: "OBB 候选",
    src: "/evidence/pipeline/proposals.webp",
    src1x: "/evidence/pipeline/proposals-1x.webp",
    alt: "真实道路区域的青色车辆标注和红色旋转框候选",
    caption:
      "青色为 GT，红色为 proposal。多数车辆附近有响应，但背景碎片和错误形状仍很明显。",
  },
  {
    id: "mask",
    label: "二值 Mask",
    src: "/evidence/pipeline/mask.webp",
    src1x: "/evidence/pipeline/mask-1x.webp",
    alt: "阈值化后的真实运动二值掩膜与车辆标注",
    caption: "琥珀色区域是进入连通域与旋转框拟合的真实二值运动响应。",
  },
];

const tubeletLayers: readonly PipelineLayer[] = [
  {
    id: "before",
    label: "连接前",
    src: "/evidence/pipeline/tubelets-before.webp",
    src1x: "/evidence/pipeline/tubelets-before-1x.webp",
    alt: "多尺度方法在第20帧连接Tubelet之前的真实候选",
    caption: "灰色框是连接前的多尺度 proposal，数量已经远超可分类范围。",
  },
  {
    id: "after",
    label: "连接后",
    src: "/evidence/pipeline/tubelets-after.webp",
    src1x: "/evidence/pipeline/tubelets-after-1x.webp",
    alt: "第18到22帧按真实tubelet编号连接后的轨迹线",
    caption:
      "同色线代表同一 tubelet。大量长距离错误连接直观说明当前连接约束过弱。",
  },
];

export const pipelineStory: readonly PipelineStage[] = [
  {
    number: "01",
    title: "残余稳像",
    status: "real",
    question: "画面已经稳定，为什么还要再对齐？",
    answer:
      "亚像素抖动仍可能在道路边缘制造伪运动，但已经稳定的视频也不能无条件重复校正。",
    inputs: "第 19、20 帧",
    process: "ECC 欧氏变换估计",
    output: "对齐后的残余差分",
    value: [
      { label: "ECC 相关系数", value: "0.9967" },
      { label: "估计平移", value: "+0.35 / −0.10 px" },
      { label: "局部平均残差", value: "3.56 → 4.36" },
    ],
    judgement:
      "这一个真实帧对中，ECC 没有降低选定区域残差；后续应增加静态差分门控，而不是无条件应用。",
    negative: true,
    visual: { kind: "evidence", layers: alignmentLayers },
  },
  {
    number: "02",
    title: "运动证据",
    status: "real",
    question: "单帧看不清的小目标，连续帧提供了什么？",
    answer:
      "位移、方向和持续性会把车辆从纹理有限的单帧背景中凸显出来。",
    inputs: "±1 / 3 / 7 / 15 帧",
    process: "变化聚合为运动分数",
    output: "运动热图与二值响应",
    value: [
      { label: "Recall@0.25", value: "91.26%" },
      { label: "中心命中", value: "99.19%" },
      { label: "Mask coverage", value: "56.16%" },
    ],
    judgement:
      "车辆确实被增强，但树影和细边缘也会响应；运动证据适合负责发现，不足以独立确认目标。",
    visual: { kind: "evidence", layers: motionLayers },
  },
  {
    number: "03",
    title: "OBB 候选",
    status: "real",
    question: "运动像素如何变成可以追踪的对象？",
    answer: "阈值化、形态学和连通域把响应区域转换为带方向的旋转框。",
    inputs: "运动热图",
    process: "阈值、连通域、旋转框拟合",
    output: "OBB proposal",
    value: [
      { label: "Proposal", value: "292,992" },
      { label: "FP / 100 GT", value: "727.72" },
      { label: "Recall@0.50", value: "7.74%" },
    ],
    judgement:
      "0.7× 相比 1.0× 使误候选下降 70.33%，召回几乎不变；但 OBB 形状仍不够准确。",
    negative: true,
    visual: { kind: "evidence", layers: proposalLayers },
  },
  {
    number: "04",
    title: "Tubelet",
    status: "real",
    question: "为什么不能把每一帧的 OBB 当成独立结果？",
    answer:
      "真实目标会连续移动，一闪而过的响应更可能是噪声；跨帧连接还可以维持短时漏检。",
    inputs: "连续帧 OBB",
    process: "位置、面积与方向连续性连接",
    output: "跨帧 Tubelet",
    value: [
      { label: "连接前", value: "2,772,669" },
      { label: "连接后", value: "2,770,752" },
      { label: "实际减少", value: "约 0.07%" },
    ],
    judgement:
      "当前 min_frames=2 和连接约束太弱，只减少约 0.07% 候选，还产生了大量错误长连接。",
    negative: true,
    visual: { kind: "evidence", layers: tubeletLayers },
  },
  {
    number: "05",
    title: "时序分类",
    status: "planned",
    question: "运动只能说明有东西在动，如何知道它是什么？",
    answer:
      "外观与运动双流共同判断交通参与者类别，并利用多帧信息细化 OBB。",
    inputs: "9–17 帧 RGB + motion crop",
    process: "双流时序分类与框回归",
    output: "类别、稳定置信度、细化 OBB",
    value: [
      { label: "未来验收", value: "各类别召回" },
      { label: "未来验收", value: "OBB R@.50" },
    ],
    judgement: "必须先把百万级候选收敛成可信 Tubelet，再进入模型训练。",
    visual: { kind: "classifier-plan" },
  },
  {
    number: "06",
    title: "轨迹管理",
    status: "planned",
    question: "短时遮挡、停车和真正离场如何区别？",
    answer:
      "显式生命周期让未观测目标短时保留，只在驶出画面或超过规则时终止。",
    inputs: "Tubelet + 类别 + OBB",
    process: "状态预测、匹配、恢复与终止",
    output: "连续且可解释的完整轨迹",
    value: [
      { label: "未来验收", value: "ID switch" },
      { label: "未来验收", value: "遮挡恢复率" },
    ],
    judgement: "轨迹状态负责落实“目标不会因一两帧漏检而凭空消失”。",
    visual: { kind: "lifecycle-plan" },
  },
];
