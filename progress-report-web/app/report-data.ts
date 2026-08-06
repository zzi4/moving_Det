export const navigation = [
  { href: "#overview", label: "项目概览", index: "01" },
  { href: "#method", label: "方法流程", index: "02" },
  { href: "#results", label: "关键结果", index: "03" },
  { href: "#engineering", label: "性能优化", index: "04" },
  { href: "#risks", label: "风险边界", index: "05" },
  { href: "#next", label: "下一步", index: "06" },
  { href: "#evidence", label: "证据入口", index: "07" },
] as const;

export const conditions = [
  ["视频", "3840 × 2160 / 30 fps"],
  ["小目标", "约 20 × 40 px"],
  ["拍摄", "悬停，已做视频稳定"],
  ["处理", "离线，暂无速度硬约束"],
  ["输出", "类别 + OBB + 连续轨迹"],
  ["数据", "约 4 小时视频级标注"],
] as const;

export const pipeline = [
  {
    number: "01",
    title: "残余稳像",
    description: "ECC 两遍估计残余抖动；真实结果表明还需增加静态残差门控。",
  },
  {
    number: "02",
    title: "运动证据",
    description: "聚合 ±1 / 3 / 7 / 15 帧的变化，用 median 与 MAD 转成稳健 Z 分数。",
  },
  {
    number: "03",
    title: "OBB 候选",
    description: "阈值化、形态学和连通域把响应区域转成旋转框 proposal。",
  },
  {
    number: "04",
    title: "Tubelet",
    description: "用位置、速度、面积与朝向连续性把跨帧候选连接起来。",
  },
  {
    number: "05",
    title: "时序分类",
    description: "后续使用 9–17 帧 RGB + motion crop 分类并细化 OBB。",
  },
  {
    number: "06",
    title: "轨迹管理",
    description: "显式处理入场、短时漏检、遮挡、停车和驶出画面。",
  },
] as const;

export const methods = [
  "frame_diff",
  "temporal_median",
  "mog2",
  "multiscale",
  "multiscale_tubelet",
] as const;

export const calibrationRows = [
  {
    method: "frame_diff / 1.0",
    threshold: "Z = 6",
    recall25: "91.32%",
    recall50: "6.96%",
    center: "99.06%",
    coverage: "59.28%",
    proposals: "910,176",
    fp: "2,452.85",
  },
  {
    method: "temporal_median / 1.0",
    threshold: "Z = 6",
    recall25: "90.62%",
    recall50: "7.50%",
    center: "96.79%",
    coverage: "75.86%",
    proposals: "1,840,565",
    fp: "5,054.21",
  },
  {
    method: "mog2 / 1.0",
    threshold: "var = 25",
    recall25: "83.10%",
    recall50: "13.94%",
    center: "99.44%",
    coverage: "46.17%",
    proposals: "722,743",
    fp: "1,937.15",
  },
] as const;

export const smokeMetrics = [
  ["范围", "4K 原图，第 16–25 帧"],
  ["方法", "multiscale_tubelet / scale 1.0 / Z 4"],
  ["Moving GT", "1,320"],
  ["Proposals", "137,749"],
  ["False proposals", "137,735"],
  ["Recall @ 0.25", "1.06%"],
  ["Mean mask coverage", "96.81%"],
  ["FP / 100 GT", "10,434.47"],
  ["耗时 / 峰值内存", "39:58 / 1.92 GiB"],
] as const;

export const engineeringRows = [
  {
    issue: "Subset 破坏时间上下文",
    evidence: "只抽中心帧时会丢掉 ±15 帧支持",
    change: "区分输出帧与处理上下文",
    effect: "小样本实验仍保留完整时间证据",
  },
  {
    issue: "Tubelet 笛卡尔积",
    evidence: "10 帧产生 1,717,147,606 次相邻比较",
    change: "STRtree 候选过滤 + 原精确判定",
    effect: "稀疏测试 360,000 次降为 0 次",
  },
  {
    issue: "连通域重复扫描",
    evidence: "512 个组件触发 512 次整图扫描",
    change: "一次稳定分组",
    effect: "避免组件数 × 4K 像素的复杂度",
  },
  {
    issue: "Mask 清理重复扫描",
    evidence: "2,048 labels 逐个扫描整张标签图",
    change: "一次 keep[labels] 映射",
    effect: "真实帧由 5m29s 未完成降到 67s 完成四阈值",
  },
  {
    issue: "OBB 匹配全量 IoU",
    evidence: "80 × 80 需要 6,400 次 rotated IoU",
    change: "空间候选过滤与跨阈值复用",
    effect: "稀疏样例降到 80 次；多阈值不超过 5 次",
  },
  {
    issue: "指标整图反复取像素",
    evidence: "单组理论触碰约 3.95 TB 像素字节",
    change: "GT 局部 ROI + AABB 候选过滤",
    effect: "mask coverage 数据访问约减少 4,147.7 倍",
  },
] as const;

export const priorities = [
  {
    level: "P0",
    title: "先闭合评估链路",
    body: "完成 calibration；修正 evaluation 源标注中的非法 OBB，并保持严格策略 A：不拟合、不跳过、不容错。",
  },
  {
    level: "P1",
    title: "把误报压下来",
    body: "扩展 Z 阈值、按尺度扫描最小组件面积，加入持续帧数、速度一致性、道路 ROI 与植被/纹理背景抑制。",
  },
  {
    level: "P2",
    title: "改善 OBB 形状",
    body: "对预测连通域做稳健核心拟合，以轨迹方向作为角度先验，并跨多帧聚合尺寸与朝向。",
  },
  {
    level: "P3",
    title: "再做时序分类",
    body: "先把百万级 proposal 收敛成 tubelet，再用 9–17 帧 RGB + motion 双流网络完成类别判断与框细化。",
  },
  {
    level: "P4",
    title: "完善轨迹生命周期",
    body: "建立边界进入、遮挡、短时漏检、停止和离场状态，平滑速度、尺度、角度与类别置信度。",
  },
  {
    level: "P5",
    title: "继续做性能工程",
    body: "增加计算组级 checkpoint，缓存 ECC，控制尺度/方法并行，评估 GPU 加速，并记录运行时间、RSS 与磁盘。",
  },
] as const;

export const experimentMatrix = [
  ["E0", "当前完整 calibration", "建立五方法、两尺度基线"],
  ["E1", "更高 Z + 最小面积", "快速压制碎片 proposal"],
  ["E2", "持续性 + 速度一致性", "验证 tubelet 级目标性"],
  ["E3", "道路 ROI + 背景抑制", "清理树木、纹理和道路边缘"],
  ["E4", "稳健 OBB + 轨迹角度", "提升 Recall@0.50"],
  ["E5", "时序分类器", "区分交通参与者类别"],
  ["E6", "20 × 40 小目标子集", "最终验证目标尺度"],
] as const;
