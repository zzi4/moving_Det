"use client";

import { useState } from "react";

const previewRoute = "/evidence/comparison.webp";
const sourcePath =
  "/home/stu1/Projects/moving_Det/.worktrees/motion-evidence-poc/runs/smoke/overlays/comparison.png";

export function EvidenceImage() {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="evidence-fallback" role="status">
        <strong>证据预览图加载失败</strong>
        <p>可以直接打开原图，或检查下面的源文件绝对路径。</p>
        <code>{previewRoute}</code>
        <code>{sourcePath}</code>
      </div>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={previewRoute}
      alt="上一帧、当前帧及下一帧的标注可视化对比"
      onError={() => setFailed(true)}
    />
  );
}
