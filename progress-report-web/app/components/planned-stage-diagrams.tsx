export function TemporalClassifierDiagram() {
  return (
    <div
      className="planned-diagram classifier-diagram"
      role="img"
      aria-label="规划中的时序分类：多帧RGB与运动裁剪进入双流模型，输出类别和细化旋转框"
    >
      <span className="plan-watermark">预期输出</span>
      <div className="classifier-inputs">
        <article className="frame-stack">
          <span aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <strong>9–17 帧 RGB</strong>
          <small>外观、颜色与目标细节</small>
        </article>
        <article className="frame-stack motion-stack">
          <span aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <strong>Motion crop</strong>
          <small>位移、方向与持续性</small>
        </article>
      </div>
      <span className="diagram-arrow" aria-hidden="true">
        →
      </span>
      <div className="diagram-output">
        <span>双流时序模型</span>
        <strong>类别判断</strong>
        <b aria-hidden="true">+</b>
        <strong>OBB 细化</strong>
      </div>
    </div>
  );
}

const lifecycleStates = [
  { name: "进入", note: "边界出现" },
  { name: "确认", note: "连续观测" },
  { name: "短时漏检 / 遮挡", note: "保留预测状态", predicted: true },
  { name: "恢复", note: "重新匹配" },
  { name: "停车", note: "运动弱但轨迹仍在" },
  { name: "离场", note: "边界驶出后终止" },
] as const;

export function TrackLifecycleDiagram() {
  return (
    <div
      className="planned-diagram lifecycle-diagram"
      role="img"
      aria-label="规划中的轨迹生命周期：进入、确认、短时漏检或遮挡、恢复、停车和离场"
    >
      <span className="plan-watermark">状态机制</span>
      <div className="lifecycle-track" aria-hidden="true" />
      {lifecycleStates.map((state, index) => (
        <article
          className={`lifecycle-state${state.predicted ? " predicted" : ""}`}
          key={state.name}
        >
          <span>{String(index + 1).padStart(2, "0")}</span>
          <strong>{state.name}</strong>
          <small>{state.note}</small>
        </article>
      ))}
    </div>
  );
}
