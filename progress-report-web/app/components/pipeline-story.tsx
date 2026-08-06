import { pipelineStory } from "../pipeline-story-data";
import { PipelineVisual } from "./pipeline-visual";

export function PipelineStory() {
  return (
    <div className="pipeline-story" aria-labelledby="pipeline-story-title">
      <header className="pipeline-story-heading">
        <p className="micro-label">SAME SCENE · REAL EVIDENCE</p>
        <h3 id="pipeline-story-title">同一场景，逐步收敛</h3>
        <p>
          固定第 20 帧和同一块道路区域，前四步展示真实产物，
          后两步只说明规划机制。
        </p>
      </header>

      <ol>
        {pipelineStory.map((stage, stageIndex) => (
          <li
            className={`pipeline-story-step${stage.negative ? " stage-judgement-negative" : ""}`}
            key={stage.number}
          >
            <div className="pipeline-story-copy">
              <header>
                <span className="story-number">{stage.number}</span>
                <span
                  className={`stage-status stage-status-${stage.status}`}
                >
                  {stage.status === "real" ? "真实结果" : "规划中"}
                </span>
              </header>
              <h4>{stage.title}</h4>
              <p className="story-question">{stage.question}</p>
              <p className="story-answer">{stage.answer}</p>

              <div className="story-flow">
                <p>输入 → 处理 → 输出</p>
                <dl>
                  <div>
                    <dt>输入</dt>
                    <dd>{stage.inputs}</dd>
                  </div>
                  <div>
                    <dt>处理</dt>
                    <dd>{stage.process}</dd>
                  </div>
                  <div>
                    <dt>输出</dt>
                    <dd>{stage.output}</dd>
                  </div>
                </dl>
              </div>

              <dl className="story-values">
                {stage.value.map((item) => (
                  <div key={`${item.label}-${item.value}`}>
                    <dt>{item.label}</dt>
                    <dd>{item.value}</dd>
                  </div>
                ))}
              </dl>

              <p className="story-judgement">{stage.judgement}</p>
            </div>

            {stage.visual.kind === "evidence" ? (
              <PipelineVisual
                layers={stage.visual.layers}
                caption={stage.title}
                eager={stageIndex === 0}
              />
            ) : (
              <div className="planned-placeholder" role="img">
                <span>机制示意</span>
                <strong>
                  {stage.visual.kind === "classifier-plan"
                    ? "时序分类机制"
                    : "轨迹生命周期"}
                </strong>
              </div>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
