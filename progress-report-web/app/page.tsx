import { LiveCalibration } from "./components/live-calibration";
import { EvidenceImage } from "./components/evidence-image";
import { PipelineStory } from "./components/pipeline-story";
import {
  calibrationRows,
  conditions,
  engineeringRows,
  experimentMatrix,
  methods,
  navigation,
  pipeline,
  priorities,
  smokeMetrics,
} from "./report-data";

function SectionHeading({
  kicker,
  title,
  summary,
}: {
  kicker: string;
  title: string;
  summary: string;
}) {
  return (
    <header className="section-heading">
      <p className="micro-label">{kicker}</p>
      <h2>{title}</h2>
      <p>{summary}</p>
    </header>
  );
}

export default function Home() {
  return (
    <div className="site-shell">
      <aside className="side-nav">
        <a className="brand" href="#top" aria-label="回到页面顶部">
          <span className="brand-mark" aria-hidden="true">
            M
          </span>
          <span>
            MOTION LAB
            <small>OBB · POC</small>
          </span>
        </a>

        <nav aria-label="报告目录">
          <p>报告目录</p>
          {navigation.map((item) => (
            <a href={item.href} key={item.href}>
              <span>{item.index}</span>
              {item.label}
            </a>
          ))}
        </nav>

        <div className="side-note">
          <span className="status-dot" aria-hidden="true" />
          <p>
            阶段快照
            <small>2026-08-05 18:15 CST</small>
          </p>
        </div>
      </aside>

      <main id="top">
        <section className="hero" id="overview">
          <div className="hero-orbit orbit-one" aria-hidden="true" />
          <div className="hero-orbit orbit-two" aria-hidden="true" />
          <div className="hero-copy">
            <p className="eyebrow">航拍视频 · 运动目标 · 旋转框追踪</p>
            <h1>
              从运动证据，
              <br />
              到可追踪的 <em>OBB</em>
            </h1>
            <p className="hero-lead">
              不是先逐帧检测再追踪。我们先把相邻帧的运动变化聚合成证据，
              再形成旋转框候选、tubelet、类别和完整轨迹。
            </p>
            <div className="hero-tags" aria-label="项目关键条件">
              <span>3840 × 2160</span>
              <span>30 FPS</span>
              <span>20 × 40 px 小目标</span>
              <span>离线处理</span>
            </div>
          </div>

          <LiveCalibration />
        </section>

        <section className="snapshot-grid" aria-label="项目阶段指标">
          <article>
            <span>工程测试</span>
            <strong>575</strong>
            <p>当前全部通过</p>
          </article>
          <article>
            <span>已完成计算组</span>
            <strong>3 / 8</strong>
            <p>报告快照时点</p>
          </article>
          <article>
            <span>运动方法</span>
            <strong>5</strong>
            <p>两个尺度公平比较</p>
          </article>
          <article className="alert-card">
            <span>最终评估</span>
            <strong>A</strong>
            <p>evaluation 标注阻塞</p>
          </article>
        </section>

        <section className="content-section overview-section">
          <SectionHeading
            kicker="01 / PROJECT"
            title="为什么从运动开始？"
            summary="单帧里的小车只有很少纹理，但在连续视频中，它的位移、方向和持续性都是额外证据。"
          />

          <div className="two-column">
            <article className="prose-card">
              <h3>核心判断</h3>
              <p>
                交通参与者通常从画面边缘或可见区域进入，不会无原因凭空出现或消失。
                相邻帧的位置、速度、面积和朝向应连续变化，短时漏检可以由轨迹状态维持。
              </p>
              <p>
                因此，本项目把运动证据作为候选生成的主要来源，
                单帧 RGB 只在候选已经被时序信息收敛后用于分类和 OBB 细化。
              </p>
              <div className="decision">
                <span>设计原则</span>
                <strong>运动负责发现，时间负责确认，外观负责分类。</strong>
              </div>
            </article>

            <article className="condition-card">
              <h3>固定条件</h3>
              <dl>
                {conditions.map(([term, value]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
            </article>
          </div>
        </section>

        <section className="content-section" id="method">
          <SectionHeading
            kicker="02 / METHOD"
            title="运动优先处理链"
            summary="31 帧时间窗口把短、中、长时间变化放到同一个证据空间，再逐步收敛成 tubelet。"
          />

          <div className="pipeline">
            {pipeline.map((step) => (
              <article key={step.number}>
                <span>{step.number}</span>
                <div>
                  <h3>{step.title}</h3>
                  <p>{step.description}</p>
                </div>
              </article>
            ))}
          </div>

          <div className="method-strip">
            <p>当前公平比较方法</p>
            <div>
              {methods.map((method) => (
                <code key={method}>{method}</code>
              ))}
            </div>
          </div>

          <PipelineStory />
        </section>

        <section className="content-section" id="results">
          <SectionHeading
            kicker="03 / EVIDENCE"
            title="关键实验结果"
            summary="这里同时保留负面结果和中间结果。工程跑通不等于检测效果达标。"
          />

          <article className="result-feature">
            <header>
              <div>
                <p className="micro-label">REAL 4K SMOKE</p>
                <h3>4K 十帧 smoke：mask 覆盖目标，但框几乎不可用</h3>
              </div>
              <span className="result-tag danger">未达约束</span>
            </header>
            <div className="smoke-layout">
              <dl className="metric-list">
                {smokeMetrics.map(([term, value]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
              <div className="interpretation">
                <span className="giant-number">96.81%</span>
                <p>Mean mask coverage 很高，说明运动响应确实“碰到了”目标。</p>
                <div className="down-arrow" aria-hidden="true">
                  ↓
                </div>
                <span className="giant-number amber">10,434</span>
                <p>
                  但每 100 个 GT 对应超过一万个误报。树木、道路边缘、建筑纹理和阴影把
                  mask 切成大量碎片 OBB。
                </p>
              </div>
            </div>
          </article>

          <article className="table-card">
            <div className="table-title">
              <div>
                <p className="micro-label">CALIBRATION · INTERIM</p>
                <h3>已完成的原生分辨率方法</h3>
              </div>
              <p>Primary moving GT = 35,775 · FP 约束 ≤ 25 / 100 GT</p>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>方法</th>
                    <th>阈值</th>
                    <th>R@.25</th>
                    <th>R@.50</th>
                    <th>中心命中</th>
                    <th>Mask 覆盖</th>
                    <th>Proposal</th>
                    <th>FP / 100 GT</th>
                  </tr>
                </thead>
                <tbody>
                  {calibrationRows.map((row) => (
                    <tr key={row.method}>
                      <th>{row.method}</th>
                      <td>{row.threshold}</td>
                      <td>{row.recall25}</td>
                      <td>{row.recall50}</td>
                      <td>{row.center}</td>
                      <td>{row.coverage}</td>
                      <td>{row.proposals}</td>
                      <td className="bad-value">{row.fp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <div className="key-conclusion">
            <p className="micro-label">WHAT THE NUMBERS SAY</p>
            <h3>目标大多在响应附近，问题出在“目标性”和“框形状”。</h3>
            <p>
              三种方法的中心命中都接近 100%，但 Recall@0.50 很低；
              这说明当前候选常常靠近车辆，却被背景碎片拉成了错误大小和方向的 OBB。
              下一步不应直接给百万级 proposal 加分类器，而要先利用时序一致性强力剪枝。
            </p>
          </div>
        </section>

        <section className="content-section" id="engineering">
          <SectionHeading
            kicker="04 / ENGINEERING"
            title="已经解决的性能瓶颈"
            summary="真实 4K 视频把隐藏的平方级和“组件数 × 全图”复杂度全部暴露了出来。"
          />

          <div className="engineering-list">
            {engineeringRows.map((row, index) => (
              <article key={row.issue}>
                <span className="row-index">{String(index + 1).padStart(2, "0")}</span>
                <div>
                  <h3>{row.issue}</h3>
                  <p>{row.evidence}</p>
                </div>
                <div>
                  <span>修复</span>
                  <p>{row.change}</p>
                </div>
                <div>
                  <span>结果</span>
                  <p>{row.effect}</p>
                </div>
              </article>
            ))}
          </div>

          <div className="before-after">
            <div>
              <span>旧版本</span>
              <strong>&gt; 3h</strong>
              <p>十帧 smoke 被中断，无法完成</p>
            </div>
            <div className="motion-line" aria-hidden="true">
              <span />
            </div>
            <div>
              <span>当前版本</span>
              <strong>39:58</strong>
              <p>完整输出 artifact 和指标</p>
            </div>
          </div>
        </section>

        <section className="content-section" id="risks">
          <SectionHeading
            kicker="05 / BOUNDARIES"
            title="当前能说什么，不能说什么"
            summary="所有结论都绑定到对应证据范围，避免把阶段性工程成果包装成最终算法效果。"
          />

          <div className="risk-grid">
            <article className="can-say">
              <span className="risk-symbol">✓</span>
              <h3>已经得到证据支持</h3>
              <ul>
                <li>视频运动信息能够生成跨帧 OBB 候选。</li>
                <li>真实 4K 主链路、artifact 和指标已经跑通。</li>
                <li>当前主要瓶颈是背景误报与 OBB 形状，而非没有响应。</li>
                <li>Tasks 1–10 已实现并通过 575 项测试。</li>
              </ul>
            </article>
            <article className="cannot-say">
              <span className="risk-symbol">!</span>
              <h3>目前不能宣称</h3>
              <ul>
                <li>不能宣称当前检测器已经满足 FP 约束。</li>
                <li>
                  多尺度与现有 Tubelet 已完成，但当前结果明显未达约束；0.7
                  尺度帧差法是目前最有价值的降噪方向。
                </li>
                <li>不能给出 frozen evaluation 的六个最终 gate。</li>
                <li>不能把当前车辆序列等同于最终 20 × 40 小目标 benchmark。</li>
              </ul>
            </article>
          </div>

          <div className="blocker">
            <span>DATA BLOCKER</span>
            <p>
              <strong>motorway_sequence2 / 000001.json</strong>{" "}
              中存在不满足严格矩形定义的 OBB。按照你选择的策略 A，
              系统明确失败，不拟合、不修补、不跳过；源标注修正后才能执行最终 evaluation。
            </p>
          </div>
        </section>

        <section className="content-section" id="next">
          <SectionHeading
            kicker="06 / ROADMAP"
            title="下一轮优化优先级"
            summary="先把运动响应收敛成可信 tubelet，再投入分类模型；顺序本身就是最重要的优化。"
          />

          <div className="priority-grid">
            {priorities.map((priority) => (
              <article key={priority.level}>
                <span>{priority.level}</span>
                <h3>{priority.title}</h3>
                <p>{priority.body}</p>
              </article>
            ))}
          </div>

          <article className="experiment-card">
            <header>
              <p className="micro-label">EXPERIMENT MATRIX</p>
              <h3>建议按单变量证据链推进</h3>
            </header>
            <div className="experiment-list">
              {experimentMatrix.map(([id, change, target]) => (
                <div key={id}>
                  <code>{id}</code>
                  <strong>{change}</strong>
                  <span>{target}</span>
                </div>
              ))}
            </div>
          </article>
        </section>

        <section className="content-section evidence-section" id="evidence">
          <SectionHeading
            kicker="07 / ARTIFACTS"
            title="证据与复现入口"
            summary="页面展示的是阶段报告的可读版本，原始图像、Markdown 与配置仍保留为可审查证据。"
          />

          <figure className="evidence-figure">
            <EvidenceImage />
            <figcaption>
              三帧对比图：上一帧 / 当前帧标注 / 下一帧
            </figcaption>
          </figure>

          <div className="artifact-links">
            <a
              href="/evidence/comparison-original.png"
              target="_blank"
              rel="noreferrer"
            >
              <span>4K</span>
              <div>
                <strong>打开 41 MiB 原始对比图</strong>
                <small>3840 × 6480 PNG · 按需加载</small>
              </div>
              <b aria-hidden="true">↗</b>
            </a>
            <a href="/evidence/report.md" target="_blank" rel="noreferrer">
              <span>MD</span>
              <div>
                <strong>打开完整阶段报告</strong>
                <small>
                  motion-evidence-poc-progress-report-2026-08-05.md
                </small>
              </div>
              <b aria-hidden="true">↗</b>
            </a>
            <div className="path-card">
              <span>RUN</span>
              <div>
                <strong>Calibration staging</strong>
                <code>
                  /home/stu1/Projects/moving_Det/.worktrees/motion-evidence-poc/runs/poc-calibration
                </code>
              </div>
            </div>
          </div>
        </section>

        <footer>
          <p>
            <strong>一句话结论：</strong>
            运动优先在工程上可行，但低阈值运动连通域会产生不可接受的背景误报；
            下一阶段要用持续性、速度一致性、背景抑制和时序分类，把响应收敛成真正的交通参与者
            tubelet。
          </p>
          <span>Motion Evidence POC · Progress Report</span>
        </footer>
      </main>
    </div>
  );
}
