function formatNumber(value, digits = 2) {
  if (value === null || value === undefined) {
    return "--";
  }

  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return "--";
  }

  return numeric.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}


function formatInteger(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: 0,
  });
}


function MetricCard({
  label,
  value,
  unit,
  emphasis = false,
}) {
  return (
    <div
      className={
        emphasis
          ? "overview-metric-card emphasis"
          : "overview-metric-card"
      }
    >
      <span>{label}</span>

      <strong>{value}</strong>

      {unit && (
        <small>{unit}</small>
      )}
    </div>
  );
}


function PhasePanel({
  type,
  title,
  subtitle,
  countLabel,
  countValue,
  children,
  footer,
}) {
  return (
    <section className={`phase-panel ${type}`}>
      <div className="phase-panel-header">
        <div>
          <div className="phase-kicker">
            {type === "prefill"
              ? "预填充 / PREFILL"
              : "解码 / DECODE"}
          </div>

          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>

        <div className="phase-count">
          <span>{countLabel}</span>
          <strong>{formatInteger(countValue)}</strong>
        </div>
      </div>

      {children}

      {footer && (
        <div className="phase-footer">
          {footer}
        </div>
      )}
    </section>
  );
}


function OverviewDashboard({
  phaseSummary,
  hardware,
}) {
  const prefill =
    phaseSummary?.prefill ?? {};

  const decode =
    phaseSummary?.decode ?? {};

  const prefillLatency =
    prefill.latency_cycles ?? {};

  const prefillPerToken =
    prefill.cycles_per_input_token ?? {};

  const promptLength =
    prefill.prompt_length ?? {};

  const decodeLatency =
    decode.cycles_per_token ?? {};

  return (
    <div className="overview-dashboard">
      <div className="overview-header">
        <div>
          <div className="overview-small-title">
            当前评估 / CURRENT EVALUATION
          </div>

          <h2>系统总览 / Overview</h2>

          <p>
            当前 Mapping 下的正式 Prefill / Decode 阶段评估结果。
          </p>
        </div>

        <div className="scope-badge">
          仅 MoE Expert / Expert Only
        </div>
      </div>

      <div className="scope-notice">
        <strong>指标范围：</strong>
        这里只统计 58 个 MoE 层的 Expert gate / up / down 计算与调度。
        Prefill 不是完整 TTFT，Decode 也不是完整 TPOT。
      </div>

      <div className="overview-phase-grid">
        <PhasePanel
          type="prefill"
          title="预填充 / MoE Prefill"
          subtitle="按 Prompt Batch 逐层执行多个输入 Token。"
          countLabel="请求 Batch"
          countValue={prefill.batch_count}
          footer={
            `输入 Token ${formatInteger(prefill.total_input_tokens)} · `
            + `平均 Prompt 长度 ${formatNumber(promptLength.mean, 2)} tokens · `
            + `P95 长度 ${formatNumber(promptLength.p95, 0)} tokens`
          }
        >
          <div className="phase-primary-metric">
            <span>平均 Batch 延迟 / Mean Batch Latency</span>
            <strong>
              {formatNumber(prefillLatency.mean, 2)}
            </strong>
            <small>cycles</small>
          </div>

          <div className="phase-metric-grid">
            <MetricCard
              label="P50 延迟"
              value={formatNumber(prefillLatency.p50, 2)}
              unit="cycles"
            />

            <MetricCard
              label="P95 延迟"
              value={formatNumber(prefillLatency.p95, 2)}
              unit="cycles"
            />

            <MetricCard
              label="P99 延迟"
              value={formatNumber(prefillLatency.p99, 2)}
              unit="cycles"
            />

            <MetricCard
              label="每输入 Token 平均周期"
              value={formatNumber(prefillPerToken.mean, 2)}
              unit="cycles/token"
              emphasis
            />
          </div>
        </PhasePanel>

        <PhasePanel
          type="decode"
          title="解码 / MoE Decode"
          subtitle="逐 Token 通过 58 个 MoE 层，统计 Expert 部分周期。"
          countLabel="Decode Token"
          countValue={decode.token_count}
          footer={
            `调度器 ${decode.scheduler_mode ?? "--"} · `
            + `Exact 校验 ${formatInteger(decode.exact_checked_tokens)} tokens`
          }
        >
          <div className="phase-primary-metric">
            <span>平均 Decode 延迟 / Mean Decode Latency</span>
            <strong>
              {formatNumber(decodeLatency.mean, 2)}
            </strong>
            <small>cycles/token</small>
          </div>

          <div className="phase-metric-grid">
            <MetricCard
              label="P50 延迟"
              value={formatNumber(decodeLatency.p50, 0)}
              unit="cycles/token"
            />

            <MetricCard
              label="P95 延迟"
              value={formatNumber(decodeLatency.p95, 0)}
              unit="cycles/token"
            />

            <MetricCard
              label="P99 延迟"
              value={formatNumber(decodeLatency.p99, 0)}
              unit="cycles/token"
            />

            <MetricCard
              label="最大值 / Max"
              value={formatNumber(decodeLatency.maximum, 0)}
              unit="cycles/token"
              emphasis
            />
          </div>
        </PhasePanel>
      </div>

      <section className="overview-context-panel">
        <div className="overview-context-heading">
          <div>
            <div className="overview-small-title">
              当前硬件 / CURRENT HARDWARE
            </div>
            <h3>映射与硬件配置 / Mapping Context</h3>
          </div>

          <span>
            {hardware.N ?? "--"} × {hardware.N ?? "--"} Sub-Cube 拓扑
          </span>
        </div>

        <div className="overview-context-grid">
          <div>
            <span>Sub-Cube 数</span>
            <strong>{hardware.num_subcubes ?? "--"}</strong>
          </div>

          <div>
            <span>Plane 尺寸</span>
            <strong>
              {hardware.H ?? "--"} × {hardware.W ?? "--"}
            </strong>
          </div>

          <div>
            <span>深度 D</span>
            <strong>{hardware.D ?? "--"}</strong>
          </div>

          <div>
            <span>Weight-Cube 数</span>
            <strong>{formatInteger(hardware.weight_cube_count)}</strong>
          </div>

          <div>
            <span>已用 Plane</span>
            <strong>{formatInteger(hardware.used_planes)}</strong>
          </div>

          <div>
            <span>空闲 Plane</span>
            <strong>{formatInteger(hardware.empty_plane_slots)}</strong>
          </div>
        </div>
      </section>

      <Style />
    </div>
  );
}


function Style() {
  return (
    <style>
      {`
        .overview-dashboard {
          width: 100%;
        }

        .overview-header {
          min-height: 54px;
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 20px;
          margin-bottom: 10px;
        }

        .overview-small-title {
          margin-bottom: 5px;
          color: #405a73;
          font-size: 15px;
          font-weight: 700;
          letter-spacing: 1.1px;
        }

        .overview-header h2 {
          margin: 0 0 5px;
          color: #102a43;
          font-size: 28px;
          font-weight: 680;
        }

        .overview-header p {
          margin: 0;
          color: #526579;
          font-size: 15px;
        }

        .scope-badge {
          padding: 7px 11px;
          border: 1px solid #60a5fa;
          border-radius: 5px;
          background: #dbeafe;
          color: #174f7d;
          font-size: 15px;
          font-weight: 700;
          letter-spacing: 0.5px;
          white-space: nowrap;
        }

        .scope-notice {
          margin-bottom: 12px;
          padding: 10px 12px;
          border: 1px solid #f2c879;
          border-radius: 6px;
          background: #fff8e8;
          color: #526579;
          font-size: 15px;
          line-height: 1.65;
        }

        .scope-notice strong {
          color: #8a4b08;
        }

        .overview-phase-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }

        .phase-panel {
          padding: 14px;
          border: 1px solid #c3d0dc;
          border-radius: 7px;
          background: #ffffff;
          min-width: 0;
        }

        .phase-panel.prefill {
          border-top: 5px solid #4F7195;
        }

        .phase-panel.decode {
          border-top: 5px solid #0f766e;
        }


        .phase-panel.prefill .phase-primary-metric {
          border-left: 5px solid #4F7195;
          background: #eff6ff;
        }

        .phase-panel.decode .phase-primary-metric {
          border-left: 5px solid #0f766e;
          background: #ecfdf5;
        }

        .phase-panel.prefill .phase-count {
          background: #dbeafe;
        }

        .phase-panel.decode .phase-count {
          background: #dff7ef;
        }

        .phase-panel.prefill .phase-kicker {
          color: #1d4ed8;
        }

        .phase-panel.decode .phase-kicker {
          color: #0f766e;
        }

        .phase-panel-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
        }

        .phase-kicker {
          margin-bottom: 5px;
          color: #526579;
          font-size: 15px;
          font-weight: 750;
          letter-spacing: 1.2px;
        }

        .phase-panel h3 {
          margin: 0 0 5px;
          color: #173b5f;
          font-size: 23px;
        }

        .phase-panel p {
          margin: 0;
          color: #5f7083;
          font-size: 15px;
          line-height: 1.55;
        }

        .phase-count {
          min-width: 88px;
          padding: 8px 10px;
          border-radius: 5px;
          background: #eaf2f8;
          text-align: right;
        }

        .phase-count span {
          display: block;
          margin-bottom: 3px;
          color: #5f7083;
          font-size: 15px;
        }

        .phase-count strong {
          color: #173b5f;
          font-size: 23px;
        }

        .phase-primary-metric {
          margin: 14px 0 10px;
          padding: 12px 14px;
          border: 1px solid #b8c8d8;
          border-radius: 6px;
          background: #f4f8fc;
        }

        .phase-primary-metric > span {
          display: block;
          margin-bottom: 7px;
          color: #49627b;
          font-size: 15px;
        }

        .phase-primary-metric strong {
          color: #0b3f6d;
          font-size: 34px;
          line-height: 1;
          font-variant-numeric: tabular-nums;
        }

        .phase-primary-metric small {
          margin-left: 7px;
          color: #49627b;
          font-size: 15px;
        }

        .phase-metric-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 8px;
        }

        .overview-metric-card {
          min-width: 0;
          padding: 10px;
          border: 1px solid #cbd5e1;
          border-radius: 5px;
          background: #ffffff;
        }

        .overview-metric-card.emphasis {
          background: #e8f2fc;
        }

        .overview-metric-card span {
          display: block;
          min-height: 38px;
          margin-bottom: 4px;
          color: #526579;
          font-size: 15px;
          line-height: 1.4;
        }

        .overview-metric-card strong {
          display: block;
          overflow: hidden;
          color: #123f70;
          font-size: 22px;
          font-variant-numeric: tabular-nums;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .overview-metric-card small {
          display: block;
          margin-top: 3px;
          color: #64748b;
          font-size: 15px;
        }

        .phase-footer {
          margin-top: 10px;
          padding-top: 9px;
          border-top: 1px solid #edf0f2;
          color: #5f7083;
          font-size: 15px;
          line-height: 1.5;
        }

        .overview-context-panel {
          margin-top: 12px;
          padding: 14px;
          border: 1px solid #c3d0dc;
          border-radius: 7px;
          background: #ffffff;
        }

        .overview-context-heading {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 12px;
        }

        .overview-context-heading h3 {
          margin: 0;
          color: #173b5f;
          font-size: 22px;
        }

        .overview-context-heading > span {
          color: #526579;
          font-size: 15px;
        }

        .overview-context-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 8px;
        }

        .overview-context-grid > div {
          padding: 9px 10px;
          border: 1px solid #e5e8eb;
          border-radius: 5px;
          background: #f4f8fc;
        }

        .overview-context-grid span {
          display: block;
          margin-bottom: 5px;
          color: #526579;
          font-size: 15px;
        }

        .overview-context-grid strong {
          color: #173b5f;
          font-size: 15px;
        }

        @media (max-width: 1180px) {
          .overview-phase-grid {
            grid-template-columns: 1fr;
          }

          .overview-context-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
        }
      `}
    </style>
  );
}


export default OverviewDashboard;
