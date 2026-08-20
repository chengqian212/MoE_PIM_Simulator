import {
  useMemo,
} from "react";


function formatNumber(value, digits = 2) {
  if (value === null || value === undefined) {
    return "--";
  }

  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return "--";
  }

  return numeric.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}


function formatInteger(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  return Number(value).toLocaleString("zh-CN", {
    maximumFractionDigits: 0,
  });
}


function percentChange(high, base) {
  const a = Number(high);
  const b = Number(base);

  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) {
    return "--";
  }

  return `${(((a - b) / b) * 100).toFixed(1)}%`;
}


function ResultMetric({
  label,
  value,
  unit,
  main = false,
}) {
  return (
    <div className={main ? "result-metric main" : "result-metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
      {unit && <small>{unit}</small>}
    </div>
  );
}


function DistributionBars({
  title,
  subtitle,
  items,
  unit,
  digits = 0,
}) {
  const maxValue = Math.max(
    ...items.map((item) => Number(item.value) || 0),
    1,
  );

  return (
    <section className="analysis-card distribution-card">
      <div className="analysis-card-header">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>

      <div className="distribution-bars">
        {items.map((item) => {
          const numeric = Number(item.value) || 0;
          const width = Math.max((numeric / maxValue) * 100, numeric > 0 ? 2 : 0);

          return (
            <div className="distribution-row" key={item.label}>
              <span className="distribution-label">{item.label}</span>

              <div className="distribution-track">
                <div
                  className={`distribution-fill ${item.kind ?? ""}`}
                  style={{ width: `${width}%` }}
                />
              </div>

              <strong>
                {formatNumber(item.value, digits)}
                <small>{unit}</small>
              </strong>
            </div>
          );
        })}
      </div>
    </section>
  );
}


function FixedResults({
  phaseSummary,
  hardware,
}) {
  const prefill = phaseSummary?.prefill ?? {};
  const decode = phaseSummary?.decode ?? {};

  const prompt = prefill.prompt_length ?? {};
  const prefillLatency = prefill.latency_cycles ?? {};
  const prefillPerToken = prefill.cycles_per_input_token ?? {};
  const decodeLatency = decode.cycles_per_token ?? {};

  const conclusions = useMemo(() => {
    const pearson = Number(prefill.prompt_length_latency_pearson);
    const decodeMean = Number(decodeLatency.mean);
    const decodeP95 = Number(decodeLatency.p95);
    const decodeMax = Number(decodeLatency.maximum);

    return [
      {
        title: "Prefill 长度相关性",
        value: Number.isFinite(pearson) ? pearson.toFixed(4) : "--",
        text: "Prompt 长度与 Prefill Batch latency 高度相关；该值描述相关性，不直接等同于因果关系。",
      },
      {
        title: "Prefill 单 Token 成本",
        value: `${formatNumber(prefillPerToken.mean, 2)} cycles/token`,
        text: `全局累计口径为 ${formatNumber(prefill.global_cycles_per_input_token, 2)} cycles/token。`,
      },
      {
        title: "Decode P95 尾部",
        value: Number.isFinite(decodeMean) && Number.isFinite(decodeP95)
          ? `+${percentChange(decodeP95, decodeMean)}`
          : "--",
        text: `P95=${formatNumber(decodeP95, 0)}，相对 Mean=${formatNumber(decodeMean, 2)} 的增幅。`,
      },
      {
        title: "Decode 最大延迟",
        value: Number.isFinite(decodeMean) && Number.isFinite(decodeMax)
          ? `+${percentChange(decodeMax, decodeMean)}`
          : "--",
        text: `Max=${formatNumber(decodeMax, 0)} cycles/token，用于观察当前 Trace 中最慢的 Decode Token。`,
      },
    ];
  }, [
    decodeLatency.maximum,
    decodeLatency.mean,
    decodeLatency.p95,
    prefill.global_cycles_per_input_token,
    prefill.prompt_length_latency_pearson,
    prefillPerToken.mean,
  ]);

  return (
    <div className="fixed-results">
      <div className="results-scope-banner">
        <strong>评估范围 / Scope：</strong>
        仅统计 58 个 MoE 层 Expert 的 Gate / Up / Down 计算与调度。
        Prefill 不是完整 TTFT，Decode 不是完整 TPOT。
      </div>

      <div className="phase-result-grid">
        <section className="phase-result-panel prefill">
          <div className="phase-result-heading">
            <div>
              <span className="phase-result-kicker">PREFILL</span>
              <h3>预填充阶段 / Prefill</h3>
            </div>

            <div className="sample-count">
              <span>Batch 数</span>
              <strong>{formatInteger(prefill.batch_count)}</strong>
            </div>
          </div>

          <div className="phase-result-primary">
            <ResultMetric
              main
              label="平均 Batch 延迟"
              value={formatNumber(prefillLatency.mean, 2)}
              unit="cycles"
            />

            <ResultMetric
              main
              label="平均输入 Token 成本"
              value={formatNumber(prefillPerToken.mean, 2)}
              unit="cycles/token"
            />
          </div>

          <div className="phase-result-mini-grid">
            <ResultMetric
              label="输入 Token 总数"
              value={formatInteger(prefill.total_input_tokens)}
              unit="tokens"
            />

            <ResultMetric
              label="Prompt 平均长度"
              value={formatNumber(prompt.mean, 2)}
              unit="tokens"
            />

            <ResultMetric
              label="P95 Prompt 长度"
              value={formatNumber(prompt.p95, 0)}
              unit="tokens"
            />

            <ResultMetric
              label="最大 Prompt 长度"
              value={formatNumber(prompt.maximum, 0)}
              unit="tokens"
            />
          </div>
        </section>

        <section className="phase-result-panel decode">
          <div className="phase-result-heading">
            <div>
              <span className="phase-result-kicker">DECODE</span>
              <h3>解码阶段 / Decode</h3>
            </div>

            <div className="sample-count">
              <span>Token 数</span>
              <strong>{formatInteger(decode.token_count)}</strong>
            </div>
          </div>

          <div className="phase-result-primary">
            <ResultMetric
              main
              label="平均 Decode 延迟"
              value={formatNumber(decodeLatency.mean, 2)}
              unit="cycles/token"
            />

            <ResultMetric
              main
              label="P95 Decode 延迟"
              value={formatNumber(decodeLatency.p95, 0)}
              unit="cycles/token"
            />
          </div>

          <div className="phase-result-mini-grid">
            <ResultMetric
              label="最小值 / Min"
              value={formatNumber(decodeLatency.minimum, 0)}
              unit="cycles/token"
            />

            <ResultMetric
              label="中位数 / P50"
              value={formatNumber(decodeLatency.p50, 0)}
              unit="cycles/token"
            />

            <ResultMetric
              label="P99"
              value={formatNumber(decodeLatency.p99, 0)}
              unit="cycles/token"
            />

            <ResultMetric
              label="最大值 / Max"
              value={formatNumber(decodeLatency.maximum, 0)}
              unit="cycles/token"
            />
          </div>
        </section>
      </div>

      <div className="distribution-grid">
        <DistributionBars
          title="Prefill Batch 延迟分位数"
          subtitle="不同 Prompt 长度下，Batch 总周期不能直接与 Decode cycles/token 横向比较。"
          items={[
            { label: "Min", value: prefillLatency.minimum, kind: "prefill" },
            { label: "P50", value: prefillLatency.p50, kind: "prefill" },
            { label: "Mean", value: prefillLatency.mean, kind: "prefill" },
            { label: "P95", value: prefillLatency.p95, kind: "prefill" },
            { label: "P99", value: prefillLatency.p99, kind: "prefill" },
            { label: "Max", value: prefillLatency.maximum, kind: "prefill" },
          ]}
          unit="cycles"
          digits={0}
        />

        <DistributionBars
          title="Decode 单 Token 延迟分位数"
          subtitle="同一 Mapping 下 255,710 个 Decode Token 的 MoE Expert 部分周期。"
          items={[
            { label: "Min", value: decodeLatency.minimum, kind: "decode" },
            { label: "P50", value: decodeLatency.p50, kind: "decode" },
            { label: "Mean", value: decodeLatency.mean, kind: "decode" },
            { label: "P95", value: decodeLatency.p95, kind: "decode" },
            { label: "P99", value: decodeLatency.p99, kind: "decode" },
            { label: "Max", value: decodeLatency.maximum, kind: "decode" },
          ]}
          unit="cycles/token"
          digits={0}
        />
      </div>

      <section className="analysis-card prompt-card">
        <div className="analysis-card-header">
          <div>
            <h3>Prefill Prompt 长度统计</h3>
            <p>用于解释为什么不同 Prefill Batch 的总周期差异较大。</p>
          </div>

          <div className="pearson-badge">
            <span>长度-延迟 Pearson</span>
            <strong>{formatNumber(prefill.prompt_length_latency_pearson, 4)}</strong>
          </div>
        </div>

        <div className="prompt-stat-grid">
          <ResultMetric label="最短" value={formatNumber(prompt.minimum, 0)} unit="tokens" />
          <ResultMetric label="P50" value={formatNumber(prompt.p50, 0)} unit="tokens" />
          <ResultMetric label="平均" value={formatNumber(prompt.mean, 2)} unit="tokens" />
          <ResultMetric label="P95" value={formatNumber(prompt.p95, 0)} unit="tokens" />
          <ResultMetric label="P99" value={formatNumber(prompt.p99, 0)} unit="tokens" />
          <ResultMetric label="最长" value={formatNumber(prompt.maximum, 0)} unit="tokens" />
        </div>
      </section>

      <section className="analysis-card conclusion-card">
        <div className="analysis-card-header">
          <div>
            <h3>当前结果要点 / Key Findings</h3>
            <p>这里只总结当前 Mapping 和当前 Trace 得到的直接结果，不代表不同 Mapping 之间的优劣。</p>
          </div>
        </div>

        <div className="conclusion-grid">
          {conclusions.map((item) => (
            <div className="conclusion-item" key={item.title}>
              <span>{item.title}</span>
              <strong>{item.value}</strong>
              <p>{item.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="analysis-card hardware-strip-card">
        <div className="analysis-card-header compact">
          <div>
            <h3>当前硬件与 Mapping 上下文</h3>
          </div>
        </div>

        <div className="hardware-strip">
          <ResultMetric label="拓扑" value={`${hardware.N ?? "--"} × ${hardware.N ?? "--"}`} unit="Sub-Cube" />
          <ResultMetric label="Sub-Cube" value={formatInteger(hardware.num_subcubes)} />
          <ResultMetric label="Plane 尺寸" value={`${hardware.H ?? "--"} × ${hardware.W ?? "--"}`} />
          <ResultMetric label="深度 D" value={formatInteger(hardware.D)} />
          <ResultMetric label="Weight-Cube" value={formatInteger(hardware.weight_cube_count)} />
          <ResultMetric label="空 Plane" value={formatInteger(hardware.empty_plane_slots)} />
        </div>
      </section>
    </div>
  );
}


function ResultsAnalysis({
  phaseSummary,
  hardware,
}) {
  return (
    <div className="results-analysis">
      <div className="results-analysis-header">
        <div>
          <div className="results-small-title">PERFORMANCE ANALYSIS</div>
          <h2>结果分析 / Results Analysis</h2>
          <p>仅展示当前 Mapping 的正式 Prefill / Decode 评估结果；旧版 Token 抽样 Workload 不再作为正式实验入口。</p>
        </div>

        <div className="results-header-badge">MoE Expert Only</div>
      </div>

      <FixedResults
        phaseSummary={phaseSummary}
        hardware={hardware}
      />

      <Style />
    </div>
  );
}


function Style() {
  return (
    <style>
      {`
        .results-analysis {
          width: 100%;
          color: #364152;
        }

        .results-analysis-header {
          min-height: 58px;
          margin-bottom: 10px;
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 20px;
        }

        .results-small-title {
          margin-bottom: 4px;
          color: #405a73;
          font-size: 16px;
          font-weight: 750;
          letter-spacing: 1px;
        }

        .results-analysis-header h2 {
          margin: 0 0 5px;
          color: #2f3b47;
          font-size: 25px;
          font-weight: 700;
        }

        .results-analysis-header p {
          margin: 0;
          color: #526579;
          font-size: 15px;
          line-height: 1.5;
        }

        .results-header-badge {
          padding: 7px 11px;
          border: 1px solid #cbd7e2;
          border-radius: 5px;
          background: #ffffff;
          color: #54718d;
          font-size: 16px;
          font-weight: 700;
          white-space: nowrap;
        }

        .results-scope-banner {
          margin-bottom: 10px;
          padding: 9px 12px;
          border: 1px solid #d8e2eb;
          border-radius: 6px;
          background: #f7fafc;
          color: #647585;
          font-size: 16px;
          line-height: 1.55;
        }

        .results-scope-banner strong {
          color: #40576c;
        }

        .phase-result-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
        }

        .phase-result-panel {
          padding: 13px;
          border: 1px solid #dfe5ea;
          border-radius: 7px;
          background: #ffffff;
        }

        .phase-result-panel.prefill {
          border-top: 4px solid #7d9bb8;
        }

        .phase-result-panel.decode {
          border-top: 4px solid #759b86;
        }

        .phase-result-heading {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 15px;
          margin-bottom: 10px;
        }

        .phase-result-kicker {
          color: #8a96a1;
          font-size: 15px;
          font-weight: 750;
          letter-spacing: 1px;
        }

        .phase-result-heading h3 {
          margin: 3px 0 0;
          color: #354250;
          font-size: 19px;
        }

        .sample-count {
          min-width: 105px;
          text-align: right;
        }

        .sample-count span {
          display: block;
          margin-bottom: 2px;
          color: #8d98a3;
          font-size: 16px;
        }

        .sample-count strong {
          color: #3e5060;
          font-size: 20px;
        }

        .phase-result-primary {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          margin-bottom: 8px;
        }

        .phase-result-mini-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 7px;
        }

        .result-metric {
          min-width: 0;
          padding: 9px 10px;
          border: 1px solid #e2e7eb;
          border-radius: 5px;
          background: #fafbfc;
        }

        .result-metric.main {
          padding: 11px 12px;
          background: #f6f9fb;
        }

        .result-metric > span {
          display: block;
          margin-bottom: 4px;
          color: #7f8b97;
          font-size: 16px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .result-metric strong {
          color: #354758;
          font-size: 18px;
          font-weight: 750;
        }

        .result-metric.main strong {
          font-size: 23px;
        }

        .result-metric small {
          margin-left: 5px;
          color: #8b98a4;
          font-size: 15px;
        }

        .distribution-grid {
          margin-top: 10px;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
        }

        .analysis-card {
          padding: 12px;
          border: 1px solid #dfe5ea;
          border-radius: 7px;
          background: #ffffff;
        }

        .analysis-card-header {
          min-height: 38px;
          margin-bottom: 10px;
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 15px;
        }

        .analysis-card-header.compact {
          min-height: auto;
          margin-bottom: 7px;
        }

        .analysis-card-header h3 {
          margin: 0 0 3px;
          color: #43515f;
          font-size: 17px;
        }

        .analysis-card-header p {
          margin: 0;
          color: #8a95a0;
          font-size: 16px;
          line-height: 1.4;
        }

        .distribution-bars {
          display: flex;
          flex-direction: column;
          gap: 7px;
        }

        .distribution-row {
          display: grid;
          grid-template-columns: 50px minmax(90px, 1fr) 126px;
          gap: 8px;
          align-items: center;
          min-height: 25px;
        }

        .distribution-label {
          color: #6f7c88;
          font-size: 16px;
          font-weight: 650;
        }

        .distribution-track {
          height: 10px;
          overflow: hidden;
          border-radius: 5px;
          background: #e8edf1;
        }

        .distribution-fill {
          height: 100%;
          border-radius: 5px;
          background: #8ea8bd;
        }

        .distribution-fill.decode {
          background: #87a895;
        }

        .distribution-row strong {
          text-align: right;
          color: #445464;
          font-size: 15px;
        }

        .distribution-row strong small {
          margin-left: 4px;
          color: #909ba5;
          font-size: 16px;
          font-weight: 500;
        }

        .prompt-card,
        .conclusion-card,
        .hardware-strip-card {
          margin-top: 10px;
        }

        .pearson-badge {
          min-width: 170px;
          padding: 6px 9px;
          border: 1px solid #d8e2e9;
          border-radius: 5px;
          background: #f7fafc;
          text-align: right;
        }

        .pearson-badge span {
          display: block;
          color: #7d8995;
          font-size: 15px;
        }

        .pearson-badge strong {
          color: #3f5e79;
          font-size: 19px;
        }

        .prompt-stat-grid {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 7px;
        }

        .conclusion-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 8px;
        }

        .conclusion-item {
          padding: 10px;
          border: 1px solid #e2e7eb;
          border-radius: 5px;
          background: #fafbfc;
        }

        .conclusion-item > span {
          color: #76838f;
          font-size: 16px;
          font-weight: 650;
        }

        .conclusion-item strong {
          display: block;
          margin: 5px 0;
          color: #3e566c;
          font-size: 20px;
        }

        .conclusion-item p {
          margin: 0;
          color: #86929d;
          font-size: 15px;
          line-height: 1.5;
        }

        .hardware-strip {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 7px;
        }

        @media (max-width: 1300px) {
          .phase-result-mini-grid,
          .conclusion-grid {
            grid-template-columns: repeat(2, 1fr);
          }

          .prompt-stat-grid,
          .hardware-strip {
            grid-template-columns: repeat(3, 1fr);
          }
        }

        @media (max-width: 980px) {
          .phase-result-grid,
          .distribution-grid {
            grid-template-columns: 1fr;
          }

          .results-mode-switch {
            height: auto;
            grid-template-columns: 1fr;
          }
        }
      `}
    </style>
  );
}


export default ResultsAnalysis;
