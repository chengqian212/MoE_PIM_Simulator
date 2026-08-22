function num(value, digits = 2) {
  const x = Number(value);
  if (!Number.isFinite(x)) return "--";
  return x.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function pct(value, digits = 2) {
  const x = Number(value);
  if (!Number.isFinite(x)) return "--";
  return `${x.toFixed(digits)}%`;
}

const MAPPING_ORDER = [
  "round_robin",
  "least_loaded",
  "frequency_aware",
  "trace_aware",
];

const PREFILL_ORDER = [
  "no_reuse",
  "switch_aware",
  "aggressive_reuse",
  "largest_batch_reuse",
];

const PREFILL_LABEL = {
  no_reuse: "No-Reuse",
  switch_aware: "Switch-Aware",
  aggressive_reuse: "Aggressive-Reuse",
  largest_batch_reuse: "Largest-Batch-Reuse",
};

function SectionTitle({ index, title, note }) {
  return (
    <div className="result-section-title">
      <span>{index}</span>
      <div>
        <h3>{title}</h3>
        {note && <p>{note}</p>}
      </div>
    </div>
  );
}

function ResultsAnalysis({ formalReference }) {
  const mapping = formalReference?.mapping ?? {};
  const mappingMetrics = mapping.metrics ?? [];
  const mappingImprove = mapping.improvements_vs_round_robin ?? [];
  const prefill = formalReference?.prefill?.results ?? {};
  const decode = formalReference?.decode ?? {};
  const ablation = formalReference?.ablation ?? [];
  const replication = formalReference?.replication ?? null;

  const mapRows = MAPPING_ORDER
    .map((mode) => mappingMetrics.find((row) => row.mode === mode))
    .filter(Boolean);

  const traceImprove = mappingImprove.find((row) => row.mode === "trace_aware") ?? {};
  const aggressiveImprove = prefill.aggressive_reuse?.improvement_vs_no_reuse ?? {};
  const decodeSummary = decode.summary ?? {};
  const rep = replication?.comparison ?? {};

  return (
    <div className="results-v2">
      <div className="page-head compact">
        <div>
          <h2>实验结果</h2>
          <p>数据按 80% 构建集 / 20% 独立评估集划分，只展示已经验证的核心结论。</p>
        </div>
        <div className="protocol-pill">MoE Expert Only</div>
      </div>

      <section className="result-highlight-strip">
        <div><span>Mapping Conflict</span><b>−{pct(traceImprove.conflict_reduction_vs_round_robin_percent)}</b></div>
        <div><span>Mapping → Prefill</span><b>−{pct(traceImprove.prefill_mean_improvement_vs_round_robin_percent)}</b></div>
        <div><span>Mapping → Decode</span><b>−{pct(traceImprove.decode_mean_improvement_vs_round_robin_percent)}</b></div>
        <div><span>Prefill Reuse</span><b>−{pct(aggressiveImprove.prefill_mean_percent)}</b></div>
      </section>

      <section className="formal-result-card">
        <SectionTitle
          index="01"
          title="Mapping Baseline"
          note="固定 Trace-aware Pairing + Local Search，只改变 Plane → Sub-Cube Mapping。"
        />
        <div className="result-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Mapping</th>
                <th>Conflict</th>
                <th>Prefill</th>
                <th>Decode</th>
                <th>P95</th>
              </tr>
            </thead>
            <tbody>
              {mapRows.map((row) => (
                <tr key={row.mode} className={row.mode === "trace_aware" ? "best" : ""}>
                  <td>{row.display_name}</td>
                  <td>{num(row.mapping_conflict_cost, 0)}</td>
                  <td>{num(row.prefill_mean_latency, 2)}</td>
                  <td>{num(row.decode_mean_cycles_per_token, 2)}</td>
                  <td>{num(row.decode_p95_cycles_per_token, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="result-conclusion">
          结论：Trace-aware Mapping 是主要静态收益来源；相比 Round-Robin，Prefill 降低 {pct(traceImprove.prefill_mean_improvement_vs_round_robin_percent)}，Decode 降低 {pct(traceImprove.decode_mean_improvement_vs_round_robin_percent)}。
        </div>
      </section>

      {ablation.length > 0 && (
        <section className="formal-result-card">
          <SectionTitle index="02" title="Pairing × Mapping 2×2 消融" note="用于判断静态性能收益到底来自 Pairing 还是 Mapping。" />
          <div className="result-table-wrap">
            <table className="research-table compact-table">
              <thead>
                <tr>
                  <th>方案</th>
                  <th>Pairing</th>
                  <th>Mapping</th>
                  <th>Prefill</th>
                  <th>Decode</th>
                </tr>
              </thead>
              <tbody>
                {ablation.map((row) => (
                  <tr key={row.experiment} className={row.experiment === "Full" ? "best" : ""}>
                    <td>{row.experiment}</td>
                    <td>{row.pairing}</td>
                    <td>{row.mapping}</td>
                    <td>{num(row.prefill_mean, 2)}</td>
                    <td>{num(row.decode_mean, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="result-conclusion">结论：Mapping Only 已获得几乎全部静态收益；Pairing 属于二级优化。</div>
        </section>
      )}

      <section className="formal-result-card">
        <SectionTitle index="03" title="Prefill Scheduler" note="同一 Trace-aware Mapping，比较 Weight-Cube reuse 策略。" />
        <div className="result-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Scheduler</th>
                <th>Mean</th>
                <th>Cycles / Token</th>
                <th>P95</th>
                <th>Switches</th>
              </tr>
            </thead>
            <tbody>
              {PREFILL_ORDER.map((mode) => {
                const row = prefill[mode];
                if (!row) return null;
                return (
                  <tr key={mode} className={mode === "aggressive_reuse" ? "best" : ""}>
                    <td>{PREFILL_LABEL[mode]}</td>
                    <td>{num(row.prefill_mean_cycles, 2)}</td>
                    <td>{num(row.mean_cycles_per_input_token, 4)}</td>
                    <td>{num(row.prefill_p95_cycles, 2)}</td>
                    <td>{num(row.mean_switches_per_batch, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="result-conclusion">
          结论：Aggressive-Reuse 最优；相比 No-Reuse，Mean Prefill 降低 {pct(aggressiveImprove.prefill_mean_percent)}，Switch 降低 {pct(aggressiveImprove.switches_percent)}。
        </div>
      </section>

      <div className="result-two-column">
        <section className="formal-result-card small-card">
          <SectionTitle index="04" title="Decode 最优性" />
          <div className="evidence-number">{pct(Number(decodeSummary.greedy_already_optimal_rate) * 100)}</div>
          <p>Greedy layer instances 已达到 CP-SAT 全局最优。</p>
          <div className="evidence-row"><span>Proven OPTIMAL</span><b>{num(decodeSummary.optimal_proven_count, 0)} / {num(decodeSummary.instance_count, 0)}</b></div>
          <div className="evidence-row"><span>Mean Gap</span><b>{pct(decodeSummary.mean_gap_vs_opt_percent, 4)}</b></div>
          <div className="result-conclusion">Decode 调度基本收尾，CP-SAT 仅作为最优性验证。</div>
        </section>

        <section className="formal-result-card small-card">
          <SectionTitle index="05" title="Expert Replication" />
          <div className="evidence-number muted">{pct(rep.oracle_improvement_percent, 4)}</div>
          <p>现有空余 Plane 下，Oracle 能达到的 Mean 理论提升上限。</p>
          <div className="evidence-row"><span>Baseline</span><b>{num(rep.baseline_mean, 2)}</b></div>
          <div className="evidence-row"><span>Balanced-All</span><b>{num(rep.balanced_all_mean, 2)}</b></div>
          <div className="evidence-row"><span>Oracle</span><b>{num(rep.oracle_mean, 2)}</b></div>
          <div className="result-conclusion">当前容量下收益可忽略，因此不纳入最终 Runtime。</div>
        </section>
      </div>
    </div>
  );
}

export default ResultsAnalysis;
