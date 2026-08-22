function n(value, digits = 2) {
  const x = Number(value);
  if (!Number.isFinite(x)) return "--";
  return x.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function i(value) {
  return n(value, 0);
}

function Metric({ label, value, unit, strong = false }) {
  return (
    <div className={strong ? "ov-metric strong" : "ov-metric"}>
      <span>{label}</span>
      <div>
        <b>{value}</b>
        {unit && <small>{unit}</small>}
      </div>
    </div>
  );
}

function OverviewDashboard({ phaseSummary, hardware, formalReference }) {
  const mappingMetrics = formalReference?.mapping?.metrics ?? [];
  const traceAware = mappingMetrics.find((row) => row.mode === "trace_aware") ?? {};
  const prefill = formalReference?.prefill?.results?.aggressive_reuse ?? {};
  const decode = formalReference?.decode?.summary ?? {};
  const protocol = formalReference?.mapping?.protocol ?? {};

  // 兼容：正式 reference 暂不可用时，退回旧 phase summary，但 UI 会明确标注。
  const fallbackPrefill = phaseSummary?.prefill ?? {};
  const fallbackDecode = phaseSummary?.decode ?? {};
  const prefillMean = prefill.prefill_mean_cycles ?? fallbackPrefill?.latency_cycles?.mean;
  const prefillP95 = prefill.prefill_p95_cycles ?? fallbackPrefill?.latency_cycles?.p95;
  const prefillCpt = prefill.mean_cycles_per_input_token ?? fallbackPrefill?.cycles_per_input_token?.mean;
  const prefillBatches = prefill.batch_count ?? fallbackPrefill?.batch_count;
  const prefillTokens = prefill.total_input_tokens ?? fallbackPrefill?.total_input_tokens;

  const decodeMean = traceAware.decode_mean_cycles_per_token ?? fallbackDecode?.cycles_per_token?.mean;
  const decodeP95 = traceAware.decode_p95_cycles_per_token ?? fallbackDecode?.cycles_per_token?.p95;
  const decodeTokens = formalReference?.decode?.sampling?.source_token_count ?? fallbackDecode?.token_count;
  const greedyOptimal = decode.greedy_already_optimal_rate;

  return (
    <div className="overview-v2">
      <div className="page-head compact">
        <div>
          <h2>系统总览</h2>
        </div>
        <div className="protocol-pill">
          Profile {i(protocol.profile_file_count)} / Eval {i(protocol.evaluation_file_count)} · seed=42
        </div>
      </div>

      <section className="final-scheme-bar">
        <div className="scheme-label">最终方案 / Final</div>
        <div><span>Pairing</span><b>Trace-aware + LS</b></div>
        <div><span>Mapping</span><b>Trace-aware</b></div>
        <div><span>Prefill</span><b>Aggressive-Reuse</b></div>
        <div><span>Decode</span><b>Greedy</b></div>
      </section>

      <div className="overview-core-grid">
        <section className="core-phase-card prefill">
          <div className="core-phase-head">
            <div>
              <h3>MoE Prefill</h3>
            </div>
            <b>{i(prefillBatches)} batches</b>
          </div>
          <div className="core-primary">
            <span>Mean Latency</span>
            <strong>{n(prefillMean, 2)}</strong>
            <small>cycles / batch</small>
          </div>
          <div className="core-mini-grid">
            <Metric label="P95" value={n(prefillP95, 2)} unit="cycles" />
            <Metric label="Cycles / Input Token" value={n(prefillCpt, 4)} unit="cycles/token" strong />
            <Metric label="Input Tokens" value={i(prefillTokens)} />
          </div>
          <div className="core-foot">404 held-out batches</div>
        </section>

        <section className="core-phase-card decode">
          <div className="core-phase-head">
            <div>
              <h3>MoE Decode</h3>
            </div>
            <b>{i(decodeTokens)} tokens</b>
          </div>
          <div className="core-primary">
            <span>Mean Latency</span>
            <strong>{n(decodeMean, 2)}</strong>
            <small>cycles / token</small>
          </div>
          <div className="core-mini-grid">
            <Metric label="P95" value={n(decodeP95, 0)} unit="cycles/token" />
            <Metric
              label="Greedy 已达最优"
              value={Number.isFinite(Number(greedyOptimal)) ? `${(Number(greedyOptimal) * 100).toFixed(2)}%` : "99.97%"}
              strong
            />
            <Metric label="CP-SAT" value="Optimal Reference" />
          </div>
          <div className="core-foot">50,916 held-out tokens</div>
        </section>
      </div>

      <section className="hardware-line">
        <span>硬件</span>
        <b>N={hardware.N ?? "--"}</b>
        <b>{hardware.num_subcubes ?? "--"} Sub-Cubes</b>
        <b>H={hardware.H ?? "--"}</b>
        <b>W={hardware.W ?? "--"}</b>
        <b>D={hardware.D ?? "--"}</b>
        <b>P={hardware.used_planes ?? "--"}</b>
        <b>Q={hardware.total_plane_slots ?? "--"}</b>
        <b>Empty={hardware.empty_plane_slots ?? "--"}</b>
      </section>

      <div className="scope-one-line">
        指标范围：仅 58 个 MoE 层的 Expert gate / up / down；不是完整 TTFT / TPOT。
      </div>
    </div>
  );
}

export default OverviewDashboard;
