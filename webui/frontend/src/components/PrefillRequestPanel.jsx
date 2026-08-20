import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE = "http://127.0.0.1:8000";


function metric(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }

  const n = Number(value);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}


function PrefillRequestPanel() {
  const [meta, setMeta] = useState(null);
  const [category, setCategory] = useState("");
  const [batch, setBatch] = useState(null);
  const [batchInput, setBatchInput] = useState("0");
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [layerData, setLayerData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [layerLoading, setLayerLoading] = useState(false);
  const [error, setError] = useState("");


  useEffect(() => {
    async function loadMeta() {
      try {
        const response = await fetch(`${API_BASE}/api/request/prefill/meta`);
        if (!response.ok) {
          throw new Error(await response.text());
        }

        const data = await response.json();
        setMeta(data);
      } catch (err) {
        console.error(err);
        setError("无法读取 Prefill 正式评估结果，请确认 prefill_evaluation.json 已生成。 ");
      }
    }

    loadMeta();
  }, []);


  async function loadRandomBatch(nextCategory = category) {
    try {
      setLoading(true);
      setError("");

      let url = `${API_BASE}/api/request/prefill/random`;
      if (nextCategory) {
        url += `?category=${encodeURIComponent(nextCategory)}`;
      }

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      setBatch(data);
      setBatchInput(String(data.batch_id));
      setSelectedLayer(0);
    } catch (err) {
      console.error(err);
      setError("读取真实 Prefill Batch 失败。" + err.message);
    } finally {
      setLoading(false);
    }
  }


  async function loadBatchById() {
    const id = Number(batchInput);
    if (!Number.isInteger(id) || id < 0) {
      setError("Batch ID 必须是非负整数。");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${API_BASE}/api/request/prefill/batches/${id}`);
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      setBatch(data);
      setCategory(data.category ?? "");
      setSelectedLayer(0);
    } catch (err) {
      console.error(err);
      setError("读取指定 Prefill Batch 失败。" + err.message);
    } finally {
      setLoading(false);
    }
  }


  useEffect(() => {
    if (!batch) {
      return;
    }

    const controller = new AbortController();

    async function loadLayer() {
      try {
        setLayerLoading(true);

        const response = await fetch(
          `${API_BASE}/api/request/prefill/batches/${batch.batch_id}/layers/${selectedLayer}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(await response.text());
        }

        setLayerData(await response.json());
      } catch (err) {
        if (err.name === "AbortError") {
          return;
        }

        console.error(err);
        setError("读取 Prefill 当前 Layer 路由失败。" + err.message);
      } finally {
        if (!controller.signal.aborted) {
          setLayerLoading(false);
        }
      }
    }

    loadLayer();

    return () => controller.abort();
  }, [batch, selectedLayer]);


  useEffect(() => {
    if (meta && !batch) {
      loadRandomBatch("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);


  const maxLayerCycles = useMemo(() => {
    const values = batch?.layer_cycles ?? [];
    return Math.max(...values.map(Number), 1);
  }, [batch]);


  const maxScBusy = useMemo(() => {
    const values = batch?.subcube_busy_cycles ?? [];
    return Math.max(...values.map(Number), 1);
  }, [batch]);


  const currentLayerCycles = batch?.layer_cycles?.[selectedLayer] ?? null;
  const topExperts = layerData?.expert_frequency?.slice(0, 16) ?? [];


  return (
    <div className="prefill-panel">
      <div className="prefill-note">
        <strong>Prefill 正式口径：</strong>
        每个 JSON 的 <b>segment0</b> 为一个多 Token Batch，整批 Token 一起完成 L0，再一起进入 L1，直到 L57。
        页面读取 exact Prefill evaluator 已保存的真实结果；逐 Task 事件未写入评估 JSON，因此这里展示 Batch、Layer、Route 与 Sub-Cube 统计。
      </div>

      <div className="prefill-controls">
        <div className="control-box">
          <label>数据集 / Dataset</label>
          <strong>Chinese-SimpleQA</strong>
        </div>

        <div className="control-box category-box">
          <label>类别 / Category</label>
          <select
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="">全部类别</option>
            {(meta?.categories ?? []).map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}（{item.batch_count}）
              </option>
            ))}
          </select>
        </div>

        <button
          className="prefill-primary"
          disabled={loading}
          onClick={() => loadRandomBatch()}
        >
          {loading ? "正在读取..." : "随机真实 Prefill"}
        </button>

        <div className="batch-jump">
          <label>Batch ID</label>
          <input
            value={batchInput}
            onChange={(event) => setBatchInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                loadBatchById();
              }
            }}
          />
          <button onClick={loadBatchById}>打开</button>
        </div>
      </div>

      {error && <div className="prefill-error">{error}</div>}

      {batch && (
        <>
          <div className="prefill-source">
            <div>
              <span>来源文件 / File</span>
              <strong title={batch.relative_file}>{batch.filename}</strong>
            </div>
            <div>
              <span>类别 / Category</span>
              <strong>{batch.category}</strong>
            </div>
            <div>
              <span>Segment</span>
              <strong>{batch.segment_index}（Prefill）</strong>
            </div>
            <div>
              <span>Batch</span>
              <strong>#{batch.batch_id}</strong>
            </div>
          </div>

          <div className="prefill-metrics">
            <MetricCard label="输入 Token" value={batch.input_tokens} unit="tokens" />
            <MetricCard label="Prefill 总周期" value={batch.total_cycles} unit="cycles" />
            <MetricCard label="平均每 Token" value={metric(batch.cycles_per_input_token)} unit="cycles/token" />
            <MetricCard label="总任务数" value={batch.total_tasks} unit="tasks" />
            <MetricCard label="切换次数" value={batch.switches} unit="switches" />
            <MetricCard label="等待周期" value={batch.wait_cycles} unit="cycles" />
          </div>

          <div className="prefill-layout">
            <section className="prefill-section layer-section">
              <SectionTitle
                title="58 层 Prefill 周期 / Layer Cycles"
                subtitle="点击任意 Layer 查看这一层所有输入 Token 的 Top-8 路由。"
              />

              <div className="layer-selector-row">
                <button
                  disabled={selectedLayer <= 0}
                  onClick={() => setSelectedLayer((value) => Math.max(value - 1, 0))}
                >‹</button>
                <strong>L{selectedLayer}</strong>
                <span>/ 57</span>
                <button
                  disabled={selectedLayer >= 57}
                  onClick={() => setSelectedLayer((value) => Math.min(value + 1, 57))}
                >›</button>
                <input
                  type="range"
                  min="0"
                  max="57"
                  value={selectedLayer}
                  onChange={(event) => setSelectedLayer(Number(event.target.value))}
                />
                <div className="selected-layer-cycle">
                  当前层 <b>{metric(currentLayerCycles, 0)}</b> cycles
                </div>
              </div>

              <div className="prefill-layer-grid">
                {(batch.layer_cycles ?? []).map((cycles, layerId) => (
                  <button
                    key={layerId}
                    className={selectedLayer === layerId ? "active" : ""}
                    onClick={() => setSelectedLayer(layerId)}
                    title={`Layer ${layerId}: ${cycles} cycles`}
                  >
                    <span>L{layerId}</span>
                    <strong>{cycles}</strong>
                    <i style={{ width: `${(Number(cycles) / maxLayerCycles) * 100}%` }} />
                  </button>
                ))}
              </div>
            </section>

            <aside className="prefill-section batch-summary-side">
              <SectionTitle title="本次 Batch 关键点" />
              <InfoRow label="最慢 Layer" value={`L${batch.slowest_layer_id ?? "--"}`} />
              <InfoRow label="最慢层周期" value={`${batch.slowest_layer_cycles ?? "--"} cycles`} />
              <InfoRow label="最忙 Sub-Cube" value={`SC-${batch.busiest_subcube_id ?? "--"}`} />
              <InfoRow label="SC Busy" value={`${batch.busiest_subcube_cycles ?? "--"} cycles`} />
              <InfoRow label="Initial Activation" value={batch.initial_activations} />
              <InfoRow label="最大 Task 等待" value={`${batch.max_task_wait_cycles} cycles`} />
            </aside>
          </div>

          <div className="prefill-layout route-layout">
            <section className="prefill-section route-section-panel">
              <SectionTitle
                title={`L${selectedLayer} 全 Batch 路由`}
                subtitle={
                  layerLoading
                    ? "正在读取真实 Router Trace..."
                    : `${layerData?.token_count ?? batch.input_tokens} 个 Token · ${layerData?.unique_routed_expert_count ?? "--"} 个不同 Routed Expert`
                }
              />

              <div className="hot-experts">
                {topExperts.map((item, index) => (
                  <div key={item.expert_id} className="hot-expert-chip">
                    <span>#{index + 1}</span>
                    <strong>E{item.expert_id}</strong>
                    <b>{item.token_count} Token</b>
                  </div>
                ))}
              </div>

              <div className="token-route-table">
                <div className="token-route-head">
                  <span>Token</span>
                  <span>Top-8 Routed Experts</span>
                </div>

                {(layerData?.token_routes ?? []).map((item) => (
                  <div className="token-route-row" key={item.token_index}>
                    <strong>T{item.token_index}</strong>
                    <div>
                      {item.routed_experts.map((expertId, rank) => (
                        <span key={`${item.token_index}-${expertId}`}>
                          <small>#{rank + 1}</small>E{expertId}
                        </span>
                      ))}
                      <span className="shared-route">E256 Shared</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <aside className="prefill-section sc-section">
              <SectionTitle
                title="16 个 Sub-Cube Busy Cycles"
                subtitle="本次完整 Prefill Batch 的累计忙碌周期。"
              />

              <div className="sc-busy-list">
                {(batch.subcube_busy_cycles ?? []).map((value, sc) => (
                  <div className="sc-busy-row" key={sc}>
                    <strong>SC-{sc}</strong>
                    <div><i style={{ width: `${(Number(value) / maxScBusy) * 100}%` }} /></div>
                    <span>{value}</span>
                  </div>
                ))}
              </div>
            </aside>
          </div>
        </>
      )}

      <Style />
    </div>
  );
}


function MetricCard({ label, value, unit }) {
  return (
    <div className="prefill-metric-card">
      <span>{label}</span>
      <strong>{value ?? "--"}</strong>
      <small>{unit}</small>
    </div>
  );
}


function SectionTitle({ title, subtitle }) {
  return (
    <div className="prefill-section-title">
      <h3>{title}</h3>
      {subtitle && <p>{subtitle}</p>}
    </div>
  );
}


function InfoRow({ label, value }) {
  return (
    <div className="prefill-info-row">
      <span>{label}</span>
      <strong>{value ?? "--"}</strong>
    </div>
  );
}


function Style() {
  return (
    <style>{`
      .prefill-panel { width: 100%; }

      .prefill-note {
        margin-bottom: 9px;
        padding: 9px 12px;
        border: 1px solid #d9e3eb;
        border-left: 4px solid #7899b7;
        border-radius: 5px;
        background: #f7fafc;
        color: #617181;
        font-size: 16px;
        line-height: 1.55;
      }

      .prefill-note strong { color: #3e5f7d; }

      .prefill-controls {
        min-height: 64px;
        padding: 8px 10px;
        display: flex;
        align-items: flex-end;
        gap: 10px;
        border: 1px solid #dce2e8;
        border-radius: 6px;
        background: #fff;
      }

      .control-box {
        min-width: 170px;
        display: flex;
        flex-direction: column;
        gap: 5px;
      }

      .control-box label,
      .batch-jump label {
        color: #798693;
        font-size: 16px;
        font-weight: 650;
      }

      .control-box > strong {
        height: 34px;
        padding: 0 10px;
        display: flex;
        align-items: center;
        border: 1px solid #dbe1e7;
        border-radius: 4px;
        background: #f7f9fb;
        color: #425263;
        font-size: 15px;
      }

      .category-box { flex: 1; max-width: 330px; }

      .category-box select {
        height: 34px;
        padding: 0 8px;
        border: 1px solid #d5dde5;
        border-radius: 4px;
        background: #fff;
        color: #435263;
        font-size: 15px;
      }

      .prefill-primary {
        height: 34px;
        padding: 0 16px;
        border: 1px solid #6587a5;
        border-radius: 4px;
        background: #7395b3;
        color: #fff;
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
      }

      .batch-jump {
        margin-left: auto;
        display: grid;
        grid-template-columns: 78px 58px;
        gap: 4px;
      }

      .batch-jump label { grid-column: 1 / 3; }

      .batch-jump input,
      .batch-jump button {
        height: 34px;
        border: 1px solid #d5dde5;
        border-radius: 4px;
        background: #fff;
        color: #445363;
        font-size: 15px;
      }

      .batch-jump input { width: 78px; padding: 0 7px; }
      .batch-jump button { cursor: pointer; font-weight: 650; }

      .prefill-error {
        margin-top: 8px;
        padding: 9px 11px;
        border: 1px solid #e0bcbc;
        border-radius: 5px;
        background: #fff5f5;
        color: #995858;
        font-size: 16px;
      }

      .prefill-source {
        min-height: 48px;
        margin-top: 9px;
        padding: 7px 10px;
        display: grid;
        grid-template-columns: minmax(220px, 1.6fr) 1fr 150px 100px;
        gap: 8px;
        border: 1px solid #dfe4e9;
        border-radius: 5px;
        background: #fff;
      }

      .prefill-source > div { min-width: 0; }
      .prefill-source span {
        display: block;
        margin-bottom: 2px;
        color: #8b95a0;
        font-size: 15px;
      }
      .prefill-source strong {
        display: block;
        overflow: hidden;
        color: #42505e;
        font-size: 15px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .prefill-metrics {
        margin-top: 8px;
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 7px;
      }

      .prefill-metric-card {
        min-height: 76px;
        padding: 9px 10px;
        border: 1px solid #dfe4e8;
        border-radius: 5px;
        background: #fff;
      }

      .prefill-metric-card span {
        display: block;
        color: #7f8995;
        font-size: 16px;
      }
      .prefill-metric-card strong {
        display: inline-block;
        margin-top: 5px;
        color: #33495d;
        font-size: 21px;
      }
      .prefill-metric-card small {
        margin-left: 5px;
        color: #5f7083;
        font-size: 15px;
      }

      .prefill-layout {
        margin-top: 8px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 260px;
        gap: 8px;
      }

      .prefill-section {
        padding: 10px;
        border: 1px solid #dfe4e8;
        border-radius: 6px;
        background: #fff;
      }

      .prefill-section-title h3 {
        margin: 0;
        color: #435362;
        font-size: 17px;
      }
      .prefill-section-title p {
        margin: 3px 0 0;
        color: #89939d;
        font-size: 16px;
      }

      .layer-selector-row {
        min-height: 42px;
        margin: 8px 0;
        padding: 5px 7px;
        display: flex;
        align-items: center;
        gap: 7px;
        border: 1px solid #e2e6ea;
        border-radius: 5px;
        background: #fafbfc;
      }

      .layer-selector-row button {
        width: 31px;
        height: 31px;
        border: 1px solid #d5dce3;
        border-radius: 4px;
        background: #fff;
        color: #536171;
        font-size: 20px;
        cursor: pointer;
      }
      .layer-selector-row > strong { color: #345a7b; font-size: 18px; }
      .layer-selector-row > span { color: #88929d; font-size: 16px; }
      .layer-selector-row input { flex: 1; min-width: 180px; }
      .selected-layer-cycle {
        min-width: 150px;
        color: #6f7b87;
        font-size: 16px;
        text-align: right;
      }
      .selected-layer-cycle b { color: #36536d; font-size: 16px; }

      .prefill-layer-grid {
        display: grid;
        grid-template-columns: repeat(10, minmax(0, 1fr));
        gap: 5px;
      }

      .prefill-layer-grid button {
        min-height: 49px;
        padding: 5px;
        position: relative;
        overflow: hidden;
        border: 1px solid #dde3e8;
        border-radius: 4px;
        background: #fafbfc;
        color: #526579;
        cursor: pointer;
        text-align: left;
      }
      .prefill-layer-grid button.active {
        border-color: #3b82f6;
        background: #dbeafe;
        color: #123f70;
      }
      .prefill-layer-grid span { display: block; font-size: 15px; font-weight: 700; }
      .prefill-layer-grid strong { display: block; margin-top: 2px; font-size: 16px; }
      .prefill-layer-grid i {
        height: 3px;
        position: absolute;
        bottom: 0;
        left: 0;
        background: #7d9fbb;
      }

      .prefill-info-row {
        min-height: 38px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        border-bottom: 1px solid #edf0f2;
        color: #526579;
        font-size: 16px;
      }
      .prefill-info-row strong { color: #3e4f60; font-size: 15px; }

      .route-layout { grid-template-columns: minmax(0, 1fr) 320px; }

      .hot-experts {
        margin: 8px 0;
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
      }
      .hot-expert-chip {
        min-width: 105px;
        padding: 5px 7px;
        display: grid;
        grid-template-columns: 24px 38px 1fr;
        gap: 3px;
        align-items: center;
        border: 1px solid #dbe2e8;
        border-radius: 4px;
        background: #f7f9fb;
        color: #72808d;
        font-size: 15px;
      }
      .hot-expert-chip strong { color: #3e5c76; font-size: 15px; }
      .hot-expert-chip b { color: #6b7784; font-size: 15px; font-weight: 600; }

      .token-route-table {
        max-height: 340px;
        overflow-y: auto;
        border: 1px solid #e1e5e9;
        border-radius: 4px;
      }
      .token-route-head,
      .token-route-row {
        min-height: 38px;
        padding: 5px 8px;
        display: grid;
        grid-template-columns: 65px minmax(0, 1fr);
        gap: 8px;
        align-items: center;
      }
      .token-route-head {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #f2f5f7;
        color: #6f7d8a;
        font-size: 16px;
        font-weight: 700;
      }
      .token-route-row { border-top: 1px solid #edf0f2; }
      .token-route-row > strong { color: #455666; font-size: 15px; }
      .token-route-row > div { display: flex; flex-wrap: wrap; gap: 4px; }
      .token-route-row span {
        padding: 3px 5px;
        border-radius: 3px;
        background: #edf1f4;
        color: #566574;
        font-size: 16px;
        font-weight: 650;
      }
      .token-route-row small { margin-right: 3px; color: #64748b; font-size: 16px; }
      .token-route-row .shared-route { background: #eee8f4; color: #6f588f; }

      .sc-busy-list { margin-top: 8px; }
      .sc-busy-row {
        min-height: 32px;
        display: grid;
        grid-template-columns: 52px minmax(0, 1fr) 62px;
        gap: 7px;
        align-items: center;
        color: #6d7986;
        font-size: 15px;
      }
      .sc-busy-row strong { color: #465766; font-size: 16px; }
      .sc-busy-row > div {
        height: 7px;
        overflow: hidden;
        border-radius: 4px;
        background: #e8ecef;
      }
      .sc-busy-row i { display: block; height: 100%; background: #84a4be; }
      .sc-busy-row span { text-align: right; font-size: 16px; }

      @media (max-width: 1200px) {
        .prefill-metrics { grid-template-columns: repeat(3, 1fr); }
        .prefill-layer-grid { grid-template-columns: repeat(8, 1fr); }
        .prefill-layout,
        .route-layout { grid-template-columns: 1fr; }
        .prefill-source { grid-template-columns: repeat(2, 1fr); }
      }
    `}</style>
  );
}


export default PrefillRequestPanel;
