import {
  useMemo,
  useState,
} from "react";


const API_BASE = "http://127.0.0.1:8000";

const PAIRING_OPTIONS = [
  ["sequential", "Sequential"],
  ["random", "Random"],
  ["frequency_aware", "Frequency-aware"],
  ["greedy", "Coactivation Greedy"],
  ["trace_aware", "Greedy + Local Search"],
  ["optimal", "Optimal Matching"],
];


function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  const n = Number(value);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}


function pctDelta(a, b) {
  const x = Number(a);
  const y = Number(b);
  if (!Number.isFinite(x) || !Number.isFinite(y) || x === 0) {
    return null;
  }
  return ((y - x) / x) * 100;
}


function sourceLabel(data) {
  const request = data?.request;
  if (!request) return "尚未抽样";
  if (data.phase === "prefill") {
    return `Batch #${request.batch_id} · ${request.input_tokens} tokens · ${request.filename}`;
  }
  return `${request.filename} · segment ${request.segment_index} · token ${request.token_index}`;
}


function PairingComparison() {
  const [phase, setPhase] = useState("decode");
  const [pairingA, setPairingA] = useState("sequential");
  const [pairingB, setPairingB] = useState("trace_aware");
  const [data, setData] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  async function randomRequest() {
    if (phase === "prefill") {
      const response = await fetch(`${API_BASE}/api/request/prefill/random`);
      if (!response.ok) throw new Error(await response.text());
      const batch = await response.json();
      return {
        prefill_batch_id: batch.batch_id,
        decode_source: null,
      };
    }

    const response = await fetch(`${API_BASE}/api/request/decode/random`);
    if (!response.ok) throw new Error(await response.text());
    const token = await response.json();
    return {
      prefill_batch_id: null,
      decode_source: {
        category: token.source?.category,
        filename: token.source?.filename,
        segment_index: token.source?.segment_index,
        token_index: token.source?.token_index ?? 0,
      },
    };
  }


  async function runRandomTest() {
    if (pairingA === pairingB) {
      setError("方案 A/B 请选择不同 Pairing。");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const requestSource = await randomRequest();
      const response = await fetch(`${API_BASE}/api/comparison/pairing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          phase,
          pairing_a: pairingA,
          pairing_b: pairingB,
          selected_layer: null,
          ...requestSource,
        }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const next = await response.json();
      setData(next);
      setSelectedLayer(next.suggested_layer ?? 0);
    } catch (err) {
      console.error(err);
      setError(`实时 Pairing 对比失败：${err.message}`);
    } finally {
      setLoading(false);
    }
  }


  const aLayer = data?.a?.pairing_layers?.[selectedLayer] ?? null;
  const bLayer = data?.b?.pairing_layers?.[selectedLayer] ?? null;
  const aResult = data?.a?.result ?? null;
  const bResult = data?.b?.result ?? null;

  const activeExperts = useMemo(() => {
    if (!aLayer) return [];
    const map = new Map();
    for (const pair of aLayer.pairs ?? []) {
      if (pair.hits_a > 0) map.set(pair.expert_a, pair.hits_a);
      if (pair.hits_b > 0) map.set(pair.expert_b, pair.hits_b);
    }
    return [...map.entries()]
      .sort((x, y) => y[1] - x[1] || x[0] - y[0])
      .slice(0, phase === "decode" ? 8 : 18);
  }, [aLayer, phase]);

  const totalDelta = pctDelta(aResult?.total_cycles, bResult?.total_cycles);
  const layerDelta = pctDelta(
    aResult?.layer_cycles?.[selectedLayer],
    bResult?.layer_cycles?.[selectedLayer]
  );


  function changePhase(next) {
    setPhase(next);
    setData(null);
    setSelectedLayer(0);
    setError("");
  }


  return (
    <div className="pairing-compare">

      <section className="pairing-controls">
        <div className="pairing-control phase-control">
          <span>测试场景</span>
          <div className="pairing-segmented">
            <button
              className={phase === "decode" ? "active" : ""}
              onClick={() => changePhase("decode")}
            >
              Decode
            </button>
            <button
              className={phase === "prefill" ? "active" : ""}
              onClick={() => changePhase("prefill")}
            >
              Prefill
            </button>
          </div>
        </div>

        <SelectStrategy
          label="方案 A"
          value={pairingA}
          onChange={setPairingA}
        />

        <div className="pairing-vs">VS</div>

        <SelectStrategy
          label="方案 B"
          value={pairingB}
          onChange={setPairingB}
        />

        <button
          className="pairing-run"
          disabled={loading}
          onClick={runRandomTest}
        >
          {loading ? "正在运行..." : data ? "再随机一个" : "随机真实请求"}
        </button>
      </section>

      <div className="pairing-fixed-note">
        <b>只改变 UP-UP Pairing</b>
        <span>Mapping algorithm = Trace-aware</span>
        <span>{phase === "prefill" ? "Scheduler = Aggressive-Reuse" : "Scheduler = Greedy"}</span>
        <span>Hardware = N4 · 7168×4096 · D1398</span>
      </div>

      {error && <div className="pairing-error">{error}</div>}

      {!data && !loading && (
        <div className="pairing-empty">
          <strong>随机抽一个真实请求，看两种 UP 配对在同一请求下发生什么。</strong>
          <span>这里不展示正式平均结果；每次点击都会现场抽样并重新跑 A/B。</span>
        </div>
      )}

      {data && (
        <>
          <section className="pairing-source-row">
            <div>
              <span>当前随机实例</span>
              <strong>{sourceLabel(data)}</strong>
            </div>
            <div>
              <span>建议观察层</span>
              <strong>L{data.suggested_layer}</strong>
            </div>
            <div>
              <span>当前观察</span>
              <strong>L{selectedLayer}</strong>
            </div>
          </section>

          <section className="pairing-result-strip">
            <ResultSide
              name={data.a.strategy.label}
              total={aResult?.total_cycles}
              layer={aResult?.layer_cycles?.[selectedLayer]}
              conflicts={aLayer?.pair_collision_count}
              planes={aLayer?.touched_up_planes}
            />

            <div className="pairing-delta-card">
              <span>当前实例 B 相对 A</span>
              <strong>
                {totalDelta === null
                  ? "--"
                  : `${totalDelta > 0 ? "+" : ""}${totalDelta.toFixed(2)}%`}
              </strong>
              <small>
                Layer L{selectedLayer}: {layerDelta === null
                  ? "--"
                  : `${layerDelta > 0 ? "+" : ""}${layerDelta.toFixed(2)}%`}
              </small>
            </div>

            <ResultSide
              name={data.b.strategy.label}
              total={bResult?.total_cycles}
              layer={bResult?.layer_cycles?.[selectedLayer]}
              conflicts={bLayer?.pair_collision_count}
              planes={bLayer?.touched_up_planes}
              right
            />
          </section>

          <section className="pairing-layer-panel">
            <div className="pairing-section-head">
              <div>
                <h3>58 层当前请求的 Pair 冲突</h3>
                <p>点击任意 Layer 查看该层到底是哪两个 UP 被配在一起。</p>
              </div>
              <div className="pairing-layer-nav">
                <button
                  disabled={selectedLayer <= 0}
                  onClick={() => setSelectedLayer(Math.max(0, selectedLayer - 1))}
                >‹</button>
                <strong>L{selectedLayer}</strong>
                <button
                  disabled={selectedLayer >= 57}
                  onClick={() => setSelectedLayer(Math.min(57, selectedLayer + 1))}
                >›</button>
              </div>
            </div>

            <LayerConflictGrid
              a={data.a.pairing_layers}
              b={data.b.pairing_layers}
              selectedLayer={selectedLayer}
              onSelect={setSelectedLayer}
            />
          </section>

          <section className="pairing-active-route">
            <div>
              <span>{phase === "decode" ? "本层 Top-8 Routed Experts" : "本层当前 Batch 高频 Routed Experts"}</span>
              <div className="pairing-expert-chips">
                {activeExperts.map(([expertId, hits]) => (
                  <b key={expertId}>
                    E{expertId}{phase === "prefill" ? <small>×{hits}</small> : null}
                  </b>
                ))}
              </div>
            </div>
            <small>
              红色 Pair 表示同一个 Token 同时命中了这两个 Expert，而它们的 UP 又共享同一 Plane。
            </small>
          </section>

          <section className="pairing-pairs-grid">
            <PairList
              title={data.a.strategy.label}
              layer={aLayer}
              phase={phase}
            />
            <PairList
              title={data.b.strategy.label}
              layer={bLayer}
              phase={phase}
            />
          </section>

          <div className="pairing-local-note">
            <strong>怎么看：</strong>
            Pairing 的目标是把经常同时激活的 Routed UP 尽量拆开。当前实例中即使 Pair 冲突明显减少，最终 cycles 仍可能相同或只差一点——这正是这个交互页要直观展示的现象。
          </div>
        </>
      )}

      <Style />
    </div>
  );
}


function SelectStrategy({ label, value, onChange }) {
  return (
    <label className="pairing-control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {PAIRING_OPTIONS.map(([id, name]) => (
          <option key={id} value={id}>{name}</option>
        ))}
      </select>
    </label>
  );
}


function ResultSide({ name, total, layer, conflicts, planes, right = false }) {
  return (
    <div className={`pairing-result-side ${right ? "right" : ""}`}>
      <span>{name}</span>
      <strong>{fmt(total)} <small>cycles</small></strong>
      <div>
        <b>L{` `}{fmt(layer)}</b>
        <b>Pair 冲突 {fmt(conflicts)}</b>
        <b>UP Plane {fmt(planes)}</b>
      </div>
    </div>
  );
}


function LayerConflictGrid({ a, b, selectedLayer, onSelect }) {
  const maxValue = Math.max(
    1,
    ...(a ?? []).map((item) => Number(item.pair_collision_count ?? 0)),
    ...(b ?? []).map((item) => Number(item.pair_collision_count ?? 0)),
  );

  return (
    <div className="pairing-layer-grid">
      {(a ?? []).map((aItem, layerId) => {
        const bItem = b?.[layerId] ?? {};
        const av = Number(aItem.pair_collision_count ?? 0);
        const bv = Number(bItem.pair_collision_count ?? 0);
        return (
          <button
            key={layerId}
            className={selectedLayer === layerId ? "selected" : ""}
            onClick={() => onSelect(layerId)}
            title={`L${layerId}: A=${av}, B=${bv}`}
          >
            <span>L{layerId}</span>
            <i className="a" style={{ height: `${Math.max(2, (av / maxValue) * 28)}px` }} />
            <i className="b" style={{ height: `${Math.max(2, (bv / maxValue) * 28)}px` }} />
          </button>
        );
      })}
    </div>
  );
}


function PairList({ title, layer, phase }) {
  const rows = (layer?.pairs ?? []).slice(0, 14);
  return (
    <div className="pairing-pair-list">
      <div className="pairing-pair-head">
        <strong>{title}</strong>
        <span>
          冲突 {fmt(layer?.pair_collision_count)} · 触及 {fmt(layer?.touched_up_planes)} Planes
        </span>
      </div>

      <div className="pairing-pair-rows">
        {rows.map((pair) => {
          const conflict = Number(pair.co_hit_tokens ?? 0) > 0;
          return (
            <div
              key={`${pair.expert_a}-${pair.expert_b}`}
              className={conflict ? "pair-row conflict" : "pair-row"}
            >
              <ExpertBox id={pair.expert_a} hits={pair.hits_a} phase={phase} />
              <div className="pair-link">
                <span />
                {conflict && <b>{pair.co_hit_tokens}</b>}
              </div>
              <ExpertBox id={pair.expert_b} hits={pair.hits_b} phase={phase} />
            </div>
          );
        })}
      </div>

      {(layer?.pairs?.length ?? 0) > rows.length && (
        <small className="pairing-more">
          当前 Layer 共触及 {layer.pairs.length} 个 UP Pair；这里只显示最相关的前 {rows.length} 个。
        </small>
      )}
    </div>
  );
}


function ExpertBox({ id, hits, phase }) {
  const active = Number(hits) > 0;
  return (
    <div className={active ? "pair-expert active" : "pair-expert"}>
      <strong>E{id}</strong>
      <small>{active ? (phase === "prefill" ? `×${hits}` : "active") : "idle"}</small>
    </div>
  );
}


function Style() {
  return (
    <style>{`
      .pairing-compare { width: 100%; }

      .pairing-controls {
        display: grid;
        grid-template-columns: 1.15fr minmax(185px, 1fr) auto minmax(185px, 1fr) auto;
        gap: 9px;
        align-items: end;
        padding: 12px;
        border: 1px solid #d8e1e8;
        border-radius: 7px;
        background: #fff;
      }

      .pairing-control { display: flex; flex-direction: column; gap: 5px; }
      .pairing-control > span { color: #6f7e8a; font-size: 16px; font-weight: 700; }
      .pairing-control select {
        height: 38px; border: 1px solid #cfd9e1; border-radius: 5px;
        padding: 0 10px; background: #fbfcfd; color: #000000; font-size: 16px; font-weight: 700;
      }

      .pairing-segmented { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 3px; border-radius: 6px; background: #eef3f6; }
      .pairing-segmented button { height: 32px; border: 0; border-radius: 4px; background: transparent; color: #697985; font-weight: 750; cursor: pointer; }
      .pairing-segmented button.active { background: #fff; color: #000000; box-shadow: 0 1px 3px rgba(43,61,75,.12); }

      .pairing-vs { align-self: end; height: 38px; display: grid; place-items: center; color: #8996a0; font-size: 15px; font-weight: 900; }
      .pairing-run { height: 38px; padding: 0 18px; border: 1px solid #718fa7; border-radius: 5px; background: #718fa7; color: #fff; font-size: 16px; font-weight: 800; cursor: pointer; white-space: nowrap; }
      .pairing-run:disabled { opacity: .55; cursor: default; }

      .pairing-fixed-note { display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 8px 11px; margin-top: 8px; border: 1px solid #dde5eb; border-radius: 5px; background: #f8fafb; color: #70808c; font-size: 14px; }
      .pairing-fixed-note b { color: #000000; }

      .pairing-error { margin-top: 9px; padding: 10px 12px; border: 1px solid #e1b9b9; border-radius: 5px; background: #fff6f6; color: #a44d4d; font-size: 15px; }
      .pairing-empty { min-height: 270px; margin-top: 10px; display: grid; place-content: center; gap: 5px; text-align: center; border: 1px dashed #ccd8e0; border-radius: 7px; background: #fbfcfd; color: #000000; }
      .pairing-empty strong { color: #000000; font-size: 17px; }

      .pairing-source-row { display: grid; grid-template-columns: 1fr 130px 130px; gap: 8px; margin-top: 10px; }
      .pairing-source-row > div { padding: 9px 11px; border: 1px solid #dbe3e9; border-radius: 5px; background: #fff; min-width: 0; }
      .pairing-source-row span { display: block; color: #89959e; font-size: 14px; font-weight: 700; }
      .pairing-source-row strong { display: block; margin-top: 2px; color: #000000; font-size: 15px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

      .pairing-result-strip { display: grid; grid-template-columns: 1fr 160px 1fr; gap: 9px; margin-top: 9px; }
      .pairing-result-side, .pairing-delta-card { min-height: 108px; padding: 12px 14px; border: 1px solid #d7e1e8; border-radius: 6px; background: #fff; }
      .pairing-result-side > span { color: #000000; font-size: 15px; font-weight: 800; }
      .pairing-result-side > strong { display: block; margin: 5px 0 10px; color: #000000; font-size: 27px; letter-spacing: -.4px; }
      .pairing-result-side > strong small { color: #8b98a1; font-size: 16px; font-weight: 700; }
      .pairing-result-side > div { display: flex; gap: 12px; flex-wrap: wrap; color: #000000; font-size: 14px; }
      .pairing-result-side.right { border-color: #b9cad7; background: #fbfdfe; }
      .pairing-delta-card { display: grid; place-content: center; text-align: center; background: #f4f8fa; }
      .pairing-delta-card span { color: #82909a; font-size: 14px; font-weight: 700; }
      .pairing-delta-card strong { color: #000000; font-size: 23px; }
      .pairing-delta-card small { color: #788a97; font-size: 14px; }

      .pairing-layer-panel { margin-top: 10px; padding: 11px 12px 12px; border: 1px solid #d9e2e8; border-radius: 6px; background: #fff; }
      .pairing-section-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
      .pairing-section-head h3 { margin: 0; color: #000000; font-size: 20px; }
      .pairing-section-head p { margin: 2px 0 0; color: #86939d; font-size: 14px; }
      .pairing-layer-nav { display: flex; align-items: center; gap: 7px; }
      .pairing-layer-nav button { width: 29px; height: 29px; border: 1px solid #d2dce3; border-radius: 4px; background: #fff; cursor: pointer; }
      .pairing-layer-nav strong { min-width: 38px; text-align: center; color: #000000; }

      .pairing-layer-grid { display: grid; grid-template-columns: repeat(29, minmax(19px, 1fr)); gap: 3px; margin-top: 10px; }
      .pairing-layer-grid button { position: relative; height: 51px; padding: 2px 1px; display: flex; justify-content: center; align-items: flex-end; gap: 2px; border: 1px solid #e1e7eb; border-radius: 3px; background: #fafcfd; cursor: pointer; overflow: hidden; }
      .pairing-layer-grid button span { position: absolute; top: 2px; left: 2px; color: #9aa5ad; font-size: 13px; }
      .pairing-layer-grid button i { width: 4px; min-height: 2px; border-radius: 2px 2px 0 0; }
      .pairing-layer-grid button i.a { background: #b8c2ca; }
      .pairing-layer-grid button i.b { background: #7694ab; }
      .pairing-layer-grid button.selected { border-color: #000000; background: #eef4f7; box-shadow: inset 0 0 0 1px #708ea5; }

      .pairing-active-route { margin-top: 9px; padding: 9px 11px; display: flex; justify-content: space-between; align-items: center; gap: 14px; border: 1px solid #dce4e9; border-radius: 5px; background: #f9fbfc; }
      .pairing-active-route > div > span { color: #748591; font-size: 14px; font-weight: 700; }
      .pairing-active-route > small { max-width: 380px; color: #84929c; font-size: 14px; text-align: right; }
      .pairing-expert-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
      .pairing-expert-chips b { padding: 3px 6px; border: 1px solid #ccd9e2; border-radius: 4px; background: #fff; color: #000000; font-size: 14px; }
      .pairing-expert-chips small { margin-left: 2px; color: #82929d; font-size: 12px; }

      .pairing-pairs-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-top: 9px; }
      .pairing-pair-list { padding: 11px; border: 1px solid #d8e1e7; border-radius: 6px; background: #fff; }
      .pairing-pair-head { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 8px; }
      .pairing-pair-head strong { color: #000000; font-size: 16px; }
      .pairing-pair-head span { color: #84919b; font-size: 13px; }
      .pairing-pair-rows { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 5px; }
      .pair-row { display: grid; grid-template-columns: minmax(58px, 1fr) 42px minmax(58px, 1fr); align-items: center; padding: 5px; border: 1px solid #e2e8ec; border-radius: 4px; background: #fbfcfd; }
      .pair-row.conflict { border-color: #dfb0ad; background: #fff8f7; }
      .pair-expert { min-width: 0; padding: 4px 5px; text-align: center; border-radius: 3px; color: #9aa4ab; }
      .pair-expert strong { display: block; font-size: 14px; }
      .pair-expert small { display: block; font-size: 11px; }
      .pair-expert.active { background: #eaf2f7; color: #000000; }
      .pair-row.conflict .pair-expert.active { background: #fdebea; color: #9b4d48; }
      .pair-link { position: relative; height: 18px; display: grid; place-items: center; }
      .pair-link span { width: 100%; height: 1px; background: #b9c5cd; }
      .pair-link b { position: absolute; padding: 1px 4px; border-radius: 7px; background: #bd6963; color: #fff; font-size: 11px; }
      .pairing-more { display: block; margin-top: 7px; color: #8c989f; font-size: 12px; }

      .pairing-local-note { margin-top: 9px; padding: 9px 11px; border-left: 3px solid #849fb3; background: #f6f9fb; color: #000000; font-size: 14px; line-height: 1.55; }
      .pairing-local-note strong { color: #000000; }

      @media (max-width: 1250px) {
        .pairing-controls { grid-template-columns: 1fr 1fr auto 1fr; }
        .pairing-run { grid-column: 1 / -1; }
        .pairing-layer-grid { grid-template-columns: repeat(15, minmax(22px, 1fr)); }
        .pairing-pair-rows { grid-template-columns: 1fr; }
      }
    `}</style>
  );
}


export default PairingComparison;
