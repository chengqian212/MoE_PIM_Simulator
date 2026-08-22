import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE = "http://127.0.0.1:8000";


function pct(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return `${(n * 100).toFixed(digits)}%`;
}


function number(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return n.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}


function area(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return number(n);
}


function candidateLabel(row) {
  if (!row) return "--";
  const orientation = row.orientation === "Transposed" ? "转置" : "原方向";
  return `#${row.spatial_rank} · ${row.H}×${row.W} · ${orientation} · N=${row.N}`;
}


function chunkText(detail) {
  const chunks = detail?.template?.chunks ?? [];
  if (!chunks.length) return "--";
  return chunks
    .map((chunk) => `${chunk.rows}×${chunk.cols}`)
    .join(" + ");
}


function metricDelta(a, b, mode = "lower") {
  const x = Number(a);
  const y = Number(b);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const delta = y - x;
  const better = mode === "higher" ? delta > 0 : delta < 0;
  const equal = Math.abs(delta) < 1e-12;
  return { delta, better, equal };
}


function PlaneDiagram({ plane }) {
  if (!plane) {
    return <div className="spatial-plane-empty">暂无 Plane 数据</div>;
  }

  const H = Number(plane.H) || 1;
  const W = Number(plane.W) || 1;
  const slots = plane.slots ?? [];

  return (
    <div className="spatial-plane-wrap">
      <div className="spatial-plane-head">
        <span>尾部 Plane #{plane.plane_id}</span>
        <b>{pct(plane.utilization)}</b>
      </div>

      <svg
        className="spatial-plane-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Plane ${plane.plane_id} occupancy`}
      >
        <rect
          className="plane-frame"
          x="0"
          y="0"
          width={W}
          height={H}
        />

        {slots.map((slot, index) => (
          <g key={slot.slot_id ?? index}>
            <rect
              className={`plane-slot slot-${index % 4}`}
              x={Number(slot.y) || 0}
              y={Number(slot.x) || 0}
              width={Number(slot.slot_cols) || 0}
              height={Number(slot.slot_rows) || 0}
            />
          </g>
        ))}
      </svg>

      <div className="spatial-plane-caption">
        尾部 Plane 用来直观看余数块与碎片；并不代表所有 Plane 的平均占用。
      </div>
    </div>
  );
}


function Metric({ label, value, sub, strong = false }) {
  return (
    <div className={strong ? "spatial-metric strong" : "spatial-metric"}>
      <span>{label}</span>
      <b>{value}</b>
      {sub && <small>{sub}</small>}
    </div>
  );
}


function SchemeCard({
  side,
  detail,
  currentKey,
}) {
  const candidate = detail?.candidate;
  if (!candidate) {
    return <div className="spatial-scheme-card loading">正在读取方案 {side}...</div>;
  }

  const isCurrent = candidate.key === currentKey;
  const chunks = detail?.template?.chunks ?? [];

  return (
    <div className={isCurrent ? "spatial-scheme-card current" : "spatial-scheme-card"}>
      <div className="spatial-scheme-title">
        <div>
          <span>方案 {side}</span>
          <h3>{candidate.H} × {candidate.W}</h3>
        </div>
        <div className="spatial-scheme-tags">
          <b>{candidate.orientation === "Transposed" ? "转置模板" : "原方向模板"}</b>
          <b>N={candidate.N}</b>
          {isCurrent && <strong>当前最终方案</strong>}
        </div>
      </div>

      <div className="spatial-partition-flow">
        <div>
          <span>原矩阵</span>
          <b>7168 × 2048</b>
        </div>
        <i>→</i>
        <div className="chunks">
          <span>切分结果 · {chunks.length || "--"} 块/矩阵</span>
          <b>{chunkText(detail)}</b>
        </div>
      </div>

      <div className="spatial-metric-grid">
        <Metric label="Packing" value={pct(candidate.packing_utilization)} strong />
        <Metric label="Hardware" value={pct(candidate.hardware_utilization)} />
        <Metric label="P / 实际 Plane" value={number(candidate.P)} />
        <Metric label="D / 每 SC 深度" value={number(candidate.D)} />
        <Metric label="空 Plane" value={number(candidate.empty_plane_slots)} />
        <Metric label="内部碎片面积" value={area(candidate.internal_fragmentation)} sub="cell-area" />
      </div>

      <PlaneDiagram plane={detail?.layout?.tail_plane} />
    </div>
  );
}


function ComparisonStrip({ a, b }) {
  if (!a || !b) return null;

  const packing = metricDelta(a.packing_utilization, b.packing_utilization, "higher");
  const hardware = metricDelta(a.hardware_utilization, b.hardware_utilization, "higher");
  const fragmentation = metricDelta(a.internal_fragmentation, b.internal_fragmentation, "lower");
  const sameN = Number(a.N) === Number(b.N);

  return (
    <div className="spatial-delta-strip">
      <div>
        <span>方案 B Packing 差值</span>
        <b className={packing?.better ? "good" : ""}>
          {packing ? `${packing.delta >= 0 ? "+" : ""}${(packing.delta * 100).toFixed(2)} pp` : "--"}
        </b>
      </div>
      <div>
        <span>Hardware 差值</span>
        <b className={hardware?.better ? "good" : ""}>
          {hardware ? `${hardware.delta >= 0 ? "+" : ""}${(hardware.delta * 100).toFixed(2)} pp` : "--"}
        </b>
      </div>
      <div>
        <span>内部碎片变化</span>
        <b className={fragmentation?.better ? "good" : ""}>
          {fragmentation ? `${fragmentation.delta > 0 ? "+" : ""}${area(fragmentation.delta)}` : "--"}
        </b>
      </div>
      <div>
        <span>N 是否一致</span>
        <b>{sameN ? `是 · N=${a.N}` : `否 · ${a.N} → ${b.N}`}</b>
      </div>
    </div>
  );
}


export default function SpatialLayoutComparison() {
  const [catalog, setCatalog] = useState(null);
  const [keyA, setKeyA] = useState("");
  const [keyB, setKeyB] = useState("");
  const [detailA, setDetailA] = useState(null);
  const [detailB, setDetailB] = useState(null);
  const [nFilter, setNFilter] = useState("4");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE}/api/comparison/spatial-catalog`);
        if (!response.ok) throw new Error("读取 spatial_candidates.json 失败");
        const data = await response.json();
        if (cancelled) return;
        setCatalog(data);
        setKeyA(data.default_a ?? "");
        setKeyB(data.default_b ?? "");
        setError("");
      } catch (err) {
        if (!cancelled) setError(err.message || "空间候选读取失败");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadCatalog();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!keyA || !keyB) return undefined;
    let cancelled = false;

    async function loadDetails() {
      try {
        setLoading(true);
        const [aResponse, bResponse] = await Promise.all([
          fetch(`${API_BASE}/api/comparison/spatial-detail?key=${encodeURIComponent(keyA)}`),
          fetch(`${API_BASE}/api/comparison/spatial-detail?key=${encodeURIComponent(keyB)}`),
        ]);
        if (!aResponse.ok || !bResponse.ok) {
          throw new Error("读取 Layout 详情失败，请检查 results/layouts。 ");
        }
        const [a, b] = await Promise.all([aResponse.json(), bResponse.json()]);
        if (cancelled) return;
        setDetailA(a);
        setDetailB(b);
        setError("");
      } catch (err) {
        if (!cancelled) setError(err.message || "空间方案详情读取失败");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadDetails();
    return () => { cancelled = true; };
  }, [keyA, keyB]);

  const candidates = catalog?.candidates ?? [];
  const filtered = useMemo(() => {
    if (nFilter === "all") return candidates;
    return candidates.filter((row) => String(row.N) === nFilter);
  }, [candidates, nFilter]);

  useEffect(() => {
    if (!filtered.length) return;

    const hasA = filtered.some((row) => row.key === keyA);
    const hasB = filtered.some((row) => row.key === keyB);

    if (!hasA) {
      setKeyA(filtered[0].key);
    }

    if (!hasB) {
      const currentInFilter = filtered.find((row) => row.key === catalog?.current_key);
      setKeyB((currentInFilter ?? filtered[Math.min(1, filtered.length - 1)]).key);
    }
  }, [filtered, keyA, keyB, catalog?.current_key]);

  if (error && !catalog) {
    return <div className="spatial-compare-error">{error}</div>;
  }

  return (
    <div className="spatial-comparison">
      <div className="spatial-compare-head">
        <div>
          <span className="page-kicker">SPATIAL PLAN COMPARISON</span>
          <h3>空间方案对比</h3>
          <p>
            装箱算法固定为 <b>面积降序 + MaxRects-BSSF + 90° 旋转</b>；这里比较 H/W、切分方向与 N 带来的空间差异。
          </p>
        </div>
        <div className="spatial-catalog-count">
          {catalog?.valid_candidate_count ?? "--"} 个合法候选
        </div>
      </div>

      <div className="spatial-controls">
        <label>
          <span>候选范围</span>
          <select value={nFilter} onChange={(event) => setNFilter(event.target.value)}>
            <option value="4">N=4 · 与当前方案同并行度</option>
            <option value="3">N=3</option>
            <option value="2">N=2</option>
            <option value="all">全部候选</option>
          </select>
        </label>

        <label>
          <span>方案 A</span>
          <select value={keyA} onChange={(event) => setKeyA(event.target.value)}>
            {filtered.map((row) => (
              <option key={row.key} value={row.key}>{candidateLabel(row)}</option>
            ))}
          </select>
        </label>

        <div className="spatial-vs">VS</div>

        <label>
          <span>方案 B</span>
          <select value={keyB} onChange={(event) => setKeyB(event.target.value)}>
            {filtered.map((row) => (
              <option key={row.key} value={row.key}>{candidateLabel(row)}</option>
            ))}
          </select>
        </label>
      </div>

      {error && <div className="spatial-inline-error">{error}</div>}

      <ComparisonStrip a={detailA?.candidate} b={detailB?.candidate} />

      <div className={loading ? "spatial-ab-grid is-loading" : "spatial-ab-grid"}>
        <SchemeCard side="A" detail={detailA} currentKey={catalog?.current_key} />
        <SchemeCard side="B" detail={detailB} currentKey={catalog?.current_key} />
      </div>

      <div className="spatial-runtime-note">
        <b>为什么最终运行周期差异不明显？</b>
        <span>
          当前执行模型中，Plane 内 x/y 几何位置不会改变 Weight-Cube 的 1-cycle Compute / 1-cycle Switch 规则。
          因此这一阶段主要优化 <strong>容量、P/D、内部碎片和硬件利用率</strong>；真正明显的 Prefill / Decode 周期收益来自后续 Trace-aware Sub-Cube Mapping 与 Scheduler。
        </span>
      </div>
    </div>
  );
}
