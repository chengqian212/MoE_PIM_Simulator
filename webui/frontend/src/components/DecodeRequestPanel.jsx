import {
  useEffect,
  useMemo,
  useState,
} from "react";

import ActiveMappingPanel from "./ActiveMappingPanel";


const API_BASE = "http://127.0.0.1:8000";



function DecodeRequestPanel() {
  const [categories, setCategories] = useState([]);
  const [category, setCategory] = useState("");
  const [token, setToken] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");



  useEffect(() => {
    async function loadCategories() {
      try {
        const response = await fetch(`${API_BASE}/api/trace/categories`);
        if (!response.ok) {
          throw new Error(await response.text());
        }

        const data = await response.json();
        setCategories(data.items ?? []);
      } catch (err) {
        console.error(err);
        setError("读取 Trace 类别失败。" + err.message);
      }
    }

    loadCategories();
  }, []);


  async function loadRandomDecode(nextCategory = category) {
    try {
      setLoading(true);
      setError("");

      let url = `${API_BASE}/api/request/decode/random`;
      if (nextCategory) {
        url += `?category=${encodeURIComponent(nextCategory)}`;
      }

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      setToken(data);
      setSelectedLayer(0);
    } catch (err) {
      console.error(err);
      setError("读取纯 Decode Token 失败。" + err.message);
    } finally {
      setLoading(false);
    }
  }


  useEffect(() => {
    loadRandomDecode("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);


  const layers = token?.layers ?? [];
  const currentLayer = layers.find((layer) => layer.layer_id === selectedLayer) ?? null;
  const routedExperts = currentLayer?.routed_experts ?? [];
  const sharedExpertId = currentLayer?.shared_expert ?? 256;


  function changeLayer(layerId) {
    const maxLayer = Math.max(layers.length - 1, 0);
    const safe = Math.max(0, Math.min(maxLayer, Number(layerId)));
    setSelectedLayer(safe);
  }


  const activeExpertCount = useMemo(
    () => routedExperts.length + (currentLayer ? 1 : 0),
    [routedExperts, currentLayer]
  );


  return (
    <div className="decode-panel">
      <div className="decode-note">
        <strong>Decode 正式口径：</strong>
        只从真实 Trace 的 <b>segment1+</b> 中抽取 singleton Token；每个 Token 依次经过 58 个 MoE Layer。
        本页只展示请求来源、逐层 Router 与基础物理映射；周期级调度与 3D 播放统一放在 04 调度可视化。
      </div>

      <div className="decode-controls">
        <div className="decode-control-item">
          <label>数据集 / Dataset</label>
          <strong>Chinese-SimpleQA</strong>
        </div>

        <div className="decode-control-item category">
          <label>类别 / Category</label>
          <select
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="">全部类别</option>
            {categories.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}（{item.file_count}）
              </option>
            ))}
          </select>
        </div>

        <button
          className="decode-primary"
          disabled={loading}
          onClick={() => loadRandomDecode()}
        >
          {loading ? "正在读取..." : "随机真实 Decode Token"}
        </button>
      </div>

      {error && <div className="decode-error">{error}</div>}

      {token && (
        <>
          <div className="decode-source">
            <SourceItem label="来源文件 / File" value={token.source?.filename} wide />
            <SourceItem label="类别 / Category" value={token.source?.category} />
            <SourceItem label="Segment" value={`${token.source?.segment_index}（Decode）`} />
            <SourceItem label="Token Index" value={token.source?.token_index} />
            <SourceItem label="MoE Layers" value={token.num_layers} />
          </div>

          <div className="decode-layer-control">
            <span>模型层 / Layer</span>
            <button disabled={selectedLayer <= 0} onClick={() => changeLayer(selectedLayer - 1)}>‹</button>
            <strong>L{selectedLayer}</strong>
            <span>/ 57</span>
            <button disabled={selectedLayer >= 57} onClick={() => changeLayer(selectedLayer + 1)}>›</button>
            <input
              type="range"
              min="0"
              max="57"
              value={selectedLayer}
              onChange={(event) => changeLayer(Number(event.target.value))}
            />
            <div>
              Trace Layer <b>{currentLayer?.trace_layer_id ?? "--"}</b>
            </div>
          </div>

          <section className="decode-route-section">
            <div className="decode-route-header">
              <div>
                <h3>L{selectedLayer} 当前路由 / Current Route</h3>
                <p>Top-8 Routed Experts + Shared Expert E256</p>
              </div>
              <div className="decode-active-count">
                激活 Expert <strong>{activeExpertCount}</strong>
              </div>
            </div>

            <div className="decode-experts">
              {routedExperts.map((expertId, index) => (
                <div className="decode-expert" key={`${selectedLayer}-${expertId}`}>
                  <span>#{index + 1}</span>
                  <strong>E{expertId}</strong>
                </div>
              ))}
              <div className="decode-expert shared">
                <span>共享</span>
                <strong>E{sharedExpertId}</strong>
              </div>
            </div>
          </section>

          <details className="decode-mapping-details">
            <summary>
              <strong>当前 Layer 物理映射 / Physical Mapping</strong>
              <span>9 Experts · 27 Matrices</span>
            </summary>
            <ActiveMappingPanel
              layerId={selectedLayer}
              routedExpertIds={routedExperts}
              sharedExpertId={sharedExpertId}
            />
          </details>

          <div className="decode-scheduler-handoff">
            <div>
              <strong>下一步：04 调度可视化 / Scheduler</strong>
              <span>当前页到 Router 与基础 Mapping 为止。</span>
            </div>
            <p>
              单层 Timeline、16 个 Sub-Cube 实时状态、Switch / Gate / Up / Down 周期，以及完整 58 层 Token Playback 均在 04 页面查看。
            </p>
          </div>
        </>
      )}

      <Style />
    </div>
  );
}


function SourceItem({ label, value, wide = false }) {
  return (
    <div className={wide ? "wide" : ""}>
      <span>{label}</span>
      <strong title={String(value ?? "")}>{value ?? "--"}</strong>
    </div>
  );
}


function Style() {
  return (
    <style>{`
      .decode-panel { width: 100%; }

      .decode-note {
        margin-bottom: 9px;
        padding: 9px 12px;
        border: 1px solid #d9e3eb;
        border-left: 4px solid #809eb8;
        border-radius: 5px;
        background: #f7fafc;
        color: #617181;
        font-size: 16px;
        line-height: 1.55;
      }
      .decode-note strong { color: #3e5f7d; }

      .decode-controls {
        min-height: 63px;
        padding: 8px 10px;
        display: flex;
        align-items: flex-end;
        gap: 10px;
        border: 1px solid #dce2e8;
        border-radius: 6px;
        background: #fff;
      }

      .decode-control-item {
        min-width: 175px;
        display: flex;
        flex-direction: column;
        gap: 5px;
      }
      .decode-control-item.category { flex: 1; max-width: 330px; }
      .decode-control-item label {
        color: #798693;
        font-size: 16px;
        font-weight: 650;
      }
      .decode-control-item > strong,
      .decode-control-item select {
        height: 34px;
        padding: 0 10px;
        display: flex;
        align-items: center;
        border: 1px solid #d7dee5;
        border-radius: 4px;
        background: #f8fafb;
        color: #435263;
        font-size: 15px;
      }
      .decode-control-item select { width: 100%; background: #fff; }

      .decode-primary {
        height: 34px;
        margin-left: auto;
        padding: 0 16px;
        border: 1px solid #6587a5;
        border-radius: 4px;
        background: #7395b3;
        color: #fff;
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
      }

      .decode-error {
        margin-top: 8px;
        padding: 9px 11px;
        border: 1px solid #e0bcbc;
        border-radius: 5px;
        background: #fff5f5;
        color: #995858;
        font-size: 16px;
      }

      .decode-source {
        min-height: 50px;
        margin-top: 9px;
        padding: 7px 10px;
        display: grid;
        grid-template-columns: minmax(230px, 1.5fr) 1fr 150px 115px 105px;
        gap: 8px;
        border: 1px solid #dfe4e9;
        border-radius: 5px;
        background: #fff;
      }
      .decode-source > div { min-width: 0; }
      .decode-source span {
        display: block;
        color: #8a94a0;
        font-size: 15px;
      }
      .decode-source strong {
        display: block;
        margin-top: 2px;
        overflow: hidden;
        color: #40505f;
        font-size: 15px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .decode-layer-control {
        min-height: 47px;
        margin-top: 8px;
        padding: 6px 9px;
        display: flex;
        align-items: center;
        gap: 7px;
        border: 1px solid #dfe4e8;
        border-radius: 5px;
        background: #fff;
        color: #77838e;
        font-size: 16px;
      }
      .decode-layer-control > button {
        width: 31px;
        height: 31px;
        border: 1px solid #d5dce3;
        border-radius: 4px;
        background: #fff;
        color: #536171;
        font-size: 20px;
        cursor: pointer;
      }
      .decode-layer-control > strong { color: #335b7e; font-size: 18px; }
      .decode-layer-control > input { flex: 1; min-width: 200px; }
      .decode-layer-control > div { min-width: 130px; text-align: right; }
      .decode-layer-control b { color: #40586e; font-size: 16px; }

      .decode-route-section {
        margin-top: 8px;
        padding: 10px;
        border: 1px solid #dfe4e8;
        border-radius: 6px;
        background: #fff;
      }
      .decode-route-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      .decode-route-header h3 { margin: 0; color: #435362; font-size: 18px; }
      .decode-route-header p { margin: 3px 0 0; color: #89939d; font-size: 16px; }
      .decode-active-count { color: #526579; font-size: 16px; }
      .decode-active-count strong { margin-left: 6px; color: #365a78; font-size: 20px; }

      .decode-experts {
        margin-top: 9px;
        display: grid;
        grid-template-columns: repeat(9, minmax(82px, 1fr));
        gap: 6px;
      }
      .decode-expert {
        min-height: 50px;
        padding: 6px 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        border: 1px solid #d8e0e7;
        border-radius: 5px;
        background: #f7f9fb;
      }
      .decode-expert span { color: #526579; font-size: 15px; }
      .decode-expert strong { color: #415d76; font-size: 18px; }
      .decode-expert.shared { border-color: #b8a6cc; background: #f3eef8; }
      .decode-expert.shared span,
      .decode-expert.shared strong { color: #735b91; }

      .decode-mapping-details {
        margin-top: 8px;
        border: 1px solid #dfe4e8;
        border-radius: 6px;
        background: #fff;
      }
      .decode-mapping-details summary {
        min-height: 45px;
        padding: 0 11px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #526170;
        font-size: 15px;
        cursor: pointer;
      }
      .decode-mapping-details summary span { color: #89939d; font-size: 16px; }

      .decode-scheduler-handoff {
        margin-top: 8px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        border: 1px solid #cfdbe5;
        border-left: 4px solid #7395b3;
        border-radius: 6px;
        background: #f7fafc;
      }
      .decode-scheduler-handoff > div {
        min-width: 285px;
      }
      .decode-scheduler-handoff strong {
        display: block;
        color: #405f7b;
        font-size: 17px;
      }
      .decode-scheduler-handoff span {
        display: block;
        margin-top: 3px;
        color: #687887;
        font-size: 15px;
      }
      .decode-scheduler-handoff p {
        margin: 0;
        color: #657585;
        font-size: 15px;
        line-height: 1.55;
      }

      @media (max-width: 1200px) {
        .decode-experts { grid-template-columns: repeat(5, 1fr); }
        .decode-source { grid-template-columns: repeat(2, 1fr); }
        .decode-scheduler-handoff { align-items: flex-start; flex-direction: column; gap: 6px; }
      }
    `}</style>
  );
}


export default DecodeRequestPanel;
