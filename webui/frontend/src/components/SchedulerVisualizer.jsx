import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import LayerTimeline from "./LayerTimeline";
import ExecutionCube3D from "./ExecutionCube3D";
import FullTokenPlaybackPanel from "./FullTokenPlaybackPanel";


const API_BASE = "http://127.0.0.1:8000";


function makeIdleStates() {
  return Array.from(
    { length: 16 },
    (_, sc) => ({
      subcube_id: sc,
      state: "idle",
      task: null,
    })
  );
}


function SchedulerVisualizer() {
  const [categories, setCategories] = useState([]);
  const [category, setCategory] = useState("");
  const [token, setToken] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [viewMode, setViewMode] = useState("layer");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [layerCycle, setLayerCycle] = useState(0);
  const [layerStates, setLayerStates] = useState(makeIdleStates());

  const [fullFrame, setFullFrame] = useState({
    globalCycle: 0,
    layerId: 0,
    localCycle: 0,
    subcubeStates: makeIdleStates(),
  });


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
        setError("读取 Trace 类别失败：" + err.message);
      }
    }

    loadCategories();
  }, []);


  const loadRandomDecode = useCallback(async (nextCategory = category) => {
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
      setViewMode("layer");
      setLayerCycle(0);
      setLayerStates(makeIdleStates());
      setFullFrame({
        globalCycle: 0,
        layerId: 0,
        localCycle: 0,
        subcubeStates: makeIdleStates(),
      });
    } catch (err) {
      console.error(err);
      setError("读取 Decode Token 失败：" + err.message);
    } finally {
      setLoading(false);
    }
  }, [category]);


  useEffect(() => {
    loadRandomDecode("");
  }, [loadRandomDecode]);


  const layers = token?.layers ?? [];

  const currentLayer = useMemo(
    () =>
      layers.find(
        (item) => item.layer_id === selectedLayer
      ) ?? null,
    [layers, selectedLayer]
  );

  const routedExperts = currentLayer?.routed_experts ?? [];


  function changeLayer(layerId) {
    const safe = Math.max(0, Math.min(57, Number(layerId)));
    setSelectedLayer(safe);
    setViewMode("layer");
  }


  const handleLayerCycle = useCallback((cycle) => {
    setLayerCycle(cycle);
  }, []);


  const handleLayerStates = useCallback((states) => {
    setLayerStates(states);
  }, []);


  const handleFullFrame = useCallback((frame) => {
    setFullFrame(frame);
  }, []);


  const cubeCycle =
    viewMode === "full"
      ? fullFrame.globalCycle
      : layerCycle;

  const cubeStates =
    viewMode === "full"
      ? fullFrame.subcubeStates
      : layerStates;

  const activeCount = useMemo(
    () =>
      cubeStates.filter(
        (item) => item.state !== "idle"
      ).length,
    [cubeStates]
  );


  return (
    <div className="scheduler-visualizer">

      <div className="scheduler-page-header">
        <div>
          <div className="scheduler-kicker">
            第五步 · 精确推理调度 / EXACT SCHEDULING
          </div>

          <h2>
            调度可视化 / Scheduler Visualizer
          </h2>

          <p>
            将同一个 Decode Token 的调度结果同时从时间维度和空间维度展开：左侧看周期图，右侧看 16 个 Sub-Cube 的实时执行状态。
          </p>
        </div>

        <div className="scheduler-scope-badge">
          MoE Expert Only
        </div>
      </div>


      <div className="scheduler-source-bar">
        <div className="scheduler-source-item dataset">
          <span>数据集 / Dataset</span>
          <strong>Chinese-SimpleQA</strong>
        </div>

        <div className="scheduler-source-item category">
          <span>类别 / Category</span>
          <select
            value={category}
            onChange={(event) =>
              setCategory(event.target.value)
            }
          >
            <option value="">全部类别</option>
            {categories.map((item) => (
              <option
                key={item.name}
                value={item.name}
              >
                {item.name}（{item.file_count}）
              </option>
            ))}
          </select>
        </div>

        {token && (
          <>
            <SourceCompact
              label="文件 / File"
              value={token.source?.filename}
              wide
            />

            <SourceCompact
              label="Segment"
              value={`${token.source?.segment_index}（Decode）`}
            />

            <SourceCompact
              label="Token"
              value={token.source?.token_index}
            />
          </>
        )}

        <button
          className="scheduler-random-button"
          disabled={loading}
          onClick={() => loadRandomDecode()}
        >
          {loading
            ? "正在读取..."
            : "换一个真实 Decode Token"}
        </button>
      </div>


      {error && (
        <div className="scheduler-error">
          {error}
        </div>
      )}


      {token && (
        <>
          <div className="scheduler-mode-tabs">
            <button
              className={
                viewMode === "layer"
                  ? "active"
                  : ""
              }
              onClick={() => setViewMode("layer")}
            >
              <strong>单层调度</strong>
              <span>Layer Timeline + 3D SC</span>
            </button>

            <button
              className={
                viewMode === "full"
                  ? "active"
                  : ""
              }
              onClick={() => setViewMode("full")}
            >
              <strong>完整 58 层</strong>
              <span>Full Token Playback</span>
            </button>

            <div className="scheduler-live-summary">
              <div>
                <span>当前 Cycle</span>
                <strong>{cubeCycle}</strong>
              </div>

              <div>
                <span>活跃 SC</span>
                <strong>{activeCount} / 16</strong>
              </div>

              <div>
                <span>当前 Layer</span>
                <strong>
                  L{viewMode === "full"
                    ? fullFrame.layerId
                    : selectedLayer}
                </strong>
              </div>
            </div>
          </div>


          {viewMode === "layer" && (
            <>
              <div className="scheduler-layer-toolbar">
                <div className="scheduler-layer-title">
                  <span>模型层 / Layer</span>
                  <strong>L{selectedLayer}</strong>
                </div>

                <button
                  disabled={selectedLayer <= 0}
                  onClick={() => changeLayer(selectedLayer - 1)}
                >
                  ‹
                </button>

                <input
                  type="range"
                  min="0"
                  max="57"
                  value={selectedLayer}
                  onChange={(event) =>
                    changeLayer(Number(event.target.value))
                  }
                />

                <button
                  disabled={selectedLayer >= 57}
                  onClick={() => changeLayer(selectedLayer + 1)}
                >
                  ›
                </button>

                <div className="scheduler-route-chips">
                  {routedExperts.map((expertId, index) => (
                    <span key={`${selectedLayer}-${expertId}`}>
                      #{index + 1} E{expertId}
                    </span>
                  ))}
                  <span className="shared">
                    Shared E{currentLayer?.shared_expert ?? 256}
                  </span>
                </div>
              </div>


              <div className="scheduler-two-view-grid">
                <section className="scheduler-view-card timeline-card">
                  <div className="scheduler-view-caption">
                    <div>
                      <h3>周期时间线 / Timeline</h3>
                      <p>
                        每一行对应一个 Sub-Cube；S 表示 Weight-Cube 切换，G/U/D 表示 Gate/Up/Down 计算。
                      </p>
                    </div>
                    <span>时间视角</span>
                  </div>

                  <LayerTimeline
                    layerId={selectedLayer}
                    routedExpertIds={routedExperts}
                    onCycleChange={handleLayerCycle}
                    onSubcubeStatesChange={handleLayerStates}
                  />
                </section>


                <section className="scheduler-view-card cube-card">
                  <div className="scheduler-view-caption">
                    <div>
                      <h3>Sub-Cube 实时状态 / 3D Execution</h3>
                      <p>
                        与左侧同一个 Cycle 同步，直接查看 16 个 SC 当前是空闲、切换还是计算。
                      </p>
                    </div>
                    <span>空间视角</span>
                  </div>

                  <ExecutionCube3D
                    currentCycle={layerCycle}
                    subcubeStates={layerStates}
                  />
                </section>
              </div>
            </>
          )}


          {viewMode === "full" && (
            <div className="scheduler-full-layout">
              <section className="scheduler-view-card full-runner-card">
                <div className="scheduler-view-caption">
                  <div>
                    <h3>完整 Token / 58 Layers</h3>
                    <p>
                      从 Layer 0 连续播放到 Layer 57，显示总周期、各层周期以及全局 Cycle 位置。
                    </p>
                  </div>
                  <span>全局时间轴</span>
                </div>

                <FullTokenPlaybackPanel
                  token={token}
                  onSelectLayer={setSelectedLayer}
                  onPlaybackFrame={handleFullFrame}
                />
              </section>

              <section className="scheduler-view-card full-cube-card">
                <div className="scheduler-view-caption">
                  <div>
                    <h3>完整 Token 的 3D 执行动画</h3>
                    <p>
                      当前 Global Cycle：{fullFrame.globalCycle}；Layer：L{fullFrame.layerId}；层内 Cycle：{fullFrame.localCycle}。
                    </p>
                  </div>
                  <span>空间联动</span>
                </div>

                <ExecutionCube3D
                  currentCycle={fullFrame.globalCycle}
                  subcubeStates={fullFrame.subcubeStates}
                />
              </section>
            </div>
          )}
        </>
      )}

      <Style />
    </div>
  );
}


function SourceCompact({
  label,
  value,
  wide = false,
}) {
  return (
    <div
      className={
        wide
          ? "scheduler-source-item file wide"
          : "scheduler-source-item"
      }
    >
      <span>{label}</span>
      <strong title={String(value ?? "")}>
        {value ?? "--"}
      </strong>
    </div>
  );
}


function Style() {
  return (
    <style>{`
      .scheduler-visualizer {
        width: 100%;
      }

      .scheduler-page-header {
        min-height: 64px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
      }

      .scheduler-kicker {
        margin-bottom: 5px;
        color: #7e91a3;
        font-size: 16px;
        font-weight: 750;
        letter-spacing: 0.7px;
      }

      .scheduler-page-header h2 {
        margin: 0 0 5px;
        color: #2f3b47;
        font-size: 24px;
        font-weight: 720;
      }

      .scheduler-page-header p {
        margin: 0;
        color: #758291;
        font-size: 15px;
        line-height: 1.55;
      }

      .scheduler-scope-badge {
        padding: 7px 11px;
        border: 1px solid #cbd8e3;
        border-radius: 5px;
        background: #f5f9fc;
        color: #526f88;
        font-size: 16px;
        font-weight: 700;
        white-space: nowrap;
      }

      .scheduler-source-bar {
        min-height: 62px;
        margin-bottom: 9px;
        padding: 8px 10px;
        display: flex;
        align-items: stretch;
        gap: 8px;
        border: 1px solid #dce3e9;
        border-radius: 7px;
        background: #ffffff;
      }

      .scheduler-source-item {
        min-width: 105px;
        padding: 5px 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-right: 1px solid #edf0f3;
      }

      .scheduler-source-item.dataset {
        min-width: 145px;
      }

      .scheduler-source-item.category {
        min-width: 180px;
      }

      .scheduler-source-item.file.wide {
        min-width: 210px;
        max-width: 280px;
      }

      .scheduler-source-item span {
        margin-bottom: 4px;
        color: #8995a1;
        font-size: 15px;
        font-weight: 650;
      }

      .scheduler-source-item strong {
        overflow: hidden;
        color: #3c4a57;
        font-size: 15px;
        font-weight: 700;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .scheduler-source-item select {
        height: 31px;
        border: 1px solid #d5dde5;
        border-radius: 4px;
        background: #ffffff;
        color: #455461;
        font-size: 16px;
      }

      .scheduler-random-button {
        min-width: 180px;
        margin-left: auto;
        padding: 0 13px;
        border: 1px solid #6c8da9;
        border-radius: 5px;
        background: #7697b3;
        color: #ffffff;
        font-size: 16px;
        font-weight: 700;
        cursor: pointer;
      }

      .scheduler-random-button:disabled {
        opacity: 0.5;
        cursor: default;
      }

      .scheduler-error {
        margin-bottom: 9px;
        padding: 9px 11px;
        border: 1px solid #e0bebe;
        border-radius: 5px;
        background: #fff5f5;
        color: #955959;
        font-size: 16px;
      }

      .scheduler-mode-tabs {
        min-height: 66px;
        margin-bottom: 9px;
        padding: 6px;
        display: flex;
        align-items: stretch;
        gap: 7px;
        border: 1px solid #dce3e9;
        border-radius: 7px;
        background: #ffffff;
      }

      .scheduler-mode-tabs > button {
        min-width: 180px;
        padding: 7px 12px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        border: 1px solid #dbe2e8;
        border-radius: 5px;
        background: #f9fafb;
        color: #65727f;
        cursor: pointer;
      }

      .scheduler-mode-tabs > button.active {
        border-color: #3b82f6;
        background: #dbeafe;
        color: #123f70;
        box-shadow: inset 4px 0 0 #4F7195;
      }

      .scheduler-mode-tabs > button strong {
        margin-bottom: 3px;
        font-size: 16px;
      }

      .scheduler-mode-tabs > button span {
        font-size: 15px;
      }

      .scheduler-live-summary {
        margin-left: auto;
        display: flex;
        align-items: stretch;
        gap: 6px;
      }

      .scheduler-live-summary > div {
        min-width: 92px;
        padding: 5px 9px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid #e0e5e9;
        border-radius: 5px;
        background: #fafbfc;
      }

      .scheduler-live-summary span {
        margin-bottom: 3px;
        color: #8a95a0;
        font-size: 15px;
      }

      .scheduler-live-summary strong {
        color: #3b4b59;
        font-size: 18px;
      }

      .scheduler-layer-toolbar {
        min-height: 56px;
        margin-bottom: 9px;
        padding: 7px 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #dce3e9;
        border-radius: 7px;
        background: #ffffff;
      }

      .scheduler-layer-title {
        min-width: 118px;
      }

      .scheduler-layer-title span {
        display: block;
        margin-bottom: 2px;
        color: #89949f;
        font-size: 15px;
      }

      .scheduler-layer-title strong {
        color: #385269;
        font-size: 20px;
      }

      .scheduler-layer-toolbar > button {
        width: 34px;
        height: 34px;
        border: 1px solid #d4dde4;
        border-radius: 4px;
        background: #ffffff;
        color: #52616e;
        font-size: 22px;
        cursor: pointer;
      }

      .scheduler-layer-toolbar > button:disabled {
        opacity: 0.35;
      }

      .scheduler-layer-toolbar > input[type="range"] {
        width: 160px;
      }

      .scheduler-route-chips {
        min-width: 0;
        margin-left: 8px;
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
      }

      .scheduler-route-chips span {
        padding: 4px 7px;
        border: 1px solid #d5e0e9;
        border-radius: 4px;
        background: #f5f9fc;
        color: #55738c;
        font-size: 15px;
        font-weight: 700;
      }

      .scheduler-route-chips span.shared {
        border-color: #b4a4c9;
        background: #f6f1fa;
        color: #765f96;
      }

      .scheduler-two-view-grid {
        display: grid;
        grid-template-columns: minmax(640px, 1.45fr) minmax(420px, 0.9fr);
        gap: 9px;
        align-items: start;
      }

      .scheduler-full-layout {
        display: grid;
        grid-template-columns: minmax(650px, 1.45fr) minmax(420px, 0.9fr);
        gap: 9px;
        align-items: start;
      }

      .scheduler-view-card {
        min-width: 0;
        padding: 10px;
        border: 1px solid #dce3e9;
        border-radius: 7px;
        background: #ffffff;
      }

      .scheduler-view-caption {
        min-height: 48px;
        margin-bottom: 5px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
      }

      .scheduler-view-caption h3 {
        margin: 0 0 4px;
        color: #344350;
        font-size: 18px;
      }

      .scheduler-view-caption p {
        margin: 0;
        color: #7f8b96;
        font-size: 16px;
        line-height: 1.45;
      }

      .scheduler-view-caption > span {
        padding: 4px 7px;
        border: 1px solid #dbe2e8;
        border-radius: 4px;
        background: #f8fafb;
        color: #87939e;
        font-size: 15px;
        white-space: nowrap;
      }

      /* ==================================================
         统一覆盖旧调度组件的小字号。
         旧组件逻辑不动，只在 Scheduler 页面放大视觉。
      ================================================== */

      .scheduler-visualizer .timeline-small-title,
      .scheduler-visualizer .full-token-small,
      .scheduler-visualizer .execution-cube-small,
      .scheduler-visualizer .full-token-section-title {
        font-size: 15px !important;
      }

      .scheduler-visualizer .timeline-root,
      .scheduler-visualizer .layer-timeline,
      .scheduler-visualizer .full-token-runner,
      .scheduler-visualizer .execution-cube-root {
        margin-top: 0 !important;
        padding-top: 0 !important;
        border-top: none !important;
      }

      .scheduler-visualizer .timeline-sc-label,
      .scheduler-visualizer .timeline-row-stat,
      .scheduler-visualizer .timeline-legend,
      .scheduler-visualizer .current-task-item,
      .scheduler-visualizer .cycle-state-card,
      .scheduler-visualizer .player-cycle,
      .scheduler-visualizer .speed-control,
      .scheduler-visualizer .full-token-header p,
      .scheduler-visualizer .full-summary-card span,
      .scheduler-visualizer .token-player-status span,
      .scheduler-visualizer .token-player-controls label,
      .scheduler-visualizer .layer-latency-item span,
      .scheduler-visualizer .slow-layer-item,
      .scheduler-visualizer .token-sc-card strong,
      .scheduler-visualizer .token-sc-card span,
      .scheduler-visualizer .full-playback-position span,
      .scheduler-visualizer .execution-cube-header p,
      .scheduler-visualizer .execution-cube-legend {
        font-size: 16px !important;
      }

      .scheduler-visualizer .full-token-header h3,
      .scheduler-visualizer .execution-cube-header h3 {
        font-size: 18px !important;
      }

      .scheduler-visualizer .full-summary-card strong,
      .scheduler-visualizer .token-player-status strong,
      .scheduler-visualizer .full-playback-position strong,
      .scheduler-visualizer .execution-cycle-info strong {
        font-size: 17px !important;
      }

      .scheduler-visualizer .run-token-button {
        min-width: 165px !important;
        height: 36px !important;
        font-size: 16px !important;
      }

      .scheduler-visualizer .layer-latency-grid {
        grid-template-columns: repeat(10, minmax(54px, 1fr)) !important;
      }

      .scheduler-visualizer .layer-latency-item {
        min-height: 44px !important;
      }

      .scheduler-visualizer .layer-latency-item strong {
        font-size: 15px !important;
      }

      .scheduler-visualizer .token-sc-grid {
        grid-template-columns: repeat(4, 1fr) !important;
      }

      .scheduler-visualizer .execution-canvas {
        height: 510px !important;
      }

      .scheduler-visualizer .execution-cube-root {
        width: 100% !important;
      }

      /* Drei <Html> 标签原组件使用了 10px/12px 内联字号；
         这里只在调度可视化页面强制放大，不改 Three.js 状态逻辑。 */
      .scheduler-visualizer .execution-cube-root div[style*="font-size: 10px"] {
        font-size: 15px !important;
      }

      .scheduler-visualizer .execution-cube-root div[style*="font-size: 16px"] {
        font-size: 16px !important;
      }

      .scheduler-visualizer .full-token-playback-panel {
        width: 100%;
      }

      @media (max-width: 1450px) {
        .scheduler-two-view-grid,
        .scheduler-full-layout {
          grid-template-columns: 1fr;
        }

        .scheduler-visualizer .execution-canvas {
          height: 460px !important;
        }
      }

      @media (max-width: 1050px) {
        .scheduler-source-bar,
        .scheduler-mode-tabs,
        .scheduler-layer-toolbar {
          flex-wrap: wrap;
        }

        .scheduler-random-button,
        .scheduler-live-summary {
          margin-left: 0;
        }
      }
    `}</style>
  );
}


export default SchedulerVisualizer;
