import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import ActiveMappingPanel
  from "./ActiveMappingPanel";

import LayerTimeline
  from "./LayerTimeline";

import ExecutionCube3D
  from "./ExecutionCube3D";

import FullTokenPlaybackPanel
  from "./FullTokenPlaybackPanel";


const API_BASE =
  "http://127.0.0.1:8000";


// ============================================================
// Expert Chip
// ============================================================


function ExpertChip({
  expertId,
  shared = false,
  rank = null,
}) {
  return (
    <div
      className={
        shared
          ? "compact-expert-chip shared"
          : "compact-expert-chip"
      }
    >
      {!shared && (
        <span className="compact-expert-rank">
          #{rank}
        </span>
      )}

      <strong>
        E{expertId}
      </strong>

      {shared && (
        <span className="compact-shared-tag">
          Shared
        </span>
      )}
    </div>
  );
}


// ============================================================
// 58 层 Route 行，仅在展开时显示
// ============================================================


function LayerRow({
  layer,
  selected,
  onClick,
}) {
  const routed =
    layer.routed_experts ?? [];


  return (
    <button
      className={
        selected
          ? "compact-layer-row selected"
          : "compact-layer-row"
      }

      onClick={
        onClick
      }
    >
      <strong>
        L{layer.layer_id}
      </strong>

      <div>
        {routed.map(
          (expertId) => (
            <span
              key={
                `${layer.layer_id}-${expertId}`
              }
            >
              E{expertId}
            </span>
          )
        )}

        <span className="mini-shared">
          E256
        </span>
      </div>
    </button>
  );
}


// ============================================================
// TokenSimulator
// ============================================================


function TokenSimulator() {

  // =========================================================
  // Trace
  // =========================================================

  const [
    categories,
    setCategories,
  ] = useState([]);


  const [
    selectedCategory,
    setSelectedCategory,
  ] = useState("");


  const [
    token,
    setToken,
  ] = useState(null);


  const [
    selectedLayer,
    setSelectedLayer,
  ] = useState(0);


  const [
    loadingCategories,
    setLoadingCategories,
  ] = useState(true);


  const [
    loadingToken,
    setLoadingToken,
  ] = useState(false);


  const [
    error,
    setError,
  ] = useState("");


  // =========================================================
  // 单 Layer 3D 状态
  // =========================================================

  const [
    executionCycle,
    setExecutionCycle,
  ] = useState(0);


  const [
    executionSubcubeStates,
    setExecutionSubcubeStates,
  ] = useState(
    Array.from(
      {
        length: 16,
      },
      (_, sc) => ({
        subcube_id: sc,
        state: "idle",
        task: null,
      })
    )
  );


  // =========================================================
  // 唯一 3D Cube 的状态源
  //
  // layer:
  //   LayerTimeline
  //
  // full:
  //   FullTokenRunner
  // =========================================================

  const [
    executionMode,
    setExecutionMode,
  ] = useState("layer");


  const [
    fullTokenFrame,
    setFullTokenFrame,
  ] = useState(
    {
      globalCycle: 0,
      layerId: 0,
      localCycle: 0,

      subcubeStates:
        Array.from(
          {
            length: 16,
          },
          (_, sc) => ({
            subcube_id: sc,
            state: "idle",
            task: null,
          })
        ),
    }
  );


  // =========================================================
  // Timeline → 3D
  // =========================================================

  const handleLayerCycleChange =
    useCallback(
      (cycle) => {

        setExecutionMode(
          "layer"
        );


        setExecutionCycle(
          cycle
        );
      },
      []
    );


  const handleLayerSubcubeStatesChange =
    useCallback(
      (states) => {

        setExecutionSubcubeStates(
          states
        );
      },
      []
    );


  // =========================================================
  // Full Token → 3D
  // =========================================================

  const handleFullTokenPlaybackFrame =
    useCallback(
      (frame) => {

        setExecutionMode(
          "full"
        );


        setFullTokenFrame(
          frame
        );
      },
      []
    );


  // =========================================================
  // Categories
  // =========================================================

  useEffect(() => {

    async function loadCategories() {

      try {

        setLoadingCategories(
          true
        );


        const response =
          await fetch(
            `${API_BASE}/api/trace/categories`
          );


        if (!response.ok) {

          throw new Error(
            `HTTP ${response.status}`
          );
        }


        const data =
          await response.json();


        setCategories(
          data.items ?? []
        );


        setError("");


      } catch (err) {

        console.error(
          err
        );


        setError(
          "无法读取 Chinese-SimpleQA 类别。请确认 Trace API 已经启动。"
        );


      } finally {

        setLoadingCategories(
          false
        );
      }
    }


    loadCategories();

  }, []);


  // =========================================================
  // 首次自动加载 Token
  // =========================================================

  useEffect(() => {

    if (
      loadingCategories
    ) {
      return;
    }


    loadRandomToken(
      ""
    );

    // eslint-disable-next-line react-hooks/exhaustive-deps

  }, [
    loadingCategories
  ]);


  // =========================================================
  // Random Token
  // =========================================================

  async function loadRandomToken(
    category = selectedCategory
  ) {

    try {

      setLoadingToken(
        true
      );


      setError("");


      let url =
        `${API_BASE}/api/trace/random-token`;


      if (category) {

        url +=
          `?category=${encodeURIComponent(category)}`;
      }


      const response =
        await fetch(
          url
        );


      if (!response.ok) {

        const body =
          await response.text();


        throw new Error(
          body ||
          `HTTP ${response.status}`
        );
      }


      const data =
        await response.json();


      setToken(
        data
      );


      setSelectedLayer(
        0
      );


      setExecutionMode(
        "layer"
      );


    } catch (err) {

      console.error(
        err
      );


      setError(
        "读取真实 Token 失败。请检查 Trace 数据路径和后端日志。"
      );


    } finally {

      setLoadingToken(
        false
      );
    }
  }


  // =========================================================
  // Layer Data
  // =========================================================

  const layers =
    token?.layers ?? [];


  const currentLayer =
    layers.find(
      (layer) =>
        layer.layer_id ===
        selectedLayer
    ) ?? null;


  const routedExperts =
    currentLayer
      ?.routed_experts ?? [];


  const sharedExpertId =
    currentLayer
      ?.shared_expert ??
    256;


  const activeExpertIds =
    useMemo(
      () => {

        if (!currentLayer) {
          return [];
        }


        return [
          ...routedExperts,
          sharedExpertId,
        ];

      },
      [
        currentLayer,
        routedExperts,
        sharedExpertId,
      ]
    );


  // =========================================================
  // Layer Change
  // =========================================================

  function changeLayer(
    layerId
  ) {

    const maxLayer =
      Math.max(
        layers.length - 1,
        0
      );


    const safeLayer =
      Math.max(
        0,
        Math.min(
          maxLayer,
          Number(layerId)
        )
      );


    setSelectedLayer(
      safeLayer
    );


    // 用户主动选择 Layer 后，
    // 3D Cube 回到单层模式。
    setExecutionMode(
      "layer"
    );
  }


  // =========================================================
  // 当前 3D 显示信息
  // =========================================================

  const cubeCycle =
    executionMode === "full"
      ? fullTokenFrame.globalCycle
      : executionCycle;


  const cubeStates =
    executionMode === "full"
      ? fullTokenFrame.subcubeStates
      : executionSubcubeStates;


  const cubeLayerId =
    executionMode === "full"
      ? fullTokenFrame.layerId
      : selectedLayer;


  const cubeLocalCycle =
    executionMode === "full"
      ? fullTokenFrame.localCycle
      : executionCycle;


  // =========================================================
  // Render
  // =========================================================

  return (
    <div className="token-simulator">

      {/* =====================================================
          Header + Trace Controls
      ====================================================== */}

      <div className="token-page-header">

        <div>

          <h2>
            Token Simulation
          </h2>

          <p>
            真实 Trace 路由、单层调度与完整 Token 执行。
          </p>

        </div>

      </div>


      <div className="trace-control">

        <div className="control-group">

          <label>
            Dataset
          </label>

          <div className="dataset-box">
            Chinese-SimpleQA
          </div>

        </div>


        <div className="control-group category-group">

          <label>
            Category
          </label>

          <select
            value={
              selectedCategory
            }

            disabled={
              loadingCategories
            }

            onChange={
              (event) => {

                const category =
                  event.target.value;


                setSelectedCategory(
                  category
                );


                loadRandomToken(
                  category
                );
              }
            }
          >
            <option value="">
              全部类别
            </option>


            {categories.map(
              (category) => (

                <option
                  key={
                    category.name
                  }

                  value={
                    category.name
                  }
                >
                  {category.name}
                  {" "}
                  ({category.file_count})
                </option>

              )
            )}
          </select>

        </div>


        <button
          className="random-token-button"

          disabled={
            loadingToken
          }

          onClick={() =>
            loadRandomToken()
          }
        >
          {
            loadingToken
              ? "Loading..."
              : "Random Real Token"
          }
        </button>

      </div>


      {error && (

        <div className="trace-error">
          {error}
        </div>

      )}


      {!token &&
       !loadingToken &&
       !error && (

        <div className="token-empty">
          暂无 Token。
        </div>

      )}


      {token && (

        <>

          {/* =================================================
              精简后的 Source
          ================================================== */}

          <div className="compact-source-bar">

            <div>
              <span>
                Source
              </span>

              <strong>
                {
                  token.source
                    ?.filename ??
                  "--"
                }
              </strong>
            </div>


            <div>
              <span>
                Segment
              </span>

              <strong>
                {
                  token.source
                    ?.segment_index ??
                  "--"
                }
              </strong>
            </div>


            <div>
              <span>
                Token
              </span>

              <strong>
                {
                  token.source
                    ?.token_index ??
                  "--"
                }
              </strong>
            </div>


            <div className="source-rule-note">
              58 Layers · Top-8 + Shared E256
            </div>

          </div>


          {/* =================================================
              Layer Controller
          ================================================== */}

          <div className="layer-controller">

            <div className="layer-control-title">
              Layer
            </div>


            <button
              className="layer-arrow"

              disabled={
                selectedLayer <= 0
              }

              onClick={() =>
                changeLayer(
                  selectedLayer - 1
                )
              }
            >
              ‹
            </button>


            <div className="current-layer-box">

              <strong>
                L{selectedLayer}
              </strong>

              <span>
                / 57
              </span>

            </div>


            <button
              className="layer-arrow"

              disabled={
                selectedLayer >=
                layers.length - 1
              }

              onClick={() =>
                changeLayer(
                  selectedLayer + 1
                )
              }
            >
              ›
            </button>


            <input
              className="layer-slider"

              type="range"

              min="0"

              max={
                Math.max(
                  layers.length - 1,
                  0
                )
              }

              value={
                selectedLayer
              }

              onChange={
                (event) =>
                  changeLayer(
                    Number(
                      event.target.value
                    )
                  )
              }
            />


            <div className="trace-layer-info">

              Trace L

              <strong>
                {
                  currentLayer
                    ?.trace_layer_id ??
                  "--"
                }
              </strong>

            </div>

          </div>


          {/* =================================================
              Current Route：合并 Top-8 + Shared
          ================================================== */}

          <section className="compact-section">

            <div className="compact-section-header">

              <div>

                <div className="small-title">
                  CURRENT ROUTE
                </div>

                <h3>
                  Layer {selectedLayer}
                </h3>

              </div>


              <div className="compact-count">
                Active Experts
                <strong>
                  {activeExpertIds.length}
                </strong>
              </div>

            </div>


            <div className="compact-route-row">

              {routedExperts.map(
                (
                  expertId,
                  index
                ) => (

                  <ExpertChip
                    key={
                      `${selectedLayer}-${expertId}`
                    }

                    expertId={
                      expertId
                    }

                    rank={
                      index + 1
                    }
                  />

                )
              )}


              <ExpertChip
                expertId={
                  sharedExpertId
                }

                shared
              />

            </div>


            <div className="route-note">
              E256 为 Shared Expert，
              不参与 Top-8 Router 选择，但始终参与当前层计算。
            </div>

          </section>


          {/* =================================================
              58-Layer Route：默认折叠
          ================================================== */}

          <details className="full-route-details">

            <summary>
              View Full 58-Layer Route
            </summary>


            <div className="full-route-grid">

              {layers.map(
                (layer) => (

                  <LayerRow
                    key={
                      layer.layer_id
                    }

                    layer={
                      layer
                    }

                    selected={
                      layer.layer_id ===
                      selectedLayer
                    }

                    onClick={() =>
                      changeLayer(
                        layer.layer_id
                      )
                    }
                  />

                )
              )}

            </div>

          </details>


          {/* =================================================
              Physical Mapping
              先默认折叠，避免 9 张大卡片一直占屏。
              后续拿到 ActiveMappingPanel.jsx 后再改成紧凑表格。
          ================================================== */}

          <details className="mapping-details">

            <summary>

              <span>
                Physical Mapping
              </span>

              <small>
                9 Experts · 27 Matrices
              </small>

            </summary>


            <ActiveMappingPanel
              layerId={
                selectedLayer
              }

              routedExpertIds={
                routedExperts
              }

              sharedExpertId={
                sharedExpertId
              }
            />

          </details>


          {/* =================================================
              核心演示：
              Timeline + 唯一 3D 并排
          ================================================== */}

          <div className="execution-grid">

            <div className="execution-grid-item timeline-item">

              <LayerTimeline
                layerId={
                  selectedLayer
                }

                routedExpertIds={
                  routedExperts
                }

                onCycleChange={
                  handleLayerCycleChange
                }

                onSubcubeStatesChange={
                  handleLayerSubcubeStatesChange
                }
              />

            </div>


            <div className="execution-grid-item cube-item">

              <ExecutionCube3D
                currentCycle={
                  cubeCycle
                }

                subcubeStates={
                  cubeStates
                }

                mode={
                  executionMode
                }

                currentLayer={
                  cubeLayerId
                }

                localCycle={
                  cubeLocalCycle
                }
              />

            </div>

          </div>


          {/* =================================================
              Full Token
          ================================================== */}

          <div className="full-token-section-wrap">

            <FullTokenPlaybackPanel
              token={
                token
              }

              onSelectLayer={
                changeLayer
              }

              onPlaybackFrame={
                handleFullTokenPlaybackFrame
              }
            />

          </div>

        </>

      )}


      <Style />

    </div>
  );
}


// ============================================================
// CSS
// ============================================================


function Style() {

  return (
    <style>
      {`

        .token-simulator {
          width: 100%;
        }


        .token-page-header {
          min-height: 44px;

          margin-bottom: 12px;
        }


        .token-page-header h2 {
          margin: 0 0 5px;

          color: #2f3945;

          font-size: 20px;

          font-weight: 650;
        }


        .token-page-header p {
          margin: 0;

          color: #8d96a1;

          font-size: 13px;
        }


        /* ================================================
           TRACE CONTROL
        ================================================ */


        .trace-control {
          min-height: 52px;

          padding: 8px 10px;

          margin-bottom: 10px;

          display: flex;

          align-items: flex-end;

          gap: 12px;

          border: 1px solid #dfe4ea;

          border-radius: 6px;

          background: #ffffff;
        }


        .control-group {
          display: flex;

          flex-direction: column;

          gap: 6px;
        }


        .control-group label {
          color: #8c96a2;

          font-size: 11px;

          font-weight: 600;

          letter-spacing: 0.5px;

          text-transform: uppercase;
        }


        .dataset-box {
          min-width: 150px;

          height: 31px;

          padding: 0 10px;

          display: flex;

          align-items: center;

          border: 1px solid #dce2e8;

          border-radius: 4px;

          background: #f7f9fb;

          color: #485666;

          font-size: 13px;

          font-weight: 600;
        }


        .category-group {
          flex: 1;

          max-width: 310px;
        }


        .category-group select {
          width: 100%;

          height: 31px;

          padding: 0 9px;

          border: 1px solid #dce2e8;

          border-radius: 4px;

          outline: none;

          background: #ffffff;

          color: #485666;

          font-size: 13px;
        }


        .random-token-button {
          height: 31px;

          padding: 0 14px;

          border: 1px solid #6f91b2;

          border-radius: 4px;

          background: #789abb;

          color: #ffffff;

          font-size: 13px;

          font-weight: 600;

          cursor: pointer;
        }


        .random-token-button:hover {
          background: #698bab;
        }


        .random-token-button:disabled {
          opacity: 0.55;

          cursor: default;
        }


        .trace-error {
          margin-bottom: 10px;

          padding: 10px 12px;

          border: 1px solid #e2bbbb;

          border-radius: 5px;

          background: #fff7f7;

          color: #9b5555;

          font-size: 12px;
        }


        .token-empty {
          min-height: 300px;

          display: flex;

          align-items: center;

          justify-content: center;

          border: 1px solid #dfe4ea;

          border-radius: 7px;

          background: #ffffff;

          color: #9ca4ad;
        }


        /* ================================================
           COMPACT SOURCE
        ================================================ */


        .compact-source-bar {
          min-height: 42px;

          margin-bottom: 10px;

          padding: 7px 11px;

          display: flex;

          align-items: center;

          gap: 24px;

          border: 1px solid #dfe4ea;

          border-radius: 6px;

          background: #ffffff;
        }


        .compact-source-bar > div:not(.source-rule-note) {
          min-width: 80px;
        }


        .compact-source-bar span {
          display: block;

          margin-bottom: 2px;

          color: #98a1aa;

          font-size: 10px;

          text-transform: uppercase;
        }


        .compact-source-bar strong {
          color: #465360;

          font-size: 12px;
        }


        .source-rule-note {
          margin-left: auto;

          color: #929ca6;

          font-size: 11px;
        }


        /* ================================================
           LAYER CONTROLLER
        ================================================ */


        .layer-controller {
          min-height: 42px;

          padding: 7px 11px;

          margin-bottom: 10px;

          display: flex;

          align-items: center;

          gap: 8px;

          border: 1px solid #dfe4ea;

          border-radius: 6px;

          background: #ffffff;
        }


        .layer-control-title {
          margin-right: 4px;

          color: #89929e;

          font-size: 11px;

          font-weight: 650;

          text-transform: uppercase;
        }


        .layer-arrow {
          width: 29px;

          height: 29px;

          border: 1px solid #d9dfe6;

          border-radius: 4px;

          background: #ffffff;

          color: #556272;

          font-size: 19px;

          cursor: pointer;
        }


        .layer-arrow:disabled {
          opacity: 0.35;

          cursor: default;
        }


        .current-layer-box {
          min-width: 74px;

          height: 29px;

          display: flex;

          align-items: center;

          justify-content: center;

          gap: 3px;

          border: 1px solid #d9dfe6;

          border-radius: 4px;

          color: #74808d;

          font-size: 12px;
        }


        .current-layer-box strong {
          color: #344152;

          font-size: 14px;
        }


        .layer-slider {
          flex: 1;

          min-width: 180px;
        }


        .trace-layer-info {
          min-width: 68px;

          color: #929ca6;

          font-size: 11px;

          text-align: right;
        }


        .trace-layer-info strong {
          color: #465361;
        }


        /* ================================================
           COMPACT SECTION
        ================================================ */


        .compact-section {
          margin-bottom: 10px;

          padding: 11px 12px;

          border: 1px solid #dfe4ea;

          border-radius: 6px;

          background: #ffffff;
        }


        .compact-section-header {
          margin-bottom: 9px;

          display: flex;

          align-items: flex-start;

          justify-content: space-between;
        }


        .small-title {
          margin-bottom: 3px;

          color: #9aa3ae;

          font-size: 10px;

          font-weight: 650;

          letter-spacing: 1px;
        }


        .compact-section-header h3 {
          margin: 0;

          color: #35414d;

          font-size: 16px;
        }


        .compact-count {
          display: flex;

          align-items: center;

          gap: 7px;

          color: #909aa4;

          font-size: 11px;
        }


        .compact-count strong {
          min-width: 26px;

          height: 24px;

          display: flex;

          align-items: center;

          justify-content: center;

          border: 1px solid #ced8e0;

          border-radius: 4px;

          background: #f6f8fa;

          color: #465563;

          font-size: 13px;
        }


        .compact-route-row {
          display: flex;

          flex-wrap: wrap;

          gap: 6px;
        }


        .compact-expert-chip {
          min-height: 32px;

          padding: 5px 8px;

          display: flex;

          align-items: center;

          gap: 5px;

          border: 1px solid #bdd0e1;

          border-radius: 4px;

          background: #eaf2f9;

          color: #3f6282;
        }


        .compact-expert-chip strong {
          font-size: 12px;
        }


        .compact-expert-rank {
          color: #8da4b9;

          font-size: 10px;
        }


        .compact-expert-chip.shared {
          border-color: #9b85ba;

          background: #f4eff9;

          color: #705a94;
        }


        .compact-shared-tag {
          padding: 1px 3px;

          border-radius: 2px;

          background: #e9e0f3;

          font-size: 9px;
        }


        .route-note {
          margin-top: 7px;

          color: #969fa8;

          font-size: 10px;
        }


        /* ================================================
           DETAILS
        ================================================ */


        .full-route-details,
        .mapping-details {
          margin-bottom: 10px;

          border: 1px solid #dfe4ea;

          border-radius: 6px;

          background: #ffffff;
        }


        .full-route-details > summary,
        .mapping-details > summary {
          min-height: 40px;

          padding: 0 12px;

          display: flex;

          align-items: center;

          gap: 8px;

          color: #586572;

          font-size: 12px;

          font-weight: 650;

          cursor: pointer;
        }


        .mapping-details > summary small {
          margin-left: auto;

          color: #9aa3ad;

          font-size: 10px;

          font-weight: 500;
        }


        .full-route-grid {
          max-height: 320px;

          padding: 7px;

          overflow-y: auto;

          border-top: 1px solid #edf0f2;
        }


        .compact-layer-row {
          width: 100%;

          min-height: 34px;

          margin-bottom: 3px;

          padding: 4px 7px;

          display: grid;

          grid-template-columns: 34px 1fr;

          align-items: center;

          border: 1px solid transparent;

          border-radius: 4px;

          background: transparent;

          text-align: left;

          cursor: pointer;
        }


        .compact-layer-row:hover {
          background: #f5f7f9;
        }


        .compact-layer-row.selected {
          border-color: #bad0e3;

          background: #edf4fa;
        }


        .compact-layer-row > strong {
          color: #65717e;

          font-size: 11px;
        }


        .compact-layer-row > div {
          display: flex;

          flex-wrap: wrap;

          gap: 3px;
        }


        .compact-layer-row span {
          padding: 1px 3px;

          border-radius: 2px;

          background: #eef1f4;

          color: #77818c;

          font-size: 10px;
        }


        .compact-layer-row .mini-shared {
          background: #eee8f4;

          color: #735e92;
        }


        /* ================================================
           EXECUTION GRID
        ================================================ */


        .execution-grid {
          display: grid;

          grid-template-columns:
            minmax(0, 3fr)
            minmax(360px, 2fr);

          gap: 12px;

          align-items: start;
        }


        .execution-grid-item {
          min-width: 0;
        }


        .timeline-item,
        .cube-item {
          border: 1px solid #dfe4ea;

          border-radius: 7px;

          background: #ffffff;

          padding: 0 10px 10px;
        }


        .full-token-section-wrap {
          margin-top: 12px;

          border: 1px solid #dfe4ea;

          border-radius: 7px;

          background: #ffffff;

          padding: 0 12px 12px;
        }


        @media (
          max-width: 1180px
        ) {

          .execution-grid {
            grid-template-columns: 1fr;
          }


          .cube-item {
            min-width: 0;
          }

        }

      `}
    </style>
  );
}


export default TokenSimulator;
