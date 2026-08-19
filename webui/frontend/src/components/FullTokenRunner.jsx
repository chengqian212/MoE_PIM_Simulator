import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE =
  "http://127.0.0.1:8000";


const NUM_SUBCUBES =
  16;


const SPEED_OPTIONS = [
  {
    value: 1,
    label: "1×",
  },
  {
    value: 2,
    label: "2×",
  },
  {
    value: 5,
    label: "5×",
  },
  {
    value: 10,
    label: "10×",
  },
  {
    value: 20,
    label: "20×",
  },
];


function speedToInterval(
  speed
) {

  const baseMs =
    500;


  return Math.max(
    40,
    baseMs / speed
  );
}


// ============================================================
// FullTokenRunner
//
// 保留：
// 1. Run Full Token
// 2. 完整 Token 播放器
// 3. Global / Layer / Local Cycle
// 4. 58 层周期图
//
// 删除：
// 1. Slowest Layers
// 2. 16 个 Sub-Cube Statistics
//
// 这些总体分析放到“结果分析”页面。
// ============================================================


function FullTokenRunner({
  token,
  onSelectLayer,
  onPlaybackFrame,
}) {

  const [
    result,
    setResult,
  ] = useState(null);


  const [
    loading,
    setLoading,
  ] = useState(false);


  const [
    error,
    setError,
  ] = useState("");


  const [
    globalCycle,
    setGlobalCycle,
  ] = useState(0);


  const [
    isPlaying,
    setIsPlaying,
  ] = useState(false);


  const [
    speed,
    setSpeed,
  ] = useState(5);


  // =========================================================
  // Run Full Token
  // =========================================================

  async function runFullToken() {

    if (!token) {

      setError(
        "请先加载一个真实 Token。"
      );

      return;
    }


    const layers =
      token.layers ?? [];


    if (
      layers.length !== 58
    ) {

      setError(
        `当前 Token 只有 ${layers.length} 层，必须包含完整的 58 层。`
      );

      return;
    }


    const routes =
      layers.map(
        (layer) =>
          layer.routed_experts ?? []
      );


    for (
      let layerId = 0;
      layerId < routes.length;
      layerId += 1
    ) {

      if (
        routes[layerId].length !==
        8
      ) {

        setError(
          `Layer ${layerId} 只有 ${routes[layerId].length} 个 Expert。`
        );

        return;
      }
    }


    try {

      setLoading(
        true
      );


      setError("");


      setIsPlaying(
        false
      );


      setGlobalCycle(
        0
      );


      const response =
        await fetch(
          `${API_BASE}/api/token-schedule/token`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                {
                  routes,

                  charge_initial_activation:
                    true,

                  include_tasks:
                    true,
                }
              ),
          }
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


      setResult(
        data
      );


      setGlobalCycle(
        0
      );


      setIsPlaying(
        false
      );


    } catch (err) {

      console.error(
        err
      );


      setError(
        "完整 Token 调度失败："
        + err.message
      );


    } finally {

      setLoading(
        false
      );
    }
  }


  const maxCycle =
    Math.max(
      (
        result
          ?.total_cycles ??
        1
      )
      - 1,
      0
    );


  // =========================================================
  // 当前 Layer
  // =========================================================

  const currentLayer =
    useMemo(
      () => {

        if (!result) {
          return null;
        }


        const layers =
          result.layers ?? [];


        for (
          const layer
          of layers
        ) {

          if (
            globalCycle >=
              layer.start_cycle
            &&
            globalCycle <
              layer.end_cycle
          ) {

            return layer;
          }
        }


        if (
          layers.length > 0
        ) {

          return (
            layers[
              layers.length - 1
            ]
          );
        }


        return null;

      },
      [
        result,
        globalCycle,
      ]
    );


  const localCycle =
    currentLayer
      ? (
          globalCycle
          -
          currentLayer.start_cycle
        )
      : 0;


  // =========================================================
  // 当前 16 个 SC 状态
  // =========================================================

  const currentSubcubeStates =
    useMemo(
      () => {

        const states =
          Array.from(
            {
              length:
                NUM_SUBCUBES,
            },
            (_, sc) => ({
              subcube_id: sc,
              state: "idle",
              task: null,
            })
          );


        if (!result) {
          return states;
        }


        const tasks =
          result.tasks ?? [];


        for (
          const task
          of tasks
        ) {

          const sc =
            task.subcube_id;


          if (
            sc < 0 ||
            sc >= NUM_SUBCUBES
          ) {
            continue;
          }


          const start =
            task.start_cycle;


          const computeStart =
            task.compute_start_cycle;


          const end =
            task.end_cycle;


          if (
            globalCycle >= start
            &&
            globalCycle < computeStart
          ) {

            states[
              sc
            ] = {
              subcube_id: sc,
              state: "switch",
              task,
            };

            continue;
          }


          if (
            globalCycle >= computeStart
            &&
            globalCycle < end
          ) {

            states[
              sc
            ] = {
              subcube_id: sc,

              state:
                task.matrix_name
                ?? "idle",

              task,
            };
          }
        }


        return states;

      },
      [
        result,
        globalCycle,
      ]
    );


  const activeSubcubeCount =
    useMemo(
      () =>
        currentSubcubeStates.filter(
          (item) =>
            item.state !==
            "idle"
        ).length,
      [
        currentSubcubeStates
      ]
    );


  // =========================================================
  // Full Token → TokenSimulator → 唯一 3D Cube
  // =========================================================

  useEffect(() => {

    if (
      !result ||
      !currentLayer ||
      !onPlaybackFrame
    ) {
      return;
    }


    onPlaybackFrame(
      {
        globalCycle,

        layerId:
          currentLayer.layer_id,

        localCycle,

        subcubeStates:
          currentSubcubeStates,
      }
    );

  }, [
    result,
    globalCycle,
    currentLayer,
    localCycle,
    currentSubcubeStates,
    onPlaybackFrame,
  ]);


  // =========================================================
  // Auto Play
  // =========================================================

  useEffect(() => {

    if (
      !isPlaying ||
      !result
    ) {
      return undefined;
    }


    const interval =
      window.setInterval(
        () => {

          setGlobalCycle(
            (previous) => {

              if (
                previous >=
                maxCycle
              ) {

                setIsPlaying(
                  false
                );

                return previous;
              }


              return (
                previous + 1
              );
            }
          );

        },
        speedToInterval(
          speed
        )
      );


    return () => {

      window.clearInterval(
        interval
      );
    };

  }, [
    isPlaying,
    speed,
    result,
    maxCycle,
  ]);


  // =========================================================
  // Controls
  // =========================================================

  function firstCycle() {

    setIsPlaying(
      false
    );


    setGlobalCycle(
      0
    );
  }


  function previousCycle() {

    setIsPlaying(
      false
    );


    setGlobalCycle(
      (previous) =>
        Math.max(
          0,
          previous - 1
        )
    );
  }


  function nextCycle() {

    setIsPlaying(
      false
    );


    setGlobalCycle(
      (previous) =>
        Math.min(
          maxCycle,
          previous + 1
        )
    );
  }


  function lastCycle() {

    setIsPlaying(
      false
    );


    setGlobalCycle(
      maxCycle
    );
  }


  function togglePlay() {

    if (!result) {
      return;
    }


    if (
      !isPlaying
      &&
      globalCycle >=
        maxCycle
    ) {

      setGlobalCycle(
        0
      );
    }


    setIsPlaying(
      (previous) =>
        !previous
    );
  }


  function jumpToLayer(
    layer
  ) {

    setIsPlaying(
      false
    );


    setGlobalCycle(
      layer.start_cycle
    );


    // 只有用户手动点 58 层周期格时，
    // 才同步上方单层查看器。
    if (onSelectLayer) {

      onSelectLayer(
        layer.layer_id
      );
    }
  }


  // =========================================================
  // Render
  // =========================================================

  return (
    <div className="full-token-runner">

      <div className="full-token-header">

        <div>

          <div className="full-token-small">
            FULL TOKEN EXECUTION
          </div>


          <h3>
            58-Layer Token Schedule
          </h3>


          <p>
            从 Layer 0 顺序执行到 Layer 57。
          </p>

        </div>


        <button
          className="run-token-button"

          disabled={
            loading ||
            !token
          }

          onClick={
            runFullToken
          }
        >
          {
            loading
              ? "Running..."
              : "▶ Run Full Token"
          }
        </button>

      </div>


      {error && (

        <div className="full-token-error">
          {error}
        </div>

      )}


      {result && (

        <>

          {/* =================================================
              核心摘要 + 播放器
          ================================================== */}

          <div className="full-token-player">

            <div className="token-player-status">

              <div className="token-total-latency">

                <span>
                  Total
                </span>

                <strong>
                  {
                    result
                      .total_cycles
                  }
                  {" "}
                  cycles
                </strong>

              </div>


              <div>

                <span>
                  Global
                </span>

                <strong>
                  {globalCycle}
                </strong>

                <small>
                  / {maxCycle}
                </small>

              </div>


              <div>

                <span>
                  Layer
                </span>

                <strong>
                  L{
                    currentLayer
                      ?.layer_id ??
                    "--"
                  }
                </strong>

              </div>


              <div>

                <span>
                  Local
                </span>

                <strong>
                  {localCycle}
                </strong>

              </div>


              <div>

                <span>
                  Active SC
                </span>

                <strong>
                  {activeSubcubeCount}
                </strong>

              </div>

            </div>


            <div className="token-player-controls">

              <button
                disabled={
                  globalCycle <= 0
                }

                onClick={
                  firstCycle
                }
              >
                |◀
              </button>


              <button
                disabled={
                  globalCycle <= 0
                }

                onClick={
                  previousCycle
                }
              >
                ◀
              </button>


              <button
                className="token-play-button"

                onClick={
                  togglePlay
                }
              >
                {
                  isPlaying
                    ? "Ⅱ"
                    : "▶"
                }
              </button>


              <button
                disabled={
                  globalCycle >=
                  maxCycle
                }

                onClick={
                  nextCycle
                }
              >
                ▶
              </button>


              <button
                disabled={
                  globalCycle >=
                  maxCycle
                }

                onClick={
                  lastCycle
                }
              >
                ▶|
              </button>


              <label>

                Speed

                <select
                  value={
                    speed
                  }

                  onChange={
                    (event) =>
                      setSpeed(
                        Number(
                          event
                            .target
                            .value
                        )
                      )
                  }
                >

                  {SPEED_OPTIONS.map(
                    (option) => (

                      <option
                        key={
                          option.value
                        }

                        value={
                          option.value
                        }
                      >
                        {option.label}
                      </option>

                    )
                  )}

                </select>

              </label>

            </div>

          </div>


          {/* =================================================
              Global Progress
          ================================================== */}

          <div className="token-progress">

            <div
              className="token-progress-bar"

              style={{
                width:
                  `${
                    maxCycle > 0
                      ? (
                          globalCycle
                          /
                          maxCycle
                          *
                          100
                        )
                      : 0
                  }%`,
              }}
            />

          </div>


          {/* =================================================
              58 Layer Cycle 图
          ================================================== */}

          <div className="full-token-layer-overview">

            <div className="full-token-section-title">
              58-Layer Cycle Overview
            </div>


            <div className="layer-latency-grid">

              {
                (
                  result.layers ??
                  []
                ).map(
                  (layer) => {

                    const active =
                      currentLayer
                        ?.layer_id
                      ===
                      layer.layer_id;


                    return (
                      <button
                        key={
                          layer.layer_id
                        }

                        className={
                          active
                            ? "layer-latency-item active"
                            : "layer-latency-item"
                        }

                        title={
                          `Layer ${layer.layer_id}: `
                          + `${layer.layer_cycles} cycles`
                        }

                        onClick={
                          () =>
                            jumpToLayer(
                              layer
                            )
                        }
                      >

                        <span>
                          L{
                            layer.layer_id
                          }
                        </span>


                        <strong>
                          {
                            layer.layer_cycles
                          }
                        </strong>

                      </button>
                    );
                  }
                )
              }

            </div>

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

        .full-token-runner {
          width: 100%;

          margin-top: 12px;

          padding-top: 12px;

          border-top: 1px solid #e3e7eb;
        }


        .full-token-header {
          margin-bottom: 10px;

          display: flex;

          align-items: center;

          justify-content: space-between;

          gap: 12px;
        }


        .full-token-small {
          margin-bottom: 4px;

          color: #99a3ad;

          font-size: 10px;

          font-weight: 650;

          letter-spacing: 1px;
        }


        .full-token-header h3 {
          margin: 0 0 4px;

          color: #34404d;

          font-size: 16px;
        }


        .full-token-header p {
          margin: 0;

          color: #929ca7;

          font-size: 10px;
        }


        .run-token-button {
          min-width: 135px;

          height: 32px;

          border: 1px solid #698ba9;

          border-radius: 5px;

          background: #7596b4;

          color: #ffffff;

          font-size: 11px;

          font-weight: 650;

          cursor: pointer;
        }


        .run-token-button:disabled {
          opacity: 0.45;

          cursor: default;
        }


        .full-token-error {
          margin-bottom: 8px;

          padding: 8px 10px;

          border: 1px solid #e1bcbc;

          border-radius: 5px;

          background: #fff5f5;

          color: #9c5757;

          font-size: 11px;
        }


        /* ================================================
           PLAYER
        ================================================ */


        .full-token-player {
          margin-bottom: 6px;

          padding: 8px 9px;

          display: flex;

          align-items: center;

          justify-content: space-between;

          gap: 12px;

          border: 1px solid #dfe4e9;

          border-radius: 6px;

          background: #ffffff;
        }


        .token-player-status {
          display: flex;

          align-items: center;

          gap: 14px;
        }


        .token-player-status > div {
          min-width: 50px;
        }


        .token-player-status
        .token-total-latency {
          min-width: 105px;
        }


        .token-player-status span {
          display: block;

          margin-bottom: 2px;

          color: #929ca7;

          font-size: 9px;
        }


        .token-player-status strong {
          color: #40505e;

          font-size: 12px;
        }


        .token-player-status small {
          margin-left: 2px;

          color: #9aa3ad;

          font-size: 9px;
        }


        .token-player-controls {
          display: flex;

          align-items: center;

          gap: 4px;
        }


        .token-player-controls button {
          width: 29px;

          height: 28px;

          padding: 0;

          border: 1px solid #d8dfe6;

          border-radius: 4px;

          background: #ffffff;

          color: #566573;

          cursor: pointer;
        }


        .token-player-controls button:disabled {
          opacity: 0.3;

          cursor: default;
        }


        .token-player-controls
        .token-play-button {
          width: 36px;

          border-color: #7596b4;

          background: #7596b4;

          color: white;
        }


        .token-player-controls label {
          margin-left: 5px;

          display: flex;

          align-items: center;

          gap: 4px;

          color: #8c96a0;

          font-size: 9px;
        }


        .token-player-controls select {
          height: 28px;

          border: 1px solid #d8dfe6;

          border-radius: 4px;

          background: white;

          color: #485562;

          font-size: 9px;
        }


        /* ================================================
           PROGRESS
        ================================================ */


        .token-progress {
          height: 4px;

          margin-bottom: 8px;

          overflow: hidden;

          border-radius: 3px;

          background: #e9edf1;
        }


        .token-progress-bar {
          height: 100%;

          background: #7596b4;

          transition: width 0.08s linear;
        }


        /* ================================================
           58 LAYERS
        ================================================ */


        .full-token-layer-overview {
          padding: 8px;

          border: 1px solid #e0e4e8;

          border-radius: 6px;

          background: #ffffff;
        }


        .full-token-section-title {
          margin-bottom: 7px;

          color: #687480;

          font-size: 10px;

          font-weight: 700;
        }


        .layer-latency-grid {
          display: grid;

          grid-template-columns:
            repeat(
              15,
              minmax(
                32px,
                1fr
              )
            );

          gap: 4px;
        }


        .layer-latency-item {
          min-height: 34px;

          padding: 3px;

          display: flex;

          flex-direction: column;

          align-items: center;

          justify-content: center;

          border: 1px solid #dfe4e8;

          border-radius: 4px;

          background: #fafbfc;

          cursor: pointer;
        }


        .layer-latency-item.active {
          border: 2px solid #6689a8;

          background: #eaf2f8;
        }


        .layer-latency-item span {
          color: #929ca6;

          font-size: 9px;
        }


        .layer-latency-item strong {
          margin-top: 2px;

          color: #465665;

          font-size: 10px;
        }

/* ================================================
   FULL TOKEN FONT ENLARGE
================================================ */


/* FULL TOKEN EXECUTION */
.full-token-small {
  font-size: 13px;
}


/* 58-Layer Token Schedule */
.full-token-header h3 {
  font-size: 21px;
}


/* 从 Layer 0 顺序执行到 Layer 57 */
.full-token-header p {
  font-size: 13px;
}


/* Run Full Token 按钮 */
.run-token-button {
  min-width: 160px;
  height: 40px;
  font-size: 13px;
}


/* Total / Global / Layer / Local / Active SC */
.token-player-status span {
  font-size: 11px;
}


/* 这些统计数字 */
.token-player-status strong {
  font-size: 15px;
}


/* 0 / 521 里面的小字 */
.token-player-status small {
  font-size: 10px;
}


/* Full Token 播放按钮 */
.token-player-controls button {
  width: 34px;
  height: 34px;
  font-size: 12px;
}


.token-player-controls
.token-play-button {
  width: 42px;
}


/* Speed */
.token-player-controls label {
  font-size: 11px;
}


.token-player-controls select {
  height: 34px;
  font-size: 11px;
}


/* 58-Layer Cycle Overview */
.full-token-section-title {
  font-size: 12px;
}


/* L0 / L1 / ... */
.layer-latency-item span {
  font-size: 10px;
}


/* Layer 周期数字 */
.layer-latency-item strong {
  font-size: 12px;
}


/* Layer 小格子也稍微增高 */
.layer-latency-item {
  min-height: 42px;
}
        @media (
          max-width: 1200px
        ) {

          .full-token-player {
            align-items: flex-start;

            flex-direction: column;
          }


          .layer-latency-grid {
            grid-template-columns:
              repeat(
                10,
                minmax(
                  32px,
                  1fr
                )
              );
          }

        }

      `}
    </style>
  );
}


export default FullTokenRunner;
