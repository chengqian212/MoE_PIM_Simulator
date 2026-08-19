import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE =
  "http://127.0.0.1:8000";


const NUM_SUBCUBES = 16;


// ============================================================
// Matrix Name
// ============================================================


function matrixShortName(
  matrixName
) {

  if (matrixName === "gate") {
    return "G";
  }


  if (matrixName === "up") {
    return "U";
  }


  if (matrixName === "down") {
    return "D";
  }


  return "?";
}


// ============================================================
// 播放速度
// ============================================================


const SPEED_OPTIONS = [
  {
    value: 0.5,
    label: "0.5×",
  },
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
];


function speedToInterval(
  speed
) {

  const baseMs =
    800;


  return Math.max(
    60,
    baseMs / speed
  );
}


// ============================================================
// Timeline Cell
// ============================================================


function TimelineCell({
  state,
  task,
  current,
}) {

  if (!state) {

    return (
      <div
        className={
          current
            ? "timeline-cell idle current-cycle"
            : "timeline-cell idle"
        }

        title="Idle"
      />
    );
  }


  if (state === "switch") {

    return (
      <div
        className={
          current
            ? "timeline-cell switching current-cycle"
            : "timeline-cell switching"
        }

        title={
          task
            ? (
                `Switch → E${task.expert_id} `
                + `${task.matrix_name} `
                + `| Cube ${task.cube_id} `
                + `| SC-${task.subcube_id}`
              )
            : "Switching"
        }
      >
        S
      </div>
    );
  }


  const shortName =
    matrixShortName(
      task?.matrix_name
    );


  return (
    <div
      className={
        current
          ? (
              `timeline-cell compute `
              + `${task?.matrix_name ?? ""} `
              + "current-cycle"
            )
          : (
              `timeline-cell compute `
              + `${task?.matrix_name ?? ""}`
            )
      }

      title={
        task
          ? (
              `Compute E${task.expert_id} `
              + `${task.matrix_name} `
              + `| SC-${task.subcube_id} `
              + `| z=${task.z} `
              + `| Cube ${task.cube_id}`
            )
          : "Compute"
      }
    >
      {shortName}
    </div>
  );
}


// ============================================================
// Summary Card
// ============================================================


function SummaryCard({
  label,
  value,
}) {

  return (
    <div className="timeline-summary-card">

      <span>
        {label}
      </span>

      <strong>
        {value ?? "--"}
      </strong>

    </div>
  );
}


// ============================================================
// Current State Badge
// ============================================================


function CurrentStateBadge({
  currentState,
}) {

  if (!currentState) {

    return (
      <span className="current-state idle">
        Idle
      </span>
    );
  }


  if (
    currentState.state ===
    "switch"
  ) {

    return (
      <span className="current-state switch">
        Switch
      </span>
    );
  }


  const matrixName =
    currentState
      .task
      ?.matrix_name;


  return (
    <span
      className={
        `current-state ${matrixName}`
      }
    >
      {matrixName}
    </span>
  );
}


// ============================================================
// Sub-Cube Row
// ============================================================


function SubcubeTimelineRow({
  subcubeId,
  layerCycles,
  cells,
  stat,
  critical,
  currentCycle,
}) {

  const currentState =
    cells[
      currentCycle
    ] ?? null;


  return (
    <div
      className={
        critical
          ? "timeline-row critical"
          : "timeline-row"
      }
    >

      <div className="timeline-sc-label">

        <div className="sc-name">
          SC-{subcubeId}
        </div>


        <div className="sc-task-count">
          {stat?.task_count ?? 0}
          {" "}
          tasks
        </div>

      </div>


      <div
        className="timeline-cells"

        style={{
          gridTemplateColumns:
            `repeat(${Math.max(layerCycles, 1)}, 54px)`,
        }}
      >

        {Array.from(
          {
            length:
              Math.max(
                layerCycles,
                1
              ),
          },
          (_, cycle) => {

            const cell =
              cells[
                cycle
              ];


            return (
              <TimelineCell
                key={
                  `${subcubeId}-${cycle}`
                }

                state={
                  cell?.state
                }

                task={
                  cell?.task
                }

                current={
                  cycle ===
                  currentCycle
                }
              />
            );
          }
        )}

      </div>


      <div className="timeline-row-stat">

        <CurrentStateBadge
          currentState={
            currentState
          }
        />


        <span>
          SW
          {" "}
          <strong>
            {
              stat
                ?.switch_count ??
              0
            }
          </strong>
        </span>


        {critical && (
          <span className="critical-tag">
            Critical
          </span>
        )}

      </div>

    </div>
  );
}


// ============================================================
// Player
// ============================================================


function TimelinePlayer({
  currentCycle,
  maxCycle,
  isPlaying,
  speed,
  setSpeed,
  onFirst,
  onPrevious,
  onTogglePlay,
  onNext,
  onLast,
}) {

  return (
    <div className="timeline-player">

      <div className="player-cycle">

        Cycle

        <strong>
          {currentCycle}
        </strong>

        <span>
          / {maxCycle}
        </span>

      </div>


      <div className="player-divider" />


      <button
        className="player-button"

        title="回到第一个 Cycle"

        disabled={
          currentCycle <= 0
        }

        onClick={
          onFirst
        }
      >
        |◀
      </button>


      <button
        className="player-button"

        title="上一个 Cycle"

        disabled={
          currentCycle <= 0
        }

        onClick={
          onPrevious
        }
      >
        ◀
      </button>


      <button
        className="player-button play"

        title={
          isPlaying
            ? "暂停"
            : "播放"
        }

        onClick={
          onTogglePlay
        }
      >
        {
          isPlaying
            ? "Ⅱ"
            : "▶"
        }
      </button>


      <button
        className="player-button"

        title="下一个 Cycle"

        disabled={
          currentCycle >=
          maxCycle
        }

        onClick={
          onNext
        }
      >
        ▶
      </button>


      <button
        className="player-button"

        title="跳到最后一个 Cycle"

        disabled={
          currentCycle >=
          maxCycle
        }

        onClick={
          onLast
        }
      >
        ▶|
      </button>


      <div className="player-divider" />


      <label className="speed-control">

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
  );
}


// ============================================================
// Compact Current Cycle Summary
//
// 删除原来 16 个大状态卡。
// 只保留当前 Cycle 的总体数量。
// ============================================================


function CurrentCycleStrip({
  currentCycle,
  timelineBySubcube,
}) {

  const summary =
    useMemo(
      () => {

        let switching = 0;
        let gate = 0;
        let up = 0;
        let down = 0;
        let idle = 0;


        for (
          let sc = 0;
          sc < NUM_SUBCUBES;
          sc += 1
        ) {

          const cell =
            timelineBySubcube[
              sc
            ][
              currentCycle
            ];


          if (!cell) {

            idle += 1;

            continue;
          }


          if (
            cell.state ===
            "switch"
          ) {

            switching += 1;

            continue;
          }


          const matrix =
            cell
              .task
              ?.matrix_name;


          if (
            matrix === "gate"
          ) {
            gate += 1;
          }


          else if (
            matrix === "up"
          ) {
            up += 1;
          }


          else if (
            matrix === "down"
          ) {
            down += 1;
          }


          else {
            idle += 1;
          }
        }


        return {
          switching,
          gate,
          up,
          down,
          idle,
        };

      },
      [
        currentCycle,
        timelineBySubcube,
      ]
    );


  return (
    <div className="current-cycle-strip">

      <strong>
        Cycle {currentCycle}
      </strong>


      <span className="strip-state switch">
        Switch {summary.switching}
      </span>


      <span className="strip-state gate">
        Gate {summary.gate}
      </span>


      <span className="strip-state up">
        Up {summary.up}
      </span>


      <span className="strip-state down">
        Down {summary.down}
      </span>


      <span className="strip-state idle">
        Idle {summary.idle}
      </span>

    </div>
  );
}


// ============================================================
// Legend
// ============================================================


function TimelineLegend() {

  return (
    <div className="timeline-legend">

      <LegendItem
        className="switching"
        text="Switch"
        symbol="S"
      />


      <LegendItem
        className="gate"
        text="gate"
        symbol="G"
      />


      <LegendItem
        className="up"
        text="up"
        symbol="U"
      />


      <LegendItem
        className="down"
        text="down"
        symbol="D"
      />


      <LegendItem
        className="idle"
        text="Idle"
        symbol=""
      />


      <div className="legend-current">
        深色边框 = 当前 Cycle
      </div>

    </div>
  );
}


function LegendItem({
  className,
  text,
  symbol,
}) {

  return (
    <div className="legend-item">

      <span
        className={
          `legend-cell ${className}`
        }
      >
        {symbol}
      </span>

      {text}

    </div>
  );
}


// ============================================================
// LayerTimeline
// ============================================================


function LayerTimeline({
  layerId,
  routedExpertIds = [],
  onCycleChange,
  onSubcubeStatesChange,
}) {

  const [
    schedule,
    setSchedule,
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
    currentCycle,
    setCurrentCycle,
  ] = useState(0);


  const [
    isPlaying,
    setIsPlaying,
  ] = useState(false);


  const [
    speed,
    setSpeed,
  ] = useState(1);


  // =========================================================
  // Scheduler
  // =========================================================

  useEffect(() => {

    if (
      layerId === null ||
      layerId === undefined
    ) {
      return;
    }


    if (
      routedExpertIds.length !==
      8
    ) {

      setSchedule(
        null
      );

      return;
    }


    const controller =
      new AbortController();


    async function loadSchedule() {

      try {

        setLoading(
          true
        );


        setError("");


        setCurrentCycle(
          0
        );


        setIsPlaying(
          false
        );


        const response =
          await fetch(
            `${API_BASE}/api/schedule/layer`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              signal:
                controller.signal,

              body:
                JSON.stringify(
                  {
                    layer_id:
                      layerId,

                    routed_expert_ids:
                      routedExpertIds,

                    charge_initial_activation:
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


        setSchedule(
          data
        );


      } catch (err) {

        if (
          err.name ===
          "AbortError"
        ) {
          return;
        }


        console.error(
          err
        );


        setError(
          `Layer ${layerId} 调度失败。`
        );


      } finally {

        if (
          !controller
            .signal
            .aborted
        ) {

          setLoading(
            false
          );
        }
      }
    }


    loadSchedule();


    return () => {

      controller.abort();
    };

  }, [
    layerId,
    routedExpertIds,
  ]);


  const layerCycles =
    schedule
      ?.layer_cycles ??
    0;


  const maxCycle =
    Math.max(
      layerCycles - 1,
      0
    );


  // =========================================================
  // Auto Play
  // =========================================================

  useEffect(() => {

    if (
      !isPlaying ||
      !schedule
    ) {
      return undefined;
    }


    const interval =
      window.setInterval(
        () => {

          setCurrentCycle(
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
    schedule,
    maxCycle,
  ]);


  // =========================================================
  // timelineBySubcube
  // =========================================================

  const timelineBySubcube =
    useMemo(
      () => {

        const result =
          Array.from(
            {
              length:
                NUM_SUBCUBES,
            },
            () => ({})
          );


        if (!schedule) {
          return result;
        }


        for (
          const task
          of (
            schedule.tasks ??
            []
          )
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
            start === null ||
            start === undefined ||
            computeStart === null ||
            computeStart === undefined ||
            end === null ||
            end === undefined
          ) {
            continue;
          }


          for (
            let cycle = start;
            cycle < computeStart;
            cycle += 1
          ) {

            result[
              sc
            ][
              cycle
            ] = {
              state: "switch",
              task,
            };
          }


          for (
            let cycle =
              computeStart;

            cycle < end;

            cycle += 1
          ) {

            result[
              sc
            ][
              cycle
            ] = {
              state: "compute",
              task,
            };
          }
        }


        return result;

      },
      [
        schedule
      ]
    );


  // =========================================================
  // Timeline → 3D
  // =========================================================

  useEffect(() => {

    if (onCycleChange) {

      onCycleChange(
        currentCycle
      );
    }


    if (!onSubcubeStatesChange) {
      return;
    }


    const states =
      Array.from(
        {
          length:
            NUM_SUBCUBES,
        },
        (_, sc) => {

          const cell =
            timelineBySubcube[
              sc
            ][
              currentCycle
            ];


          if (!cell) {

            return {
              subcube_id: sc,
              state: "idle",
              task: null,
            };
          }


          if (
            cell.state ===
            "switch"
          ) {

            return {
              subcube_id: sc,
              state: "switch",
              task:
                cell.task ??
                null,
            };
          }


          const matrixName =
            cell.task
              ?.matrix_name;


          return {
            subcube_id: sc,

            state:
              matrixName ??
              "idle",

            task:
              cell.task ??
              null,
          };
        }
      );


    onSubcubeStatesChange(
      states
    );

  }, [
    currentCycle,
    timelineBySubcube,
    onCycleChange,
    onSubcubeStatesChange,
  ]);


  // =========================================================
  // SC Stats
  // =========================================================

  const subcubeStats =
    useMemo(
      () => {

        const result =
          new Map();


        for (
          const stat
          of (
            schedule
              ?.subcubes ??
            []
          )
        ) {

          result.set(
            stat.subcube_id,
            stat
          );
        }


        return result;

      },
      [
        schedule
      ]
    );


  const criticalSet =
    useMemo(
      () =>
        new Set(
          schedule
            ?.critical_subcubes ??
          []
        ),
      [
        schedule
      ]
    );


  // =========================================================
  // Player Functions
  // =========================================================

  function firstCycle() {

    setIsPlaying(
      false
    );


    setCurrentCycle(
      0
    );
  }


  function previousCycle() {

    setIsPlaying(
      false
    );


    setCurrentCycle(
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


    setCurrentCycle(
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


    setCurrentCycle(
      maxCycle
    );
  }


  function togglePlay() {

    if (
      !isPlaying &&
      currentCycle >=
      maxCycle
    ) {

      setCurrentCycle(
        0
      );
    }


    setIsPlaying(
      (previous) =>
        !previous
    );
  }


  // =========================================================
  // Render
  // =========================================================

  return (
    <div className="layer-timeline">

      <div className="timeline-header">

        <div>

          <div className="timeline-small-title">
            LAYER EXECUTION
          </div>


          <h3>
            Layer {layerId}
          </h3>


          <p>
            16 个 Sub-Cube 的并行执行时间线。
          </p>

        </div>


        {schedule && (

          <div className="timeline-header-summary">

            <SummaryCard
              label="Cycles"

              value={
                schedule
                  .layer_cycles
              }
            />


            <SummaryCard
              label="Tasks"

              value={
                schedule
                  .task_count
              }
            />


            <SummaryCard
              label="Active SC"

              value={
                schedule
                  .active_subcube_count
              }
            />

          </div>

        )}

      </div>


      {loading && (

        <div className="timeline-loading">

          正在执行 Layer {layerId} Scheduler...

        </div>

      )}


      {error && (

        <div className="timeline-error">
          {error}
        </div>

      )}


      {!loading &&
       !error &&
       schedule && (

        <>

          <TimelinePlayer
            currentCycle={
              currentCycle
            }

            maxCycle={
              maxCycle
            }

            isPlaying={
              isPlaying
            }

            speed={
              speed
            }

            setSpeed={
              setSpeed
            }

            onFirst={
              firstCycle
            }

            onPrevious={
              previousCycle
            }

            onTogglePlay={
              togglePlay
            }

            onNext={
              nextCycle
            }

            onLast={
              lastCycle
            }
          />


          <CurrentCycleStrip
            currentCycle={
              currentCycle
            }

            timelineBySubcube={
              timelineBySubcube
            }
          />


          <div className="timeline-scroll">

            <div className="cycle-header-row">

              <div className="cycle-label-space">
                SC
              </div>


              <div
                className="cycle-header-cells"

                style={{
                  gridTemplateColumns:
                    `repeat(${Math.max(layerCycles, 1)}, 54px)`,
                }}
              >

                {Array.from(
                  {
                    length:
                      Math.max(
                        layerCycles,
                        1
                      ),
                  },
                  (_, cycle) => (

                    <div
                      key={
                        cycle
                      }

                      className={
                        cycle ===
                        currentCycle
                          ? "cycle-number current"
                          : "cycle-number"
                      }
                    >
                      {cycle}
                    </div>

                  )
                )}

              </div>


              <div className="cycle-stat-space">
                State
              </div>

            </div>


            {Array.from(
              {
                length:
                  NUM_SUBCUBES,
              },
              (_, sc) => (

                <SubcubeTimelineRow
                  key={
                    sc
                  }

                  subcubeId={
                    sc
                  }

                  layerCycles={
                    layerCycles
                  }

                  cells={
                    timelineBySubcube[
                      sc
                    ]
                  }

                  stat={
                    subcubeStats.get(
                      sc
                    )
                  }

                  critical={
                    criticalSet.has(
                      sc
                    )
                  }

                  currentCycle={
                    currentCycle
                  }
                />

              )
            )}

          </div>


          <TimelineLegend />

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

        .layer-timeline {
          margin-top: 12px;

          padding-top: 12px;

          border-top: 1px solid #e4e8ec;
        }


        .timeline-header {
          margin-bottom: 10px;

          display: flex;

          align-items: flex-start;

          justify-content: space-between;

          gap: 8px;
        }


        .timeline-small-title {
          margin-bottom: 4px;

          color: #9aa3ad;

          font-size: 16px;

          font-weight: 650;

          letter-spacing: 1px;
        }


        .timeline-header h3 {
          margin: 0 0 4px;

          color: #34404d;

          font-size: 18px;
        }


        .timeline-header p {
          margin: 0;

          color: #929ca7;

          font-size: 16px;
        }


        .timeline-header-summary {
          display: flex;

          gap: 5px;
        }


        .timeline-summary-card {
          min-width: 68px;

          padding: 7px 8px;

          border: 1px solid #dce2e8;

          border-radius: 4px;

          background: #f8fafb;
        }


        .timeline-summary-card span {
          display: block;

          margin-bottom: 2px;

          color: #929ca7;

          font-size: 12px;
        }


        .timeline-summary-card strong {
          color: #40505f;

          font-size: 16px;
        }


        .timeline-loading,
        .timeline-error {
          padding: 9px 10px;

          margin-bottom: 8px;

          border-radius: 5px;

          font-size: 12px;
        }


        .timeline-loading {
          border: 1px solid #dce3e9;

          background: #f8fafb;

          color: #7f8a96;
        }


        .timeline-error {
          border: 1px solid #e2bbbb;

          background: #fff6f6;

          color: #9d5959;
        }


        /* ================================================
           PLAYER
        ================================================ */


        .timeline-player {
          min-height: 50px;

          margin-bottom: 7px;

          padding: 8px 10px;

          display: flex;

          align-items: center;

          gap: 5px;

          border: 1px solid #dfe4e9;

          border-radius: 5px;

          background: #ffffff;
        }


        .player-cycle {
          min-width: 118px;

          color: #84909c;

          font-size: 13px;
        }


        .player-cycle strong {
          margin-left: 5px;

          color: #31475b;

          font-size: 16px;
        }


        .player-cycle span {
          margin-left: 2px;

          color: #a0a8b1;
        }


        .player-divider {
          width: 1px;

          height: 36px;

          margin: 0 3px;

          background: #e2e6ea;
        }


        .player-button {
          width: 34px;

          height: 36px;

          padding: 0;

          display: flex;

          align-items: center;

          justify-content: center;

          border: 1px solid #d8dfe6;

          border-radius: 4px;

          background: #ffffff;

          color: #566573;

          font-size: 12px;

          cursor: pointer;
        }


        .player-button:disabled {
          opacity: 0.3;

          cursor: default;
        }


        .player-button.play {
          width: 42px;

          border-color: #7898b5;

          background: #7798b7;

          color: #ffffff;
        }


        .speed-control {
          display: flex;

          align-items: center;

          gap: 4px;

          color: #89939e;

          font-size: 13px;
        }


        .speed-control select {
          width: 64px;

          height: 32px;

          border: 1px solid #d8dfe6;

          border-radius: 4px;

          background: #ffffff;

          color: #485562;

          font-size: 13px;
        }


        /* ================================================
           CURRENT CYCLE STRIP
        ================================================ */


        .current-cycle-strip {
          min-height: 42px;

          margin-bottom: 7px;

          padding: 7px 10px;

          display: flex;

          align-items: center;

          gap: 8px;

          border: 1px solid #dfe4e9;

          border-radius: 5px;

          background: #fafbfc;
        }


        .current-cycle-strip > strong {
          margin-right: 4px;

          color: #485664;

          font-size: 12px;
        }


        .strip-state {
          padding: 5px 7px;

          border-radius: 3px;

          font-size: 12px;

          font-weight: 650;
        }


        .strip-state.switch {
          background: #f7dede;

          color: #995c5c;
        }


        .strip-state.gate {
          background: #dceaf7;

          color: #426c91;
        }


        .strip-state.up {
          background: #f7e8d7;

          color: #8b653b;
        }


        .strip-state.down {
          background: #dfeee4;

          color: #477451;
        }


        .strip-state.idle {
          background: #f1f3f5;

          color: #7f8994;
        }


        /* ================================================
           TIMELINE
        ================================================ */


        .timeline-scroll {
          width: 100%;

          max-height: 620px;

          overflow: auto;

          padding-bottom: 4px;

          border: 1px solid #dfe4e9;

          border-radius: 5px;

          background: #ffffff;
        }


        .cycle-header-row {
          width: max-content;

          min-width: 100%;

          height: 36px;

          display: grid;

          grid-template-columns:
            78px auto 190px;

          align-items: stretch;

          position: sticky;

          top: 0;

          z-index: 4;

          border-bottom: 1px solid #e2e6ea;

          background: #f7f9fa;
        }


        .cycle-label-space,
        .cycle-stat-space {
          display: flex;

          align-items: center;

          padding: 0 10px;

          color: #8c96a0;

          font-size: 16px;

          font-weight: 650;
        }


        .cycle-stat-space {
          justify-content: flex-end;
        }


        .cycle-header-cells,
        .timeline-cells {
          display: grid;
        }


        .cycle-number {
          width: 54px;

          display: flex;

          align-items: center;

          justify-content: center;

          border-left: 1px solid #edf0f2;

          color: #9aa3ac;

          font-size: 13px;
        }


        .cycle-number.current {
          background: #dfe8f0;

          color: #324f69;

          font-weight: 700;

          box-shadow:
            inset 2px 0 0 #557998,
            inset -2px 0 0 #557998;
        }


        .timeline-row {
          width: max-content;

          min-width: 100%;

          height: 50px;

          display: grid;

          grid-template-columns:
            78px auto 190px;

          align-items: stretch;

          border-bottom: 1px solid #f0f2f4;
        }


        .timeline-row.critical {
          background: #fffaf5;
        }


        .timeline-sc-label {
          padding: 7px 8px;

          display: flex;

          flex-direction: column;

          justify-content: center;

          border-right: 1px solid #e7eaed;
        }


        .sc-name {
          color: #485563;

          font-size: 13px;

          font-weight: 700;
        }


        .timeline-row.critical
        .sc-name {
          color: #9a6840;
        }


        .sc-task-count {
          margin-top: 4px;

          color: #a0a8b1;

          font-size: 10px;
        }


        .timeline-cell {
          width: 54px;

          display: flex;

          align-items: center;

          justify-content: center;

          position: relative;

          border-left: 1px solid #f0f2f4;

          font-size: 13px;

          font-weight: 700;
        }


        .timeline-cell.current-cycle {
          box-shadow:
            inset 2px 0 0 #4c6f8c,
            inset -2px 0 0 #4c6f8c;

          z-index: 2;
        }


        .timeline-cell.idle {
          background: #fafbfc;

          color: transparent;
        }


        .timeline-cell.switching {
          background: #f7dede;

          color: #9e5858;
        }


        .timeline-cell.compute.gate {
          background: #dceaf7;

          color: #315d86;
        }


        .timeline-cell.compute.up {
          background: #f7e8d7;

          color: #855d32;
        }


        .timeline-cell.compute.down {
          background: #dfeee4;

          color: #386746;
        }


        .timeline-row-stat {
          padding: 0 10px;

          display: flex;

          align-items: center;

          gap: 5px;

          border-left: 1px solid #e7eaed;

          color: #9099a4;

          font-size: 12px;
        }


        .timeline-row-stat > * {
          flex: 1 1 0;

          min-width: 0;

          display: flex;

          align-items: center;

          justify-content: center;
        }


        .timeline-row-stat strong {
          color: #4d5966;

          font-size: 12px;
        }


        .current-state {
          padding: 4px 6px;

          border-radius: 3px;

          text-align: center;

          font-size: 10px;

          font-weight: 650;

          text-transform: uppercase;
        }


        .current-state.idle {
          background: #f1f3f5;

          color: #9099a3;
        }


        .current-state.switch {
          background: #f7dede;

          color: #985858;
        }


        .current-state.gate {
          background: #dceaf7;

          color: #315d86;
        }


        .current-state.up {
          background: #f7e8d7;

          color: #855d32;
        }


        .current-state.down {
          background: #dfeee4;

          color: #386746;
        }


        .critical-tag {
          padding: 4px 6px;

          border: 1px solid #dbb48e;

          border-radius: 3px;

          background: #fff3e7;

          color: #99653d;

          font-size: 10px;

          font-weight: 650;
        }


        /* ================================================
           LEGEND
        ================================================ */


        .timeline-legend {
          min-height: 40px;

          margin-top: 6px;

          padding: 0 10px;

          display: flex;

          align-items: center;

          gap: 14px;

          border: 1px solid #e0e4e8;

          border-radius: 5px;

          background: #ffffff;

          color: #7d8791;

          font-size: 12px;
        }


        .timeline-legend
        .legend-item {
          display: flex;

          align-items: center;

          gap: 4px;
        }


        .legend-cell {
          width: 18px;

          height: 18px;

          display: flex;

          align-items: center;

          justify-content: center;

          border: 1px solid #d6dce2;

          font-size: 12px;

          font-weight: 700;
        }


        .legend-cell.switching {
          background: #f7dede;

          color: #9e5858;
        }


        .legend-cell.gate {
          background: #dceaf7;

          color: #315d86;
        }


        .legend-cell.up {
          background: #f7e8d7;

          color: #855d32;
        }


        .legend-cell.down {
          background: #dfeee4;

          color: #386746;
        }


        .legend-cell.idle {
          background: #fafbfc;
        }


        .legend-current {
          margin-left: auto;

          color: #8d97a1;
        }



        /* ================================================
           READABILITY OVERRIDES
           更大字号、更舒展的行高、单元格严格居中
        ================================================ */

        .timeline-header h3 {
          font-size: 20px;
        }

        .timeline-header p {
          font-size: 13px;
        }

        .timeline-summary-card span {
          font-size: 11px;
        }

        .timeline-summary-card strong {
          font-size: 15px;
        }

        .timeline-player {
          min-height: 52px;
        }

        .player-cycle {
          font-size: 12px;
        }

        .player-cycle strong {
          font-size: 17px;
        }

        .speed-control {
          font-size: 12px;
        }

        .speed-control select {
          width: 70px;
          height: 34px;
          font-size: 12px;
        }

        .current-cycle-strip {
          min-height: 44px;
        }

        .current-cycle-strip > strong {
          font-size: 13px;
        }

        .strip-state {
          font-size: 12px;
          padding: 5px 8px;
        }

        .cycle-header-row {
          grid-template-columns: 82px auto 210px;
          height: 38px;
        }

        .timeline-row {
          grid-template-columns: 82px auto 210px;
          height: 52px;
        }

        .cycle-label-space,
        .cycle-stat-space {
          font-size: 12px;
          padding: 0 10px;
        }

        .cycle-number {
          width: 54px;
          font-size: 12px;
          line-height: 1;
        }

        .timeline-sc-label {
          padding: 6px 8px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 4px;
          line-height: 1.15;
          overflow: hidden;
        }

        .sc-name {
          font-size: 13px;
          line-height: 1.15;
          white-space: nowrap;
        }

        .sc-task-count {
          margin-top: 0;
          font-size: 10px;
          line-height: 1.15;
          white-space: nowrap;
        }

        .timeline-cells {
          align-items: stretch;
        }

        .timeline-cell {
          width: 54px;
          height: 52px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0;
          line-height: 1;
          font-size: 14px;
          font-weight: 750;
          text-align: center;
        }

        .timeline-row-stat {
          min-width: 0;
          padding: 0 10px;
          display: grid;
          grid-template-columns: 88px 54px 62px;
          align-items: center;
          justify-content: center;
          gap: 4px;
          font-size: 11px;
        }

        .timeline-row-stat > * {
          min-width: 0;
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .timeline-row-stat > span:not(.current-state):not(.critical-tag) {
          white-space: nowrap;
          font-size: 11px;
        }

        .timeline-row-stat strong {
          font-size: 11px;
        }

        .current-state {
          min-height: 30px;
          padding: 0 8px;
          font-size: 11px;
          line-height: 1;
        }

        .critical-tag {
          min-height: 30px;
          padding: 0 7px;
          font-size: 10px;
          line-height: 1;
        }

        .timeline-legend {
          min-height: 42px;
          font-size: 11px;
        }

        .legend-cell {
          width: 19px;
          height: 19px;
          font-size: 11px;
        }

      `}
    </style>
  );
}


export default LayerTimeline;