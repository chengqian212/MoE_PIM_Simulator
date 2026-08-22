import {
  useEffect,
  useMemo,
  useState,
} from "react";

import ExecutionCube3D from "./ExecutionCube3D";


const API_BASE = "http://127.0.0.1:8000";

const MAPPING_OPTIONS = [
  { id: "round_robin", label: "Round-Robin" },
  { id: "least_loaded", label: "Least-Loaded" },
  { id: "frequency_aware", label: "Frequency-aware" },
  { id: "trace_aware", label: "Trace-aware" },
];

const PREFILL_OPTIONS = [
  { id: "no_reuse", label: "No-Reuse" },
  { id: "switch_aware", label: "Switch-Aware" },
  { id: "aggressive_reuse", label: "Aggressive-Reuse" },
  { id: "largest_batch_reuse", label: "Largest-Batch-Reuse" },
];

const PREFILL_LABELS = Object.fromEntries(
  PREFILL_OPTIONS.map((item) => [item.id, item.label])
);

const LAYER_OPTIONS = Array.from({ length: 58 }, (_, layer) => ({
  id: String(layer),
  label: `L${layer}`,
}));


function optionLabel(items, id) {
  return items.find((item) => item.id === id)?.label ?? id;
}


function formatNumber(value, digits = 2) {
  if (!Number.isFinite(Number(value))) {
    return "--";
  }

  return Number(value).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}


function improvementPercent(a, b) {
  const left = Number(a);
  const right = Number(b);

  if (!Number.isFinite(left) || !Number.isFinite(right) || left === 0) {
    return null;
  }

  return ((left - right) / left) * 100;
}


function taskArray(layerResult) {
  return Array.isArray(layerResult?.tasks) ? layerResult.tasks : [];
}


function aggregateTasksBySubcube(layerResult) {
  const rows = Array.from({ length: 16 }, (_, subcubeId) => ({
    subcube_id: subcubeId,
    task_count: 0,
    busy_cycles: 0,
    switch_count: 0,
    wait_cycles: 0,
  }));

  for (const task of taskArray(layerResult)) {
    const sc = Number(task.subcube_id);
    if (!Number.isInteger(sc) || sc < 0 || sc >= rows.length) {
      continue;
    }

    const start = Number(task.start_cycle ?? 0);
    const end = Number(task.end_cycle ?? start);
    rows[sc].task_count += 1;
    rows[sc].busy_cycles += Math.max(0, end - start);
    rows[sc].wait_cycles += Math.max(0, Number(task.wait_cycles ?? 0));
    rows[sc].switch_count += Number(task.activation_cycles ?? 0) > 0 ? 1 : 0;
  }

  return rows;
}


function stateAtCycle(layerResult, subcubeId, cycle) {
  const tasks = taskArray(layerResult).filter(
    (task) => Number(task.subcube_id) === subcubeId
  );

  const running = tasks.find((task) => {
    const start = Number(task.start_cycle ?? 0);
    const end = Number(task.end_cycle ?? start);
    return start <= cycle && cycle < end;
  });

  if (running) {
    const computeStart = Number(
      running.compute_start_cycle ?? running.start_cycle ?? 0
    );
    const phase = cycle < computeStart
      ? "Switch"
      : String(running.matrix_name ?? "Run");

    return {
      kind: "running",
      label: phase,
      detail: `E${running.expert_id}`,
      waiting: tasks.filter((task) => {
        const ready = Number(task.ready_time ?? task.start_cycle ?? 0);
        const start = Number(task.start_cycle ?? 0);
        return ready <= cycle && cycle < start;
      }).length,
    };
  }

  const waiting = tasks.filter((task) => {
    const ready = Number(task.ready_time ?? task.start_cycle ?? 0);
    const start = Number(task.start_cycle ?? 0);
    return ready <= cycle && cycle < start;
  }).length;

  if (waiting > 0) {
    return {
      kind: "waiting",
      label: "Waiting",
      detail: `${waiting} tasks`,
      waiting,
    };
  }

  return {
    kind: "idle",
    label: "Idle",
    detail: "",
    waiting: 0,
  };
}


function cubeStatesAtCycle(layerResult, cycle) {
  const tasks = taskArray(layerResult);

  return Array.from({ length: 16 }, (_, subcubeId) => {
    const subcubeTasks = tasks.filter(
      (task) => Number(task.subcube_id) === subcubeId
    );

    const running = subcubeTasks.find((task) => {
      const start = Number(task.start_cycle ?? 0);
      const end = Number(task.end_cycle ?? start);
      return start <= cycle && cycle < end;
    });

    if (!running) {
      return {
        subcube_id: subcubeId,
        state: "idle",
        task: null,
      };
    }

    const start = Number(running.start_cycle ?? 0);
    const computeStart = Number(running.compute_start_cycle ?? start);
    let state = cycle < computeStart
      ? "switch"
      : String(running.matrix_name ?? "idle").toLowerCase();

    if (!["gate", "up", "down", "switch"].includes(state)) {
      state = "idle";
    }

    return {
      subcube_id: subcubeId,
      state,
      task: running,
    };
  });
}


async function fetchJson(url, options = undefined) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload?.detail ? String(payload.detail) : "";
    } catch {
      detail = await response.text();
    }

    throw new Error(detail || `HTTP ${response.status}`);
  }

  return response.json();
}


function StrategyComparison() {
  const [activeTab, setActiveTab] = useState("mapping");

  const [mappingPhase, setMappingPhase] = useState("decode");
  const [mappingA, setMappingA] = useState("round_robin");
  const [mappingB, setMappingB] = useState("trace_aware");

  const [prefillA, setPrefillA] = useState("no_reuse");
  const [prefillB, setPrefillB] = useState("aggressive_reuse");

  const [selectedLayer, setSelectedLayer] = useState(48);
  const [cycleCursor, setCycleCursor] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const [decodeToken, setDecodeToken] = useState(null);
  const [prefillBatch, setPrefillBatch] = useState(null);
  const [reference, setReference] = useState(null);
  const [runResult, setRunResult] = useState(null);

  const [sourceLoading, setSourceLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");


  async function loadDecodeToken() {
    try {
      setSourceLoading(true);
      setError("");
      setRunResult(null);
      const data = await fetchJson(`${API_BASE}/api/request/decode/random`);
      setDecodeToken(data);
    } catch (err) {
      console.error(err);
      setError(`读取真实 Decode Token 失败：${err.message}`);
    } finally {
      setSourceLoading(false);
    }
  }


  async function loadPrefillBatch() {
    try {
      setSourceLoading(true);
      setError("");
      setRunResult(null);
      const data = await fetchJson(`${API_BASE}/api/request/prefill/random`);
      setPrefillBatch(data);
    } catch (err) {
      console.error(err);
      setError(`读取真实 Prefill Batch 失败：${err.message}`);
    } finally {
      setSourceLoading(false);
    }
  }


  useEffect(() => {
    async function initialize() {
      try {
        const [decodeData, prefillData] = await Promise.all([
          fetchJson(`${API_BASE}/api/request/decode/random`),
          fetchJson(`${API_BASE}/api/request/prefill/random`),
        ]);

        setDecodeToken(decodeData);
        setPrefillBatch(prefillData);
      } catch (err) {
        console.error(err);
        setError(`真实请求初始化失败：${err.message}`);
      }

      try {
        const referenceData = await fetchJson(
          `${API_BASE}/api/comparison/reference`
        );
        setReference(referenceData);
      } catch (err) {
        console.warn("正式参考结果暂不可用：", err);
        setReference({ unavailable: true });
      }
    }

    initialize();
  }, []);


  function invalidate(next) {
    setRunResult(null);
    setError("");
    next();
  }


  async function runComparison() {
    try {
      setRunning(true);
      setError("");

      let url = "";
      let body = {};

      if (activeTab === "mapping") {
        url = `${API_BASE}/api/comparison/mapping`;

        if (mappingPhase === "decode") {
          if (!decodeToken?.source) {
            throw new Error("还没有可用的 Decode Token。");
          }

          body = {
            phase: "decode",
            mapping_a: mappingA,
            mapping_b: mappingB,
            selected_layer: selectedLayer,
            decode_source: {
              category: decodeToken.source.category,
              filename: decodeToken.source.filename,
              segment_index: decodeToken.source.segment_index,
              token_index: decodeToken.source.token_index ?? 0,
            },
          };
        } else {
          if (prefillBatch?.batch_id === undefined) {
            throw new Error("还没有可用的 Prefill Batch。");
          }

          body = {
            phase: "prefill",
            mapping_a: mappingA,
            mapping_b: mappingB,
            selected_layer: selectedLayer,
            prefill_batch_id: prefillBatch.batch_id,
          };
        }
      } else if (activeTab === "prefill") {
        if (prefillBatch?.batch_id === undefined) {
          throw new Error("还没有可用的 Prefill Batch。");
        }

        url = `${API_BASE}/api/comparison/prefill`;
        body = {
          batch_id: prefillBatch.batch_id,
          mode_a: prefillA,
          mode_b: prefillB,
          selected_layer: selectedLayer,
        };
      } else {
        if (!decodeToken?.source) {
          throw new Error("还没有可用的 Decode Token。");
        }

        url = `${API_BASE}/api/comparison/decode-optimality`;
        body = {
          source: {
            category: decodeToken.source.category,
            filename: decodeToken.source.filename,
            segment_index: decodeToken.source.segment_index,
            token_index: decodeToken.source.token_index ?? 0,
          },
          layer_id: selectedLayer,
          time_limit_seconds: 5.0,
          solver_workers: 8,
        };
      }

      const data = await fetchJson(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      setRunResult(data);
    } catch (err) {
      console.error(err);
      setRunResult(null);
      setError(`A/B 运行失败：${err.message}`);
    } finally {
      setRunning(false);
    }
  }


  useEffect(() => {
    setCycleCursor(0);
    setIsPlaying(false);
  }, [runResult]);


  const currentSource = useMemo(() => {
    const usePrefill =
      activeTab === "prefill" ||
      (activeTab === "mapping" && mappingPhase === "prefill");

    if (usePrefill) {
      if (!prefillBatch) {
        return null;
      }

      return {
        kind: "Prefill Batch",
        title: `Batch-${prefillBatch.batch_id}`,
        parts: [
          prefillBatch.category,
          prefillBatch.filename,
          `${prefillBatch.input_tokens} tokens`,
        ],
      };
    }

    if (!decodeToken?.source) {
      return null;
    }

    return {
      kind: "Decode Token",
      title: `${decodeToken.source.filename} · Seg-${decodeToken.source.segment_index}`,
      parts: [
        decodeToken.source.category,
        `Token-${decodeToken.source.token_index ?? 0}`,
        activeTab === "decode" ? `Layer L${selectedLayer}` : "58 Layers",
      ],
    };
  }, [
    activeTab,
    decodeToken,
    mappingPhase,
    prefillBatch,
    selectedLayer,
  ]);


  const formalReference = useMemo(() => {
    if (!reference) {
      return "正式 Held-out 参考加载中...";
    }

    if (reference.unavailable) {
      return "正式汇总暂不可用，不影响当前真实 A/B。";
    }

    if (activeTab === "mapping") {
      const metrics = reference.mapping?.metrics ?? [];
      const a = metrics.find((item) => item.mode === mappingA);
      const b = metrics.find((item) => item.mode === mappingB);

      if (!a || !b) {
        return "正式 Mapping 参考结果不可用。";
      }

      if (mappingPhase === "decode") {
        return `Held-out：${a.display_name} ${formatNumber(a.decode_mean_cycles_per_token, 2)} → ${b.display_name} ${formatNumber(b.decode_mean_cycles_per_token, 2)} cycles/token`;
      }

      return `Held-out：${a.display_name} ${formatNumber(a.prefill_mean_latency, 2)} → ${b.display_name} ${formatNumber(b.prefill_mean_latency, 2)} cycles/batch`;
    }

    if (activeTab === "prefill") {
      const results = reference.prefill?.results ?? {};
      const a = results[prefillA];
      const b = results[prefillB];

      if (!a || !b) {
        return "正式 Prefill 参考结果不可用。";
      }

      return `Held-out 404 batches：${PREFILL_LABELS[prefillA]} ${formatNumber(a.prefill_mean_cycles, 2)} → ${PREFILL_LABELS[prefillB]} ${formatNumber(b.prefill_mean_cycles, 2)} cycles/batch`;
    }

    const summary = reference.decode?.summary;
    if (!summary) {
      return "正式 Decode Optimality 参考结果不可用。";
    }

    return `Held-out：${formatNumber(summary.greedy_already_optimal_rate * 100, 2)}% Layer 已最优 · ${formatNumber(summary.optimal_proven_count, 0)} / ${formatNumber(summary.instance_count, 0)} CP-SAT proven OPTIMAL`;
  }, [
    activeTab,
    mappingA,
    mappingB,
    mappingPhase,
    prefillA,
    prefillB,
    reference,
  ]);


  const liveComparison = useMemo(() => {
    if (!runResult?.a?.result || !runResult?.b?.result) {
      return null;
    }

    const a = runResult.a.result;
    const b = runResult.b.result;

    if (activeTab === "mapping") {
      const isPrefill = mappingPhase === "prefill";
      return {
        title: isPrefill ? "当前 Prefill Batch 总周期" : "当前 Decode Token 总周期",
        unit: isPrefill ? "cycles / batch" : "cycles / token",
        aLabel: optionLabel(MAPPING_OPTIONS, mappingA),
        bLabel: optionLabel(MAPPING_OPTIONS, mappingB),
        aValue: a.total_cycles,
        bValue: b.total_cycles,
        digits: 0,
        secondary: isPrefill
          ? [
              {
                label: "Cycles / Input Token",
                a: formatNumber(a.cycles_per_input_token, 2),
                b: formatNumber(b.cycles_per_input_token, 2),
              },
              {
                label: "Switches",
                a: formatNumber(a.total_switches, 0),
                b: formatNumber(b.total_switches, 0),
              },
              {
                label: "Mapping Conflict",
                a: formatNumber(runResult.a.strategy?.mapping_conflict_cost, 0),
                b: formatNumber(runResult.b.strategy?.mapping_conflict_cost, 0),
              },
            ]
          : [
              {
                label: "Switches",
                a: formatNumber(a.total_switches, 0),
                b: formatNumber(b.total_switches, 0),
              },
              {
                label: "Max Layer Cycles",
                a: formatNumber(a.max_layer_cycles, 0),
                b: formatNumber(b.max_layer_cycles, 0),
              },
              {
                label: "Mapping Conflict",
                a: formatNumber(runResult.a.strategy?.mapping_conflict_cost, 0),
                b: formatNumber(runResult.b.strategy?.mapping_conflict_cost, 0),
              },
            ],
      };
    }

    if (activeTab === "prefill") {
      return {
        title: "当前 Prefill Batch 总周期",
        unit: "cycles / batch",
        aLabel: PREFILL_LABELS[prefillA],
        bLabel: PREFILL_LABELS[prefillB],
        aValue: a.total_cycles,
        bValue: b.total_cycles,
        digits: 0,
        secondary: [
          {
            label: "Cycles / Input Token",
            a: formatNumber(a.cycles_per_input_token, 2),
            b: formatNumber(b.cycles_per_input_token, 2),
          },
          {
            label: "Switches",
            a: formatNumber(a.total_switches, 0),
            b: formatNumber(b.total_switches, 0),
          },
          {
            label: "Wait Cycles",
            a: formatNumber(a.total_wait_cycles, 0),
            b: formatNumber(b.total_wait_cycles, 0),
          },
        ],
      };
    }

    return {
      title: `当前 Decode Token · Layer L${selectedLayer}`,
      unit: "cycles / layer",
      aLabel: "Greedy",
      bLabel: "CP-SAT Optimal",
      aValue: a.total_cycles,
      bValue: b.total_cycles,
      digits: 0,
      secondary: [
        {
          label: "CP-SAT Status",
          a: "Greedy",
          b: b.status ?? "--",
        },
        {
          label: "Proven Optimal",
          a: "Runtime",
          b: b.proven_optimal ? "YES" : "NO",
        },
        {
          label: "Solver Time",
          a: "--",
          b: `${formatNumber(b.wall_time_seconds, 3)} s`,
        },
      ],
    };
  }, [
    activeTab,
    mappingA,
    mappingB,
    mappingPhase,
    prefillA,
    prefillB,
    runResult,
    selectedLayer,
  ]);


  const layerDelta = useMemo(() => {
    const a = runResult?.a?.result?.layer_cycles;
    const b = runResult?.b?.result?.layer_cycles;

    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return null;
    }

    let best = null;

    for (let layer = 0; layer < a.length; layer += 1) {
      const delta = Number(a[layer]) - Number(b[layer]);
      const score = Math.abs(delta);

      if (!best || score > best.score) {
        best = {
          layer,
          a: Number(a[layer]),
          b: Number(b[layer]),
          delta,
          score,
        };
      }
    }

    return best;
  }, [runResult]);


  const delta = liveComparison
    ? Number(liveComparison.aValue) - Number(liveComparison.bValue)
    : null;

  const improvement = liveComparison
    ? improvementPercent(liveComparison.aValue, liveComparison.bValue)
    : null;

  const usesPrefillSource =
    activeTab === "prefill" ||
    (activeTab === "mapping" && mappingPhase === "prefill");


  const visualData = useMemo(() => {
    const resultA = runResult?.a?.result;
    const resultB = runResult?.b?.result;

    if (!resultA || !resultB || !liveComparison) {
      return null;
    }

    const isDecodeOptimality = activeTab === "decode";
    const layerA = isDecodeOptimality ? resultA : resultA.selected_layer;
    const layerB = isDecodeOptimality ? resultB : resultB.selected_layer;

    if (!layerA || !layerB) {
      return null;
    }

    const layerCyclesA = Array.isArray(resultA.layer_cycles)
      ? resultA.layer_cycles.map(Number)
      : null;
    const layerCyclesB = Array.isArray(resultB.layer_cycles)
      ? resultB.layer_cycles.map(Number)
      : null;

    const aggregateA = Array.isArray(resultA.subcubes)
      ? resultA.subcubes
      : aggregateTasksBySubcube(layerA);
    const aggregateB = Array.isArray(resultB.subcubes)
      ? resultB.subcubes
      : aggregateTasksBySubcube(layerB);

    const maxCycles = Math.max(
      Number(layerA.total_cycles ?? 0),
      Number(layerB.total_cycles ?? 0),
      1
    );

    return {
      aLabel: liveComparison.aLabel,
      bLabel: liveComparison.bLabel,
      layerA,
      layerB,
      layerCyclesA,
      layerCyclesB,
      aggregateA,
      aggregateB,
      maxCycles,
    };
  }, [activeTab, liveComparison, runResult]);


  useEffect(() => {
    if (!isPlaying || !visualData) {
      return undefined;
    }

    const delay = Math.max(80, Math.round(500 / playbackSpeed));
    const timer = window.setInterval(() => {
      setCycleCursor((current) => {
        if (current >= visualData.maxCycles) {
          setIsPlaying(false);
          return visualData.maxCycles;
        }
        return Math.min(visualData.maxCycles, current + 1);
      });
    }, delay);

    return () => window.clearInterval(timer);
  }, [isPlaying, playbackSpeed, visualData]);


  function togglePlayback() {
    if (!visualData) {
      return;
    }
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    if (cycleCursor >= visualData.maxCycles) {
      setCycleCursor(0);
    }
    setIsPlaying(true);
  }


  function stepPlayback() {
    if (!visualData) {
      return;
    }
    setIsPlaying(false);
    setCycleCursor((current) => Math.min(visualData.maxCycles, current + 1));
  }


  function resetPlayback() {
    setIsPlaying(false);
    setCycleCursor(0);
  }


  return (
    <div className="strategy-page">
      <div className="strategy-header">
        <div>
          <h2>策略对比</h2>
        </div>
        <div className="strategy-header-badges">
          <span>Real Trace</span>
          <span>Held-out Reference</span>
          <span>MoE Expert Only</span>
        </div>
      </div>

      <div className="strategy-tabs">
        <TabButton
          active={activeTab === "mapping"}
          label="Mapping"
          sub="矩阵放哪里"
          onClick={() => invalidate(() => setActiveTab("mapping"))}
        />

        <TabButton
          active={activeTab === "prefill"}
          label="Prefill 调度"
          sub="任务怎么排"
          onClick={() => invalidate(() => setActiveTab("prefill"))}
        />

        <TabButton
          active={activeTab === "decode"}
          label="Decode 最优性"
          sub="还有多少空间"
          onClick={() => invalidate(() => setActiveTab("decode"))}
        />
      </div>

      <section className="strategy-control-panel">
        {activeTab === "mapping" && (
          <>
            <ControlSelect
              label="阶段 / Phase"
              value={mappingPhase}
              onChange={(value) => invalidate(() => setMappingPhase(value))}
              options={[
                { id: "decode", label: "Decode" },
                { id: "prefill", label: "Prefill" },
              ]}
            />

            <ControlSelect
              label="方案 A"
              value={mappingA}
              onChange={(value) => invalidate(() => setMappingA(value))}
              options={MAPPING_OPTIONS}
            />

            <CompareMark />

            <ControlSelect
              label="方案 B"
              value={mappingB}
              onChange={(value) => invalidate(() => setMappingB(value))}
              options={MAPPING_OPTIONS}
            />

            <FixedControl
              label="固定条件"
              value={mappingPhase === "prefill"
                ? "Pairing Trace+LS · Aggressive-Reuse"
                : "Pairing Trace+LS · Greedy"}
              wide
            />

            <ControlSelect
              label="观察 Layer"
              value={String(selectedLayer)}
              onChange={(value) =>
                invalidate(() => setSelectedLayer(Number(value)))
              }
              options={LAYER_OPTIONS}
            />
          </>
        )}

        {activeTab === "prefill" && (
          <>
            <FixedControl label="阶段 / Phase" value="Prefill" />

            <ControlSelect
              label="方案 A"
              value={prefillA}
              onChange={(value) => invalidate(() => setPrefillA(value))}
              options={PREFILL_OPTIONS}
            />

            <CompareMark />

            <ControlSelect
              label="方案 B"
              value={prefillB}
              onChange={(value) => invalidate(() => setPrefillB(value))}
              options={PREFILL_OPTIONS}
            />

            <FixedControl label="固定 Mapping" value="Trace-aware" />


            <ControlSelect
              label="观察 Layer"
              value={String(selectedLayer)}
              onChange={(value) =>
                invalidate(() => setSelectedLayer(Number(value)))
              }
              options={LAYER_OPTIONS}
            />
          </>
        )}

        {activeTab === "decode" && (
          <>
            <FixedControl label="阶段 / Phase" value="Decode" />
            <FixedControl label="方案 A" value="Greedy" />
            <CompareMark />
            <FixedControl label="方案 B" value="CP-SAT Optimal" />

            <ControlSelect
              label="模型层 / Layer"
              value={String(selectedLayer)}
              onChange={(value) =>
                invalidate(() => setSelectedLayer(Number(value)))
              }
              options={LAYER_OPTIONS}
            />
          </>
        )}
      </section>

      <section className="request-bar">
        <div className="request-source">
          <span>{currentSource?.kind ?? "真实请求"}</span>
          <strong>{currentSource?.title ?? "读取中..."}</strong>
          <div>
            {(currentSource?.parts ?? []).map((part) => (
              <em key={part}>{part}</em>
            ))}
          </div>
        </div>

        <button
          type="button"
          className="secondary-button"
          disabled={sourceLoading || running}
          onClick={usesPrefillSource ? loadPrefillBatch : loadDecodeToken}
        >
          {sourceLoading ? "读取中..." : usesPrefillSource ? "换一个 Batch" : "换一个 Token"}
        </button>

        <button
          type="button"
          className="run-button"
          disabled={sourceLoading || running || !currentSource}
          onClick={runComparison}
        >
          {running ? "正在运行 A/B..." : "运行真实 A/B"}
        </button>
      </section>

      <div className="reference-strip">
        <strong>正式参考</strong>
        <span>{formalReference}</span>
      </div>

      {error && (
        <div className="strategy-error">{error}</div>
      )}

      {!liveComparison ? (
        <section className="empty-result">
          <strong>当前实例尚未运行</strong>
          <span>选择方案后点击“运行真实 A/B”，这里会显示当前 Token / Batch / Layer 的真实周期。</span>
        </section>
      ) : (
        <section className="comparison-result">
          <div className="comparison-title-row">
            <div>
              <span>当前真实实例</span>
              <strong>{liveComparison.title}</strong>
            </div>
            <small>{liveComparison.unit}</small>
          </div>

          <div className="comparison-main-grid">
            <ResultCard
              side="A"
              label={liveComparison.aLabel}
              value={formatNumber(liveComparison.aValue, liveComparison.digits)}
              preferred={Number(liveComparison.aValue) < Number(liveComparison.bValue)}
            />

            <div className="comparison-delta">
              <span>差值 / Δ</span>
              <strong>
                {delta >= 0 ? "−" : "+"}
                {formatNumber(Math.abs(delta), liveComparison.digits)}
              </strong>
              <em className={improvement >= 0 ? "better" : "worse"}>
                {improvement >= 0 ? "↓" : "↑"} {formatNumber(Math.abs(improvement), 2)}%
              </em>
            </div>

            <ResultCard
              side="B"
              label={liveComparison.bLabel}
              value={formatNumber(liveComparison.bValue, liveComparison.digits)}
              preferred={Number(liveComparison.bValue) < Number(liveComparison.aValue)}
            />
          </div>

          <div className={`secondary-grid secondary-${liveComparison.secondary.length}`}>
            {liveComparison.secondary.map((item) => (
              <div className="secondary-item" key={item.label}>
                <span>{item.label}</span>
                <div>
                  <strong>{item.a}</strong>
                  <b>→</b>
                  <strong>{item.b}</strong>
                </div>
              </div>
            ))}
          </div>

          {layerDelta && (
            <div className="layer-delta-strip">
              <span>本次差异最大的 Layer</span>
              <strong>L{layerDelta.layer}</strong>
              <em>A {layerDelta.a}</em>
              <b>→</b>
              <em>B {layerDelta.b}</em>
              <small>
                Δ {layerDelta.delta > 0 ? "−" : layerDelta.delta < 0 ? "+" : ""}
                {Math.abs(layerDelta.delta)} cycles
              </small>
            </div>
          )}
        </section>
      )}

      {visualData && (
        <section className="visual-stack">
          {visualData.layerCyclesA && visualData.layerCyclesB ? (
            <LayerDifferenceChart
              a={visualData.layerCyclesA}
              b={visualData.layerCyclesB}
              aLabel={visualData.aLabel}
              bLabel={visualData.bLabel}
              selectedLayer={selectedLayer}
            />
          ) : (
            <div className="single-layer-strip">
              <span>Decode 最优性只求当前选择层</span>
              <strong>L{selectedLayer}</strong>
              <em>Greedy / CP-SAT 共用同一真实 Token 与 Mapping</em>
            </div>
          )}

          <SubcubeLoadComparison
            a={visualData.aggregateA}
            b={visualData.aggregateB}
            aLabel={visualData.aLabel}
            bLabel={visualData.bLabel}
          />

          <TimelineComparison
            a={visualData.layerA}
            b={visualData.layerB}
            aLabel={visualData.aLabel}
            bLabel={visualData.bLabel}
            maxCycles={visualData.maxCycles}
            cursor={cycleCursor}
            selectedLayer={selectedLayer}
          />

          <SubcubeSnapshotComparison
            a={visualData.layerA}
            b={visualData.layerB}
            aLabel={visualData.aLabel}
            bLabel={visualData.bLabel}
            maxCycles={visualData.maxCycles}
            cycle={cycleCursor}
            onCycleChange={(value) => {
              setIsPlaying(false);
              setCycleCursor(value);
            }}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            onTogglePlayback={togglePlayback}
            onStep={stepPlayback}
            onReset={resetPlayback}
            onSpeedChange={setPlaybackSpeed}
          />
        </section>
      )}

      <Style />
    </div>
  );
}


function TabButton({ active, label, sub, onClick }) {
  return (
    <button
      type="button"
      className={active ? "strategy-tab active" : "strategy-tab"}
      onClick={onClick}
    >
      <strong>{label}</strong>
      <span>{sub}</span>
    </button>
  );
}


function ControlSelect({ label, value, onChange, options }) {
  return (
    <label className="control-field">
      <span>{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((item) => (
          <option key={item.id} value={item.id}>
            {item.label}
          </option>
        ))}
      </select>
    </label>
  );
}


function FixedControl({ label, value, wide = false }) {
  return (
    <div className={wide ? "control-field fixed wide" : "control-field fixed"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}


function CompareMark() {
  return <div className="compare-mark" aria-hidden="true">VS</div>;
}


function ResultCard({ side, label, value, preferred = false }) {
  return (
    <div className={preferred ? "result-card preferred" : "result-card"}>
      <div className="result-card-top">
        <span className="side-badge">{side}</span>
        {preferred && <em>当前更低</em>}
      </div>
      <strong className="result-card-label">{label}</strong>
      <div className="result-card-value">{value}</div>
    </div>
  );
}


function SectionHeading({ title, sub }) {
  return (
    <div className="visual-section-heading">
      <strong>{title}</strong>
      <span>{sub}</span>
    </div>
  );
}


function LayerDifferenceChart({ a, b, aLabel, bLabel, selectedLayer }) {
  const maxValue = Math.max(...a, ...b, 1);
  const deltas = a
    .map((value, layer) => ({
      layer,
      a: Number(value),
      b: Number(b[layer]),
      delta: Number(value) - Number(b[layer]),
    }))
    .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta));

  const top = deltas.slice(0, 3);

  return (
    <section className="visual-card">
      <SectionHeading
        title="58 Layer 周期差异"
        sub={`A ${aLabel} · B ${bLabel} · 当前观察 L${selectedLayer}`}
      />

      <div className="layer-chart">
        {a.map((value, layer) => {
          const aValue = Number(value);
          const bValue = Number(b[layer]);
          const active = layer === selectedLayer;

          return (
            <div
              className={active ? "layer-column active" : "layer-column"}
              key={layer}
              title={`L${layer} · A ${aValue} cycles · B ${bValue} cycles · Δ ${aValue - bValue}`}
            >
              <div className="layer-bars">
                <span
                  className="layer-bar bar-a"
                  style={{ height: `${Math.max(3, (aValue / maxValue) * 100)}%` }}
                />
                <span
                  className="layer-bar bar-b"
                  style={{ height: `${Math.max(3, (bValue / maxValue) * 100)}%` }}
                />
              </div>
              <small>
                {active || layer === 0 || layer === 57 || layer % 10 === 0
                  ? `L${layer}`
                  : ""}
              </small>
            </div>
          );
        })}
      </div>

      <div className="layer-chart-footer">
        <div className="chart-legend">
          <span><i className="legend-a" />A</span>
          <span><i className="legend-b" />B</span>
        </div>

        <div className="top-layer-deltas">
          {top.map((item) => (
            <span key={item.layer}>
              L{item.layer} · {item.a}→{item.b}
              <b>{item.delta > 0 ? ` −${item.delta}` : item.delta < 0 ? ` +${Math.abs(item.delta)}` : " ="}</b>
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}


function normalizeSubcubeRows(rows) {
  const byId = new Map(
    (Array.isArray(rows) ? rows : []).map((row) => [Number(row.subcube_id), row])
  );

  return Array.from({ length: 16 }, (_, sc) => ({
    subcube_id: sc,
    busy_cycles: Number(byId.get(sc)?.busy_cycles ?? 0),
    task_count: Number(byId.get(sc)?.task_count ?? 0),
    switch_count: Number(byId.get(sc)?.switch_count ?? 0),
    wait_cycles: Number(byId.get(sc)?.wait_cycles ?? 0),
  }));
}


function SubcubeLoadComparison({ a, b, aLabel, bLabel }) {
  const rowsA = normalizeSubcubeRows(a);
  const rowsB = normalizeSubcubeRows(b);
  const maxValue = Math.max(
    ...rowsA.map((row) => row.busy_cycles),
    ...rowsB.map((row) => row.busy_cycles),
    1
  );

  return (
    <section className="visual-card">
      <SectionHeading
        title="16 Sub-Cube 负载"
        sub="当前请求的累计 busy cycles；只看负载分布，不重复展示空间装箱对比"
      />

      <div className="sc-load-grid">
        {rowsA.map((rowA, sc) => {
          const rowB = rowsB[sc];
          return (
            <div className="sc-load-cell" key={sc}>
              <strong>SC-{sc}</strong>
              <div className="sc-load-line">
                <span>A</span>
                <div><i className="load-a" style={{ width: `${(rowA.busy_cycles / maxValue) * 100}%` }} /></div>
                <em>{rowA.busy_cycles}</em>
              </div>
              <div className="sc-load-line">
                <span>B</span>
                <div><i className="load-b" style={{ width: `${(rowB.busy_cycles / maxValue) * 100}%` }} /></div>
                <em>{rowB.busy_cycles}</em>
              </div>
            </div>
          );
        })}
      </div>

      <div className="visual-footnote">
        A {aLabel} · B {bLabel}
      </div>
    </section>
  );
}


function TimelineComparison({ a, b, aLabel, bLabel, maxCycles, cursor, selectedLayer }) {
  return (
    <CollapsibleVisualSection
      title={`L${selectedLayer} A/B Timeline`}
      sub={`统一绝对横轴 0–${maxCycles} cycles；较快方案结束后右侧保持空白`}
      defaultOpen
    >
      <div className="timeline-legend">
        <span className="matrix-gate">Gate</span>
        <span className="matrix-up">Up</span>
        <span className="matrix-down">Down</span>
        <span className="matrix-switch">Switch</span>
      </div>

      <TimelinePanel
        side="A"
        label={aLabel}
        layer={a}
        maxCycles={maxCycles}
        cursor={cursor}
      />

      <TimelinePanel
        side="B"
        label={bLabel}
        layer={b}
        maxCycles={maxCycles}
        cursor={cursor}
      />
    </CollapsibleVisualSection>
  );
}


function TimelinePanel({ side, label, layer, maxCycles, cursor }) {
  const tasks = taskArray(layer);
  const ticks = Array.from({ length: Math.min(maxCycles, 10) + 1 }, (_, index) => {
    const ratio = index / Math.min(maxCycles, 10);
    return Math.round(ratio * maxCycles);
  });

  return (
    <div className="timeline-panel">
      <div className="timeline-panel-title">
        <span>{side}</span>
        <strong>{label}</strong>
        <em>{formatNumber(layer?.total_cycles, 0)} cycles</em>
      </div>

      <div className="timeline-axis">
        <span className="axis-label">SC</span>
        <div>
          {ticks.map((tick, index) => (
            <i key={`${tick}-${index}`} style={{ left: `${(tick / maxCycles) * 100}%` }}>
              {tick}
            </i>
          ))}
        </div>
      </div>

      <div className="timeline-rows">
        {Array.from({ length: 16 }, (_, sc) => {
          const rowTasks = tasks.filter((task) => Number(task.subcube_id) === sc);
          return (
            <div className="timeline-row" key={sc}>
              <span className="timeline-sc-label">{sc}</span>
              <div className="timeline-track">
                <span
                  className="timeline-cursor"
                  style={{ left: `${Math.min(100, Math.max(0, (cursor / maxCycles) * 100))}%` }}
                />
                {rowTasks.map((task, index) => {
                  const start = Number(task.start_cycle ?? 0);
                  const end = Number(task.end_cycle ?? start);
                  const computeStart = Number(task.compute_start_cycle ?? start);
                  const width = Math.max(0.35, ((end - start) / maxCycles) * 100);
                  const activationWidth = Math.max(0, ((computeStart - start) / Math.max(1, end - start)) * 100);
                  const matrix = String(task.matrix_name ?? "run");

                  return (
                    <span
                      className={`timeline-task task-${matrix}`}
                      key={`${task.expert_id}-${matrix}-${start}-${index}`}
                      style={{
                        left: `${(start / maxCycles) * 100}%`,
                        width: `${width}%`,
                      }}
                      title={`SC-${sc} · E${task.expert_id} · ${matrix} · ${start}–${end}`}
                    >
                      {activationWidth > 0 && (
                        <i className="task-activation" style={{ width: `${activationWidth}%` }} />
                      )}
                    </span>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


function SubcubeSnapshotComparison({
  a,
  b,
  aLabel,
  bLabel,
  maxCycles,
  cycle,
  onCycleChange,
  isPlaying,
  playbackSpeed,
  onTogglePlayback,
  onStep,
  onReset,
  onSpeedChange,
}) {
  const cubeStatesA = cubeStatesAtCycle(a, cycle);
  const cubeStatesB = cubeStatesAtCycle(b, cycle);

  return (
    <CollapsibleVisualSection
      title="调度动画"
      sub="A / B 共用同一个 Cycle；2D Timeline、3D Cube 与状态明细完全同步"
      defaultOpen
    >
      <div className="playback-toolbar">
        <button type="button" className={isPlaying ? "playback-button active" : "playback-button"} onClick={onTogglePlayback}>
          {isPlaying ? "暂停" : "播放"}
        </button>
        <button type="button" className="playback-button" onClick={onStep}>单步 +1</button>
        <button type="button" className="playback-button" onClick={onReset}>重置</button>
        <label className="playback-speed">
          <span>速度</span>
          <select value={String(playbackSpeed)} onChange={(event) => onSpeedChange(Number(event.target.value))}>
            <option value="0.5">0.5×</option>
            <option value="1">1×</option>
            <option value="2">2×</option>
            <option value="4">4×</option>
          </select>
        </label>
        <div className="playback-now">
          <span>{isPlaying ? "播放中" : "已暂停"}</span>
          <strong>Cycle {Math.min(cycle, maxCycles)} / {maxCycles}</strong>
        </div>
      </div>

      <div className="cycle-control-row">
        <span>Cycle</span>
        <input
          type="range"
          min="0"
          max={maxCycles}
          step="1"
          value={Math.min(cycle, maxCycles)}
          onChange={(event) => onCycleChange(Number(event.target.value))}
        />
        <strong>{Math.min(cycle, maxCycles)} / {maxCycles}</strong>
      </div>

      <div className="cube-comparison-grid">
        <div className="cube-comparison-side">
          <div className="cube-side-title">
            <span>A</span>
            <strong>{aLabel}</strong>
          </div>
          <ExecutionCube3D
            currentCycle={cycle}
            currentLayer={Number(a?.layer_id ?? 0)}
            subcubeStates={cubeStatesA}
          />
        </div>

        <div className="cube-comparison-side">
          <div className="cube-side-title">
            <span>B</span>
            <strong>{bLabel}</strong>
          </div>
          <ExecutionCube3D
            currentCycle={cycle}
            currentLayer={Number(b?.layer_id ?? 0)}
            subcubeStates={cubeStatesB}
          />
        </div>
      </div>

      <div className="snapshot-detail-heading">16 个 Sub-Cube 状态明细</div>
      <div className="snapshot-columns">
        <SnapshotPanel side="A" label={aLabel} layer={a} cycle={cycle} />
        <SnapshotPanel side="B" label={bLabel} layer={b} cycle={cycle} />
      </div>
    </CollapsibleVisualSection>
  );
}


function CollapsibleVisualSection({ title, sub, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className={open ? "visual-card collapsible-card open" : "visual-card collapsible-card"}>
      <button
        type="button"
        className="collapsible-header"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <div>
          <strong>{title}</strong>
          <span>{sub}</span>
        </div>
        <b>{open ? "收起 ▲" : "展开 ▼"}</b>
      </button>

      {open && (
        <div className="collapsible-body">
          {children}
        </div>
      )}
    </section>
  );
}


function SnapshotPanel({ side, label, layer, cycle }) {
  return (
    <div className="snapshot-panel">
      <div className="snapshot-title">
        <span>{side}</span>
        <strong>{label}</strong>
      </div>

      <div className="snapshot-grid">
        {Array.from({ length: 16 }, (_, sc) => {
          const state = stateAtCycle(layer, sc, cycle);
          return (
            <div className={`snapshot-cell ${state.kind}`} key={sc}>
              <span>SC-{sc}</span>
              <strong>{state.label}</strong>
              <em>{state.detail || (state.waiting ? `${state.waiting} waiting` : "—")}</em>
            </div>
          );
        })}
      </div>
    </div>
  );
}


function Style() {
  return (
    <style>{`
      .strategy-page {
        width: 100%;
        color: #000000;
      }

      .strategy-kicker {
        margin-bottom: 5px;
        color: #000000;
        font-size: 15px;
        font-weight: 750;
        letter-spacing: 0.7px;
      }

      .strategy-header {
        min-height: 60px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
      }

      .strategy-header h2 {
        margin: 0 0 4px;
        color: #000000;
        font-size: 28px;
        font-weight: 760;
      }

      .strategy-header p {
        margin: 0;
        color: #000000;
        font-size: 16px;
        line-height: 1.5;
      }

      .strategy-header-badges {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 6px;
      }

      .strategy-header-badges span {
        padding: 6px 9px;
        border: 1px solid #c2d1dd;
        border-radius: 4px;
        background: #f2f7fa;
        color: #000000;
        font-size: 15px;
        font-weight: 700;
      }

      .strategy-tabs {
        margin-bottom: 9px;
        padding: 5px;
        display: grid;
        grid-template-columns: repeat(3, minmax(180px, 1fr));
        gap: 6px;
        border: 1px solid #d8e1e8;
        border-radius: 6px;
        background: #ffffff;
      }

      .strategy-tab {
        min-height: 52px;
        padding: 7px 12px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        border: 1px solid transparent;
        border-radius: 4px;
        background: #f7f9fa;
        color: #000000;
        cursor: pointer;
      }

      .strategy-tab strong {
        margin-bottom: 2px;
        font-size: 17px;
      }

      .strategy-tab span {
        font-size: 15px;
      }

      .strategy-tab.active {
        border-color: #86a6bf;
        background: #e9f1f7;
        color: #000000;
        box-shadow: inset 4px 0 0 #5f83a3;
      }

      .strategy-control-panel {
        min-height: 67px;
        margin-bottom: 8px;
        padding: 8px 10px;
        display: flex;
        align-items: stretch;
        gap: 8px;
        border: 1px solid #d8e1e8;
        border-radius: 6px;
        background: #ffffff;
      }

      .control-field,
      .layer-control {
        min-width: 155px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .control-field > span,
      .layer-control > span {
        margin-bottom: 4px;
        color: #7d8c9a;
        font-size: 15px;
        font-weight: 650;
      }

      .control-field select {
        min-width: 170px;
        height: 34px;
        padding: 0 8px;
        border: 1px solid #cfd9e1;
        border-radius: 4px;
        background: #ffffff;
        color: #000000;
        font-size: 16px;
        font-weight: 650;
      }

      .control-field.fixed {
        min-width: 165px;
        padding: 0 9px;
        border: 1px solid #e1e6ea;
        border-radius: 4px;
        background: #f8fafb;
      }

      .control-field.fixed.wide {
        min-width: 270px;
      }

      .control-field.fixed strong {
        color: #000000;
        font-size: 16px;
      }

      .compare-mark {
        width: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #000000;
        font-size: 15px;
        font-weight: 800;
      }

      .layer-control {
        min-width: 225px;
        margin-left: auto;
      }

      .layer-control > div {
        display: flex;
        align-items: center;
        gap: 9px;
      }

      .layer-control input {
        width: 165px;
      }

      .layer-control strong {
        min-width: 38px;
        color: #000000;
        font-size: 18px;
      }

      .request-bar {
        min-height: 64px;
        margin-bottom: 8px;
        padding: 8px 10px;
        display: flex;
        align-items: stretch;
        gap: 8px;
        border: 1px solid #d8e1e8;
        border-radius: 6px;
        background: #ffffff;
      }

      .request-source {
        min-width: 0;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .request-source > span {
        color: #7d8c9a;
        font-size: 15px;
        font-weight: 650;
      }

      .request-source > strong {
        margin-top: 2px;
        overflow: hidden;
        color: #000000;
        font-size: 17px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .request-source > div {
        margin-top: 4px;
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
      }

      .request-source em {
        padding: 2px 6px;
        border: 1px solid #e1e6ea;
        border-radius: 3px;
        background: #fafbfc;
        color: #000000;
        font-size: 14px;
        font-style: normal;
      }

      .secondary-button,
      .run-button {
        min-width: 145px;
        padding: 0 13px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: 750;
        cursor: pointer;
      }

      .secondary-button {
        border: 1px solid #bdcbd6;
        background: #ffffff;
        color: #000000;
      }

      .run-button {
        min-width: 160px;
        border: 1px solid #6487a3;
        background: #6f91ad;
        color: #ffffff;
      }

      .secondary-button:disabled,
      .run-button:disabled {
        opacity: 0.5;
        cursor: default;
      }

      .reference-strip {
        min-height: 38px;
        margin-bottom: 8px;
        padding: 7px 10px;
        display: flex;
        align-items: center;
        gap: 10px;
        border: 1px solid #c7d8e4;
        border-radius: 5px;
        background: #f0f6fa;
      }

      .reference-strip strong {
        color: #000000;
        font-size: 15px;
        white-space: nowrap;
      }

      .reference-strip span {
        color: #000000;
        font-size: 15px;
        line-height: 1.4;
      }

      .strategy-error {
        margin-bottom: 8px;
        padding: 9px 11px;
        border: 1px solid #dfbcbc;
        border-radius: 5px;
        background: #fff5f5;
        color: #955858;
        font-size: 16px;
      }

      .empty-result {
        min-height: 142px;
        margin-bottom: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        border: 1px dashed #cbd7e0;
        border-radius: 7px;
        background: #fbfcfd;
        text-align: center;
      }

      .empty-result strong {
        color: #000000;
        font-size: 18px;
      }

      .empty-result span {
        color: #788997;
        font-size: 16px;
      }

      .comparison-result {
        margin-bottom: 8px;
        padding: 12px;
        border: 1px solid #d8e1e8;
        border-radius: 7px;
        background: #ffffff;
      }

      .comparison-title-row {
        min-height: 44px;
        margin-bottom: 8px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
      }

      .comparison-title-row > div {
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .comparison-title-row span {
        color: #8493a0;
        font-size: 15px;
        font-weight: 650;
      }

      .comparison-title-row strong {
        color: #000000;
        font-size: 18px;
      }

      .comparison-title-row small {
        padding: 4px 7px;
        border: 1px solid #e0e5e9;
        border-radius: 4px;
        background: #fafbfc;
        color: #000000;
        font-size: 15px;
      }

      .comparison-main-grid {
        display: grid;
        grid-template-columns: minmax(250px, 1fr) 170px minmax(250px, 1fr);
        gap: 10px;
        align-items: stretch;
      }

      .result-card {
        min-height: 128px;
        padding: 13px 15px;
        border: 1px solid #d8e0e7;
        border-radius: 6px;
        background: #fbfcfd;
      }

      .result-card.preferred {
        border-color: #83a7c1;
        background: #f0f6fa;
        box-shadow: inset 4px 0 0 #6289a7;
      }

      .result-card-top {
        min-height: 26px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
      }

      .side-badge {
        width: 25px;
        height: 25px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #b8c7d3;
        border-radius: 4px;
        background: #ffffff;
        color: #000000;
        font-size: 15px;
        font-weight: 800;
      }

      .result-card-top em {
        color: #4f7b63;
        font-size: 15px;
        font-style: normal;
        font-weight: 700;
      }

      .result-card-label {
        display: block;
        margin-top: 8px;
        color: #000000;
        font-size: 17px;
      }

      .result-card-value {
        margin-top: 7px;
        color: #000000;
        font-size: 31px;
        font-weight: 780;
        line-height: 1;
        letter-spacing: -0.6px;
      }

      .comparison-delta {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-left: 1px solid #edf0f2;
        border-right: 1px solid #edf0f2;
        text-align: center;
      }

      .comparison-delta span {
        color: #84919d;
        font-size: 15px;
        font-weight: 650;
      }

      .comparison-delta strong {
        margin: 5px 0 3px;
        color: #000000;
        font-size: 23px;
      }

      .comparison-delta em {
        font-size: 16px;
        font-style: normal;
        font-weight: 780;
      }

      .comparison-delta em.better {
        color: #4d7a60;
      }

      .comparison-delta em.worse {
        color: #a16565;
      }

      .secondary-grid {
        margin-top: 9px;
        display: grid;
        gap: 7px;
      }

      .secondary-grid.secondary-3 {
        grid-template-columns: repeat(3, 1fr);
      }

      .secondary-item {
        min-height: 59px;
        padding: 8px 10px;
        border: 1px solid #e2e7eb;
        border-radius: 5px;
        background: #fafbfc;
      }

      .secondary-item > span {
        display: block;
        margin-bottom: 6px;
        color: #7c8a97;
        font-size: 15px;
      }

      .secondary-item > div {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .secondary-item strong {
        color: #000000;
        font-size: 16px;
      }

      .secondary-item b {
        color: #9aa6b0;
        font-size: 15px;
      }

      .layer-delta-strip {
        min-height: 42px;
        margin-top: 8px;
        padding: 7px 10px;
        display: flex;
        align-items: center;
        gap: 10px;
        border: 1px solid #e0e6ea;
        border-radius: 5px;
        background: #fafbfc;
      }

      .layer-delta-strip > span {
        color: #000000;
        font-size: 15px;
      }

      .layer-delta-strip strong {
        color: #000000;
        font-size: 18px;
      }

      .layer-delta-strip em {
        color: #000000;
        font-size: 15px;
        font-style: normal;
        font-weight: 700;
      }

      .layer-delta-strip b {
        color: #98a5af;
      }

      .layer-delta-strip small {
        margin-left: auto;
        color: #000000;
        font-size: 15px;
        font-weight: 700;
      }

      .visual-stack {
        display: flex;
        flex-direction: column;
        gap: 9px;
      }

      .visual-card {
        padding: 11px 12px;
        border: 1px solid #d8e1e8;
        border-radius: 6px;
        background: #ffffff;
      }

      .visual-section-heading {
        margin-bottom: 9px;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
      }

      .visual-section-heading strong {
        color: #000000;
        font-size: 18px;
        font-weight: 760;
      }

      .visual-section-heading span {
        color: #7a8a97;
        font-size: 14px;
      }

      .single-layer-strip {
        min-height: 44px;
        padding: 8px 11px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid #d8e1e8;
        border-radius: 6px;
        background: #ffffff;
      }

      .single-layer-strip span {
        color: #000000;
        font-size: 15px;
      }

      .single-layer-strip strong {
        color: #000000;
        font-size: 19px;
      }

      .single-layer-strip em {
        color: #000000;
        font-size: 14px;
        font-style: normal;
      }

      .layer-chart {
        height: 138px;
        padding: 6px 5px 0;
        display: grid;
        grid-template-columns: repeat(58, minmax(7px, 1fr));
        gap: 2px;
        border-bottom: 1px solid #dce4ea;
        background:
          repeating-linear-gradient(
            to top,
            transparent 0,
            transparent 33px,
            #eef2f5 34px
          );
      }

      .layer-column {
        min-width: 0;
        display: grid;
        grid-template-rows: 1fr 18px;
        gap: 2px;
        border-radius: 3px 3px 0 0;
      }

      .layer-column.active {
        background: #eef4f8;
        box-shadow: inset 0 0 0 1px #a8bdcc;
      }

      .layer-bars {
        min-height: 0;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        gap: 1px;
      }

      .layer-bar {
        width: 42%;
        min-height: 3px;
        border-radius: 2px 2px 0 0;
      }

      .bar-a,
      .legend-a,
      .load-a {
        background: #7b93a7;
      }

      .bar-b,
      .legend-b,
      .load-b {
        background: #6f9b91;
      }

      .layer-column small {
        overflow: visible;
        color: #7a8996;
        font-size: 14px;
        line-height: 16px;
        text-align: center;
        white-space: nowrap;
      }

      .layer-column.active small {
        color: #000000;
        font-weight: 800;
      }

      .layer-chart-footer {
        margin-top: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }

      .chart-legend {
        display: flex;
        gap: 10px;
      }

      .chart-legend span {
        display: flex;
        align-items: center;
        gap: 5px;
        color: #000000;
        font-size: 14px;
        font-weight: 700;
      }

      .chart-legend i {
        width: 9px;
        height: 9px;
        border-radius: 2px;
      }

      .top-layer-deltas {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 5px;
      }

      .top-layer-deltas span {
        padding: 3px 6px;
        border: 1px solid #e0e6ea;
        border-radius: 3px;
        background: #fafbfc;
        color: #000000;
        font-size: 14px;
      }

      .top-layer-deltas b {
        color: #000000;
      }

      .sc-load-grid {
        display: grid;
        grid-template-columns: repeat(8, minmax(120px, 1fr));
        gap: 5px;
      }

      .sc-load-cell {
        padding: 6px 7px;
        border: 1px solid #e3e8ec;
        border-radius: 4px;
        background: #fafbfc;
      }

      .sc-load-cell > strong {
        display: block;
        margin-bottom: 4px;
        color: #000000;
        font-size: 14px;
      }

      .sc-load-line {
        display: grid;
        grid-template-columns: 13px 1fr 34px;
        align-items: center;
        gap: 4px;
      }

      .sc-load-line + .sc-load-line {
        margin-top: 3px;
      }

      .sc-load-line > span {
        color: #7b8995;
        font-size: 14px;
        font-weight: 800;
      }

      .sc-load-line > div {
        height: 6px;
        overflow: hidden;
        border-radius: 2px;
        background: #e8edf0;
      }

      .sc-load-line i {
        height: 100%;
        display: block;
        border-radius: 2px;
      }

      .sc-load-line em {
        color: #6b7b88;
        font-size: 14px;
        font-style: normal;
        text-align: right;
      }

      .visual-footnote {
        margin-top: 6px;
        color: #87939d;
        font-size: 14px;
        text-align: right;
      }

      .timeline-legend {
        margin: -1px 0 8px 42px;
        display: flex;
        gap: 6px;
      }

      .timeline-legend span {
        padding: 2px 7px;
        border-radius: 3px;
        color: #ffffff;
        font-size: 14px;
        font-weight: 750;
      }

      .matrix-gate,
      .task-gate {
        background: #708fa7;
      }

      .matrix-up,
      .task-up {
        background: #7b9e91;
      }

      .matrix-down,
      .task-down {
        background: #9b866e;
      }

      .matrix-switch {
        background: #a8b1b8;
      }

      .timeline-panel + .timeline-panel {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #e7ecef;
      }

      .timeline-panel-title {
        min-height: 28px;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .timeline-panel-title > span,
      .snapshot-title > span {
        width: 24px;
        height: 22px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 3px;
        background: #edf2f5;
        color: #000000;
        font-size: 14px;
        font-weight: 800;
      }

      .timeline-panel-title strong,
      .snapshot-title strong {
        color: #000000;
        font-size: 15px;
      }

      .timeline-panel-title em {
        margin-left: auto;
        color: #718290;
        font-size: 14px;
        font-style: normal;
        font-weight: 700;
      }

      .timeline-axis {
        height: 25px;
        display: grid;
        grid-template-columns: 34px 1fr;
        gap: 5px;
      }

      .axis-label {
        color: #87949e;
        font-size: 14px;
        text-align: center;
      }

      .timeline-axis > div {
        position: relative;
        border-bottom: 1px solid #ccd7df;
      }

      .timeline-axis i {
        position: absolute;
        bottom: 1px;
        transform: translateX(-50%);
        color: #7c8b98;
        font-size: 14px;
        font-style: normal;
      }

      .timeline-rows {
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .timeline-row {
        height: 18px;
        display: grid;
        grid-template-columns: 34px 1fr;
        gap: 5px;
      }

      .timeline-sc-label {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #71818e;
        font-size: 14px;
        font-weight: 700;
      }

      .timeline-track {
        position: relative;
        overflow: hidden;
        border-radius: 2px;
        background:
          repeating-linear-gradient(
            to right,
            #f6f8f9 0,
            #f6f8f9 9.8%,
            #edf1f3 10%
          );
      }

      .timeline-task {
        position: absolute;
        top: 2px;
        height: 14px;
        overflow: hidden;
        border-radius: 2px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.22);
      }

      .timeline-task.task-run {
        background: #7e949f;
      }

      .task-activation {
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        background: rgba(255, 255, 255, 0.46);
        border-right: 1px solid rgba(62, 77, 89, 0.25);
      }

      .timeline-cursor {
        position: absolute;
        top: 0;
        bottom: 0;
        z-index: 6;
        width: 2px;
        background: #000000;
        opacity: 0.72;
        pointer-events: none;
        transition: left 0.1s linear;
      }

      .collapsible-card {
        padding: 0;
        overflow: hidden;
      }

      .collapsible-header {
        width: 100%;
        min-height: 62px;
        padding: 10px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        border: 0;
        background: #ffffff;
        color: #000000;
        text-align: left;
        cursor: pointer;
      }

      .collapsible-header:hover {
        background: #f8fafb;
      }

      .collapsible-header > div {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 3px;
      }

      .collapsible-header strong {
        color: #000000;
        font-size: 18px;
        font-weight: 800;
      }

      .collapsible-header span {
        color: #758592;
        font-size: 15px;
        line-height: 1.4;
      }

      .collapsible-header b {
        flex: 0 0 auto;
        padding: 5px 8px;
        border: 1px solid #d6dfe5;
        border-radius: 4px;
        background: #f8fafb;
        color: #000000;
        font-size: 14px;
      }

      .collapsible-body {
        padding: 0 12px 12px;
        border-top: 1px solid #edf1f3;
      }

      .collapsible-body .timeline-legend {
        margin-top: 10px;
      }

      .cube-comparison-grid {
        margin-top: 10px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 10px;
      }

      .cube-comparison-side {
        min-width: 0;
        padding: 8px;
        border: 1px solid #dfe5e9;
        border-radius: 6px;
        background: #fbfcfd;
      }

      .cube-side-title {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .cube-side-title span {
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #cad5dc;
        border-radius: 4px;
        background: #ffffff;
        color: #000000;
        font-size: 14px;
        font-weight: 800;
      }

      .cube-side-title strong {
        color: #000000;
        font-size: 16px;
      }

      .cube-comparison-side .execution-cube-root {
        margin-top: 6px !important;
        padding-top: 0 !important;
        border-top: 0 !important;
      }

      .cube-comparison-side .execution-cube-header {
        display: none !important;
      }

      .cube-comparison-side .execution-canvas {
        height: 330px !important;
      }

      .cube-comparison-side .execution-cube-legend {
        color: #000000 !important;
        font-size: 13px !important;
      }

      .cube-comparison-side .execution-sc-label,
      .cube-comparison-side .execution-hover-title,
      .cube-comparison-side .execution-hover-row strong {
        color: #000000 !important;
      }

      .cube-comparison-side .execution-sc-label {
        font-size: 12px !important;
      }

      .snapshot-detail-heading {
        margin: 12px 0 7px;
        color: #000000;
        font-size: 16px;
        font-weight: 800;
      }

      @media (max-width: 1350px) {
        .cube-comparison-grid {
          grid-template-columns: 1fr;
        }
      }

      .playback-toolbar {
        margin-bottom: 9px;
        padding: 8px;
        display: flex;
        align-items: center;
        gap: 7px;
        border: 1px solid #dce3e8;
        border-radius: 5px;
        background: #f8fafb;
      }

      .playback-button {
        min-width: 82px;
        height: 34px;
        padding: 0 11px;
        border: 1px solid #bfcbd4;
        border-radius: 4px;
        background: #ffffff;
        color: #000000;
        font-size: 15px;
        font-weight: 750;
        cursor: pointer;
      }

      .playback-button.active {
        border-color: #6f9388;
        background: #edf6f2;
      }

      .playback-speed {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #000000;
        font-size: 15px;
        font-weight: 700;
      }

      .playback-speed select {
        height: 34px;
        padding: 0 8px;
        border: 1px solid #c8d2d9;
        border-radius: 4px;
        background: #ffffff;
        color: #000000;
        font-size: 15px;
        font-weight: 700;
      }

      .playback-now {
        margin-left: auto;
        display: flex;
        align-items: baseline;
        gap: 9px;
      }

      .playback-now span {
        color: #6f808e;
        font-size: 14px;
        font-weight: 700;
      }

      .playback-now strong {
        color: #000000;
        font-size: 16px;
      }

      .cycle-control-row {
        margin-bottom: 9px;
        display: grid;
        grid-template-columns: 45px 1fr 78px;
        align-items: center;
        gap: 8px;
      }

      .cycle-control-row span {
        color: #6d7f8d;
        font-size: 14px;
        font-weight: 700;
      }

      .cycle-control-row input {
        width: 100%;
      }

      .cycle-control-row strong {
        color: #000000;
        font-size: 15px;
        text-align: right;
      }

      .snapshot-columns {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 9px;
      }

      .snapshot-panel {
        min-width: 0;
        padding: 8px;
        border: 1px solid #e0e6ea;
        border-radius: 5px;
        background: #fafbfc;
      }

      .snapshot-title {
        margin-bottom: 7px;
        display: flex;
        align-items: center;
        gap: 7px;
      }

      .snapshot-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(100px, 1fr));
        gap: 5px;
      }

      .snapshot-cell {
        min-height: 57px;
        padding: 6px 7px;
        display: grid;
        grid-template-columns: 1fr auto;
        grid-template-rows: auto auto;
        gap: 1px 5px;
        border: 1px solid #e0e5e9;
        border-left-width: 4px;
        border-radius: 4px;
        background: #ffffff;
        transition: background 0.12s ease, border-color 0.12s ease, transform 0.12s ease;
      }

      .snapshot-cell.running {
        transform: translateY(-1px);
      }

      .snapshot-cell > span {
        color: #6f808e;
        font-size: 14px;
        font-weight: 750;
      }

      .snapshot-cell > strong {
        color: #000000;
        font-size: 14px;
        text-align: right;
      }

      .snapshot-cell > em {
        grid-column: 1 / -1;
        overflow: hidden;
        color: #84919b;
        font-size: 14px;
        font-style: normal;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .snapshot-cell.idle {
        border-left-color: #c7d0d6;
      }

      .snapshot-cell.waiting {
        border-left-color: #b29a74;
        background: #fbf8f2;
      }

      .snapshot-cell.running {
        border-left-color: #6f9388;
        background: #f3f8f6;
      }

      @media (max-width: 1250px) {
        .strategy-control-panel,
        .request-bar {
          flex-wrap: wrap;
        }

        .layer-control {
          margin-left: 0;
        }

        .comparison-main-grid {
          grid-template-columns: 1fr 135px 1fr;
        }

        .sc-load-grid {
          grid-template-columns: repeat(4, minmax(120px, 1fr));
        }
      }

      @media (max-width: 900px) {
        .strategy-header {
          flex-direction: column;
        }

        .strategy-header-badges {
          justify-content: flex-start;
        }

        .strategy-tabs,
        .comparison-main-grid,
        .secondary-grid.secondary-3,
        .snapshot-columns {
          grid-template-columns: 1fr;
        }

        .sc-load-grid {
          grid-template-columns: repeat(2, minmax(120px, 1fr));
        }

        .snapshot-grid {
          grid-template-columns: repeat(2, minmax(100px, 1fr));
        }

        .comparison-delta {
          min-height: 82px;
          border: none;
          border-top: 1px solid #edf0f2;
          border-bottom: 1px solid #edf0f2;
        }

        .layer-delta-strip {
          flex-wrap: wrap;
        }

        .layer-delta-strip small {
          margin-left: 0;
        }
      }
    `}</style>
  );
}


export default StrategyComparison;
