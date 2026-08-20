import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE =
  "http://127.0.0.1:8000";


const SCOPE_OPTIONS = [
  {
    id: "smoke",
    title: "快速检查 / Smoke Test",
    description: "严格取前 10 个完整 JSON Request；Prefill 与 Decode 使用同一批请求。",
    requestCount: 10,
  },
  {
    id: "full",
    title: "全量 Trace / Full Trace",
    description: "使用全部 2020 个真实 Request，作为正式全量实验口径。",
    requestCount: 2020,
  },
];


const ALGORITHM_ORDER = [
  "partition",
  "placement",
  "plane_pairing",
  "mapping",
  "prefill_scheduler",
  "decode_scheduler",
];


const FALLBACK_DEFAULTS = {
  partition: "anonymous_template_baseline",
  placement: "maxrects_bssf",
  plane_pairing: "trace_greedy_local_search",
  mapping: "trace_aware_mapping",
  prefill_scheduler: "exact_batch_scheduler",
  decode_scheduler: "formal_auto",
};


const FALLBACK_GROUPS = {
  partition: {
    label: "矩阵切分 / Matrix Partition",
    stage: "空间规划",
    options: [
      {
        id: "anonymous_template_baseline",
        label: "匿名切分模板 / Baseline",
        description: "当前第二步匿名几何切分模板。",
        implemented: true,
      },
    ],
  },
  placement: {
    label: "空间放置 / Placement",
    stage: "空间规划",
    options: [
      {
        id: "maxrects_bssf",
        label: "MaxRects-BSSF",
        description: "面积降序 + MaxRects-BSSF。",
        implemented: true,
      },
    ],
  },
  plane_pairing: {
    label: "Plane 配对 / Plane Pairing",
    stage: "逻辑映射",
    options: [
      {
        id: "trace_greedy_local_search",
        label: "Trace-aware Greedy + Local Search",
        description: "当前正式 Plane Pairing。",
        implemented: true,
      },
    ],
  },
  mapping: {
    label: "Sub-Cube 映射 / Mapping",
    stage: "逻辑映射",
    options: [
      {
        id: "trace_aware_mapping",
        label: "Trace-aware Mapping / 当前正式方法",
        description: "当前正式 Sub-Cube Mapping。",
        implemented: true,
      },
    ],
  },
  prefill_scheduler: {
    label: "Prefill 调度器 / Scheduler",
    stage: "推理调度",
    options: [
      {
        id: "exact_batch_scheduler",
        label: "Exact Batch Scheduler",
        description: "当前正式 Prefill Scheduler。",
        implemented: true,
      },
    ],
  },
  decode_scheduler: {
    label: "Decode 调度器 / Scheduler",
    stage: "推理调度",
    options: [
      {
        id: "formal_auto",
        label: "Formal Auto / 正式模式",
        description: "Smoke=Exact Continuous；Full=Fast Exact-validated。",
        implemented: true,
      },
    ],
  },
};


function ConfigRow({
  label,
  value,
  note,
}) {
  return (
    <div className="exp-config-row">
      <div>
        <span>
          {label}
        </span>

        {note && (
          <small>
            {note}
          </small>
        )}
      </div>

      <strong>
        {value}
      </strong>
    </div>
  );
}


function AlgorithmSelect({
  groupId,
  group,
  value,
  onChange,
  disabled,
}) {
  const options = group?.options ?? [];
  const selected = options.find(
    (item) => item.id === value
  );

  return (
    <div className="algorithm-select-row">
      <div className="algorithm-select-label">
        <strong>
          {group?.label ?? groupId}
        </strong>

        <span>
          {group?.stage ?? ""}
        </span>
      </div>

      <div className="algorithm-select-control">
        <select
          value={value}
          disabled={disabled}
          onChange={(event) =>
            onChange(
              groupId,
              event.target.value,
            )
          }
        >
          {options.map(
            (option) => (
              <option
                key={option.id}
                value={option.id}
                disabled={!option.implemented}
              >
                {option.label}
                {!option.implemented
                  ? "（待接入）"
                  : ""}
              </option>
            )
          )}
        </select>

        <small>
          {selected?.description ?? "--"}
        </small>
      </div>
    </div>
  );
}


function MetricCard({
  label,
  value,
  unit,
}) {
  return (
    <div className="experiment-metric-card">
      <span>
        {label}
      </span>

      <strong>
        {value ?? "--"}
      </strong>

      {unit && (
        <small>
          {unit}
        </small>
      )}
    </div>
  );
}


function formatNumber(
  value,
  decimals = 2,
) {
  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return "--";
  }

  if (Number.isInteger(numeric)) {
    return numeric.toLocaleString("zh-CN");
  }

  return numeric.toLocaleString(
    "zh-CN",
    {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }
  );
}


function Experiments({
  phaseSummary,
  hardware,
  mappingFile,
}) {
  const [
    scope,
    setScope,
  ] = useState("smoke");

  const [
    runPrefill,
    setRunPrefill,
  ] = useState(true);

  const [
    runDecode,
    setRunDecode,
  ] = useState(true);

  const [
    job,
    setJob,
  ] = useState(null);

  const [
    starting,
    setStarting,
  ] = useState(false);

  const [
    actionError,
    setActionError,
  ] = useState("");


  const [
    algorithmGroups,
    setAlgorithmGroups,
  ] = useState(FALLBACK_GROUPS);

  const [
    algorithms,
    setAlgorithms,
  ] = useState(FALLBACK_DEFAULTS);

  const [
    algorithmLoading,
    setAlgorithmLoading,
  ] = useState(true);


  const selectedScope = useMemo(
    () => (
      SCOPE_OPTIONS.find(
        (item) => item.id === scope
      ) ?? SCOPE_OPTIONS[0]
    ),
    [scope],
  );


  const fullRequestCount = Number(
    phaseSummary?.prefill?.batch_count ?? 2020
  );

  const decodeTokenCount = Number(
    phaseSummary?.decode?.token_count ?? 255710
  );

  const effectiveRequestCount =
    scope === "full"
      ? fullRequestCount
      : Math.min(
          selectedScope.requestCount,
          fullRequestCount,
        );


  const running =
    job?.state === "queued"
    || job?.state === "running";


  // =========================================================
  // 读取后端算法注册表
  // =========================================================

  useEffect(() => {
    const controller = new AbortController();

    async function loadAlgorithms() {
      try {
        setAlgorithmLoading(true);

        const response = await fetch(
          `${API_BASE}/api/experiments/algorithms`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        setAlgorithmGroups(
          data.groups ?? FALLBACK_GROUPS
        );

        setAlgorithms(
          data.defaults ?? FALLBACK_DEFAULTS
        );
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error(
            "Load experiment algorithms failed:",
            err
          );
          // 保留 fallback，页面仍然可以使用当前 Baseline。
        }
      } finally {
        if (!controller.signal.aborted) {
          setAlgorithmLoading(false);
        }
      }
    }

    loadAlgorithms();

    return () => controller.abort();
  }, []);


  function changeAlgorithm(
    groupId,
    algorithmId,
  ) {
    setAlgorithms(
      (previous) => ({
        ...previous,
        [groupId]: algorithmId,
      })
    );
  }


  // =========================================================
  // 页面打开时，如果后端已有活动实验，自动接回状态。
  // =========================================================

  useEffect(() => {
    const controller =
      new AbortController();

    async function reconnect() {
      try {
        const response = await fetch(
          `${API_BASE}/api/experiments/status`,
          {
            signal: controller.signal,
          }
        );

        if (!response.ok) {
          return;
        }

        const data =
          await response.json();

        if (
          data.busy
          && data.active_job_id
        ) {
          const jobResponse = await fetch(
            `${API_BASE}/api/experiments/${data.active_job_id}`,
            {
              signal: controller.signal,
            }
          );

          if (jobResponse.ok) {
            setJob(
              await jobResponse.json()
            );
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error(
            "Reconnect experiment failed:",
            err
          );
        }
      }
    }

    reconnect();

    return () => {
      controller.abort();
    };
  }, []);


  // =========================================================
  // Poll
  // =========================================================

  useEffect(() => {
    if (
      !job?.job_id
      || !running
    ) {
      return undefined;
    }

    const interval = window.setInterval(
      async () => {
        try {
          const response = await fetch(
            `${API_BASE}/api/experiments/${job.job_id}`
          );

          if (!response.ok) {
            return;
          }

          const data =
            await response.json();

          setJob(data);
        } catch (err) {
          console.error(
            "Poll experiment failed:",
            err
          );
        }
      },
      1000,
    );

    return () => {
      window.clearInterval(interval);
    };
  }, [
    job?.job_id,
    running,
  ]);


  async function startExperiment() {
    if (!runPrefill && !runDecode) {
      setActionError(
        "至少选择 Prefill 或 Decode 一个阶段。"
      );
      return;
    }

    try {
      setStarting(true);
      setActionError("");
      setJob(null);

      const response = await fetch(
        `${API_BASE}/api/experiments/run`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            scope,
            run_prefill: runPrefill,
            run_decode: runDecode,
            algorithms,
          }),
        }
      );

      if (!response.ok) {
        let message =
          `HTTP ${response.status}`;

        try {
          const body =
            await response.json();

          message =
            body.detail ?? message;
        } catch {
          // ignore
        }

        throw new Error(message);
      }

      setJob(
        await response.json()
      );

    } catch (err) {
      console.error(err);

      setActionError(
        `实验启动失败：${err.message}`
      );

    } finally {
      setStarting(false);
    }
  }


  async function cancelExperiment() {
    if (!job?.job_id) {
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE}/api/experiments/${job.job_id}/cancel`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status}`
        );
      }

      setJob(
        await response.json()
      );
    } catch (err) {
      setActionError(
        `取消失败：${err.message}`
      );
    }
  }


  const prefillResult =
    job?.result?.prefill ?? null;

  const decodeResult =
    job?.result?.decode ?? null;


  return (
    <div className="experiments-page">

      {/* ===================================================
          Header
      ==================================================== */}

      <div className="experiments-header">
        <div>
          <div className="experiments-kicker">
            EXPERIMENTS
          </div>

          <h2>
            实验配置与运行 / Experiments
          </h2>

          <p>
            按完整 Request 运行当前 Prefill / Decode 正式评估链，实验结果单独保存，不覆盖当前 Baseline。
          </p>
        </div>

        <div className="experiments-badge">
          Request-level Evaluation
        </div>
      </div>


      <div className="exp-rule-banner">
        <strong>
          统一请求语义：
        </strong>
        一个 JSON = 一个 Request。Smoke 模式中 Prefill 与 Decode 严格使用前 10 个相同 JSON；Decode Exact 会先执行该 Request 的 segment0 Prefill 生成正确 SC 激活状态，再连续执行 segment1+。
      </div>


      <div className="experiments-grid">

        {/* =================================================
            Scope
        ================================================== */}

        <section className="exp-card">
          <div className="exp-card-header">
            <div>
              <h3>
                ① 评估范围 / Scope
              </h3>

              <p>
                这里限制的是完整 JSON Request 数量，不是 Decode Token Batch Size。
              </p>
            </div>
          </div>

          <div className="scope-options">
            {SCOPE_OPTIONS.map(
              (option) => (
                <button
                  key={option.id}
                  disabled={running || starting}
                  className={
                    scope === option.id
                      ? "scope-option active"
                      : "scope-option"
                  }
                  onClick={() =>
                    setScope(option.id)
                  }
                >
                  <div>
                    <strong>
                      {option.title}
                    </strong>

                    <span>
                      {option.description}
                    </span>
                  </div>

                  <b>
                    {option.id === "full"
                      ? fullRequestCount
                      : option.requestCount}
                    {" "}
                    Requests
                  </b>
                </button>
              )
            )}
          </div>

          <div className="scope-summary">
            当前范围：
            <strong>
              {effectiveRequestCount.toLocaleString("zh-CN")}
            </strong>
            {" "}
            个完整 Request。
          </div>

          {scope === "smoke" && (
            <div className="scope-detail-note">
              <strong>Smoke：</strong>
              Prefill = 前 10 个 Request 的 Exact Prefill；Decode = 同一批 10 个 Request 的 Exact Continuous Request State。不会使用旧的“1000 个随机 Decode Token”口径。
            </div>
          )}

          {scope === "full" && (
            <div className="scope-detail-note important">
              <strong>Full：</strong>
              Prefill 使用 Exact 全量；Decode 使用当前正式 Fast Exact-validated 全量路径，并对前 100 个 Token 做 FAST == EXACT 校验。
            </div>
          )}
        </section>


        {/* =================================================
            Phase
        ================================================== */}

        <section className="exp-card">
          <div className="exp-card-header">
            <div>
              <h3>
                ② 推理阶段 / Phase
              </h3>

              <p>
                两阶段可以分别统计。Smoke Decode 即使单独勾选，也会内部先执行 Prefill 只用于生成正确初始状态。
              </p>
            </div>
          </div>

          <label className="phase-check">
            <input
              type="checkbox"
              disabled={running || starting}
              checked={runPrefill}
              onChange={(event) =>
                setRunPrefill(
                  event.target.checked
                )
              }
            />

            <div>
              <strong>
                Prefill / 预填充
              </strong>

              <span>
                Exact Batch Scheduler；统计 Batch latency、Cycles/Input Token、Layer 与 SC 负载。
              </span>
            </div>
          </label>


          <label className="phase-check">
            <input
              type="checkbox"
              disabled={running || starting}
              checked={runDecode}
              onChange={(event) =>
                setRunDecode(
                  event.target.checked
                )
              }
            />

            <div>
              <strong>
                Decode / 解码
              </strong>

              <span>
                Smoke 使用 Exact Continuous Request State；Full 使用正式 Fast Exact-validated 全量评估。
              </span>
            </div>
          </label>


          {!runPrefill && !runDecode && (
            <div className="phase-warning">
              至少需要选择一个评估阶段。
            </div>
          )}
        </section>
      </div>


      <div className="experiments-grid lower">

        {/* =================================================
            Algorithms
        ================================================== */}

        <section className="exp-card algorithm-card">
          <div className="exp-card-header">
            <div>
              <h3>
                ③ 算法配置 / Algorithm Configuration
              </h3>

              <p>
                每个阶段独立选择算法。当前源码中已实现的选项可以直接运行；规划中的消融算法显示为“待接入”，实现后无需重做页面。
              </p>
            </div>
          </div>

          {algorithmLoading && (
            <div className="algorithm-loading">
              正在读取算法注册表…
            </div>
          )}

          <div className="algorithm-select-list">
            {ALGORITHM_ORDER.map(
              (groupId) => (
                <AlgorithmSelect
                  key={groupId}
                  groupId={groupId}
                  group={algorithmGroups[groupId]}
                  value={algorithms[groupId] ?? ""}
                  onChange={changeAlgorithm}
                  disabled={running || starting}
                />
              )
            )}
          </div>

          <div className="algorithm-note">
            <strong>当前行为：</strong>
            选择值会随实验一起提交到后端、写入 config.json / result_summary.json，并在 Worker 中作为算法 ID 接收。当前只有 Baseline 分支可运行；后续新增算法时只需要补算法实现与注册项。
          </div>
        </section>


        {/* =================================================
            Context
        ================================================== */}

        <section className="exp-card">
          <div className="exp-card-header">
            <div>
              <h3>
                ④ 当前硬件与数据 / Context
              </h3>

              <p>
                后续不同算法必须固定硬件、Mapping 输入和 Trace 数据口径，才能公平比较。
              </p>
            </div>
          </div>

          <div className="exp-config-list">
            <ConfigRow
              label="Mapping"
              value={mappingFile ?? "--"}
            />

            <ConfigRow
              label="Sub-Cube"
              value={`${hardware?.num_subcubes ?? "--"}`}
            />

            <ConfigRow
              label="Plane Size"
              value={`${hardware?.H ?? "--"} × ${hardware?.W ?? "--"}`}
            />

            <ConfigRow
              label="Depth D"
              value={`${hardware?.D ?? "--"}`}
            />

            <ConfigRow
              label="正式 Request"
              value={`${fullRequestCount.toLocaleString("zh-CN")} Requests`}
            />

            <ConfigRow
              label="正式 Decode Token"
              value={`${decodeTokenCount.toLocaleString("zh-CN")} Tokens`}
            />
          </div>
        </section>
      </div>


      {/* ===================================================
          Request semantics
      ==================================================== */}

      <section className="exp-flow-card">
        <div className="exp-card-header">
          <div>
            <h3>
              实验执行语义 / Execution Semantics
            </h3>

            <p>
              Exact Decode 的 Request 状态按下面顺序连续继承；不同 JSON Request 之间重新开始。
            </p>
          </div>
        </div>

        <div className="request-flow">
          <div className="flow-node prefill">
            Request
            <br />
            <strong>
              Prefill
            </strong>
          </div>

          <span>→</span>

          <div className="flow-node state">
            Final SC State
          </div>

          <span>→</span>

          <div className="flow-node decode">
            Decode-1
          </div>

          <span>→</span>

          <div className="flow-node state">
            Final State
          </div>

          <span>→</span>

          <div className="flow-node decode">
            Decode-2
          </div>

          <span>→</span>

          <div className="flow-more">
            …
          </div>
        </div>

        {scope === "full" && runDecode && (
          <div className="fast-semantic-note">
            <strong>
              Full Decode 说明：
            </strong>
            当前正式 Fast 路径不会为 25 万级 Token 重复构造 Prefill event，但在当前 Mapping 中跨层 Weight-Cube 不同，同时 initial activation 与 switch 都是 1 cycle，已经用 Exact 连续状态验证两种方式的 Decode latency 完全一致。以后若修改这些硬件代价或 WC 复用规则，需要重新验证 Fast 路径。
          </div>
        )}
      </section>


      {/* ===================================================
          Action
      ==================================================== */}

      <div className="experiment-action-bar">
        <div>
          <strong>
            当前配置：
          </strong>
          {effectiveRequestCount.toLocaleString("zh-CN")}
          {" "}
          Requests ·
          {runPrefill ? " Prefill" : ""}
          {runPrefill && runDecode ? " +" : ""}
          {runDecode ? " Decode" : ""}
          <span className="action-algorithm-summary">
            · Pairing: {algorithms.plane_pairing ?? "--"}
            · Mapping: {algorithms.mapping ?? "--"}
          </span>
        </div>

        <div className="experiment-action-buttons">
          {running && (
            <button
              className="cancel-experiment-button"
              onClick={cancelExperiment}
            >
              取消实验 / Cancel
            </button>
          )}

          <button
            className="run-experiment-button"
            disabled={
              starting
              || running
              || (!runPrefill && !runDecode)
            }
            onClick={startExperiment}
          >
            {starting
              ? "正在启动…"
              : running
                ? "实验运行中…"
                : "▶ 运行实验 / Run Experiment"}
          </button>
        </div>
      </div>


      {actionError && (
        <div className="experiment-error">
          {actionError}
        </div>
      )}


      {/* ===================================================
          Running status
      ==================================================== */}

      {job && (
        <section className="experiment-status-card">
          <div className="experiment-status-header">
            <div>
              <div className="status-kicker">
                EXPERIMENT STATUS
              </div>

              <h3>
                实验状态 / Run Status
              </h3>
            </div>

            <div
              className={
                `experiment-state ${job.state ?? "unknown"}`
              }
            >
              {job.state === "queued" && "等待启动 / Queued"}
              {job.state === "running" && "运行中 / Running"}
              {job.state === "completed" && "已完成 / Completed"}
              {job.state === "failed" && "失败 / Failed"}
              {job.state === "cancelled" && "已取消 / Cancelled"}
              {!job.state && "--"}
            </div>
          </div>

          <div className="experiment-status-grid">
            <ConfigRow
              label="当前阶段 / Stage"
              value={job.stage ?? "--"}
            />

            <ConfigRow
              label="任务 ID / Job ID"
              value={job.job_id ?? "--"}
            />

            <ConfigRow
              label="范围 / Scope"
              value={
                job.scope === "full"
                  ? "Full · 2020 Requests"
                  : "Smoke · 10 Requests"
              }
            />

            <ConfigRow
              label="结果目录"
              value={
                job.job_id
                  ? `results/webui_experiments/${job.job_id}`
                  : "--"
              }
            />
          </div>

          {job.algorithms && (
            <div className="status-algorithm-strip">
              <strong>本次算法：</strong>
              <span>Partition: {job.algorithms.partition}</span>
              <span>Placement: {job.algorithms.placement}</span>
              <span>Pairing: {job.algorithms.plane_pairing}</span>
              <span>Mapping: {job.algorithms.mapping}</span>
              <span>Prefill: {job.algorithms.prefill_scheduler}</span>
              <span>Decode: {job.algorithms.decode_scheduler}</span>
            </div>
          )}

          <div className="experiment-status-message">
            {job.message ?? ""}
          </div>

          {job.error && (
            <div className="experiment-error status-error">
              {job.error}
            </div>
          )}

          {(job.log_tail ?? []).length > 0 && (
            <div className="experiment-log-panel">
              <div className="experiment-log-title">
                运行日志 / Recent Log
              </div>

              {(job.log_tail ?? []).map(
                (line, index) => (
                  <div
                    className="experiment-log-line"
                    key={`${index}-${line}`}
                  >
                    {line}
                  </div>
                )
              )}
            </div>
          )}
        </section>
      )}


      {/* ===================================================
          Result
      ==================================================== */}

      {job?.state === "completed" && (
        <section className="experiment-result-card">
          <div className="exp-card-header">
            <div>
              <h3>
                本次实验结果 / Result
              </h3>

              <p>
                结果保存在独立实验目录，不会覆盖 05 结果分析使用的当前正式 Baseline。
              </p>
            </div>
          </div>

          {prefillResult && (
            <div className="experiment-result-phase">
              <div className="result-phase-title">
                Prefill / 预填充
              </div>

              <div className="experiment-metric-grid">
                <MetricCard
                  label="Requests / Batches"
                  value={formatNumber(prefillResult.batch_count, 0)}
                />

                <MetricCard
                  label="Input Tokens"
                  value={formatNumber(prefillResult.total_input_tokens, 0)}
                />

                <MetricCard
                  label="Mean Latency"
                  value={formatNumber(prefillResult.latency_cycles?.mean)}
                  unit="cycles"
                />

                <MetricCard
                  label="P95 Latency"
                  value={formatNumber(prefillResult.latency_cycles?.p95)}
                  unit="cycles"
                />

                <MetricCard
                  label="Mean Cycles / Input Token"
                  value={formatNumber(prefillResult.cycles_per_input_token?.mean)}
                  unit="cycles/token"
                />

                <MetricCard
                  label="Global Weighted"
                  value={formatNumber(prefillResult.global_cycles_per_input_token)}
                  unit="cycles/token"
                />
              </div>
            </div>
          )}


          {decodeResult && (
            <div className="experiment-result-phase decode-result-phase">
              <div className="result-phase-title">
                Decode / 解码
                <span>
                  {decodeResult.mode}
                </span>
              </div>

              <div className="experiment-metric-grid">
                <MetricCard
                  label="Requests"
                  value={formatNumber(decodeResult.request_count, 0)}
                />

                <MetricCard
                  label="Decode Tokens"
                  value={formatNumber(decodeResult.token_count, 0)}
                />

                <MetricCard
                  label="Mean"
                  value={formatNumber(decodeResult.cycles_per_token?.mean)}
                  unit="cycles/token"
                />

                <MetricCard
                  label="P95"
                  value={formatNumber(decodeResult.cycles_per_token?.p95)}
                  unit="cycles/token"
                />

                <MetricCard
                  label="P99"
                  value={formatNumber(decodeResult.cycles_per_token?.p99)}
                  unit="cycles/token"
                />

                <MetricCard
                  label="Max"
                  value={formatNumber(decodeResult.cycles_per_token?.maximum)}
                  unit="cycles/token"
                />
              </div>

              {decodeResult.semantic_note && (
                <div className="result-semantic-note">
                  {decodeResult.semantic_note}
                </div>
              )}
            </div>
          )}
        </section>
      )}


      <Style />
    </div>
  );
}


function Style() {
  return (
    <style>{`
      .experiments-page { width: 100%; color: #364152; }
      .algorithm-card { min-height: 100%; }
      .algorithm-select-list { display: grid; gap: 7px; }
      .algorithm-select-row { min-height: 68px; padding: 8px 9px; display: grid; grid-template-columns: minmax(190px, .75fr) minmax(280px, 1.25fr); gap: 12px; align-items: center; border: 1px solid #e2e7eb; border-radius: 6px; background: #fafbfc; }
      .algorithm-select-label { display: flex; flex-direction: column; gap: 3px; }
      .algorithm-select-label strong { color: #425363; font-size: 15px; }
      .algorithm-select-label span { color: #526579; font-size: 15px; }
      .algorithm-select-control { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
      .algorithm-select-control select { width: 100%; height: 36px; padding: 0 9px; border: 1px solid #cfd9e1; border-radius: 5px; background: #fff; color: #3f5060; font-size: 15px; font-weight: 650; outline: none; }
      .algorithm-select-control select:focus { border-color: #7f9fba; box-shadow: 0 0 0 2px rgba(111,148,180,.12); }
      .algorithm-select-control select:disabled { background: #f4f6f8; color: #7f8993; }
      .algorithm-select-control small { overflow: hidden; color: #5f7083; font-size: 15px; line-height: 1.35; text-overflow: ellipsis; white-space: nowrap; }
      .algorithm-loading { margin-bottom: 7px; padding: 7px 9px; border-radius: 5px; background: #f4f7f9; color: #768491; font-size: 16px; }
      .algorithm-note { margin-top: 8px; padding: 8px 10px; border: 1px solid #e0e6eb; border-radius: 5px; background: #f7f9fb; color: #71808d; font-size: 16px; line-height: 1.45; }
      .algorithm-note strong { color: #4a5f72; }
      .action-algorithm-summary { margin-left: 6px; color: #526579; font-size: 15px; }
      .status-algorithm-strip { margin: 9px 0; padding: 9px 10px; display: flex; flex-wrap: wrap; gap: 6px 12px; border: 1px solid #e0e6eb; border-radius: 5px; background: #f8fafb; color: #526579; font-size: 15px; }
      .status-algorithm-strip strong { color: #465867; }
      .experiments-header { min-height: 58px; margin-bottom: 10px; display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }
      .experiments-kicker, .status-kicker { margin-bottom: 4px; color: #405a73; font-size: 16px; font-weight: 750; letter-spacing: 1px; }
      .experiments-header h2 { margin: 0 0 5px; color: #2f3b47; font-size: 25px; font-weight: 700; }
      .experiments-header p { margin: 0; color: #526579; font-size: 15px; line-height: 1.5; }
      .experiments-badge { padding: 7px 11px; border: 1px solid #cbd7e2; border-radius: 5px; background: #fff; color: #54718d; font-size: 16px; font-weight: 700; white-space: nowrap; }
      .exp-rule-banner { margin-bottom: 10px; padding: 10px 12px; border: 1px solid #d4e0ea; border-radius: 6px; background: #f5f9fc; color: #5f7283; font-size: 15px; line-height: 1.55; }
      .exp-rule-banner strong { color: #3e566b; }
      .experiments-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
      .experiments-grid.lower { margin-top: 10px; }
      .exp-card, .exp-flow-card, .experiment-status-card, .experiment-result-card { padding: 13px; border: 1px solid #dfe5ea; border-radius: 7px; background: #fff; }
      .exp-card-header { margin-bottom: 10px; }
      .exp-card-header h3, .experiment-status-header h3 { margin: 0 0 4px; color: #42515f; font-size: 18px; }
      .exp-card-header p { margin: 0; color: #5f7083; font-size: 16px; line-height: 1.45; }
      .scope-options { display: grid; gap: 8px; }
      .scope-option { min-height: 72px; padding: 10px 11px; display: flex; align-items: center; justify-content: space-between; gap: 14px; text-align: left; border: 1px solid #dfe5ea; border-radius: 6px; background: #fafbfc; color: #52606d; cursor: pointer; }
      .scope-option:hover:not(:disabled) { background: #f5f8fa; }
      .scope-option:disabled { cursor: default; opacity: .72; }
      .scope-option.active { border-color: #3b82f6; background: #dbeafe; color: #123f70; box-shadow: inset 5px 0 0 #4F7195; }
      .scope-option div { display: flex; flex-direction: column; gap: 3px; }
      .scope-option strong { color: #3d4f5f; font-size: 16px; }
      .scope-option span { color: #5f7083; font-size: 16px; line-height: 1.35; }
      .scope-option b { color: #496d8a; font-size: 15px; white-space: nowrap; }
      .scope-summary { margin-top: 8px; padding: 8px 10px; border-radius: 5px; background: #f5f7f9; color: #6e7b87; font-size: 15px; }
      .scope-summary strong { color: #365b79; font-size: 17px; }
      .scope-detail-note { margin-top: 8px; padding: 8px 10px; border: 1px solid #dce4ea; border-radius: 5px; background: #fbfcfd; color: #5f7083; font-size: 16px; line-height: 1.5; }
      .scope-detail-note.important { border-color: #d8dfca; background: #fafcf6; color: #68735d; }
      .phase-check { min-height: 74px; margin-bottom: 8px; padding: 10px 11px; display: grid; grid-template-columns: 22px 1fr; gap: 10px; align-items: center; border: 1px solid #dfe5ea; border-radius: 6px; background: #fafbfc; cursor: pointer; }
      .phase-check input { width: 20px; height: 20px; accent-color: #4F7195; }
      .phase-check div { display: flex; flex-direction: column; gap: 3px; }
      .phase-check strong { color: #41515f; font-size: 16px; }
      .phase-check span { color: #5f7083; font-size: 16px; line-height: 1.35; }
      .phase-warning { padding: 8px 10px; border: 1px solid #e2b9b9; border-radius: 5px; background: #fff5f5; color: #9a5757; font-size: 16px; }
      .exp-config-list { border: 1px solid #e2e7eb; border-radius: 6px; overflow: hidden; }
      .exp-config-row { min-height: 48px; padding: 7px 10px; display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid #edf0f2; background: #fbfcfd; }
      .exp-config-row:last-child { border-bottom: none; }
      .exp-config-row > div { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
      .exp-config-row span { color: #526579; font-size: 16px; }
      .exp-config-row small { color: #64748b; font-size: 15px; }
      .exp-config-row strong { max-width: 56%; color: #3e5060; font-size: 15px; text-align: right; overflow-wrap: anywhere; }
      .exp-flow-card { margin-top: 10px; }
      .request-flow { min-height: 84px; padding: 10px; display: flex; align-items: center; justify-content: center; gap: 10px; border: 1px solid #e2e7eb; border-radius: 6px; background: #fafbfc; color: #5f7083; font-size: 18px; }
      .flow-node, .flow-more { min-width: 94px; min-height: 52px; padding: 7px 9px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid #d8e0e6; border-radius: 5px; background: #fff; color: #52616e; font-size: 16px; text-align: center; }
      .flow-node strong { font-size: 16px; }
      .flow-node.prefill { border-color: #9db4ca; background: #f1f6fa; }
      .flow-node.decode { border-color: #9fbaa9; background: #f1f7f3; }
      .flow-node.state { min-width: 112px; color: #687887; }
      .flow-more { min-width: 48px; font-size: 24px; }
      .fast-semantic-note, .result-semantic-note { margin-top: 9px; padding: 9px 11px; border: 1px solid #e2dcc5; border-radius: 5px; background: #fffdf6; color: #756b4e; font-size: 16px; line-height: 1.55; }
      .experiment-action-bar { min-height: 62px; margin-top: 10px; padding: 9px 11px; display: flex; align-items: center; justify-content: space-between; gap: 15px; border: 1px solid #dfe4e9; border-radius: 6px; background: #f7f9fa; color: #74818d; font-size: 15px; }
      .experiment-action-bar strong { color: #4b5c6b; }
      .experiment-action-buttons { display: flex; gap: 8px; }
      .run-experiment-button, .cancel-experiment-button { min-width: 200px; height: 40px; border-radius: 5px; font-size: 15px; font-weight: 700; cursor: pointer; }
      .run-experiment-button { border: 1px solid #6689a7; background: #7596b4; color: #fff; }
      .run-experiment-button:hover:not(:disabled) { background: #6789a8; }
      .run-experiment-button:disabled { border-color: #cbd3da; background: #e9edf0; color: #89939c; cursor: default; }
      .cancel-experiment-button { min-width: 150px; border: 1px solid #d2a8a8; background: #fff; color: #965e5e; }
      .experiment-error { margin-top: 9px; padding: 9px 11px; border: 1px solid #e1bcbc; border-radius: 5px; background: #fff5f5; color: #995858; font-size: 16px; }
      .experiment-status-card, .experiment-result-card { margin-top: 10px; }
      .experiment-status-header { margin-bottom: 9px; display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }
      .experiment-state { padding: 6px 10px; border-radius: 5px; font-size: 16px; font-weight: 750; white-space: nowrap; }
      .experiment-state.queued { border: 1px solid #d6dce2; background: #f5f7f9; color: #73808c; }
      .experiment-state.running { border: 1px solid #9eb8cf; background: #edf5fb; color: #456d8e; }
      .experiment-state.completed { border: 1px solid #a7c2ae; background: #f0f7f2; color: #52745b; }
      .experiment-state.failed { border: 1px solid #deb1b1; background: #fff3f3; color: #9a5555; }
      .experiment-state.cancelled { border: 1px solid #d6c6a8; background: #fffaf0; color: #8a7350; }
      .experiment-status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
      .experiment-status-grid .exp-config-row { border: 1px solid #e4e8eb; border-radius: 5px; }
      .experiment-status-message { margin-top: 8px; padding: 8px 10px; border-radius: 5px; background: #f6f8fa; color: #6c7985; font-size: 15px; line-height: 1.45; }
      .experiment-log-panel { max-height: 210px; margin-top: 9px; padding: 9px 10px; overflow: auto; border: 1px solid #dfe4e8; border-radius: 5px; background: #f7f8fa; }
      .experiment-log-title { margin-bottom: 6px; color: #60707e; font-size: 16px; font-weight: 750; }
      .experiment-log-line { padding: 2px 0; color: #687783; font-family: ui-monospace, Consolas, monospace; font-size: 16px; line-height: 1.4; overflow-wrap: anywhere; }
      .experiment-result-phase { padding: 10px; border: 1px solid #dce4ea; border-radius: 6px; background: #fafcfd; }
      .decode-result-phase { margin-top: 9px; border-color: #dae5dc; background: #fbfdfb; }
      .result-phase-title { margin-bottom: 9px; display: flex; align-items: center; gap: 9px; color: #40515f; font-size: 17px; font-weight: 750; }
      .result-phase-title span { padding: 3px 6px; border-radius: 4px; background: #eef2f5; color: #74818d; font-size: 15px; font-weight: 650; }
      .experiment-metric-grid { display: grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); gap: 7px; }
      .experiment-metric-card { min-height: 78px; padding: 9px 10px; border: 1px solid #e1e6ea; border-radius: 5px; background: #fff; }
      .experiment-metric-card span { display: block; margin-bottom: 6px; color: #5f7083; font-size: 16px; }
      .experiment-metric-card strong { color: #3c5060; font-size: 20px; font-weight: 750; }
      .experiment-metric-card small { display: block; margin-top: 2px; color: #64748b; font-size: 15px; }
      @media (max-width: 1250px) { .experiment-metric-grid { grid-template-columns: repeat(3, 1fr); } }
      @media (max-width: 1150px) { .experiments-grid { grid-template-columns: 1fr; } .request-flow { overflow-x: auto; justify-content: flex-start; } .experiment-status-grid { grid-template-columns: 1fr; } }
    `}</style>
  );
}


export default Experiments;
