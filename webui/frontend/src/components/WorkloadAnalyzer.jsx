import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE =
  "http://127.0.0.1:8000";


const TOKEN_OPTIONS = [
  10,
  100,
  1000,
];


// ============================================================
// WorkloadAnalyzer
// ============================================================


function WorkloadAnalyzer() {

  // =========================================================
  // Categories
  // =========================================================

  const [
    categories,
    setCategories,
  ] = useState([]);


  const [
    selectedCategory,
    setSelectedCategory,
  ] = useState("ALL");


  // =========================================================
  // Token Count
  // =========================================================

  const [
    tokenCount,
    setTokenCount,
  ] = useState(100);


  // =========================================================
  // Result
  // =========================================================

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


  // =========================================================
  // 加载 Trace Categories
  // =========================================================

  useEffect(() => {

    const controller =
      new AbortController();


    async function loadCategories() {

      try {

        const response =
          await fetch(
            `${API_BASE}/api/workload/categories`,
            {
              signal:
                controller.signal,
            }
          );


        if (!response.ok) {

          throw new Error(
            `HTTP ${response.status}`
          );
        }


        const data =
          await response.json();


        setCategories(
          data.categories ?? []
        );


      } catch (err) {

        if (
          err.name ===
          "AbortError"
        ) {
          return;
        }


        console.error(
          "Load workload categories failed:",
          err
        );
      }
    }


    loadCategories();


    return () => {

      controller.abort();
    };

  }, []);


  // =========================================================
  // Run Workload
  // =========================================================

  async function runWorkload() {

    try {

      setLoading(
        true
      );


      setError("");


      setResult(
        null
      );


      const response =
        await fetch(
          `${API_BASE}/api/workload/evaluate`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                {
                  token_count:
                    tokenCount,

                  category:
                    selectedCategory ===
                    "ALL"
                      ? null
                      : selectedCategory,

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


      setResult(
        data
      );


    } catch (err) {

      console.error(
        err
      );


      setError(
        "Workload 评估失败："
        + err.message
      );


    } finally {

      setLoading(
        false
      );
    }
  }


  // =========================================================
  // Histogram 最大值
  // =========================================================

  const histogramMax =
    useMemo(
      () => {

        const histogram =
          result?.histogram ?? [];


        if (
          histogram.length === 0
        ) {
          return 1;
        }


        return Math.max(
          ...histogram.map(
            (item) =>
              item.count
          ),
          1
        );

      },
      [
        result
      ]
    );


  // =========================================================
  // Layer 最大平均周期
  // =========================================================

  const maxLayerMean =
    useMemo(
      () => {

        const layers =
          result?.layers ?? [];


        if (
          layers.length === 0
        ) {
          return 1;
        }


        return Math.max(
          ...layers.map(
            (layer) =>
              Number(
                layer.mean_cycles
              )
          ),
          1
        );

      },
      [
        result
      ]
    );


  // =========================================================
  // SC 最大 Critical
  // =========================================================

  const maxCriticalCount =
    useMemo(
      () => {

        const subcubes =
          result?.subcubes ?? [];


        if (
          subcubes.length === 0
        ) {
          return 1;
        }


        return Math.max(
          ...subcubes.map(
            (sc) =>
              sc.critical_layer_count
          ),
          1
        );

      },
      [
        result
      ]
    );


  // =========================================================
  // Render
  // =========================================================

  return (
    <div className="workload-analyzer">

      {/* =====================================================
          Header
      ====================================================== */}

      <div className="workload-header">

        <div>

          <div className="workload-small-title">
            MULTI-TOKEN EVALUATION
          </div>


          <h2>
            Workload Results
          </h2>


          <p>
            使用真实 Chinese-SimpleQA
            Router Trace 评估多个 Token
            的完整 58 层推理延迟。
          </p>

        </div>

      </div>


      {/* =====================================================
          Controls
      ====================================================== */}

      <div className="workload-control-panel">

        {/* ===================================================
            Category
        ==================================================== */}

        <div className="workload-control-group">

          <label>
            Trace Category
          </label>


          <select
            value={
              selectedCategory
            }

            onChange={
              (event) =>
                setSelectedCategory(
                  event.target.value
                )
            }
          >

            <option value="ALL">
              全部类别
            </option>


            {categories.map(
              (category) => (

                <option
                  key={
                    category
                  }

                  value={
                    category
                  }
                >
                  {category}
                </option>

              )
            )}

          </select>

        </div>


        {/* ===================================================
            Token Count
        ==================================================== */}

        <div className="workload-control-group">

          <label>
            Token Count
          </label>


          <div className="token-count-buttons">

            {TOKEN_OPTIONS.map(
              (count) => (

                <button
                  key={
                    count
                  }

                  className={
                    tokenCount ===
                    count
                      ? "active"
                      : ""
                  }

                  onClick={
                    () =>
                      setTokenCount(
                        count
                      )
                  }
                >
                  {count}
                </button>

              )
            )}

          </div>

        </div>


        {/* ===================================================
            Run
        ==================================================== */}

        <button
          className="run-workload-button"

          disabled={
            loading
          }

          onClick={
            runWorkload
          }
        >

          {
            loading
              ? `Evaluating ${tokenCount} Tokens...`
              : "▶ Run Workload"
          }

        </button>

      </div>


      {/* =====================================================
          Loading
      ====================================================== */}

      {loading && (

        <div className="workload-loading">

          正在评估
          {" "}
          <strong>
            {tokenCount}
          </strong>
          {" "}
          个真实 Token。

          Token 数量越多，
          所需时间越长。

        </div>

      )}


      {/* =====================================================
          Error
      ====================================================== */}

      {error && (

        <div className="workload-error">
          {error}
        </div>

      )}


      {/* =====================================================
          Result
      ====================================================== */}

      {result && (

        <>

          {/* =================================================
              Run Information
          ================================================== */}

          <div className="workload-run-info">

            <span>
              Evaluated
              {" "}
              <strong>
                {
                  result
                    .evaluated_token_count
                }
              </strong>
              {" "}
              Tokens
            </span>


            <span>
              Category
              {" "}
              <strong>
                {
                  result.category
                }
              </strong>
            </span>


            <span>
              Runtime
              {" "}
              <strong>
                {
                  Number(
                    result.elapsed_seconds
                  ).toFixed(2)
                }
                s
              </strong>
            </span>

          </div>


          {/* =================================================
              Latency Cards
          ================================================== */}

          <div className="latency-summary-grid">

            <LatencyCard
              label="Mean"
              value={
                result.latency?.mean
              }
            />


            <LatencyCard
              label="Min"
              value={
                result.latency?.min
              }
            />


            <LatencyCard
              label="P50"
              value={
                result.latency?.p50
              }
            />


            <LatencyCard
              label="P95"
              value={
                result.latency?.p95
              }
            />


            <LatencyCard
              label="P99"
              value={
                result.latency?.p99
              }
            />


            <LatencyCard
              label="Max"
              value={
                result.latency?.max
              }
            />

          </div>


          {/* =================================================
              Histogram
          ================================================== */}

          <ResultSection
            title="Token Latency Distribution"
            subtitle="每个柱子表示一个 10-cycle 延迟区间。"
          >

            <div className="latency-histogram">

              {
                (
                  result.histogram ??
                  []
                ).map(
                  (
                    bin,
                    index
                  ) => {

                    const height =
                      (
                        bin.count
                        /
                        histogramMax
                      )
                      * 100;


                    return (
                      <div
                        className="histogram-column"
                        key={
                          `${bin.start}-${bin.end}`
                        }
                      >

                        <div className="histogram-bar-area">

                          <div
                            className="histogram-bar"

                            style={{
                              height:
                                `${height}%`,
                            }}

                            title={
                              `${bin.start}-${bin.end} cycles: `
                              + `${bin.count} tokens`
                            }
                          >

                            {bin.count > 0 && (

                              <span>
                                {
                                  bin.count
                                }
                              </span>

                            )}

                          </div>

                        </div>


                        <div className="histogram-label">

                          {
                            index % 2 === 0
                              ? bin.start
                              : ""
                          }

                        </div>

                      </div>
                    );
                  }
                )
              }

            </div>

          </ResultSection>


          {/* =================================================
              58 Layer Mean Latency
          ================================================== */}

          <ResultSection
            title="58-Layer Mean Latency"
            subtitle="每个格子表示该 Layer 在当前 Workload 中的平均周期。"
          >

            <div className="workload-layer-grid">

              {
                (
                  result.layers ??
                  []
                ).map(
                  (layer) => {

                    const ratio =
                      Number(
                        layer.mean_cycles
                      )
                      /
                      maxLayerMean;


                    return (
                      <div
                        className="workload-layer-card"

                        key={
                          layer.layer_id
                        }

                        title={
                          `Layer ${layer.layer_id}`
                          + ` | Mean ${Number(layer.mean_cycles).toFixed(3)}`
                          + ` | Max ${layer.max_cycles}`
                        }
                      >

                        <div className="layer-card-header">

                          <span>
                            L{
                              layer.layer_id
                            }
                          </span>


                          <strong>
                            {
                              Number(
                                layer.mean_cycles
                              ).toFixed(2)
                            }
                          </strong>

                        </div>


                        <div className="layer-card-bar">

                          <div
                            style={{
                              width:
                                `${
                                  ratio
                                  * 100
                                }%`,
                            }}
                          />

                        </div>


                        <small>
                          Max
                          {" "}
                          {
                            layer.max_cycles
                          }
                        </small>

                      </div>
                    );
                  }
                )
              }

            </div>

          </ResultSection>


          {/* =================================================
              Slowest Layers
          ================================================== */}

          <ResultSection
            title="Slowest Layers"
            subtitle="按平均 Layer latency 从高到低排序。"
          >

            <div className="slowest-workload-layers">

              {
                (
                  result.slowest_layers ??
                  []
                ).map(
                  (
                    layer,
                    index
                  ) => (

                    <div
                      className="slow-workload-layer"

                      key={
                        layer.layer_id
                      }
                    >

                      <span className="slow-workload-rank">
                        #{index + 1}
                      </span>


                      <strong>
                        Layer
                        {" "}
                        {
                          layer.layer_id
                        }
                      </strong>


                      <span>
                        Mean
                        {" "}
                        {
                          Number(
                            layer.mean_cycles
                          ).toFixed(3)
                        }
                      </span>


                      <small>
                        Max
                        {" "}
                        {
                          layer.max_cycles
                        }
                      </small>

                    </div>

                  )
                )
              }

            </div>

          </ResultSection>


          {/* =================================================
              16 SC Critical Distribution
          ================================================== */}

          <ResultSection
            title="Critical Sub-Cube Distribution"
            subtitle="某个 Sub-Cube 成为当前 Layer 最晚完成资源的次数。"
          >

            <div className="workload-sc-grid">

              {
                (
                  result.subcubes ??
                  []
                ).map(
                  (sc) => {

                    const ratio =
                      sc.critical_layer_count
                      /
                      maxCriticalCount;


                    return (
                      <div
                        className="workload-sc-card"

                        key={
                          sc.subcube_id
                        }
                      >

                        <div className="workload-sc-header">

                          <strong>
                            SC-{
                              sc.subcube_id
                            }
                          </strong>


                          <span>
                            {
                              (
                                sc
                                  .critical_layer_rate
                                * 100
                              ).toFixed(2)
                            }
                            %
                          </span>

                        </div>


                        <div className="workload-sc-bar">

                          <div
                            style={{
                              width:
                                `${
                                  ratio
                                  * 100
                                }%`,
                            }}
                          />

                        </div>


                        <div className="workload-sc-details">

                          <span>
                            Critical
                            {" "}
                            {
                              sc
                                .critical_layer_count
                            }
                          </span>


                          <span>
                            Switch
                            {" "}
                            {
                              sc
                                .switch_count
                            }
                          </span>


                          <span>
                            Tasks
                            {" "}
                            {
                              sc
                                .task_count
                            }
                          </span>

                        </div>

                      </div>
                    );
                  }
                )
              }

            </div>

          </ResultSection>


          {/* =================================================
              Slowest Tokens
          ================================================== */}

          <ResultSection
            title="Slowest Tokens"
            subtitle="当前评估样本中 latency 最高的 Token。"
          >

            <div className="slow-token-table">

              <div className="slow-token-header">

                <span>
                  Rank
                </span>

                <span>
                  Latency
                </span>

                <span>
                  Category
                </span>

                <span>
                  File
                </span>

                <span>
                  Segment
                </span>

                <span>
                  Token
                </span>

              </div>


              {
                (
                  result.slowest_tokens ??
                  []
                ).map(
                  (
                    token,
                    index
                  ) => (

                    <div
                      className="slow-token-row"

                      key={
                        `${token.file_name}-${token.segment_index}-${token.token_index}`
                      }
                    >

                      <span>
                        #{index + 1}
                      </span>


                      <strong>
                        {
                          token.latency
                        }
                        {" "}
                        cycles
                      </strong>


                      <span>
                        {
                          token.category
                        }
                      </span>


                      <span
                        className="file-name-cell"

                        title={
                          token.file_name
                        }
                      >
                        {
                          token.file_name
                        }
                      </span>


                      <span>
                        {
                          token.segment_index
                        }
                      </span>


                      <span>
                        {
                          token.token_index
                        }
                      </span>

                    </div>

                  )
                )
              }

            </div>

          </ResultSection>

        </>

      )}


      {/* =====================================================
          Empty
      ====================================================== */}

      {!result &&
       !loading &&
       !error && (

        <div className="workload-empty">

          <div className="empty-icon">
            ∿
          </div>


          <strong>
            尚未运行 Workload
          </strong>


          <p>
            选择 Trace 类别和 Token 数量，
            然后点击 Run Workload。
          </p>

        </div>

      )}


      <Style />

    </div>
  );
}


// ============================================================
// Latency Card
// ============================================================


function LatencyCard({
  label,
  value,
}) {

  const numeric =
    Number(
      value ?? 0
    );


  return (
    <div className="latency-stat-card">

      <span>
        {label}
      </span>


      <strong>
        {
          Number.isInteger(
            numeric
          )
            ? numeric
            : numeric.toFixed(2)
        }
      </strong>


      <small>
        cycles
      </small>

    </div>
  );
}


// ============================================================
// Result Section
// ============================================================


function ResultSection({
  title,
  subtitle,
  children,
}) {

  return (
    <section className="workload-result-section">

      <div className="workload-section-header">

        <div>

          <h3>
            {title}
          </h3>


          {subtitle && (

            <p>
              {subtitle}
            </p>

          )}

        </div>

      </div>


      {children}

    </section>
  );
}


// ============================================================
// CSS
// ============================================================


function Style() {

  return (
    <style>
      {`

        .workload-analyzer {
          width: 100%;
        }


        /* ================================================
           HEADER
        ================================================ */


        .workload-header {
          margin-bottom: 12px;
        }


        .workload-small-title {
          margin-bottom: 6px;

          color: #9aa3ad;

          font-size: 12px;

          font-weight: 700;

          letter-spacing: 1px;
        }


        .workload-header h2 {
          margin: 0 0 6px;

          color: #34404d;

          font-size: 22px;
        }


        .workload-header p {
          margin: 0;

          color: #909ba6;

          font-size: 12px;
        }


        /* ================================================
           CONTROL PANEL
        ================================================ */


        .workload-control-panel {
          padding: 10px;

          display: flex;

          align-items: flex-end;

          gap: 14px;

          border: 1px solid #dfe4e9;

          border-radius: 6px;

          background: #ffffff;
        }


        .workload-control-group {
          display: flex;

          flex-direction: column;

          gap: 6px;
        }


        .workload-control-group label {
          color: #87929d;

          font-size: 12px;

          font-weight: 650;
        }


        .workload-control-group select {
          min-width: 180px;

          height: 32px;

          padding: 0 8px;

          border: 1px solid #d7dee5;

          border-radius: 4px;

          background: #ffffff;

          color: #485562;

          font-size: 12px;

          outline: none;
        }


        .token-count-buttons {
          display: flex;

          gap: 5px;
        }


        .token-count-buttons button {
          width: 55px;

          height: 32px;

          border: 1px solid #d7dee5;

          border-radius: 4px;

          background: #ffffff;

          color: #66727e;

          font-size: 12px;

          cursor: pointer;
        }


        .token-count-buttons button.active {
          border-color: #7596b4;

          background: #eaf2f8;

          color: #466985;

          font-weight: 700;
        }


        .run-workload-button {
          min-width: 135px;

          height: 32px;

          margin-left: auto;

          border: 1px solid #698ba9;

          border-radius: 4px;

          background: #7596b4;

          color: #ffffff;

          font-size: 12px;

          font-weight: 650;

          cursor: pointer;
        }


        .run-workload-button:disabled {
          opacity: 0.45;

          cursor: default;
        }


        /* ================================================
           STATE
        ================================================ */


        .workload-loading,
        .workload-error {
          margin-top: 10px;

          padding: 10px 12px;

          border-radius: 5px;

          font-size: 12px;
        }


        .workload-loading {
          border: 1px solid #dae2e8;

          background: #f7fafc;

          color: #75818d;
        }


        .workload-error {
          border: 1px solid #e1bcbc;

          background: #fff5f5;

          color: #995858;
        }


        /* ================================================
           RUN INFO
        ================================================ */


        .workload-run-info {
          min-height: 35px;

          margin-top: 10px;

          padding: 0 10px;

          display: flex;

          align-items: center;

          gap: 25px;

          border: 1px solid #e1e5e9;

          border-radius: 5px;

          background: #fafbfc;

          color: #87919b;

          font-size: 12px;
        }


        .workload-run-info strong {
          color: #485662;

          font-size: 12px;
        }


        /* ================================================
           LATENCY SUMMARY
        ================================================ */


        .latency-summary-grid {
          margin-top: 10px;

          display: grid;

          grid-template-columns:
            repeat(6, 1fr);

          gap: 7px;
        }


        .latency-stat-card {
          padding: 10px;

          border: 1px solid #dfe4e8;

          border-radius: 5px;

          background: #ffffff;
        }


        .latency-stat-card > span {
          display: block;

          margin-bottom: 5px;

          color: #919ba5;

          font-size: 12px;
        }


        .latency-stat-card strong {
          color: #3f5060;

          font-size: 17px;
        }


        .latency-stat-card small {
          margin-left: 4px;

          color: #9ba4ad;

          font-size: 12px;
        }


        /* ================================================
           SECTION
        ================================================ */


        .workload-result-section {
          margin-top: 11px;

          padding: 11px;

          border: 1px solid #e0e4e8;

          border-radius: 6px;

          background: #ffffff;
        }


        .workload-section-header {
          margin-bottom: 10px;
        }


        .workload-section-header h3 {
          margin: 0 0 4px;

          color: #566470;

          font-size: 13px;
        }


        .workload-section-header p {
          margin: 0;

          color: #9aa3ac;

          font-size: 12px;
        }


        /* ================================================
           HISTOGRAM
        ================================================ */


        .latency-histogram {
          height: 190px;

          display: flex;

          align-items: stretch;

          gap: 3px;

          overflow-x: auto;
        }


        .histogram-column {
          min-width: 31px;

          flex: 1;

          display: flex;

          flex-direction: column;
        }


        .histogram-bar-area {
          height: 160px;

          display: flex;

          align-items: flex-end;

          justify-content: center;

          border-bottom: 1px solid #dce2e7;
        }


        .histogram-bar {
          width: 72%;

          min-height: 1px;

          position: relative;

          border-radius: 2px 2px 0 0;

          background: #90abc1;
        }


        .histogram-bar span {
          position: absolute;

          top: -13px;

          left: 50%;

          transform: translateX(-50%);

          color: #71808e;

          font-size: 12px;
        }


        .histogram-label {
          padding-top: 5px;

          text-align: center;

          color: #939da7;

          font-size: 12px;
        }


        /* ================================================
           LAYER GRID
        ================================================ */


        .workload-layer-grid {
          display: grid;

          grid-template-columns:
            repeat(10, 1fr);

          gap: 5px;
        }


        .workload-layer-card {
          padding: 6px;

          border: 1px solid #e0e4e8;

          border-radius: 4px;

          background: #fafbfc;
        }


        .layer-card-header {
          display: flex;

          justify-content: space-between;

          align-items: center;
        }


        .layer-card-header span {
          color: #8e98a2;

          font-size: 12px;
        }


        .layer-card-header strong {
          color: #495966;

          font-size: 12px;
        }


        .layer-card-bar {
          height: 3px;

          margin: 6px 0 4px;

          overflow: hidden;

          border-radius: 2px;

          background: #e5e9ed;
        }


        .layer-card-bar div {
          height: 100%;

          background: #87a5bd;
        }


        .workload-layer-card small {
          color: #9aa3ac;

          font-size: 12px;
        }


        /* ================================================
           SLOW LAYERS
        ================================================ */


        .slowest-workload-layers {
          display: flex;

          flex-wrap: wrap;

          gap: 6px;
        }


        .slow-workload-layer {
          min-width: 150px;

          padding: 7px 8px;

          display: grid;

          grid-template-columns:
            22px 55px 1fr;

          gap: 5px;

          align-items: center;

          border: 1px solid #e5d6c6;

          border-radius: 4px;

          background: #fffaf5;

          color: #896f57;

          font-size: 12px;
        }


        .slow-workload-layer small {
          grid-column: 2 / 4;

          color: #a08c79;
        }


        .slow-workload-rank {
          font-weight: 700;

          color: #b48660;
        }


        /* ================================================
           SUBCUBE
        ================================================ */


        .workload-sc-grid {
          display: grid;

          grid-template-columns:
            repeat(4, 1fr);

          gap: 7px;
        }


        .workload-sc-card {
          padding: 8px;

          border: 1px solid #e0e4e8;

          border-radius: 4px;

          background: #fafbfc;
        }


        .workload-sc-header {
          display: flex;

          justify-content: space-between;
        }


        .workload-sc-header strong {
          color: #465663;

          font-size: 12px;
        }


        .workload-sc-header span {
          color: #8796a3;

          font-size: 12px;
        }


        .workload-sc-bar {
          height: 4px;

          margin: 7px 0;

          overflow: hidden;

          border-radius: 2px;

          background: #e5e9ed;
        }


        .workload-sc-bar div {
          height: 100%;

          background: #8ea9bf;
        }


        .workload-sc-details {
          display: flex;

          gap: 9px;

          color: #929ca6;

          font-size: 12px;
        }


        /* ================================================
           SLOW TOKEN TABLE
        ================================================ */


        .slow-token-table {
          border: 1px solid #e3e6e9;

          border-radius: 4px;

          overflow: hidden;
        }


        .slow-token-header,
        .slow-token-row {
          min-height: 31px;

          padding: 0 8px;

          display: grid;

          grid-template-columns:
            45px
            85px
            130px
            minmax(150px, 1fr)
            70px
            60px;

          gap: 8px;

          align-items: center;
        }


        .slow-token-header {
          background: #f5f7f9;

          color: #84909b;

          font-size: 12px;

          font-weight: 700;
        }


        .slow-token-row {
          border-top: 1px solid #edf0f2;

          color: #84909a;

          font-size: 12px;
        }


        .slow-token-row strong {
          color: #4f5f6c;
        }


        .file-name-cell {
          overflow: hidden;

          text-overflow: ellipsis;

          white-space: nowrap;
        }


        /* ================================================
           EMPTY
        ================================================ */


        .workload-empty {
          min-height: 300px;

          display: flex;

          flex-direction: column;

          align-items: center;

          justify-content: center;

          color: #929ca6;
        }


        .empty-icon {
          margin-bottom: 10px;

          color: #b2bbc3;

          font-size: 36px;
        }


        .workload-empty strong {
          margin-bottom: 5px;

          color: #66727e;

          font-size: 13px;
        }


        .workload-empty p {
          margin: 0;

          font-size: 12px;
        }


        @media (
          max-width: 1200px
        ) {

          .latency-summary-grid {
            grid-template-columns:
              repeat(3, 1fr);
          }


          .workload-layer-grid {
            grid-template-columns:
              repeat(8, 1fr);
          }


          .workload-sc-grid {
            grid-template-columns:
              repeat(2, 1fr);
          }

        }

      `}
    </style>
  );
}


export default WorkloadAnalyzer;