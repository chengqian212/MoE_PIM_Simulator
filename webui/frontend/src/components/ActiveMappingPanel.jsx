import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE =
  "http://127.0.0.1:8000";


// ============================================================
// Matrix Name
// ============================================================


function normalizeMatrixName(
  name
) {

  if (name === "gate_proj") {
    return "gate";
  }


  if (name === "up_proj") {
    return "up";
  }


  if (name === "down_proj") {
    return "down";
  }


  return name;
}


// ============================================================
// 从 API 返回结果中提取 placements
// ============================================================


function extractPlacements(
  data
) {

  if (Array.isArray(data)) {
    return data;
  }


  if (
    Array.isArray(
      data?.subcubes
    )
  ) {

    return data.subcubes.flatMap(
      (subcube) => {

        if (
          !Array.isArray(
            subcube?.weights
          )
        ) {
          return [];
        }


        return subcube.weights.map(
          (weight) => ({
            ...weight,

            subcube_id:
              weight.subcube_id ??
              subcube.subcube_id,
          })
        );
      }
    );
  }


  if (
    Array.isArray(
      data?.items
    )
  ) {
    return data.items;
  }


  if (
    Array.isArray(
      data?.placements
    )
  ) {
    return data.placements;
  }


  if (
    Array.isArray(
      data?.weights
    )
  ) {
    return data.weights;
  }


  return [];
}


// ============================================================
// 单个 Matrix 的紧凑位置显示
// ============================================================


function MatrixCell({
  location,
  matrixName,
}) {

  if (!location) {

    return (
      <div className="compact-matrix-cell missing">
        Not Found
      </div>
    );
  }


  return (
    <div
      className={
        `compact-matrix-cell ${matrixName}`
      }

      title={
        `SC-${location.subcube_id}`
        + ` | z=${location.z}`
        + ` | Plane ${location.physical_plane_id ?? "--"}`
        + ` | Slot ${location.slot_id ?? "--"}`
      }
    >

      <div className="matrix-main-location">

        <strong>
          SC-{location.subcube_id}
        </strong>

        <span>
          z={location.z}
        </span>

      </div>


      <div className="matrix-secondary-location">

        P{
          location
            .physical_plane_id ??
          "--"
        }

        <span>
          ·
        </span>

        Slot {
          location
            .slot_id ??
          "--"
        }

      </div>

    </div>
  );
}


// ============================================================
// Constraint Check
// ============================================================


function MappingCheckCell({
  expert,
}) {

  const gate =
    expert.matrices.gate;


  const up =
    expert.matrices.up;


  const down =
    expert.matrices.down;


  const gateDownSameSC =
    Boolean(
      gate
      &&
      down
      &&
      gate.subcube_id ===
        down.subcube_id
    );


  const gateUpSeparated =
    Boolean(
      gate
      &&
      up
      &&
      gate.subcube_id !==
        up.subcube_id
    );


  const allFound =
    Boolean(
      gate &&
      up &&
      down
    );


  const passed =
    allFound
    &&
    gateDownSameSC
    &&
    gateUpSeparated;


  return (
    <div
      className={
        passed
          ? "compact-check pass"
          : "compact-check fail"
      }

      title={
        [
          `gate/down: ${
            gateDownSameSC
              ? "Same SC"
              : "Fail"
          }`,

          `gate/up: ${
            gateUpSeparated
              ? "Separated"
              : "Fail"
          }`,
        ].join(" | ")
      }
    >

      <strong>
        {
          passed
            ? "✓ Pass"
            : "⚠ Check"
        }
      </strong>


      <span>
        GD {
          gateDownSameSC
            ? "✓"
            : "×"
        }
        {"  "}
        GU {
          gateUpSeparated
            ? "✓"
            : "×"
        }
      </span>

    </div>
  );
}


// ============================================================
// ActiveMappingPanel
// ============================================================


function ActiveMappingPanel({
  layerId,

  routedExpertIds = [],

  sharedExpertId = 256,
}) {

  const [
    layerPlacements,
    setLayerPlacements,
  ] = useState([]);


  const [
    loading,
    setLoading,
  ] = useState(false);


  const [
    error,
    setError,
  ] = useState("");


  // =========================================================
  // 请求当前 Layer Mapping
  // =========================================================

  useEffect(() => {

    if (
      layerId === null ||
      layerId === undefined
    ) {
      return;
    }


    const controller =
      new AbortController();


    async function loadLayer() {

      try {

        setLoading(
          true
        );


        setError("");


        const response =
          await fetch(
            `${API_BASE}/api/layers/${layerId}`,
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


        setLayerPlacements(
          extractPlacements(
            data
          )
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
          `读取 Layer ${layerId} Mapping 失败。`
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


    loadLayer();


    return () => {

      controller.abort();
    };

  }, [
    layerId
  ]);


  // =========================================================
  // 当前 Layer 的 8 Routed + 1 Shared
  // =========================================================

  const activeExperts =
    useMemo(
      () => {

        const routeRanks =
          new Map();


        routedExpertIds.forEach(
          (
            expertId,
            index
          ) => {

            routeRanks.set(
              Number(
                expertId
              ),
              index + 1
            );
          }
        );


        const activeIds = [
          ...routedExpertIds.map(
            Number
          ),

          Number(
            sharedExpertId
          ),
        ];


        return activeIds.map(
          (expertId) => {

            const matrices = {
              gate: null,
              up: null,
              down: null,
            };


            const placements =
              layerPlacements.filter(
                (placement) =>
                  Number(
                    placement.expert_id
                  ) ===
                  expertId
              );


            for (
              const placement
              of placements
            ) {

              const matrixName =
                normalizeMatrixName(
                  placement.matrix_name ??
                  placement.matrix
                );


              if (
                matrixName === "gate"
                ||
                matrixName === "up"
                ||
                matrixName === "down"
              ) {

                matrices[
                  matrixName
                ] = placement;
              }
            }


            return {
              expert_id:
                expertId,

              is_shared:
                expertId ===
                Number(
                  sharedExpertId
                ),

              route_rank:
                routeRanks.get(
                  expertId
                ) ?? null,

              matrices,
            };
          }
        );

      },
      [
        layerPlacements,
        routedExpertIds,
        sharedExpertId,
      ]
    );


  // =========================================================
  // Missing Matrix
  // =========================================================

  const missingMatrixCount =
    useMemo(
      () => {

        let count =
          0;


        for (
          const expert
          of activeExperts
        ) {

          for (
            const matrixName
            of [
              "gate",
              "up",
              "down",
            ]
          ) {

            if (
              !expert
                .matrices[
                  matrixName
                ]
            ) {

              count += 1;
            }
          }
        }


        return count;

      },
      [
        activeExperts
      ]
    );


  // =========================================================
  // Active SC Count
  // =========================================================

  const activeSubcubeCount =
    useMemo(
      () => {

        const ids =
          new Set();


        for (
          const expert
          of activeExperts
        ) {

          for (
            const matrixName
            of [
              "gate",
              "up",
              "down",
            ]
          ) {

            const location =
              expert
                .matrices[
                  matrixName
                ];


            if (location) {

              ids.add(
                location.subcube_id
              );
            }
          }
        }


        return ids.size;

      },
      [
        activeExperts
      ]
    );


  // =========================================================
  // Mapping Constraint Summary
  // =========================================================

  const mappingCheckSummary =
    useMemo(
      () => {

        let passed =
          0;


        for (
          const expert
          of activeExperts
        ) {

          const gate =
            expert.matrices.gate;


          const up =
            expert.matrices.up;


          const down =
            expert.matrices.down;


          if (
            gate
            &&
            up
            &&
            down
            &&
            gate.subcube_id ===
              down.subcube_id
            &&
            gate.subcube_id !==
              up.subcube_id
          ) {

            passed += 1;
          }
        }


        return {
          passed,

          total:
            activeExperts.length,
        };

      },
      [
        activeExperts
      ]
    );


  // =========================================================
  // Render
  // =========================================================

  return (
    <div className="active-mapping-panel compact">

      {/* =====================================================
          紧凑摘要
      ====================================================== */}

      <div className="compact-map-summary">

        <div>

          <span>
            Layer
          </span>

          <strong>
            L{layerId}
          </strong>

        </div>


        <div>

          <span>
            Active SC
          </span>

          <strong>
            {activeSubcubeCount}/16
          </strong>

        </div>


        <div>

          <span>
            Experts
          </span>

          <strong>
            {activeExperts.length}
          </strong>

        </div>


        <div>

          <span>
            Matrices
          </span>

          <strong>
            {
              activeExperts.length
              * 3
            }
          </strong>

        </div>


        <div
          className={
            mappingCheckSummary.passed ===
            mappingCheckSummary.total
              ? "mapping-summary-check pass"
              : "mapping-summary-check fail"
          }
        >

          <span>
            Mapping Check
          </span>

          <strong>
            {
              mappingCheckSummary.passed ===
              mappingCheckSummary.total
                ? "✓ "
                : "⚠ "
            }

            {
              mappingCheckSummary.passed
            }
            /
            {
              mappingCheckSummary.total
            }
          </strong>

        </div>

      </div>


      {/* =====================================================
          Loading / Error
      ====================================================== */}

      {loading && (

        <div className="mapping-loading">

          正在读取 Layer
          {" "}
          {layerId}
          {" "}
          Mapping...

        </div>

      )}


      {error && (

        <div className="mapping-error">
          {error}
        </div>

      )}


      {!loading &&
       !error &&
       missingMatrixCount > 0 && (

        <div className="mapping-warning">

          有
          {" "}
          <strong>
            {missingMatrixCount}
          </strong>
          {" "}
          个矩阵未找到，请检查
          `/api/layers/{layerId}`
          的返回字段。

        </div>

      )}


      {/* =====================================================
          紧凑 Mapping Table
      ====================================================== */}

      {!loading &&
       !error && (

        <div className="compact-mapping-table">

          <div className="compact-mapping-header">

            <span>
              Expert
            </span>

            <span>
              Gate
            </span>

            <span>
              Up
            </span>

            <span>
              Down
            </span>

            <span>
              Check
            </span>

          </div>


          {activeExperts.map(
            (expert) => (

              <div
                className={
                  expert.is_shared
                    ? "compact-mapping-row shared"
                    : "compact-mapping-row"
                }

                key={
                  expert.expert_id
                }
              >

                <div className="compact-expert-cell">

                  <strong>
                    E{
                      expert.expert_id
                    }
                  </strong>


                  <span>
                    {
                      expert.is_shared
                        ? "Shared"
                        : `Routed #${expert.route_rank}`
                    }
                  </span>

                </div>


                <MatrixCell
                  matrixName="gate"

                  location={
                    expert.matrices.gate
                  }
                />


                <MatrixCell
                  matrixName="up"

                  location={
                    expert.matrices.up
                  }
                />


                <MatrixCell
                  matrixName="down"

                  location={
                    expert.matrices.down
                  }
                />


                <MappingCheckCell
                  expert={
                    expert
                  }
                />

              </div>

            )
          )}

        </div>

      )}


      <div className="mapping-footnote">

        GD = gate/down 同 Sub-Cube；
        GU = gate/up 分离。
        鼠标停在位置单元格上可查看 Plane 和 Slot。

      </div>


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

        .active-mapping-panel.compact {
          padding:
            8px 10px 10px;

          border-top:
            1px solid #edf0f2;
        }


        /* ================================================
           SUMMARY
        ================================================ */


        .compact-map-summary {
          margin-bottom:
            8px;

          padding:
            6px 8px;

          display:
            flex;

          align-items:
            center;

          gap:
            18px;

          border:
            1px solid #e2e6ea;

          border-radius:
            5px;

          background:
            #fafbfc;
        }


        .compact-map-summary > div {
          min-width:
            58px;
        }


        .compact-map-summary span {
          display:
            block;

          margin-bottom:
            2px;

          color:
            #97a0aa;

          font-size:
            9px;
        }


        .compact-map-summary strong {
          color:
            #455563;

          font-size:
            11px;
        }


        .compact-map-summary
        .mapping-summary-check {
          margin-left:
            auto;

          padding-left:
            12px;

          border-left:
            1px solid #e1e5e9;
        }


        .mapping-summary-check.pass
        strong {
          color:
            #55765f;
        }


        .mapping-summary-check.fail
        strong {
          color:
            #9a5c5c;
        }


        /* ================================================
           STATE
        ================================================ */


        .mapping-loading,
        .mapping-error,
        .mapping-warning {
          margin-bottom:
            8px;

          padding:
            8px 9px;

          border-radius:
            5px;

          font-size:
            10px;
        }


        .mapping-loading {
          border:
            1px solid #dce3e9;

          background:
            #f8fafb;

          color:
            #7f8a96;
        }


        .mapping-error {
          border:
            1px solid #e1bcbc;

          background:
            #fff6f6;

          color:
            #9e5757;
        }


        .mapping-warning {
          border:
            1px solid #e4d3aa;

          background:
            #fffaf0;

          color:
            #8a6e32;
        }


        /* ================================================
           TABLE
        ================================================ */


        .compact-mapping-table {
          overflow:
            hidden;

          border:
            1px solid #dfe4ea;

          border-radius:
            5px;

          background:
            #ffffff;
        }


        .compact-mapping-header,
        .compact-mapping-row {
          display:
            grid;

          grid-template-columns:
            minmax(78px, 0.75fr)
            repeat(
              3,
              minmax(
                125px,
                1.25fr
              )
            )
            minmax(92px, 0.8fr);

          align-items:
            stretch;
        }


        .compact-mapping-header {
          min-height:
            30px;

          background:
            #f5f7f9;

          border-bottom:
            1px solid #dfe4ea;
        }


        .compact-mapping-header > span {
          padding:
            0 8px;

          display:
            flex;

          align-items:
            center;

          border-right:
            1px solid #e5e8eb;

          color:
            #84909b;

          font-size:
            9px;

          font-weight:
            700;

          text-transform:
            uppercase;
        }


        .compact-mapping-row {
          min-height:
            44px;

          border-bottom:
            1px solid #edf0f2;
        }


        .compact-mapping-row:last-child {
          border-bottom:
            0;
        }


        .compact-mapping-row.shared {
          background:
            #fbf8fd;

          box-shadow:
            inset 3px 0 0
            #8f78ad;
        }


        .compact-mapping-row > * {
          border-right:
            1px solid #edf0f2;
        }


        .compact-mapping-row > *:last-child {
          border-right:
            0;
        }


        /* ================================================
           EXPERT
        ================================================ */


        .compact-expert-cell {
          padding:
            6px 8px;

          display:
            flex;

          flex-direction:
            column;

          justify-content:
            center;
        }


        .compact-expert-cell strong {
          color:
            #35414d;

          font-size:
            11px;
        }


        .compact-expert-cell span {
          margin-top:
            2px;

          color:
            #98a1aa;

          font-size:
            8px;
        }


        .compact-mapping-row.shared
        .compact-expert-cell strong {
          color:
            #725c91;
        }


        /* ================================================
           MATRIX
        ================================================ */


        .compact-matrix-cell {
          padding:
            5px 8px;

          display:
            flex;

          flex-direction:
            column;

          justify-content:
            center;
        }


        .compact-matrix-cell.missing {
          color:
            #a0a8b1;

          font-size:
            10px;
        }


        .matrix-main-location {
          display:
            flex;

          align-items:
            center;

          gap:
            6px;
        }


        .matrix-main-location strong {
          color:
            #3f5060;

          font-size:
            10px;
        }


        .matrix-main-location span {
          color:
            #87929d;

          font-size:
            9px;
        }


        .matrix-secondary-location {
          margin-top:
            2px;

          display:
            flex;

          gap:
            4px;

          color:
            #a0a8b1;

          font-size:
            8px;
        }


        .compact-matrix-cell.gate {
          background:
            rgba(
              227,
              238,
              247,
              0.28
            );
        }


        .compact-matrix-cell.up {
          background:
            rgba(
              249,
              236,
              221,
              0.3
            );
        }


        .compact-matrix-cell.down {
          background:
            rgba(
              228,
              240,
              232,
              0.32
            );
        }


        /* ================================================
           CHECK
        ================================================ */


        .compact-check {
          padding:
            5px 7px;

          display:
            flex;

          flex-direction:
            column;

          justify-content:
            center;

          align-items:
            center;

          text-align:
            center;
        }


        .compact-check strong {
          font-size:
            9px;
        }


        .compact-check span {
          margin-top:
            2px;

          color:
            #929ca6;

          font-size:
            8px;
        }


        .compact-check.pass {
          background:
            #f3f8f4;
        }


        .compact-check.pass strong {
          color:
            #54755d;
        }


        .compact-check.fail {
          background:
            #fff6f6;
        }


        .compact-check.fail strong {
          color:
            #9a5c5c;
        }


        /* ================================================
           FOOTNOTE
        ================================================ */


        .mapping-footnote {
          margin-top:
            6px;

          color:
            #9aa3ac;

          font-size:
            8px;
        }

        /* ================================================
          FONT SIZE ENLARGE
        ================================================ */


        /* 顶部 Layer / Active SC / Experts / Matrices */
        .compact-map-summary span {
          font-size: 11px;
        }

        .compact-map-summary strong {
          font-size: 14px;
        }


        /* 表头：EXPERT / GATE / UP / DOWN / CHECK */
        .compact-mapping-header > span {
          font-size: 12px;
        }


        /* 每一行增高一点 */
        .compact-mapping-row {
          min-height: 58px;
        }


        /* E17 / E21 / E256 */
        .compact-expert-cell strong {
          font-size: 15px;
        }


        /* Routed #1 / Shared */
        .compact-expert-cell span {
          margin-top: 4px;
          font-size: 11px;
        }


        /* SC-10 / SC-6 等 */
        .matrix-main-location strong {
          font-size: 14px;
        }


        /* z=2 / z=935 */
        .matrix-main-location span {
          font-size: 12px;
        }


        /* P17 · Slot 34 */
        .matrix-secondary-location {
          margin-top: 5px;
          font-size: 11px;
        }


        /* ✓ Pass */
        .compact-check strong {
          font-size: 13px;
        }


        /* GD ✓  GU ✓ */
        .compact-check span {
          margin-top: 5px;
          font-size: 11px;
        }


        /* 最下面的 GD/GU 说明 */
        .mapping-footnote {
          margin-top: 8px;
          font-size: 11px;
        }
        @media (
          max-width: 1050px
        ) {

          .compact-mapping-table {
            overflow-x:
              auto;
          }


          .compact-mapping-header,
          .compact-mapping-row {
            min-width:
              720px;
          }

        }

      `}
    </style>
  );
}


export default ActiveMappingPanel;
