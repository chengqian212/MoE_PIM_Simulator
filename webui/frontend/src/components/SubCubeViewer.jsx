import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE =
  "http://127.0.0.1:8000";


/*
============================================================
SubCubeViewer

作用：

Global Cube
    ↓ 点击 SC-x
SubCubeViewer
    ↓
选择 Physical z
    ↓
显示该 Plane 中的真实 Weight-Cube

例如：

SC-3 / z=523

┌────────────────────┬────────────────────┐
│ Layer 17           │ Layer 17           │
│ Expert 83          │ Expert 83          │
│ gate               │ down               │
│                    │                    │
└────────────────────┴────────────────────┘

坐标关系：

PhysicalSlot：

    x -> H 方向
    y -> W 方向

网页：

    top    = x / H
    left   = y / W
    height = slot_rows / H
    width  = slot_cols / W

因此可以真实还原第三步二维 Plane 中
Weight-Cube 的空间位置。
============================================================
*/


// ============================================================
// 矩阵名称统一
// ============================================================


function normalizeMatrixName(
  matrixName
) {
  if (
    matrixName === "gate_proj"
  ) {
    return "gate";
  }

  if (
    matrixName === "up_proj"
  ) {
    return "up";
  }

  if (
    matrixName === "down_proj"
  ) {
    return "down";
  }

  return matrixName;
}


// ============================================================
// 矩阵显示样式
// ============================================================


function matrixStyle(
  matrixName,
  isShared
) {
  const type =
    normalizeMatrixName(
      matrixName
    );


  let background =
    "#e8edf3";

  let border =
    "#aab4c0";

  let text =
    "#475569";


  if (type === "gate") {
    background =
      "#dceaf7";

    border =
      "#7fa7cb";

    text =
      "#315d86";
  }


  if (type === "up") {
    background =
      "#f7e8d7";

    border =
      "#cf9f68";

    text =
      "#855d32";
  }


  if (type === "down") {
    background =
      "#dfeee4";

    border =
      "#80aa8e";

    text =
      "#386746";
  }


  /*
  Shared Expert：

  不改变 gate/up/down 本身颜色，
  只增加一圈紫色内边框。
  */

  const boxShadow =
    isShared
      ? "inset 0 0 0 3px #8b73b7"
      : "none";


  return {
    background,
    border,
    text,
    boxShadow,
  };
}


// ============================================================
// 单个 Weight-Cube
// ============================================================


function WeightBlock({
  weight,

  H,
  W,

  selected,

  onClick,
}) {
  /*
  ------------------------------------------------------------
  PhysicalSlot 坐标：

      x 沿 H
      y 沿 W

  网页：

      vertical   <- H
      horizontal <- W
  ------------------------------------------------------------
  */


  const top =
    (
      (weight.x ?? 0)
      / H
    )
    * 100;


  const left =
    (
      (weight.y ?? 0)
      / W
    )
    * 100;


  const slotRows =
    weight.slot_rows ??
    weight.logical_rows ??
    0;


  const slotCols =
    weight.slot_cols ??
    weight.logical_cols ??
    0;


  const height =
    (
      slotRows
      / H
    )
    * 100;


  const width =
    (
      slotCols
      / W
    )
    * 100;


  const style =
    matrixStyle(
      weight.matrix_name,
      weight.is_shared
    );


  const matrixName =
    normalizeMatrixName(
      weight.matrix_name
    );


  return (
    <button
      className={
        selected
          ? "weight-block selected"
          : "weight-block"
      }

      style={{
        top:
          `${top}%`,

        left:
          `${left}%`,

        width:
          `${width}%`,

        height:
          `${height}%`,

        background:
          style.background,

        borderColor:
          style.border,

        color:
          style.text,

        boxShadow:
          style.boxShadow,
      }}

      onClick={() =>
        onClick(weight)
      }
    >
      <div className="weight-title">
        L{weight.layer_id}
        {" / "}
        E{weight.expert_id}
      </div>


      <div className="weight-matrix">
        {matrixName}
      </div>


      <div className="weight-size">
        {
          weight.logical_rows ??
          "?"
        }
        ×
        {
          weight.logical_cols ??
          "?"
        }
      </div>


      {weight.is_shared && (
        <div className="shared-tag">
          Shared
        </div>
      )}
    </button>
  );
}


// ============================================================
// Plane Viewer
// ============================================================


function PhysicalPlane({
  plane,

  H,
  W,

  selectedWeight,

  setSelectedWeight,
}) {
  const weights =
    plane?.weights ?? [];


  return (
    <div className="plane-area">

      {/* =====================================================
          H / W 标注
      ====================================================== */}

      <div className="plane-w-label">
        W = {W}
      </div>


      <div className="plane-h-label">
        H = {H}
      </div>


      {/* =====================================================
          Plane
      ====================================================== */}

      <div className="physical-plane">

        {weights.length === 0 ? (

          <div className="empty-plane">
            Empty Physical Plane
          </div>

        ) : (

          weights.map(
            (weight) => (
              <WeightBlock
                key={
                  weight.cube_id
                }

                weight={
                  weight
                }

                H={H}

                W={W}

                selected={
                  selectedWeight
                    ?.cube_id ===
                  weight.cube_id
                }

                onClick={
                  setSelectedWeight
                }
              />
            )
          )

        )}

      </div>


      {/* =====================================================
          坐标说明
      ====================================================== */}

      <div className="plane-origin">
        (0, 0)
      </div>

    </div>
  );
}


// ============================================================
// Weight 信息
// ============================================================


function WeightDetail({
  weight,
}) {
  if (!weight) {
    return (
      <div className="weight-detail-empty">
        点击 Plane 中的 Weight-Cube
        查看详细映射信息。
      </div>
    );
  }


  return (
    <div className="weight-detail">

      <div className="detail-header">
        <div>
          <div className="detail-small">
            WEIGHT-CUBE
          </div>

          <h3>
            Cube-{weight.cube_id}
          </h3>
        </div>


        <div
          className={
            weight.is_shared
              ? "expert-badge shared"
              : "expert-badge"
          }
        >
          {
            weight.is_shared
              ? "Shared"
              : "Routed"
          }
        </div>
      </div>


      <DetailRow
        label="Model Layer"
        value={
          weight.layer_id
        }
      />


      <DetailRow
        label="Expert"
        value={
          weight.expert_id
        }
      />


      <DetailRow
        label="Matrix"
        value={
          normalizeMatrixName(
            weight.matrix_name
          )
        }
      />


      <div className="detail-divider" />


      <DetailRow
        label="Logical Plane"
        value={
          weight.logical_plane_id
        }
      />


      <DetailRow
        label="Physical Plane"
        value={
          weight.physical_plane_id
        }
      />


      <DetailRow
        label="Slot ID"
        value={
          weight.slot_id
        }
      />


      <DetailRow
        label="Sub-Cube"
        value={
          weight.subcube_id
        }
      />


      <DetailRow
        label="z"
        value={
          weight.z
        }
      />


      <div className="detail-divider" />


      <DetailRow
        label="Position"
        value={
          `(${weight.x}, ${weight.y})`
        }
      />


      <DetailRow
        label="Logical Size"
        value={
          `${weight.logical_rows} × ${weight.logical_cols}`
        }
      />


      <DetailRow
        label="Physical Size"
        value={
          `${weight.slot_rows} × ${weight.slot_cols}`
        }
      />


      <DetailRow
        label="Rotated"
        value={
          weight.rotated
            ? "Yes"
            : "No"
        }
      />

    </div>
  );
}


// ============================================================
// Detail Row
// ============================================================


function DetailRow({
  label,
  value,
}) {
  return (
    <div className="detail-row">
      <span>
        {label}
      </span>

      <strong>
        {
          value ??
          "--"
        }
      </strong>
    </div>
  );
}


// ============================================================
// Legend
// ============================================================


function Legend() {
  return (
    <div className="plane-legend">

      <LegendItem
        className="legend-gate"
        label="gate"
      />


      <LegendItem
        className="legend-up"
        label="up"
      />


      <LegendItem
        className="legend-down"
        label="down"
      />


      <LegendItem
        className="legend-shared"
        label="Shared Expert"
      />

    </div>
  );
}


function LegendItem({
  className,
  label,
}) {
  return (
    <div className="legend-item">

      <span
        className={
          `legend-box ${className}`
        }
      />

      {label}

    </div>
  );
}


// ============================================================
// 主组件
// ============================================================


function SubCubeViewer({
  subcubeId,

  hardware,

  onBack,
}) {
  const H =
    hardware?.H ?? 7168;


  const W =
    hardware?.W ?? 4096;


  const D =
    hardware?.D ?? 1;


  // =========================================================
  // 当前 z
  // =========================================================

  const [
    z,
    setZ,
  ] = useState(0);


  // =========================================================
  // Plane Data
  // =========================================================

  const [
    plane,
    setPlane,
  ] = useState(null);


  // =========================================================
  // 当前选择的 Weight
  // =========================================================

  const [
    selectedWeight,
    setSelectedWeight,
  ] = useState(null);


  // =========================================================
  // Loading / Error
  // =========================================================

  const [
    loading,
    setLoading,
  ] = useState(false);


  const [
    error,
    setError,
  ] = useState("");


  // =========================================================
  // 手动输入 z
  // =========================================================

  const [
    zInput,
    setZInput,
  ] = useState("0");


  // =========================================================
  // SC 改变以后回 z=0
  // =========================================================

  useEffect(() => {
    setZ(0);

    setZInput("0");

    setSelectedWeight(
      null
    );
  }, [
    subcubeId
  ]);


  // =========================================================
  // Load Plane
  // =========================================================

  useEffect(() => {
    /*
    ----------------------------------------------------------
    Slider 快速拖动时：

    z 会连续变化。

    AbortController 可以取消上一次尚未结束的请求，
    防止旧请求晚回来覆盖新 Plane。
    ----------------------------------------------------------
    */

    const controller =
      new AbortController();


    async function loadPlane() {
      try {
        setLoading(true);

        setError("");


        const response =
          await fetch(
            `${API_BASE}/api/subcubes/${subcubeId}/planes/${z}`,
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


        setPlane(
          data
        );


        setSelectedWeight(
          null
        );

      } catch (err) {

        if (
          err.name ===
          "AbortError"
        ) {
          return;
        }


        console.error(err);


        setError(
          `无法读取 SC-${subcubeId} / z=${z}`
        );

      } finally {

        if (
          !controller.signal
            .aborted
        ) {
          setLoading(false);
        }
      }
    }


    loadPlane();


    return () => {
      controller.abort();
    };

  }, [
    subcubeId,
    z,
  ]);


  // =========================================================
  // Plane 简短统计
  // =========================================================

  const planeStats =
    useMemo(() => {
      const weights =
        plane?.weights ?? [];


      return {
        weightCount:
          weights.length,

        gateCount:
          weights.filter(
            (weight) =>
              normalizeMatrixName(
                weight.matrix_name
              ) === "gate"
          ).length,

        upCount:
          weights.filter(
            (weight) =>
              normalizeMatrixName(
                weight.matrix_name
              ) === "up"
          ).length,

        downCount:
          weights.filter(
            (weight) =>
              normalizeMatrixName(
                weight.matrix_name
              ) === "down"
          ).length,
      };

    }, [
      plane
    ]);


  // =========================================================
  // z 更新
  // =========================================================

  function changeZ(
    newZ
  ) {
    const safeZ =
      Math.max(
        0,
        Math.min(
          D - 1,
          Number(newZ)
        )
      );


    setZ(
      safeZ
    );


    setZInput(
      String(
        safeZ
      )
    );
  }


  // =========================================================
  // z Input
  // =========================================================

  function submitZInput() {
    const parsed =
      Number(
        zInput
      );


    if (
      Number.isNaN(parsed)
    ) {
      setZInput(
        String(z)
      );

      return;
    }


    changeZ(
      Math.floor(parsed)
    );
  }


  // =========================================================
  // Render
  // =========================================================

  return (
    <div className="subcube-viewer">

      {/* =====================================================
          Header
      ====================================================== */}

      <div className="subcube-header">

        <div className="header-left">

          <button
            className="back-button"

            onClick={
              onBack
            }
          >
            ← Global Cube
          </button>


          <div>
            <h2>
              Sub-Cube {subcubeId}
            </h2>

            <p>
              查看该 Sub-Cube
              中不同深度的 Physical Plane。
            </p>
          </div>

        </div>


        <div className="subcube-tag">
          SC-{subcubeId}
        </div>

      </div>


      {/* =====================================================
          z Controller
      ====================================================== */}

      <div className="depth-controller">

        <div className="depth-title">
          Physical Plane
        </div>


        <button
          className="depth-button"

          disabled={
            z <= 0
          }

          onClick={() =>
            changeZ(
              z - 1
            )
          }
        >
          ‹
        </button>


        <div className="z-box">
          z =
          <input
            value={
              zInput
            }

            onChange={
              (event) =>
                setZInput(
                  event.target.value
                )
            }

            onBlur={
              submitZInput
            }

            onKeyDown={
              (event) => {
                if (
                  event.key ===
                  "Enter"
                ) {
                  submitZInput();
                }
              }
            }
          />

          <span>
            / {D - 1}
          </span>
        </div>


        <button
          className="depth-button"

          disabled={
            z >= D - 1
          }

          onClick={() =>
            changeZ(
              z + 1
            )
          }
        >
          ›
        </button>


        <input
          className="depth-slider"

          type="range"

          min="0"

          max={
            Math.max(
              D - 1,
              0
            )
          }

          value={
            z
          }

          onChange={
            (event) =>
              changeZ(
                Number(
                  event.target.value
                )
              )
          }
        />


        <div className="depth-state">

          {loading ? (
            "Loading..."
          ) : (
            plane?.empty
              ? "Empty Plane"
              : `${planeStats.weightCount} Weight-Cubes`
          )}

        </div>

      </div>


      {/* =====================================================
          Main
      ====================================================== */}

      <div className="subcube-main">

        {/* ===================================================
            左：Plane
        ==================================================== */}

        <div className="plane-panel">

          <div className="plane-panel-header">

            <div>
              <div className="plane-title">
                SC-{subcubeId}
                {" / "}
                z={z}
              </div>

              <div className="plane-subtitle">
                Physical Plane
                {" · "}
                {H} × {W}
              </div>
            </div>


            <div className="plane-stats">

              <span>
                Gate
                <strong>
                  {
                    planeStats
                      .gateCount
                  }
                </strong>
              </span>


              <span>
                Up
                <strong>
                  {
                    planeStats
                      .upCount
                  }
                </strong>
              </span>


              <span>
                Down
                <strong>
                  {
                    planeStats
                      .downCount
                  }
                </strong>
              </span>

            </div>

          </div>


          {error ? (

            <div className="plane-error">
              {error}
            </div>

          ) : (

            <PhysicalPlane
              plane={
                plane
              }

              H={
                H
              }

              W={
                W
              }

              selectedWeight={
                selectedWeight
              }

              setSelectedWeight={
                setSelectedWeight
              }
            />

          )}


          <Legend />

        </div>


        {/* ===================================================
            右：Weight Detail
        ==================================================== */}

        <div className="weight-panel">

          <div className="weight-panel-title">
            Weight Information
          </div>


          <WeightDetail
            weight={
              selectedWeight
            }
          />

        </div>

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

        /* ================================================
           MAIN
        ================================================ */


        .subcube-viewer {
          width: 100%;
          height: 100%;
        }


        /* ================================================
           HEADER
        ================================================ */


        .subcube-header {
          min-height: 52px;

          display: flex;
          align-items: flex-start;
          justify-content: space-between;

          margin-bottom: 12px;
        }


        .header-left {
          display: flex;
          align-items: flex-start;
          gap: 14px;
        }


        .back-button {
          height: 29px;

          padding: 0 10px;

          border: 1px solid #d9e0e7;
          border-radius: 4px;

          background: #ffffff;

          color: #657181;

          font-size: 13px;

          cursor: pointer;
        }


        .back-button:hover {
          background: #f3f5f7;
        }


        .subcube-header h2 {
          margin: 0 0 5px;

          color: #2f3945;

          font-size: 20px;
          font-weight: 650;
        }


        .subcube-header p {
          margin: 0;

          color: #8d96a1;

          font-size: 14px;
        }


        .subcube-tag {
          padding: 6px 9px;

          border: 1px solid #cad5df;
          border-radius: 4px;

          background: #ffffff;

          color: #59738e;

          font-size: 13px;
          font-weight: 650;
        }


        /* ================================================
           DEPTH
        ================================================ */


        .depth-controller {
          min-height: 44px;

          padding: 6px 10px;

          display: flex;
          align-items: center;
          gap: 9px;

          border: 1px solid #dfe4ea;
          border-radius: 6px;

          background: #ffffff;

          margin-bottom: 12px;
        }


        .depth-title {
          margin-right: 7px;

          color: #89929e;

          font-size: 13px;
          font-weight: 600;

          text-transform: uppercase;
          letter-spacing: 0.7px;
        }


        .depth-button {
          width: 29px;
          height: 29px;

          border: 1px solid #d9dfe6;
          border-radius: 4px;

          background: #ffffff;

          color: #556272;

          font-size: 20px;

          cursor: pointer;
        }


        .depth-button:disabled {
          cursor: default;

          opacity: 0.35;
        }


        .z-box {
          height: 29px;

          padding: 0 8px;

          display: flex;
          align-items: center;

          border: 1px solid #d9dfe6;
          border-radius: 4px;

          color: #6e7884;

          font-size: 13px;
        }


        .z-box input {
          width: 48px;

          margin: 0 4px;

          border: none;
          outline: none;

          text-align: center;

          color: #344152;

          font-size: 14px;
          font-weight: 650;

          background: transparent;
        }


        .depth-slider {
          flex: 1;

          min-width: 180px;

          cursor: pointer;
        }


        .depth-state {
          min-width: 100px;

          text-align: right;

          color: #7c8794;

          font-size: 13px;
        }


        /* ================================================
           CONTENT
        ================================================ */


        .subcube-main {
          min-height: 440px;

          display: grid;

          grid-template-columns:
            minmax(500px, 1fr)
            230px;

          gap: 12px;
        }


        /* ================================================
           PLANE PANEL
        ================================================ */


        .plane-panel {
          padding: 12px;

          border: 1px solid #dfe4ea;
          border-radius: 7px;

          background: #ffffff;
        }


        .plane-panel-header {
          display: flex;

          align-items: center;
          justify-content: space-between;

          min-height: 43px;

          margin-bottom: 13px;
        }


        .plane-title {
          color: #34404d;

          font-size: 15px;
          font-weight: 650;
        }


        .plane-subtitle {
          margin-top: 4px;

          color: #979faa;

          font-size: 13px;
        }


        .plane-stats {
          display: flex;

          gap: 15px;

          color: #8a939e;

          font-size: 13px;
        }


        .plane-stats span {
          display: flex;

          gap: 5px;
        }


        .plane-stats strong {
          color: #3c4855;
        }


        /* ================================================
           PLANE DRAWING
        ================================================ */


        .plane-area {
          position: relative;

          width: min(
            420px,
            75%
          );

          /*
          H=7168
          W=4096

          网页宽度对应 W，
          高度对应 H。

          比例：

              H/W = 7168/4096
          */

          aspect-ratio:
            4096 / 7168;

          margin:
            25px auto 15px;
        }


        .physical-plane {
          position: absolute;

          inset: 0;

          overflow: hidden;

          border: 2px solid #7f8996;

          background:
            repeating-linear-gradient(
              0deg,
              #fafbfc,
              #fafbfc 24px,
              #f5f7f9 25px
            );
        }


        .plane-w-label {
          position: absolute;

          top: -23px;

          left: 50%;

          transform:
            translateX(-50%);

          color: #8a939e;

          font-size: 12px;
        }


        .plane-h-label {
          position: absolute;

          left: -50px;

          top: 50%;

          transform:
            translateY(-50%)
            rotate(-90deg);

          color: #8a939e;

          font-size: 12px;
        }


        .plane-origin {
          position: absolute;

          left: 2px;
          top: 100%;

          margin-top: 4px;

          color: #a0a7b0;

          font-size: 12px;
        }


        /* ================================================
           WEIGHT BLOCK
        ================================================ */


        .weight-block {
          position: absolute;

          padding: 9px;

          overflow: hidden;

          display: flex;
          flex-direction: column;
          align-items: flex-start;
          justify-content: center;

          border: 1px solid;

          cursor: pointer;

          text-align: left;

          transition:
            filter 0.1s,
            outline 0.1s;

          appearance: none;
        }


        .weight-block:hover {
          filter:
            brightness(0.97);
        }


        .weight-block.selected {
          outline:
            3px solid #3f678d;

          outline-offset:
            -3px;

          z-index: 4;
        }


        .weight-title {
          color: inherit;

          font-size: 14px;
          font-weight: 700;
        }


        .weight-matrix {
          margin-top: 5px;

          font-size: 15px;
          font-weight: 600;

          text-transform: uppercase;
        }


        .weight-size {
          margin-top: 5px;

          opacity: 0.72;

          font-size: 12px;
        }


        .shared-tag {
          margin-top: 7px;

          padding: 2px 4px;

          border: 1px solid #836dae;
          border-radius: 3px;

          background: #f3eef9;

          color: #745c9d;

          font-size: 12px;
          font-weight: 650;
        }


        .empty-plane {
          width: 100%;
          height: 100%;

          display: flex;
          align-items: center;
          justify-content: center;

          color: #acb2ba;

          font-size: 14px;
        }


        .plane-error {
          min-height: 400px;

          display: flex;
          align-items: center;
          justify-content: center;

          color: #a75d5d;

          font-size: 14px;
        }


        /* ================================================
           LEGEND
        ================================================ */


        .plane-legend {
          min-height: 32px;

          margin-top: 23px;

          display: flex;
          align-items: center;
          justify-content: center;

          gap: 18px;

          color: #7d8792;

          font-size: 12px;
        }


        .legend-item {
          display: flex;
          align-items: center;

          gap: 5px;
        }


        .legend-box {
          width: 11px;
          height: 11px;

          display: inline-block;

          border: 1px solid;
        }


        .legend-gate {
          background: #dceaf7;
          border-color: #7fa7cb;
        }


        .legend-up {
          background: #f7e8d7;
          border-color: #cf9f68;
        }


        .legend-down {
          background: #dfeee4;
          border-color: #80aa8e;
        }


        .legend-shared {
          background: #ffffff;

          border:
            2px solid #8b73b7;
        }


        /* ================================================
           RIGHT WEIGHT PANEL
        ================================================ */


        .weight-panel {
          padding: 15px;

          border: 1px solid #dfe4ea;
          border-radius: 7px;

          background: #ffffff;
        }


        .weight-panel-title {
          margin-bottom: 16px;

          color: #8e97a2;

          font-size: 12px;
          font-weight: 650;

          letter-spacing: 1px;
          text-transform: uppercase;
        }


        .weight-detail-empty {
          padding-top: 20px;

          color: #a0a8b1;

          font-size: 13px;

          line-height: 1.8;
        }


        .detail-header {
          display: flex;

          justify-content: space-between;
          align-items: flex-start;

          margin-bottom: 14px;
        }


        .detail-small {
          margin-bottom: 4px;

          color: #a0a8b1;

          font-size: 12px;
          font-weight: 650;

          letter-spacing: 1px;
        }


        .detail-header h3 {
          margin: 0;

          color: #34404d;

          font-size: 17px;
        }


        .expert-badge {
          padding: 3px 5px;

          border: 1px solid #cad3dd;
          border-radius: 3px;

          background: #f6f8fa;

          color: #718091;

          font-size: 12px;
        }


        .expert-badge.shared {
          border-color: #a18abe;

          background: #f5f0fa;

          color: #795f9e;
        }


        .detail-row {
          min-height: 31px;

          display: flex;
          align-items: center;
          justify-content: space-between;

          border-bottom:
            1px solid #f0f2f4;

          color: #8b949f;

          font-size: 12px;
        }


        .detail-row strong {
          color: #3f4b58;

          font-size: 12px;
          font-weight: 600;
        }


        .detail-divider {
          height: 1px;

          margin: 12px 0;

          background: #e7eaee;
        }

      `}
    </style>
  );
}


export default SubCubeViewer;