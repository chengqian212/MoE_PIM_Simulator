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


function displayMatrixName(matrixName) {
  const type = normalizeMatrixName(matrixName);

  if (type === "gate") {
    return "Gate / 门控";
  }

  if (type === "up") {
    return "Up / 上投影";
  }

  if (type === "down") {
    return "Down / 下投影";
  }

  return type;
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
      "#cfe8ff";

    border =
      "#4f8fc9";

    text =
      "#174f7d";
  }


  if (type === "up") {
    background =
      "#ffedd5";

    border =
      "#d68a32";

    text =
      "#7c4a0b";
  }


  if (type === "down") {
    background =
      "#dcfce7";

    border =
      "#4d9c68";

    text =
      "#245d38";
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
        {displayMatrixName(matrixName)}
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
          共享 / Shared
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
            空物理 Plane / Empty Plane
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
        点击左侧 Plane 中的 Weight-Cube，查看该权重块的详细映射信息。
      </div>
    );
  }


  return (
    <div className="weight-detail">

      <div className="detail-header">
        <div>
          <div className="detail-small">
            WEIGHT-CUBE / 权重块
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
              ? "共享 / Shared"
              : "路由 / Routed"
          }
        </div>
      </div>


      <DetailRow
        label="模型层 / Layer"
        value={
          weight.layer_id
        }
      />


      <DetailRow
        label="专家 / Expert"
        value={
          weight.expert_id
        }
      />


      <DetailRow
        label="矩阵 / Matrix"
        value={
          displayMatrixName(
            weight.matrix_name
          )
        }
      />


      <div className="detail-divider" />


      <DetailRow
        label="逻辑 Plane"
        value={
          weight.logical_plane_id
        }
      />


      <DetailRow
        label="物理 Plane"
        value={
          weight.physical_plane_id
        }
      />


      <DetailRow
        label="槽位 ID / Slot"
        value={
          weight.slot_id
        }
      />


      <DetailRow
        label="Sub-Cube / 子立方"
        value={
          weight.subcube_id
        }
      />


      <DetailRow
        label="深度 z"
        value={
          weight.z
        }
      />


      <div className="detail-divider" />


      <DetailRow
        label="坐标 / Position"
        value={
          `(${weight.x}, ${weight.y})`
        }
      />


      <DetailRow
        label="逻辑尺寸"
        value={
          `${weight.logical_rows} × ${weight.logical_cols}`
        }
      />


      <DetailRow
        label="物理尺寸"
        value={
          `${weight.slot_rows} × ${weight.slot_cols}`
        }
      />


      <DetailRow
        label="旋转 / Rotated"
        value={
          weight.rotated
            ? "是 / Yes"
            : "否 / No"
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
        label="Gate / 门控"
      />


      <LegendItem
        className="legend-up"
        label="Up / 上投影"
      />


      <LegendItem
        className="legend-down"
        label="Down / 下投影"
      />


      <LegendItem
        className="legend-shared"
        label="Shared Expert / 共享专家"
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

  initialZ,

  focusCubeId,

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
  ] = useState(
    Number.isFinite(Number(initialZ))
      ? Number(initialZ)
      : 0
  );


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
  ] = useState(
    Number.isFinite(Number(initialZ))
      ? String(initialZ)
      : "0"
  );


  // =========================================================
  // SC 改变以后回 z=0
  // =========================================================

  useEffect(() => {
    const targetZ =
      Number.isFinite(Number(initialZ))
        ? Math.max(
            0,
            Math.min(
              D - 1,
              Number(initialZ)
            )
          )
        : 0;

    setZ(targetZ);
    setZInput(String(targetZ));
    setSelectedWeight(null);
  }, [
    subcubeId,
    initialZ,
    D,
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


        const focusedWeight =
          focusCubeId === undefined ||
          focusCubeId === null
            ? null
            : (data.weights ?? []).find(
                (weight) =>
                  weight.cube_id ===
                  focusCubeId
              ) ?? null;


        setSelectedWeight(
          focusedWeight
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
    focusCubeId,
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
            ← 返回全局 Cube
          </button>


          <div>
            <h2>
              Sub-Cube {subcubeId} / 子立方
            </h2>

            <p>
              查看该 Sub-Cube 不同深度上的物理 Plane 与 Weight-Cube 映射。
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
          物理 Plane / Physical Plane
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
            "加载中..."
          ) : (
            plane?.empty
              ? "空 Plane"
              : `${planeStats.weightCount} 个 Weight-Cube`
          )}

        </div>

      </div>


      {focusCubeId !== undefined &&
       focusCubeId !== null && (
        <div className="mapping-focus-banner">
          <strong>已自动定位：</strong>
          Cube-{focusCubeId}
          {" · "}
          SC-{subcubeId}
          {" · "}
          z={z}
          <span>目标 Weight-Cube 已在下方 Plane 中高亮。</span>
        </div>
      )}


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
                物理 Plane / Physical Plane
                {" · "}
                {H} × {W}
              </div>
            </div>


            <div className="plane-stats">

              <span>
                Gate / 门控
                <strong>
                  {
                    planeStats
                      .gateCount
                  }
                </strong>
              </span>


              <span>
                Up / 上投影
                <strong>
                  {
                    planeStats
                      .upCount
                  }
                </strong>
              </span>


              <span>
                Down / 下投影
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
            权重信息 / Weight Info
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

          margin-bottom: 8px;
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

          font-size: 16px;
          font-weight: 700;

          cursor: pointer;
        }


        .back-button:hover {
          background: #f3f5f7;
        }


        .subcube-header h2 {
          margin: 0 0 5px;

          color: #000000;

          font-size: 24px;
          font-weight: 700;
        }


        .subcube-header p {
          margin: 0;

          color: #000000;

          font-size: 15px;
          line-height: 1.55;
        }


        .subcube-tag {
          padding: 6px 9px;

          border: 1px solid #cad5df;
          border-radius: 4px;

          background: #ffffff;

          color: #000000;

          font-size: 15px;
          font-weight: 700;
        }


        /* ================================================
           DEPTH
        ================================================ */


        .depth-controller {
          min-height: 52px;

          padding: 8px 12px;

          display: flex;
          align-items: center;
          gap: 9px;

          border: 1px solid #dfe4ea;
          border-radius: 6px;

          background: #ffffff;

          margin-bottom: 8px;
        }


        .depth-title {
          margin-right: 7px;

          color: #000000;

          font-size: 15px;
          font-weight: 700;

          text-transform: uppercase;
          letter-spacing: 0.7px;
        }


        .depth-button {
          width: 34px;
          height: 34px;

          border: 1px solid #d9dfe6;
          border-radius: 4px;

          background: #ffffff;

          color: #000000;

          font-size: 20px;

          cursor: pointer;
        }


        .depth-button:disabled {
          cursor: default;

          opacity: 0.35;
        }


        .z-box {
          height: 34px;

          padding: 0 10px;

          display: flex;
          align-items: center;

          border: 1px solid #d9dfe6;
          border-radius: 4px;

          color: #000000;

          font-size: 15px;
        }


        .z-box input {
          width: 56px;

          margin: 0 4px;

          border: none;
          outline: none;

          text-align: center;

          color: #000000;

          font-size: 16px;
          font-weight: 700;

          background: transparent;
        }


        .depth-slider {
          flex: 1;

          min-width: 180px;

          cursor: pointer;
        }


        .depth-state {
          min-width: 135px;

          text-align: right;

          color: #000000;

          font-size: 16px;
          font-weight: 700;
        }


        .mapping-focus-banner {
          min-height: 38px;
          margin-bottom: 10px;
          padding: 7px 11px;
          display: flex;
          align-items: center;
          gap: 7px;
          border: 1px solid #b9cede;
          border-radius: 6px;
          background: #eef6fb;
          color: #000000;
          font-size: 16px;
        }

        .mapping-focus-banner strong {
          color: #000000;
          font-size: 15px;
        }

        .mapping-focus-banner span {
          margin-left: auto;
          color: #000000;
          font-size: 15px;
        }


        /* ================================================
           CONTENT
        ================================================ */


        .subcube-main {
          min-height: 500px;

          display: grid;

          grid-template-columns:
            minmax(560px, 1fr)
            300px;

          gap: 10px;
        }


        /* ================================================
           PLANE PANEL
        ================================================ */


        .plane-panel {
          padding: 14px;

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
          color: #000000;

          font-size: 18px;
          font-weight: 700;
        }


        .plane-subtitle {
          margin-top: 4px;

          color: #000000;

          font-size: 16px;
        }


        .plane-stats {
          display: flex;

          gap: 15px;

          color: #000000;

          font-size: 16px;
        }


        .plane-stats span {
          display: flex;

          gap: 5px;
        }


        .plane-stats strong {
          color: #000000;
        }


        /* ================================================
           PLANE DRAWING
        ================================================ */


        .plane-area {
          position: relative;

          width: min(
            540px,
            92%
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
            28px auto 10px;
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

          color: #000000;

          font-size: 16px;
          font-weight: 600;
        }


        .plane-h-label {
          position: absolute;

          left: -50px;

          top: 50%;

          transform:
            translateY(-50%)
            rotate(-90deg);

          color: #000000;

          font-size: 16px;
          font-weight: 600;
        }


        .plane-origin {
          position: absolute;

          left: 2px;
          top: 100%;

          margin-top: 4px;

          color: #000000;

          font-size: 15px;
        }


        /* ================================================
           WEIGHT BLOCK
        ================================================ */


        .weight-block {
          position: absolute;

          padding: 10px;

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

          font-size: 16px;
          font-weight: 700;
        }


        .weight-matrix {
          margin-top: 5px;

          font-size: 16px;
          font-weight: 700;

          text-transform: uppercase;
        }


        .weight-size {
          margin-top: 5px;

          opacity: 0.78;

          font-size: 16px;
          font-weight: 600;
        }


        .shared-tag {
          margin-top: 7px;

          padding: 2px 4px;

          border: 1px solid #836dae;
          border-radius: 3px;

          background: #f3eef9;

          color: #745c9d;

          font-size: 15px;
          font-weight: 700;
        }


        .empty-plane {
          width: 100%;
          height: 100%;

          display: flex;
          align-items: center;
          justify-content: center;

          color: #acb2ba;

          font-size: 16px;
          font-weight: 600;
        }


        .plane-error {
          min-height: 400px;

          display: flex;
          align-items: center;
          justify-content: center;

          color: #a75d5d;

          font-size: 16px;
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

          font-size: 16px;
          font-weight: 600;
        }


        .legend-item {
          display: flex;
          align-items: center;

          gap: 5px;
        }


        .legend-box {
          width: 14px;
          height: 14px;

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
          padding: 16px;

          border: 1px solid #dfe4ea;
          border-radius: 7px;

          background: #ffffff;
        }


        .weight-panel-title {
          margin-bottom: 16px;

          color: #000000;

          font-size: 15px;
          font-weight: 700;

          letter-spacing: 1px;
          text-transform: uppercase;
        }


        .weight-detail-empty {
          padding-top: 8px;

          color: #000000;

          font-size: 15px;

          line-height: 1.7;
        }


        .detail-header {
          display: flex;

          justify-content: space-between;
          align-items: flex-start;

          margin-bottom: 14px;
        }


        .detail-small {
          margin-bottom: 4px;

          color: #000000;

          font-size: 15px;
          font-weight: 700;

          letter-spacing: 1px;
        }


        .detail-header h3 {
          margin: 0;

          color: #000000;

          font-size: 20px;
        }


        .expert-badge {
          padding: 3px 5px;

          border: 1px solid #cad3dd;
          border-radius: 3px;

          background: #f6f8fa;

          color: #718091;

          font-size: 15px;
          font-weight: 650;
        }


        .expert-badge.shared {
          border-color: #a18abe;

          background: #f5f0fa;

          color: #795f9e;
        }


        .detail-row {
          min-height: 36px;

          display: flex;
          align-items: center;
          justify-content: space-between;

          border-bottom:
            1px solid #f0f2f4;

          color: #000000;

          font-size: 16px;
        }


        .detail-row strong {
          color: #000000;

          font-size: 16px;
          font-weight: 700;
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