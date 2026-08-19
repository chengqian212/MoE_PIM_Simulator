import {
  useEffect,
  useState,
} from "react";

import GlobalCube3D
  from "./components/GlobalCube3D";

import SubCubeViewer
  from "./components/SubCubeViewer";

import TokenSimulator
  from "./components/TokenSimulator";

import WorkloadAnalyzer from "./components/WorkloadAnalyzer";
  
const API_BASE =
  "http://127.0.0.1:8000";


// ============================================================
// 顶部统计卡
// ============================================================


function StatCard({
  label,
  value,
}) {
  return (
    <div className="stat-card">

      <div className="stat-label">
        {label}
      </div>

      <div className="stat-value">
        {value ?? "--"}
      </div>

    </div>
  );
}


// ============================================================
// App
// ============================================================


function App() {

  // =========================================================
  // 后端数据
  // =========================================================

  const [
    summary,
    setSummary,
  ] = useState(null);


  const [
    subcubes,
    setSubcubes,
  ] = useState([]);


  // =========================================================
  // 一级页面
  //
  // cube
  // token
  // result
  // =========================================================

  const [
    activePage,
    setActivePage,
  ] = useState("cube");


  // =========================================================
  // Global Cube 当前选中的 SC
  // =========================================================

  const [
    selectedSubcube,
    setSelectedSubcube,
  ] = useState(null);


  // =========================================================
  // 当前是否已经真正进入某个 SC
  //
  // null:
  //      Global Cube
  //
  // 3:
  //      当前正在查看 SC-3
  // =========================================================

  const [
    openedSubcube,
    setOpenedSubcube,
  ] = useState(null);


  // =========================================================
  // Loading / Error
  // =========================================================

  const [
    loading,
    setLoading,
  ] = useState(true);


  const [
    error,
    setError,
  ] = useState("");


  // =========================================================
  // 读取后端
  // =========================================================

  useEffect(() => {

    async function loadData() {

      try {

        setLoading(true);


        // ====================================================
        // System Summary
        // ====================================================

        const summaryResponse =
          await fetch(
            `${API_BASE}/api/system/summary`
          );


        if (!summaryResponse.ok) {

          throw new Error(
            "读取 System Summary 失败。"
          );
        }


        const summaryData =
          await summaryResponse.json();


        // ====================================================
        // Sub-Cubes
        // ====================================================

        const subcubeResponse =
          await fetch(
            `${API_BASE}/api/subcubes`
          );


        if (!subcubeResponse.ok) {

          throw new Error(
            "读取 Sub-Cube 数据失败。"
          );
        }


        const subcubeData =
          await subcubeResponse.json();


        setSummary(
          summaryData
        );


        setSubcubes(
          subcubeData.items ?? []
        );


        setError("");


      } catch (err) {

        console.error(
          err
        );


        setError(
          "无法从 FastAPI 后端读取 Mapping 数据。"
        );


      } finally {

        setLoading(
          false
        );
      }
    }


    loadData();

  }, []);


  // =========================================================
  // Loading
  // =========================================================

  if (loading) {

    return (
      <>
        <div className="center-page">

          <div className="loading-card">
            正在读取 MoE-PIM Mapping...
          </div>

        </div>

        <Style />
      </>
    );
  }


  // =========================================================
  // Error
  // =========================================================

  if (error) {

    return (
      <>
        <div className="center-page">

          <div className="error-card">

            <h2>
              后端连接失败
            </h2>

            <p>
              {error}
            </p>

            <p>
              请确认后端正在运行：
            </p>

            <code>
              python -m uvicorn
              webui.backend.main:app
              --reload --port 8000
            </code>

          </div>

        </div>

        <Style />
      </>
    );
  }


  // =========================================================
  // Hardware
  // =========================================================

  const hardware =
    summary?.hardware ?? {};


  // =========================================================
  // 当前 SC 信息
  // =========================================================

  const selectedInfo =
    selectedSubcube === null
      ? null
      : (
          subcubes.find(
            (item) =>
              item.subcube_id ===
              selectedSubcube
          ) ?? null
        );


  // =========================================================
  // 切换一级页面
  // =========================================================

  function switchPage(
    page
  ) {

    setActivePage(
      page
    );


    /*
    离开 Cube 页面时，
    自动退出 Sub-Cube 内部。
    */

    if (page !== "cube") {

      setOpenedSubcube(
        null
      );
    }
  }


  // =========================================================
  // 打开 SC
  // =========================================================

  function openSubcube(
    subcubeId
  ) {

    setSelectedSubcube(
      subcubeId
    );


    setOpenedSubcube(
      subcubeId
    );
  }


  // =========================================================
  // 返回 Global Cube
  // =========================================================

  function backToGlobalCube() {

    setOpenedSubcube(
      null
    );
  }


  // =========================================================
  // Main
  // =========================================================

  return (
    <div className="app">


      {/* =====================================================
          顶部
      ====================================================== */}

      <header className="top-header">

        <div>

          <h1>
            MoE-PIM Simulator
          </h1>

          <div className="subtitle">
            3D 存算一体资源映射与推理模拟
          </div>

        </div>


        <div className="mapping-info">

          <div>
            Current Mapping
          </div>

          <strong>
            {summary?.mapping_file}
          </strong>

        </div>

      </header>


      {/* =====================================================
          Hardware Stats
      ====================================================== */}

      <section className="stats-bar">

        <StatCard
          label="N"
          value={
            hardware.N
          }
        />


        <StatCard
          label="Sub-Cubes"
          value={
            hardware.num_subcubes
          }
        />


        <StatCard
          label="H"
          value={
            hardware.H
          }
        />


        <StatCard
          label="W"
          value={
            hardware.W
          }
        />


        <StatCard
          label="Depth D"
          value={
            hardware.D
          }
        />


        <StatCard
          label="Used Planes"
          value={
            hardware.used_planes
          }
        />


        <StatCard
          label="Total Planes"
          value={
            hardware.total_plane_slots
          }
        />


        <StatCard
          label="Empty Planes"
          value={
            hardware.empty_plane_slots
          }
        />

      </section>


      {/* =====================================================
          Main
      ====================================================== */}
      <main
        className={
          activePage === "cube"
            ? "main-layout cube-layout"
            : "main-layout wide-layout"
        }
      >


        {/* ===================================================
            左侧菜单
        ==================================================== */}

        <aside className="sidebar">

          <div className="sidebar-title">
            Visualization
          </div>


          <NavButton
            number="01"
            label="Cube 总览"

            active={
              activePage ===
              "cube"
            }

            onClick={() =>
              switchPage(
                "cube"
              )
            }
          />


          <NavButton
            number="02"
            label="Token 模拟"

            active={
              activePage ===
              "token"
            }

            onClick={() =>
              switchPage(
                "token"
              )
            }
          />


          <NavButton
            number="03"
            label="结果分析"

            active={
              activePage ===
              "result"
            }

            onClick={() =>
              switchPage(
                "result"
              )
            }
          />


          <div className="sidebar-divider" />


          <div className="model-info">

            <div className="info-label">
              Model
            </div>


            <div>

              <span>
                MoE Layers
              </span>

              <strong>
                {
                  hardware.layer_count
                }
              </strong>

            </div>


            <div>

              <span>
                Weight-Cubes
              </span>

              <strong>
                {
                  hardware.weight_cube_count
                }
              </strong>

            </div>

          </div>

        </aside>


        {/* ===================================================
            中间 Workspace
        ==================================================== */}

        <section className="workspace">


          {/* =================================================
              Cube 页面
          ================================================== */}

          {activePage === "cube" && (

            openedSubcube === null ? (

              /*
              ----------------------------------------------
              第一层：
              Global Cube
              ----------------------------------------------
              */

              <CubePage
                hardware={
                  hardware
                }

                subcubes={
                  subcubes
                }

                selectedSubcube={
                  selectedSubcube
                }

                setSelectedSubcube={
                  setSelectedSubcube
                }
              />

            ) : (

              /*
              ----------------------------------------------
              第二层：
              Sub-Cube
              ----------------------------------------------
              */

              <SubCubeViewer
                subcubeId={
                  openedSubcube
                }

                hardware={
                  hardware
                }

                onBack={
                  backToGlobalCube
                }
              />

            )

          )}


          {/* =================================================
              Token Simulation
          ================================================== */}

          {activePage === "token" && (

            <TokenSimulator />

          )}


          {/* =================================================
              Results
          ================================================== */}

          {activePage === "result" && (
            <WorkloadAnalyzer />
          )}

        </section>


        {/* ===================================================
            最右侧 Panel
        ==================================================== */}
{activePage === "cube" && (
        <aside className="right-panel">

          {selectedInfo === null ? (

            <HardwarePanel
              hardware={
                hardware
              }
            />

          ) : (

            <SelectedSubcubePanel
              info={
                selectedInfo
              }

              opened={
                openedSubcube ===
                selectedInfo.subcube_id
              }

              onClear={() => {

                setSelectedSubcube(
                  null
                );

                setOpenedSubcube(
                  null
                );
              }}

              onOpen={() =>
                openSubcube(
                  selectedInfo.subcube_id
                )
              }
            />

          )}

        </aside>
)}
      </main>


      {/* =====================================================
          Footer
      ====================================================== */}

      <footer className="footer">

        <span>
          Mapping Loaded
        </span>


        <span className="status">

          <span className="status-dot" />

          Backend Connected

        </span>

      </footer>


      <Style />

    </div>
  );
}


// ============================================================
// Navigation
// ============================================================


function NavButton({
  number,
  label,
  active,
  onClick,
}) {

  return (
    <button
      className={
        active
          ? "nav-button active"
          : "nav-button"
      }

      onClick={
        onClick
      }
    >

      <span className="nav-number">
        {number}
      </span>

      {label}

    </button>
  );
}


// ============================================================
// Global Cube Page
// ============================================================


function CubePage({
  hardware,

  subcubes,

  selectedSubcube,

  setSelectedSubcube,
}) {

  return (
    <>

      <div className="workspace-header">

        <div>

          <h2>
            Global Cube
          </h2>

          <p>
            点击一个 Sub-Cube，
            再从右侧进入内部查看 Plane。
          </p>

        </div>


        <div className="header-actions">

          {selectedSubcube !==
            null && (

            <button
              className="reset-button"

              onClick={() =>
                setSelectedSubcube(
                  null
                )
              }
            >
              Clear Selection
            </button>

          )}


          <div className="view-tag">
            3D GLOBAL VIEW
          </div>

        </div>

      </div>


      <div className="cube-stage">

        <GlobalCube3D
          subcubes={
            subcubes
          }

          selectedSubcube={
            selectedSubcube
          }

          onSelectSubcube={
            setSelectedSubcube
          }
        />

      </div>


      <div className="cube-bottom-bar">

        <div>
          <strong>
            {
              hardware.num_subcubes
            }
          </strong>

          {" "}Sub-Cubes
        </div>


        <div>
          Physical Depth：
          <strong>
            {hardware.D}
          </strong>
        </div>


        <div>
          Plane Size：
          <strong>
            {hardware.H}
            ×
            {hardware.W}
          </strong>
        </div>


        <div className="mouse-hint">
          Drag 旋转 ·
          Wheel 缩放 ·
          Click 选择
        </div>

      </div>

    </>
  );
}


// ============================================================
// Hardware Panel
// ============================================================


function HardwarePanel({
  hardware,
}) {

  return (
    <>

      <div className="panel-title">
        Hardware
      </div>


      <InfoRow
        label="Topology"

        value={
          hardware.N
            ? (
                `${hardware.N} × ${hardware.N}`
              )
            : "--"
        }
      />


      <InfoRow
        label="Sub-Cubes"
        value={
          hardware.num_subcubes
        }
      />


      <InfoRow
        label="Plane Size"

        value={
          hardware.H &&
          hardware.W
            ? (
                `${hardware.H} × ${hardware.W}`
              )
            : "--"
        }
      />


      <InfoRow
        label="Depth D"
        value={
          hardware.D
        }
      />


      <div className="panel-divider" />


      <div className="panel-title">
        Storage
      </div>


      <InfoRow
        label="Used Planes"
        value={
          hardware.used_planes
        }
      />


      <InfoRow
        label="Total Planes"
        value={
          hardware.total_plane_slots
        }
      />


      <InfoRow
        label="Empty"
        value={
          hardware.empty_plane_slots
        }
      />


      <div className="panel-help">
        点击中央任意 Sub-Cube，
        右侧会显示该 SC 的详细信息。
      </div>

    </>
  );
}


// ============================================================
// Selected Sub-Cube Panel
// ============================================================


function SelectedSubcubePanel({
  info,

  opened,

  onClear,

  onOpen,
}) {

  const matrixCounts =
    info.matrix_counts ?? {};


  return (
    <>

      <div className="selected-panel-header">

        <div>

          <div className="selected-small-title">
            SELECTED
          </div>

          <h3>
            Sub-Cube
            {" "}
            {info.subcube_id}
          </h3>

        </div>


        <button
          className="close-button"

          onClick={
            onClear
          }
        >
          ×
        </button>

      </div>


      <InfoRow
        label="Used Planes"
        value={
          info.used_planes
        }
      />


      <InfoRow
        label="Depth"
        value={
          info.depth_capacity
        }
      />


      <InfoRow
        label="Empty Planes"
        value={
          info.empty_planes
        }
      />


      <InfoRow
        label="Weight-Cubes"
        value={
          info.weight_cube_count
        }
      />


      <InfoRow
        label="Shared Weight"
        value={
          info.shared_weight_count
        }
      />


      <div className="panel-divider" />


      <div className="panel-title">
        Matrix Type
      </div>


      <InfoRow
        label="Gate"
        value={
          matrixCounts.gate ??
          matrixCounts.gate_proj ??
          0
        }
      />


      <InfoRow
        label="Up"
        value={
          matrixCounts.up ??
          matrixCounts.up_proj ??
          0
        }
      />


      <InfoRow
        label="Down"
        value={
          matrixCounts.down ??
          matrixCounts.down_proj ??
          0
        }
      />


      {!opened && (

        <button
          className="open-subcube-button"

          onClick={
            onOpen
          }
        >
          Open Sub-Cube
        </button>

      )}


      {opened && (

        <div className="opened-state">
          Viewing SC-{info.subcube_id}
        </div>

      )}


      <div className="panel-help">

        {opened
          ? (
              "当前已经进入该 Sub-Cube，可以通过中间的 z Slider 查看 Physical Plane。"
            )
          : (
              "点击 Open Sub-Cube 后进入该 SC 内部。"
            )
        }

      </div>

    </>
  );
}


// ============================================================
// Info Row
// ============================================================


function InfoRow({
  label,
  value,
}) {

  return (
    <div className="info-row">

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
// Placeholder
// ============================================================


function PlaceholderPage({
  title,
  description,
}) {

  return (
    <div className="placeholder-page">

      <div>

        <div className="placeholder-icon">
          +
        </div>

        <h2>
          {title}
        </h2>

        <p>
          {description}
        </p>

      </div>

    </div>
  );
}


// ============================================================
// Style
// ============================================================


function Style() {

  return (
    <style>
      {`

        * {
          box-sizing: border-box;
        }


        html,
        body,
        #root {
          margin: 0;

          width: 100%;

          min-width: 0;

          min-height: 100%;

          font-family:
            Inter,
            "Microsoft YaHei",
            Arial,
            sans-serif;

          color: #1f2937;

          background: #f5f7fa;
        }


        button {
          font-family: inherit;
        }


        /* ================================================
           APP
        ================================================ */


        .app {
          min-height: 100vh;

          display: flex;

          flex-direction: column;
        }


        /* ================================================
           HEADER
        ================================================ */


        .top-header {
          height: 64px;

          padding: 0 20px;

          display: flex;

          align-items: center;

          justify-content:
            space-between;

          background:
            #ffffff;

          border-bottom:
            1px solid #e5e7eb;
        }


        .top-header h1 {
          margin: 0;

          font-size: 26px;

          font-weight: 650;
        }


        .subtitle {
          margin-top: 6px;

          color: #8a94a3;

          font-size: 15px;
        }


        .mapping-info {
          max-width: 430px;

          text-align: right;

          color: #8a94a3;

          font-size: 14px;
        }


        .mapping-info strong {
          display: block;

          margin-top: 5px;

          color: #374151;

          font-size: 15px;

          font-weight: 500;
        }


        /* ================================================
           STATS
        ================================================ */


        .stats-bar {
          min-height: 70px;

          padding: 10px 20px;

          display: grid;

          grid-template-columns:
            repeat(
              8,
              minmax(
                95px,
                1fr
              )
            );

          gap: 10px;

          background:
            #ffffff;

          border-bottom:
            1px solid #e5e7eb;
        }


        .stat-card {
          padding:
            11px 13px;

          background:
            #fafbfc;

          border:
            1px solid #e5e7eb;

          border-radius:
            6px;
        }


        .stat-label {
          margin-bottom:
            7px;

          color:
            #8a94a3;

          font-size:
            14px;
        }


        .stat-value {
          color:
            #252d38;

          font-size:
            20px;

          font-weight:
            650;
        }


        /* ================================================
           MAIN
        ================================================ */


        .main-layout {
          flex: 1;

          display: grid;

          min-width: 0;

          min-height: 0;
        }


        /* Cube 页面需要右侧 Hardware Panel */

        .main-layout.cube-layout {
          grid-template-columns:
            200px
            minmax(0, 1fr)
            280px;
        }


        /* Token / Results 页面给中间区域最大空间 */

        .main-layout.wide-layout {
          grid-template-columns:
            200px
            minmax(0, 1fr);
        }


        /* ================================================
           SIDEBAR
        ================================================ */


        .sidebar {
          padding: 14px 12px;

          background:
            #ffffff;

          border-right:
            1px solid #e5e7eb;
        }


        .sidebar-title {
          margin:
            0 9px 14px;

          color:
            #9aa3af;

          font-size:
            13px;

          font-weight:
            600;

          letter-spacing:
            1.3px;

          text-transform:
            uppercase;
        }


        .nav-button {
          width: 100%;

          height: 44px;

          margin-bottom:
            5px;

          padding:
            0 11px;

          display: flex;

          align-items:
            center;

          gap: 10px;

          border:
            none;

          border-radius:
            5px;

          background:
            transparent;

          color:
            #606b79;

          font-size:
            15px;

          cursor:
            pointer;

          text-align:
            left;
        }


        .nav-button:hover {
          background:
            #f3f5f7;
        }


        .nav-button.active {
          background:
            #edf3fb;

          color:
            #234b7a;

          font-weight:
            600;
        }


        .nav-number {
          width: 20px;

          color:
            #a4acb6;

          font-size:
            13px;
        }


        .sidebar-divider {
          height: 1px;

          margin:
            23px 8px;

          background:
            #eceff2;
        }


        .model-info {
          padding:
            0 9px;

          color:
            #7b8490;

          font-size:
            14px;
        }


        .info-label {
          margin-bottom:
            12px;

          color:
            #a0a8b2;

          font-size:
            13px;

          letter-spacing:
            1px;

          text-transform:
            uppercase;
        }


        .model-info
        > div:not(.info-label) {
          display: flex;

          justify-content:
            space-between;

          margin-bottom:
            10px;
        }


        /* ================================================
           WORKSPACE
        ================================================ */


        .workspace {
          min-width: 0;

          width: 100%;

          overflow-x: hidden;

          padding: 16px;

          background: #f6f8fa;
        }


        .workspace-header {
          min-height: 52px;

          display: flex;

          align-items:
            flex-start;

          justify-content:
            space-between;
        }


        .workspace-header h2 {
          margin:
            0 0 5px;

          font-size:
            20px;

          font-weight:
            650;
        }


        .workspace-header p {
          margin: 0;

          color:
            #87919e;

          font-size:
            14px;
        }


        .header-actions {
          display: flex;

          align-items:
            center;

          gap:
            8px;
        }


        .view-tag {
          padding:
            5px 8px;

          border:
            1px solid #dbe2ea;

          border-radius:
            4px;

          background:
            #ffffff;

          color:
            #8b95a1;

          font-size:
            12px;

          letter-spacing:
            1px;
        }


        .reset-button {
          height:
            27px;

          padding:
            0 9px;

          border:
            1px solid #d9e0e7;

          border-radius:
            4px;

          background:
            #ffffff;

          color:
            #687381;

          font-size:
            13px;

          cursor:
            pointer;
        }


        /* ================================================
           3D
        ================================================ */


        .cube-stage {
          height: 560px;

          min-height: 560px;

          overflow: hidden;

          border:
            1px solid #dfe4ea;

          border-radius:
            7px;

          background:
            #ffffff;
        }


        .cube-bottom-bar {
          height: 32px;

          margin-top:
            8px;

          padding:
            0 12px;

          display: flex;

          align-items:
            center;

          gap:
            24px;

          border:
            1px solid #e1e5ea;

          border-radius:
            5px;

          background:
            #ffffff;

          color:
            #7f8996;

          font-size:
            13px;
        }


        .cube-bottom-bar strong {
          color:
            #3e4a57;
        }


        .mouse-hint {
          margin-left:
            auto;

          color:
            #a0a8b1;
        }


        /* ================================================
           RIGHT PANEL
        ================================================ */


        .right-panel {
          padding: 16px 14px;

          background:
            #ffffff;

          border-left:
            1px solid #e5e7eb;
        }


        .panel-title {
          margin-bottom:
            13px;

          color:
            #89929f;

          font-size:
            13px;

          font-weight:
            600;

          letter-spacing:
            1px;

          text-transform:
            uppercase;
        }


        .info-row {
          min-height: 30px;

          display: flex;

          align-items:
            center;

          justify-content:
            space-between;

          border-bottom:
            1px solid #f0f1f3;

          color:
            #828b97;

          font-size:
            14px;
        }


        .info-row strong {
          color:
            #364152;

          font-weight:
            600;
        }


        .panel-divider {
          height: 1px;

          margin:
            23px 0;

          background:
            #e8ebee;
        }


        .panel-help {
          margin-top: 12px;

          color:
            #a0a8b1;

          font-size:
            13px;

          line-height:
            1.7;
        }


        /* ================================================
           SELECTED
        ================================================ */


        .selected-panel-header {
          display: flex;

          align-items:
            flex-start;

          justify-content:
            space-between;

          margin-bottom:
            15px;
        }


        .selected-small-title {
          margin-bottom:
            5px;

          color:
            #9ba4af;

          font-size:
            12px;

          font-weight:
            600;

          letter-spacing:
            1px;
        }


        .selected-panel-header h3 {
          margin: 0;

          color:
            #2f3a46;

          font-size:
            19px;
        }


        .close-button {
          width: 26px;

          height: 26px;

          border:
            1px solid #dfe4e8;

          border-radius:
            4px;

          background:
            #ffffff;

          color:
            #8a939e;

          cursor:
            pointer;
        }


        .open-subcube-button {
          width: 100%;

          height: 36px;

          margin-top: 14px;

          border:
            1px solid #6e8ead;

          border-radius:
            5px;

          background:
            #7799bb;

          color:
            #ffffff;

          font-size:
            14px;

          font-weight:
            600;

          cursor:
            pointer;
        }


        .open-subcube-button:hover {
          background:
            #6789ab;
        }


        .opened-state {
          margin-top: 14px;

          padding:
            10px;

          border:
            1px solid #bdd0e1;

          border-radius:
            5px;

          background:
            #eef5fb;

          color:
            #4b6f91;

          font-size:
            13px;

          text-align:
            center;
        }


        /* ================================================
           PLACEHOLDER
        ================================================ */


        .placeholder-page {
          min-height:
            600px;

          display: flex;

          align-items:
            center;

          justify-content:
            center;

          border:
            1px solid #dfe4ea;

          border-radius:
            7px;

          background:
            #ffffff;

          text-align:
            center;
        }


        .placeholder-icon {
          width: 36px;

          height: 36px;

          margin:
            0 auto 15px;

          display: flex;

          align-items:
            center;

          justify-content:
            center;

          border-radius:
            50%;

          background:
            #f0f3f6;

          color:
            #84909d;

          font-size:
            22px;
        }


        .placeholder-page h2 {
          margin:
            0 0 9px;

          font-size:
            19px;
        }


        .placeholder-page p {
          max-width:
            410px;

          margin: 0;

          color:
            #929ba6;

          font-size:
            15px;
        }


        /* ================================================
           FOOTER
        ================================================ */


        .footer {
          height: 30px;

          padding:
            0 20px;

          display: flex;

          align-items:
            center;

          justify-content:
            space-between;

          background:
            #ffffff;

          border-top:
            1px solid #e5e7eb;

          color:
            #9aa2ac;

          font-size:
            13px;
        }


        .status {
          display: flex;

          align-items:
            center;

          gap:
            6px;
        }


        .status-dot {
          width: 6px;

          height: 6px;

          border-radius:
            50%;

          background:
            #56a56f;
        }


        /* ================================================
           Loading
        ================================================ */


        .center-page {
          min-height:
            100vh;

          display: flex;

          align-items:
            center;

          justify-content:
            center;

          background:
            #f5f7fa;
        }


        .loading-card,
        .error-card {
          padding:
            26px 30px;

          border:
            1px solid #dfe4ea;

          border-radius:
            7px;

          background:
            #ffffff;

          color:
            #586270;

          font-size:
            15px;
        }


        .error-card {
          width:
            550px;
        }


        .error-card code {
          display:
            block;

          margin-top:
            12px;

          padding:
            12px;

          background:
            #f3f4f6;
        }

      `}
    </style>
  );
}


export default App;