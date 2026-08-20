import {
  useEffect,
  useState,
} from "react";

import GlobalCube3D
  from "./components/GlobalCube3D";

import SubCubeViewer
  from "./components/SubCubeViewer";

import RequestSimulator
  from "./components/RequestSimulator";

import OverviewDashboard from "./components/OverviewDashboard";

import MappingLocator from "./components/MappingLocator";

import SchedulerVisualizer from "./components/SchedulerVisualizer";

import ResultsAnalysis from "./components/ResultsAnalysis";

import Experiments from "./components/Experiments";
  
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
    phaseSummary,
    setPhaseSummary,
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
  ] = useState("overview");


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
  // Mapping 快速定位目标
  // =========================================================

  const [
    mappingTarget,
    setMappingTarget,
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
        // Prefill / Decode Phase Summary
        // ====================================================

        const phaseResponse =
          await fetch(
            `${API_BASE}/api/phase-summary`
          );


        if (!phaseResponse.ok) {

          throw new Error(
            "读取 Phase Summary 失败。"
          );
        }


        const phaseData =
          await phaseResponse.json();


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


        setPhaseSummary(
          phaseData
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
          "无法从 FastAPI 后端读取 Mapping 或阶段评估数据。"
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
            正在读取 MoE-PIM Mapping 与阶段评估结果...
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

      setMappingTarget(
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

    setMappingTarget(
      null
    );
  }


  // =========================================================
  // Mapping 快速定位
  // =========================================================

  function locateWeight(
    weight
  ) {
    if (!weight) {
      return;
    }

    setSelectedSubcube(
      weight.subcube_id
    );

    setMappingTarget({
      cube_id:
        weight.cube_id,
      subcube_id:
        weight.subcube_id,
      z:
        weight.z,
    });

    setOpenedSubcube(
      weight.subcube_id
    );
  }


  // =========================================================
  // 返回 Global Cube
  // =========================================================

  function backToGlobalCube() {

    setOpenedSubcube(
      null
    );

    setMappingTarget(
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
            当前映射 / Current Mapping
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
          label="Sub-Cube 数"
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
          label="深度 D"
          value={
            hardware.D
          }
        />


        <StatCard
          label="已用 Plane"
          value={
            hardware.used_planes
          }
        />


        <StatCard
          label="总 Plane"
          value={
            hardware.total_plane_slots
          }
        />


        <StatCard
          label="空 Plane"
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
          activePage === "cube" &&
          openedSubcube === null
            ? "main-layout cube-layout"
            : "main-layout wide-layout"
        }
      >


        {/* ===================================================
            左侧菜单
        ==================================================== */}

        <aside className="sidebar">

          <div className="sidebar-title">
            功能导航 / Navigation
          </div>


          <NavButton
            number="01"
            label="总览 / Overview"

            active={
              activePage ===
              "overview"
            }

            onClick={() =>
              switchPage(
                "overview"
              )
            }
          />


          <NavButton
            number="02"
            label="映射空间 / Cube"

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
            number="03"
            label="请求模拟 / Request"

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
            number="04"
            label="调度可视化 / Scheduler"

            active={
              activePage ===
              "scheduler"
            }

            onClick={() =>
              switchPage(
                "scheduler"
              )
            }
          />


          <NavButton
            number="05"
            label="结果分析 / Results"

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


          <NavButton
            number="06"
            label="实验配置 / Experiments"

            active={
              activePage ===
              "experiments"
            }

            onClick={() =>
              switchPage(
                "experiments"
              )
            }
          />


          <div className="sidebar-divider" />


          <div className="model-info">

            <div className="info-label">
              模型 / Model
            </div>


            <div>

              <span>
                MoE 层数
              </span>

              <strong>
                {
                  hardware.layer_count
                }
              </strong>

            </div>


            <div>

              <span>
                Weight-Cube 数
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
              Overview
          ================================================== */}

          {activePage === "overview" && (

            <OverviewDashboard
              phaseSummary={
                phaseSummary
              }

              hardware={
                hardware
              }
            />

          )}


          {/* =================================================
              Cube 页面
          ================================================== */}

          {activePage === "cube" && (

            openedSubcube === null ? (

              /*
              ----------------------------------------------
              第一层：
              全局映射空间 / Global Cube
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

                onLocateWeight={
                  locateWeight
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

                initialZ={
                  mappingTarget?.z
                }

                focusCubeId={
                  mappingTarget?.cube_id
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

            <RequestSimulator />

          )}


          {/* =================================================
              Scheduler Visualizer
          ================================================== */}

          {activePage === "scheduler" && (

            <SchedulerVisualizer />

          )}


          {/* =================================================
              Results
          ================================================== */}

          {activePage === "result" && (
            <ResultsAnalysis
              phaseSummary={phaseSummary}
              hardware={hardware}
            />
          )}


          {/* =================================================
              Experiments
          ================================================== */}

          {activePage === "experiments" && (
            <Experiments
              phaseSummary={phaseSummary}
              hardware={hardware}
              mappingFile={summary?.mapping_file}
            />
          )}

        </section>


        {/* ===================================================
            最右侧 Panel
        ==================================================== */}
{activePage === "cube" &&
 openedSubcube === null && (
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
          Mapping 已加载
        </span>


        <span className="status">

          <span className="status-dot" />

          后端已连接 / Connected

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

      <span className="nav-label">
        <strong>
          {label.split("/")[0]?.trim()}
        </strong>

        {label.includes("/") && (
          <small>
            {label.split("/").slice(1).join("/").trim()}
          </small>
        )}
      </span>

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

  onLocateWeight,
}) {

  return (
    <>

      <div className="workspace-header">

        <div>

          <h2>
            全局映射空间 / Global Cube
          </h2>

          <p>
            点击 Sub-Cube 查看概况，或使用下方定位器直接查找某个 Layer / Expert / Matrix。
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
              清除选择
            </button>

          )}


          <div className="view-tag">
            3D 全局视图
          </div>

        </div>

      </div>


      <MappingLocator
        layerCount={
          hardware.layer_count
        }

        onLocate={
          onLocateWeight
        }
      />


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

          {" "}个 Sub-Cube
        </div>


        <div>
          物理深度 D：
          <strong>
            {hardware.D}
          </strong>
        </div>


        <div>
          Plane 尺寸：
          <strong>
            {hardware.H}
            ×
            {hardware.W}
          </strong>
        </div>


        <div className="mouse-hint">
          拖拽旋转 · 滚轮缩放 · 单击选择
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
        硬件信息 / Hardware
      </div>


      <InfoRow
        label="拓扑 / Topology"

        value={
          hardware.N
            ? (
                `${hardware.N} × ${hardware.N}`
              )
            : "--"
        }
      />


      <InfoRow
        label="Sub-Cube 数"
        value={
          hardware.num_subcubes
        }
      />


      <InfoRow
        label="Plane 尺寸"

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
        label="深度 D"
        value={
          hardware.D
        }
      />


      <div className="panel-divider" />


      <div className="panel-title">
        存储空间 / Storage
      </div>


      <InfoRow
        label="已用 Plane"
        value={
          hardware.used_planes
        }
      />


      <InfoRow
        label="总 Plane"
        value={
          hardware.total_plane_slots
        }
      />


      <InfoRow
        label="空 Plane"
        value={
          hardware.empty_plane_slots
        }
      />


      <div className="panel-help">
        点击中央任意 Sub-Cube，右侧会显示该 SC 的空间占用和矩阵统计。
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
            已选择 / SELECTED
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
        label="已用 Plane"
        value={
          info.used_planes
        }
      />


      <InfoRow
        label="深度容量"
        value={
          info.depth_capacity
        }
      />


      <InfoRow
        label="空 Plane"
        value={
          info.empty_planes
        }
      />


      <InfoRow
        label="Weight-Cube 数"
        value={
          info.weight_cube_count
        }
      />


      <InfoRow
        label="Shared 权重"
        value={
          info.shared_weight_count
        }
      />


      <div className="panel-divider" />


      <div className="panel-title">
        矩阵类型 / Matrix
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
          进入 Sub-Cube
        </button>

      )}


      {opened && (

        <div className="opened-state">
          正在查看 SC-{info.subcube_id}
        </div>

      )}


      <div className="panel-help">

        {opened
          ? (
              "当前已经进入该 Sub-Cube，可以使用 z 滑块查看不同物理 Plane。"
            )
          : (
              "点击进入 Sub-Cube，可查看具体 Plane 和 Weight-Cube。"
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

          color: #17263a;

          background: #edf3f8;
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
          height: 76px;

          padding: 0 22px;

          display: flex;

          align-items: center;

          justify-content:
            space-between;

          background:
            #ffffff;

          border-bottom:
            1px solid #cbd5e1;

          box-shadow:
            inset 0 4px 0 #4F7195;
        }


        .top-header h1 {
          margin: 0;

          font-size: 32px;

          font-weight: 700;
        }


        .subtitle {
          margin-top: 6px;

          color: #526579;

          font-size: 17px;
        }


        .mapping-info {
          max-width: 470px;

          padding: 7px 10px;

          text-align: right;

          color: #7f8b98;

          font-size: 16px;

          border: 1px solid #e2e7ec;

          border-radius: 5px;

          background: #fafbfd;
        }


        .mapping-info strong {
          display: block;

          margin-top: 5px;

          color: #102a43;

          font-size: 17px;

          font-weight: 500;
        }


        /* ================================================
           STATS
        ================================================ */


        .stats-bar {
          min-height: 82px;

          padding: 9px 22px;

          display: grid;

          grid-template-columns:
            repeat(
              8,
              minmax(
                95px,
                1fr
              )
            );

          gap: 12px;

          background:
            #ffffff;

          border-bottom:
            1px solid #cbd5e1;
        }


        .stat-card {
          padding:
            10px 13px;

          background:
            #ffffff;

          border:
            1px solid #cbd5e1;

          border-left:
            4px solid #3b82f6;

          border-radius:
            5px;
        }


        .stat-label {
          margin-bottom:
            7px;

          color:
            #526579;

          font-size:
            16px;
        }


        .stat-value {
          color:
            #0f2744;

          font-size:
            26px;

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
            270px
            minmax(0, 1fr)
            300px;
        }


        /* Token / Results 页面给中间区域最大空间 */

        .main-layout.wide-layout {
          grid-template-columns:
            270px
            minmax(0, 1fr);
        }


        /* ================================================
           SIDEBAR
        ================================================ */


        .sidebar {
          padding: 16px 14px;

          background:
            #f8fafc;

          border-right:
            1px solid #cbd5e1;
        }


        .sidebar-title {
          margin:
            0 10px 12px;

          color:
            #244a6b;

          font-size:
            16px;

          font-weight:
            650;

          letter-spacing:
            1.3px;

          text-transform:
            uppercase;
        }


        .nav-button {
          width: 100%;

          min-height: 56px;

          margin-bottom:
            6px;

          padding:
            7px 12px;

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
            #536171;

          font-size:
            16px;

          cursor:
            pointer;

          text-align:
            left;
        }


        .nav-button:hover {
          background:
            #e8f1fb;

          color:
            #174a78;
        }


        .nav-button.active {
          background:
            #dbeafe;

          color:
            #123f70;

          font-weight:
            650;

          box-shadow:
            inset 5px 0 0 #4F7195;
        }


        .nav-number {
          width: 32px;
          min-width: 32px;

          color:
            #526579;

          font-size:
            15px;

          font-weight:
            700;

          font-variant-numeric:
            tabular-nums;
        }


        .nav-label {
          min-width: 0;

          display: flex;

          flex-direction: column;

          justify-content: center;

          gap: 2px;

          line-height: 1.18;
        }


        .nav-label strong {
          color: inherit;

          font-size: 18px;

          font-weight: 650;

          white-space: nowrap;
        }


        .nav-label small {
          color: #64748b;

          font-size: 15px;

          font-weight: 500;

          white-space: nowrap;
        }


        .nav-button.active .nav-label small {
          color: #315b85;
        }


        .sidebar-divider {
          height: 1px;

          margin:
            18px 8px;

          background:
            #eceff2;
        }


        .model-info {
          padding:
            0 10px;

          color:
            #687583;

          font-size:
            15px;
        }


        .info-label {
          margin-bottom:
            12px;

          color:
            #a0a8b2;

          font-size:
            14px;

          letter-spacing:
            0.8px;

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

          padding: 14px 16px 16px;

          background: #edf3f8;
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
            26px;

          font-weight:
            700;
        }


        .workspace-header p {
          margin: 0;

          color:
            #5f7083;

          font-size:
            15px;
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
            1px solid #9fb3c8;

          border-radius:
            4px;

          background:
            #ffffff;

          color:
            #49627b;

          font-size:
            13px;

          letter-spacing:
            0.7px;
        }


        .reset-button {
          height:
            32px;

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
            14px;

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
            1px solid #c3d0dc;

          border-radius:
            7px;

          background:
            #ffffff;
        }


        .cube-bottom-bar {
          min-height: 38px;

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
            #526579;

          font-size:
            14px;
        }


        .cube-bottom-bar strong {
          color:
            #3e4a57;
        }


        .mouse-hint {
          margin-left:
            auto;

          color:
            #64748b;
        }


        /* ================================================
           RIGHT PANEL
        ================================================ */


        .right-panel {
          padding: 16px;

          background:
            #ffffff;

          border-left:
            1px solid #cbd5e1;
        }


        .panel-title {
          margin-bottom:
            13px;

          color:
            #244a6b;

          font-size:
            15px;

          font-weight:
            650;

          letter-spacing:
            0.8px;

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
            #526579;

          font-size:
            15px;
        }


        .info-row strong {
          color:
            #17263a;

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
            #64748b;

          font-size:
            14px;

          line-height:
            1.65;
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
            20px;
        }


        .close-button {
          width: 30px;

          height: 30px;

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

          height: 40px;

          margin-top: 14px;

          border:
            1px solid #1d4f8a;

          border-radius:
            5px;

          background:
            #2563a8;

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
            #1d4f8a;
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
            1px solid #c3d0dc;

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
            26px;
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
          height: 32px;

          padding:
            0 22px;

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
            1px solid #c3d0dc;

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


        /* ================================================
           GLOBAL READABILITY + EMPHASIS
           统一提升 01–06 页面可读性与重点层级
        ================================================ */

        .workspace h2 {
          color: #102a43 !important;
          font-size: 26px !important;
          font-weight: 750 !important;
        }

        .workspace h3 {
          color: #173b5f !important;
          font-size: 20px !important;
          font-weight: 700 !important;
        }

        .workspace p,
        .workspace label {
          color: #526579;
          font-size: 16px !important;
          line-height: 1.55;
        }

        .workspace button,
        .workspace input,
        .workspace select {
          min-height: 36px;
          font-size: 16px !important;
        }

        .workspace input,
        .workspace select {
          border-color: #9fb3c8 !important;
          color: #17263a !important;
          background: #ffffff !important;
        }

        .workspace small {
          font-size: 14px !important;
        }

        .workload-small-title,
        .overview-small-title,
        .request-small-title,
        .scheduler-small-title,
        .results-small-title,
        .experiments-small-title,
        .detail-small,
        .selected-small-title {
          color: #405a73 !important;
          font-size: 15px !important;
          font-weight: 750 !important;
        }

        .panel-title,
        .weight-panel-title,
        .depth-title {
          color: #1f4f78 !important;
          font-size: 16px !important;
          font-weight: 750 !important;
        }

        .info-row {
          min-height: 36px;
          font-size: 16px !important;
        }

        .info-row strong {
          color: #102a43;
          font-size: 17px;
        }

        .scope-badge,
        .view-tag,
        .subcube-tag {
          border-color: #7aa7d2 !important;
          background: #e6f1fb !important;
          color: #174f7d !important;
          font-weight: 750 !important;
        }


        @media (max-width: 1320px) {
          .main-layout.cube-layout {
            grid-template-columns:
              250px
              minmax(0, 1fr)
              286px;
          }

          .main-layout.wide-layout {
            grid-template-columns:
              250px
              minmax(0, 1fr);
          }

          .nav-button {
            padding-left: 10px;
            padding-right: 10px;
          }
        }

      `}
    </style>
  );
}


export default App;