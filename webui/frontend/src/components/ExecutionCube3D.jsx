import {
  useMemo,
  useState,
} from "react";

import {
  Canvas,
} from "@react-three/fiber";

import {
  Edges,
  Html,
  OrbitControls,
} from "@react-three/drei";


// ============================================================
// 状态配置
// ============================================================


const STATE_CONFIG = {
  idle: {
    label: "Idle",
    color: "#d8dee5",
    border: "#8793a0",
  },

  switch: {
    label: "Switch",
    color: "#efcaca",
    border: "#b66e6e",
  },

  gate: {
    label: "Gate",
    color: "#bcd8ee",
    border: "#5f8eb5",
  },

  up: {
    label: "Up",
    color: "#efd6b6",
    border: "#b9874d",
  },

  down: {
    label: "Down",
    color: "#c8e3d1",
    border: "#63916f",
  },
};


function getStateConfig(
  state
) {

  return (
    STATE_CONFIG[
      state
    ]
    ??
    STATE_CONFIG.idle
  );
}


// ============================================================
// Hover Row
// ============================================================


function HoverRow({
  label,
  value,
}) {

  return (
    <div className="execution-hover-row">

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
// 单个 Sub-Cube
// ============================================================


function ExecutionSubCube({
  id,
  position,
  size,
  stateInfo,
  hovered,
  selected,
  onHover,
  onLeave,
  onSelect,
}) {

  const state =
    stateInfo?.state ??
    "idle";


  const task =
    stateInfo?.task ??
    null;


  const config =
    getStateConfig(
      state
    );


  const isShared =
    Boolean(
      task?.is_shared
    );


  let color =
    config.color;


  if (hovered) {

    color =
      "#d6e2ed";
  }


  const opacity =
    state === "idle"
      ? 0.36
      : 0.88;


  return (
    <group
      position={
        position
      }
    >

      <mesh
        onClick={
          (event) => {

            event.stopPropagation();

            onSelect(
              id
            );
          }
        }

        onPointerEnter={
          (event) => {

            event.stopPropagation();

            document.body.style.cursor =
              "pointer";


            onHover(
              id
            );
          }
        }

        onPointerLeave={
          (event) => {

            event.stopPropagation();

            document.body.style.cursor =
              "default";


            onLeave();
          }
        }
      >

        <boxGeometry
          args={[
            size[0],
            size[1],
            size[2],
          ]}
        />


        <meshStandardMaterial
          color={
            color
          }

          transparent

          opacity={
            opacity
          }

          roughness={
            0.75
          }

          metalness={
            0.02
          }
        />


        <Edges
          color={
            selected
              ? "#344f68"
              : config.border
          }
        />

      </mesh>


      {isShared && (

        <mesh
          scale={[
            1.025,
            1.025,
            1.025,
          ]}
        >

          <boxGeometry
            args={[
              size[0],
              size[1],
              size[2],
            ]}
          />


          <meshBasicMaterial
            transparent

            opacity={
              0
            }
          />


          <Edges
            color="#866aa8"
          />

        </mesh>

      )}


      <Html
        position={[
          0,
          size[1] / 2
            + 0.13,
          0,
        ]}

        center

        distanceFactor={
          10
        }

        style={{
          pointerEvents:
            "none",

          userSelect:
            "none",
        }}
      >

        <div className="execution-sc-label">
          SC-{id}
        </div>

      </Html>


      {state !== "idle" && (

        <Html
          position={[
            0,
            0,
            size[2] / 2
              + 0.05,
          ]}

          center

          distanceFactor={
            9
          }

          style={{
            pointerEvents:
              "none",
          }}
        >

          <div
            className="execution-state-label"

            style={{
              borderColor:
                config.border,

              color:
                config.border,
            }}
          >
            {config.label}
          </div>

        </Html>

      )}


      {hovered && (

        <Html
          position={[
            size[0] / 2
              + 0.15,

            0.3,

            0,
          ]}

          distanceFactor={
            8
          }

          style={{
            pointerEvents:
              "none",
          }}
        >

          <div className="execution-hover-card">

            <div className="execution-hover-title">
              SC-{id}
            </div>


            <HoverRow
              label="State"

              value={
                config.label
              }
            />


            {task && (

              <>

                <HoverRow
                  label="Expert"

                  value={
                    `E${task.expert_id}`
                  }
                />


                <HoverRow
                  label="Matrix"

                  value={
                    task.matrix_name
                  }
                />


                <HoverRow
                  label="z"

                  value={
                    task.z
                  }
                />


                <HoverRow
                  label="Cube"

                  value={
                    task.cube_id
                  }
                />


                <HoverRow
                  label="Type"

                  value={
                    task.is_shared
                      ? "Shared"
                      : "Routed"
                  }
                />

              </>

            )}

          </div>

        </Html>

      )}

    </group>
  );
}


// ============================================================
// Global 外框
// ============================================================


function GlobalFrame({
  width,
  height,
  depth,
}) {

  return (
    <mesh>

      <boxGeometry
        args={[
          width,
          height,
          depth,
        ]}
      />


      <meshBasicMaterial
        transparent

        opacity={
          0.015
        }

        depthWrite={
          false
        }
      />


      <Edges
        color="#7f8993"
      />

    </mesh>
  );
}


// ============================================================
// Scene
// ============================================================


function ExecutionScene({
  subcubeStates,
  selectedSubcube,
  setSelectedSubcube,
}) {

  const [
    hoveredSubcube,
    setHoveredSubcube,
  ] = useState(null);


  const layout =
    useMemo(
      () => {

        const columns =
          4;


        const rows =
          4;


        const width =
          1.22;


        const depth =
          1.22;


        const height =
          3.8;


        const gap =
          0.18;


        const totalWidth =
          columns * width
          +
          (
            columns - 1
          ) * gap;


        const totalDepth =
          rows * depth
          +
          (
            rows - 1
          ) * gap;


        const items =
          [];


        for (
          let id = 0;
          id < 16;
          id += 1
        ) {

          const row =
            Math.floor(
              id / columns
            );


          const column =
            id % columns;


          const x =
            -totalWidth / 2
            +
            width / 2
            +
            column
            * (
              width + gap
            );


          const z =
            -totalDepth / 2
            +
            depth / 2
            +
            row
            * (
              depth + gap
            );


          items.push(
            {
              id,

              position: [
                x,
                0,
                z,
              ],

              size: [
                width,
                height,
                depth,
              ],
            }
          );
        }


        return {
          items,

          frameWidth:
            totalWidth + 0.4,

          frameDepth:
            totalDepth + 0.4,

          frameHeight:
            height + 0.4,
        };

      },
      []
    );


  return (
    <>

      <ambientLight
        intensity={
          1.2
        }
      />


      <directionalLight
        position={[
          6,
          8,
          7,
        ]}

        intensity={
          1.35
        }
      />


      <directionalLight
        position={[
          -5,
          3,
          -4,
        ]}

        intensity={
          0.4
        }
      />


      <GlobalFrame
        width={
          layout.frameWidth
        }

        height={
          layout.frameHeight
        }

        depth={
          layout.frameDepth
        }
      />


      {layout.items.map(
        (item) => {

          const stateInfo =
            subcubeStates?.[
              item.id
            ] ?? {
              state: "idle",
              task: null,
            };


          return (
            <ExecutionSubCube
              key={
                item.id
              }

              id={
                item.id
              }

              position={
                item.position
              }

              size={
                item.size
              }

              stateInfo={
                stateInfo
              }

              hovered={
                hoveredSubcube ===
                item.id
              }

              selected={
                selectedSubcube ===
                item.id
              }

              onHover={
                setHoveredSubcube
              }

              onLeave={() =>
                setHoveredSubcube(
                  null
                )
              }

              onSelect={
                setSelectedSubcube
              }
            />
          );
        }
      )}


      <OrbitControls
        enableDamping={
          false
        }

        enableZoom={
          true
        }

        enableRotate={
          true
        }

        enablePan={
          false
        }

        minDistance={
          7
        }

        maxDistance={
          15
        }

        zoomToCursor={
          false
        }

        maxPolarAngle={
          Math.PI * 0.49
        }

        target={[
          0,
          0,
          0,
        ]}
      />

    </>
  );
}


// ============================================================
// Legend
// ============================================================


function ExecutionLegend() {

  return (
    <div className="execution-cube-legend">

      <LegendItem
        state="idle"
        label="Idle"
      />


      <LegendItem
        state="switch"
        label="Switch"
      />


      <LegendItem
        state="gate"
        label="gate"
      />


      <LegendItem
        state="up"
        label="up"
      />


      <LegendItem
        state="down"
        label="down"
      />


      <div className="execution-shared-legend">

        <span />

        Shared

      </div>

    </div>
  );
}


function LegendItem({
  state,
  label,
}) {

  const config =
    getStateConfig(
      state
    );


  return (
    <div className="execution-legend-item">

      <span
        style={{
          background:
            config.color,

          borderColor:
            config.border,
        }}
      />

      {label}

    </div>
  );
}


// ============================================================
// ExecutionCube3D
// ============================================================


function ExecutionCube3D({
  subcubeStates = [],
  currentCycle = 0,
  mode = "layer",
  currentLayer = 0,
  localCycle = 0,
}) {

  const [
    selectedSubcube,
    setSelectedSubcube,
  ] = useState(null);


  const activeSubcubeCount =
    useMemo(
      () => {

        let count =
          0;


        for (
          let sc = 0;
          sc < 16;
          sc += 1
        ) {

          const state =
            subcubeStates?.[
              sc
            ]?.state;


          if (
            state &&
            state !== "idle"
          ) {

            count += 1;
          }
        }


        return count;

      },
      [
        subcubeStates
      ]
    );


  return (
    <div className="execution-cube-root">

      <div className="execution-cube-header">

        <div>

          <div className="execution-cube-small">
            3D EXECUTION
          </div>


          <h3>
            {
              mode === "full"
                ? "Full Token"
                : `Layer ${currentLayer}`
            }
          </h3>


          <p>
            16 个 Sub-Cube 的当前执行状态。
          </p>

        </div>


        <div className="execution-cycle-info">

          {mode === "full" && (

            <div>

              <span>
                Layer
              </span>

              <strong>
                L{currentLayer}
              </strong>

            </div>

          )}


          <div>

            <span>
              {
                mode === "full"
                  ? "Global"
                  : "Cycle"
              }
            </span>

            <strong>
              {currentCycle}
            </strong>

          </div>


          {mode === "full" && (

            <div>

              <span>
                Local
              </span>

              <strong>
                {localCycle}
              </strong>

            </div>

          )}


          <div>

            <span>
              Active
            </span>

            <strong>
              {activeSubcubeCount}
            </strong>

          </div>

        </div>

      </div>


      <div className="execution-canvas">

        <Canvas
          camera={{
            position: [
              8,
              6.5,
              9,
            ],

            fov: 38,

            near: 0.1,

            far: 100,
          }}

          dpr={[
            1,
            1.5,
          ]}

          gl={{
            antialias: true,
            alpha: true,
          }}
        >

          <ExecutionScene
            subcubeStates={
              subcubeStates
            }

            selectedSubcube={
              selectedSubcube
            }

            setSelectedSubcube={
              setSelectedSubcube
            }
          />

        </Canvas>

      </div>


      <ExecutionLegend />


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

        .execution-cube-root {
          width: 100%;

          margin-top: 12px;

          padding-top: 12px;

          border-top: 1px solid #e4e8ec;
        }


        .execution-cube-header {
          margin-bottom: 9px;

          display: flex;

          align-items: flex-start;

          justify-content: space-between;

          gap: 8px;
        }


        .execution-cube-small {
          margin-bottom: 4px;

          color: #9aa3ad;

          font-size: 10px;

          font-weight: 650;

          letter-spacing: 1px;
        }


        .execution-cube-header h3 {
          margin: 0 0 4px;

          color: #34404d;

          font-size: 16px;
        }


        .execution-cube-header p {
          margin: 0;

          color: #929ca7;

          font-size: 10px;
        }


        .execution-cycle-info {
          display: flex;

          gap: 5px;

          flex-wrap: wrap;

          justify-content: flex-end;
        }


        .execution-cycle-info > div {
          min-width: 48px;

          padding: 5px 6px;

          border: 1px solid #dce2e8;

          border-radius: 4px;

          background: #f8fafb;
        }


        .execution-cycle-info span {
          display: block;

          margin-bottom: 2px;

          color: #929ca7;

          font-size: 9px;
        }


        .execution-cycle-info strong {
          color: #40505f;

          font-size: 12px;
        }


        .execution-canvas {
          width: 100%;

          height: 465px;

          overflow: hidden;

          border: 1px solid #dfe4e9;

          border-radius: 6px;

          background: #ffffff;
        }


        .execution-cube-legend {
          min-height: 31px;

          margin-top: 6px;

          padding: 0 7px;

          display: flex;

          align-items: center;

          flex-wrap: wrap;

          gap: 9px;

          border: 1px solid #e0e4e8;

          border-radius: 5px;

          background: #ffffff;

          color: #7f8993;

          font-size: 9px;
        }


        .execution-legend-item,
        .execution-shared-legend {
          display: flex;

          align-items: center;

          gap: 4px;
        }


        .execution-legend-item > span,
        .execution-shared-legend span {
          width: 11px;

          height: 11px;

          display: inline-block;
        }


        .execution-legend-item > span {
          border: 1px solid;
        }


        .execution-shared-legend span {
          border: 2px solid #866aa8;

          background: #ffffff;
        }


        .execution-sc-label {
          min-width: 39px;

          padding: 2px 4px;

          border: 1px solid #d5dbe1;

          border-radius: 3px;

          background: rgba(255,255,255,0.92);

          color: #697582;

          font-size: 9px;

          font-weight: 700;

          text-align: center;

          white-space: nowrap;
        }


        .execution-state-label {
          min-width: 43px;

          padding: 2px 4px;

          border: 1px solid;

          border-radius: 3px;

          background: rgba(255,255,255,0.94);

          font-size: 9px;

          font-weight: 700;

          text-align: center;

          text-transform: uppercase;
        }


        .execution-hover-card {
          width: 160px;

          padding: 8px 9px;

          border: 1px solid #d7dde4;

          border-radius: 5px;

          background: rgba(255,255,255,0.97);

          box-shadow: 0 5px 16px rgba(0,0,0,0.08);

          color: #4b5563;

          font-family:
            Inter,
            "Microsoft YaHei",
            Arial,
            sans-serif;

          font-size: 10px;
        }


        .execution-hover-title {
          margin-bottom: 6px;

          color: #33404c;

          font-size: 12px;

          font-weight: 700;
        }


        .execution-hover-row {
          min-height: 18px;

          display: flex;

          align-items: center;

          justify-content: space-between;

          gap: 9px;

          border-bottom: 1px solid #f0f2f4;
        }


        .execution-hover-row span {
          color: #919aa4;
        }


        .execution-hover-row strong {
          color: #3e4b58;

          font-weight: 600;
        }
        /* ================================================
          3D EXECUTION FONT ENLARGE
        ================================================ */


        /* 3D EXECUTION */
        .execution-cube-small {
          font-size: 13px;
        }


        /* Layer 0 / Full Token */
        .execution-cube-header h3 {
          font-size: 20px;
        }


        /* 16 个 Sub-Cube 的当前执行状态 */
        .execution-cube-header p {
          font-size: 13px;
        }


        /* 右上角 Cycle / Active 的标题 */
        .execution-cycle-info span {
          font-size: 11px;
        }


        /* 右上角 Cycle / Active 的数字 */
        .execution-cycle-info strong {
          font-size: 16px;
        }


        /* 右上角卡片稍微放大 */
        .execution-cycle-info > div {
          min-width: 60px;
          padding: 8px 10px;
        }


        /* 3D Cube 上面的 SC-0 / SC-1 标签 */
        .execution-sc-label {
          min-width: 46px;
          padding: 4px 6px;
          font-size: 11px;
        }


        /* SWITCH / GATE / UP / DOWN */
        .execution-state-label {
          min-width: 52px;
          padding: 4px 6px;
          font-size: 11px;
        }


        /* 底部图例文字 */
        .execution-cube-legend {
          min-height: 42px;
          gap: 14px;
          padding: 0 12px;
          font-size: 12px;
        }


        /* 底部图例色块 */
        .execution-legend-item > span,
        .execution-shared-legend span {
          width: 15px;
          height: 15px;
        }
      `}
    </style>
  );
}


export default ExecutionCube3D;
