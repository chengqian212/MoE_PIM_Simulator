import { useMemo, useState } from "react";

import {
  Canvas,
} from "@react-three/fiber";

import {
  OrbitControls,
  Edges,
  Html,
} from "@react-three/drei";


/*
============================================================
GlobalCube3D

作用：

1. 显示整个 Global Cube；
2. 内部显示 16 个 Sub-Cube；
3. 采用 4 × 4 排列；
4. 鼠标可以旋转、缩放；
5. 鼠标悬停显示 SC 信息；
6. 点击某个 SC 后高亮；
7. 后面可以继续接：
      - Sub-Cube Plane 查看
      - Token 执行状态
      - active / waiting / switching 状态

注意：

N=4 表示：

    4 × 4 = 16 个 Sub-Cube

不是：

    4 × 4 × 4

Sub-Cube 自身的竖直方向用于表现：

    Depth D
============================================================
*/


// ============================================================
// 一个 Sub-Cube
// ============================================================

function SubCube({
  id,

  position,

  size,

  selected,

  hovered,

  onClick,

  onPointerEnter,

  onPointerLeave,

  info,
}) {
  /*
  ------------------------------------------------------------
  默认颜色

  不做很花的渐变。

  selected：
      选中的 SC

  hovered：
      鼠标当前指向的 SC
  ------------------------------------------------------------
  */

  let color = "#cbd5e1";

  if (hovered) {
    color = "#9fb8d3";
  }

  if (selected) {
    color = "#6f94bc";
  }


  return (
    <group
      position={position}
    >
      {/* =====================================================
          Sub-Cube 本体
      ====================================================== */}

      <mesh
        onClick={(event) => {
          event.stopPropagation();

          onClick(id);
        }}

        onPointerEnter={(event) => {
          event.stopPropagation();

          document.body.style.cursor =
            "pointer";

          onPointerEnter(id);
        }}

        onPointerLeave={(event) => {
          event.stopPropagation();

          document.body.style.cursor =
            "default";

          onPointerLeave();
        }}
      >
        <boxGeometry
          args={[
            size[0],
            size[1],
            size[2],
          ]}
        />

        <meshStandardMaterial
          color={color}
          transparent
          opacity={
            selected
              ? 0.88
              : 0.68
          }
          roughness={0.75}
          metalness={0.05}
        />

        <Edges
          color={
            selected
              ? "#315d89"
              : "#738395"
          }
          threshold={15}
        />
      </mesh>


      {/* =====================================================
          SC 编号
      ====================================================== */}

      <Html
        position={[
          0,
          size[1] / 2 + 0.12,
          0,
        ]}
        center
        distanceFactor={10}
        style={{
          pointerEvents: "none",
          userSelect: "none",
        }}
      >
        <div
          style={{
            color: selected
              ? "#1f4f7b"
              : "#505b68",

            fontSize: "13px",

            fontWeight: selected
              ? 700
              : 600,

            whiteSpace: "nowrap",

            background:
              "rgba(255,255,255,0.85)",

            border:
              "1px solid rgba(210,216,224,0.9)",

            borderRadius: "3px",

            padding: "2px 4px",
          }}
        >
          SC-{id}
        </div>
      </Html>


      {/* =====================================================
          Hover 信息
      ====================================================== */}

      {hovered && (
        <Html
          position={[
            size[0] / 2 + 0.15,
            0.25,
            0,
          ]}
          center={false}
          distanceFactor={8}
          style={{
            pointerEvents: "none",
          }}
        >
          <div
            style={{
              width: "150px",

              padding: "9px 10px",

              background:
                "rgba(255,255,255,0.96)",

              border:
                "1px solid #d7dde5",

              borderRadius: "5px",

              boxShadow:
                "0 5px 18px rgba(0,0,0,0.08)",

              fontFamily:
                'Inter, "Microsoft YaHei", Arial, sans-serif',

              fontSize: "13px",

              color: "#4b5563",
            }}
          >
            <div
              style={{
                fontSize: "14px",
                fontWeight: 700,
                marginBottom: "7px",
                color: "#2f3b48",
              }}
            >
              Sub-Cube {id}
            </div>

            <InfoLine
              label="Planes"
              value={
                info?.used_planes ?? "--"
              }
            />

            <InfoLine
              label="Depth"
              value={
                info?.depth_capacity ??
                "--"
              }
            />

            <InfoLine
              label="Weights"
              value={
                info?.weight_cube_count ??
                "--"
              }
            />

            <InfoLine
              label="Empty"
              value={
                info?.empty_planes ??
                "--"
              }
            />
          </div>
        </Html>
      )}
    </group>
  );
}


// ============================================================
// Hover 信息字段
// ============================================================

function InfoLine({
  label,
  value,
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent:
          "space-between",

        marginTop: "4px",
      }}
    >
      <span
        style={{
          color: "#8a94a1",
        }}
      >
        {label}
      </span>

      <strong
        style={{
          color: "#394553",
          fontWeight: 600,
        }}
      >
        {value}
      </strong>
    </div>
  );
}


// ============================================================
// Global Cube 外框
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
        color="#8b98a7"
        transparent
        opacity={0.025}
        depthWrite={false}
      />

      <Edges
        color="#7d8996"
      />
    </mesh>
  );
}


// ============================================================
// Ground
// ============================================================

function Ground({
  size,
}) {
  return (
    <mesh
      rotation={[
        -Math.PI / 2,
        0,
        0,
      ]}
      position={[
        0,
        -2.32,
        0,
      ]}
    >
      <planeGeometry
        args={[
          size,
          size,
        ]}
      />

      <meshStandardMaterial
        color="#eef1f4"
        roughness={1}
      />
    </mesh>
  );
}


// ============================================================
// 3D Scene
// ============================================================

function CubeScene({
  subcubes,

  selectedSubcube,

  setSelectedSubcube,
}) {
  const [
    hoveredSubcube,
    setHoveredSubcube,
  ] = useState(null);


  /*
  ------------------------------------------------------------
  16 个 SC：

      SC0  SC1  SC2  SC3
      SC4  SC5  SC6  SC7
      SC8  SC9  SC10 SC11
      SC12 SC13 SC14 SC15

  映射到：

      X × Z

  Y 方向用于显示 Sub-Cube 的 Depth。
  ------------------------------------------------------------
  */

  const cubeData = useMemo(
    () => {
      const result = [];

      const columns = 4;
      const rows = 4;

      const subWidth = 1.25;
      const subDepth = 1.25;

      const subHeight = 4.0;

      const gap = 0.18;

      const totalWidth =
        columns * subWidth +
        (columns - 1) * gap;

      const totalDepth =
        rows * subDepth +
        (rows - 1) * gap;


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
          -totalWidth / 2 +
          subWidth / 2 +
          column *
            (
              subWidth +
              gap
            );


        const z =
          -totalDepth / 2 +
          subDepth / 2 +
          row *
            (
              subDepth +
              gap
            );


        result.push({
          id,

          position: [
            x,
            0,
            z,
          ],

          size: [
            subWidth,
            subHeight,
            subDepth,
          ],

          info:
            subcubes?.find(
              (item) =>
                item.subcube_id ===
                id
            ) ?? null,
        });
      }


      return {
        items: result,

        frameWidth:
          totalWidth + 0.45,

        frameDepth:
          totalDepth + 0.45,

        frameHeight:
          subHeight + 0.45,
      };
    },

    [
      subcubes
    ]
  );


  return (
    <>
      {/* =====================================================
          灯光
      ====================================================== */}

      <ambientLight
        intensity={1.1}
      />

      <directionalLight
        position={[
          6,
          9,
          8,
        ]}
        intensity={1.4}
      />

      <directionalLight
        position={[
          -5,
          3,
          -4,
        ]}
        intensity={0.45}
      />


      {/* =====================================================
          Global Cube 外框
      ====================================================== */}

      <GlobalFrame
        width={
          cubeData.frameWidth
        }

        height={
          cubeData.frameHeight
        }

        depth={
          cubeData.frameDepth
        }
      />


      {/* =====================================================
          16 个 Sub-Cube
      ====================================================== */}

      {cubeData.items.map(
        (item) => (
          <SubCube
            key={item.id}

            id={item.id}

            position={
              item.position
            }

            size={
              item.size
            }

            info={
              item.info
            }

            selected={
              selectedSubcube ===
              item.id
            }

            hovered={
              hoveredSubcube ===
              item.id
            }

            onClick={
              setSelectedSubcube
            }

            onPointerEnter={
              setHoveredSubcube
            }

            onPointerLeave={() =>
              setHoveredSubcube(
                null
              )
            }
          />
        )
      )}


      {/* =====================================================
          地面
      ====================================================== */}

      <Ground
        size={9}
      />


      {/* =====================================================
          鼠标控制
      ====================================================== */}

      <OrbitControls
        enableDamping

        dampingFactor={0.08}

        minDistance={6}

        maxDistance={18}

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
// 对外组件
// ============================================================

function GlobalCube3D({
  subcubes = [],

  selectedSubcube: externalSelectedSubcube,

  onSelectSubcube,
}) {
  /*
  ------------------------------------------------------------
  支持两种使用方式：

  1. 父组件不管理 selected：
       组件自己管理；

  2. 父组件传入 selectedSubcube：
       由父组件管理。

  后面我们做 Sub-Cube 下钻时会用第二种。
  ------------------------------------------------------------
  */

  const [
    internalSelected,
    setInternalSelected,
  ] = useState(null);


  const selectedSubcube =
    externalSelectedSubcube !==
    undefined
      ? externalSelectedSubcube
      : internalSelected;


  function handleSelect(
    subcubeId
  ) {
    setInternalSelected(
      subcubeId
    );

    if (onSelectSubcube) {
      onSelectSubcube(
        subcubeId
      );
    }
  }


  return (
    <div
      style={{
        position: "relative",

        width: "100%",
        height: "100%",

        minHeight: "540px",

        background: "#ffffff",
      }}
    >
      {/* =====================================================
          左上角说明
      ====================================================== */}

      <div
        style={{
          position: "absolute",

          left: "16px",
          top: "14px",

          zIndex: 10,

          padding:
            "7px 9px",

          background:
            "rgba(255,255,255,0.9)",

          border:
            "1px solid #e0e5ea",

          borderRadius: "4px",

          fontSize: "13px",

          lineHeight: 1.7,

          color: "#7a8490",

          pointerEvents: "none",
        }}
      >
        <div>
          Drag：旋转
        </div>

        <div>
          Wheel：缩放
        </div>

        <div>
          Click：选择 Sub-Cube
        </div>
      </div>


      {/* =====================================================
          当前选中
      ====================================================== */}

      {selectedSubcube !==
        null && (
        <div
          style={{
            position:
              "absolute",

            right: "16px",
            top: "14px",

            zIndex: 10,

            padding:
              "8px 11px",

            background:
              "#ffffff",

            border:
              "1px solid #dce2e8",

            borderRadius: "4px",

            fontSize: "14px",

            color: "#4a5563",
          }}
        >
          Selected：
          <strong
            style={{
              marginLeft: "5px",
              color: "#315d89",
            }}
          >
            SC-
            {
              selectedSubcube
            }
          </strong>
        </div>
      )}


      {/* =====================================================
          Three.js Canvas
      ====================================================== */}

      <Canvas
        camera={{
          position: [
            8,
            7,
            9,
          ],

          fov: 38,

          near: 0.1,

          far: 100,
        }}

        dpr={[
          1,
          1.8,
        ]}

        gl={{
          antialias: true,
          alpha: true,
        }}
      >
        <CubeScene
          subcubes={
            subcubes
          }

          selectedSubcube={
            selectedSubcube
          }

          setSelectedSubcube={
            handleSelect
          }
        />
      </Canvas>
    </div>
  );
}


export default GlobalCube3D;