import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API_BASE =
  "http://127.0.0.1:8000";


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


function matrixLabel(
  matrixName
) {
  const name =
    normalizeMatrixName(
      matrixName
    );

  if (name === "gate") {
    return "Gate / 门控投影";
  }

  if (name === "up") {
    return "Up / 上投影";
  }

  if (name === "down") {
    return "Down / 下投影";
  }

  return name;
}


function MappingLocator({
  layerCount = 58,
  onLocate,
}) {
  const [
    layerId,
    setLayerId,
  ] = useState(0);

  const [
    layerData,
    setLayerData,
  ] = useState(null);

  const [
    expertKey,
    setExpertKey,
  ] = useState("");

  const [
    matrixName,
    setMatrixName,
  ] = useState("gate");

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState("");


  useEffect(() => {
    const controller =
      new AbortController();

    async function loadLayer() {
      try {
        setLoading(true);
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

        setLayerData(data);

      } catch (err) {
        if (
          err.name ===
          "AbortError"
        ) {
          return;
        }

        console.error(
          "Load mapping layer failed:",
          err
        );

        setError(
          `无法读取 Layer ${layerId} 的映射。`
        );

        setLayerData(null);

      } finally {
        if (
          !controller.signal.aborted
        ) {
          setLoading(false);
        }
      }
    }

    loadLayer();

    return () => {
      controller.abort();
    };

  }, [layerId]);


  const weights =
    useMemo(
      () => {
        const subcubes =
          layerData?.subcubes ?? [];

        return subcubes.flatMap(
          (subcube) =>
            subcube.weights ?? []
        );
      },
      [layerData]
    );


  const experts =
    useMemo(
      () => {
        const map =
          new Map();

        for (const weight of weights) {
          const key =
            `${weight.expert_id}|${weight.is_shared ? 1 : 0}`;

          if (!map.has(key)) {
            map.set(
              key,
              {
                key,
                expert_id:
                  weight.expert_id,
                is_shared:
                  Boolean(
                    weight.is_shared
                  ),
              }
            );
          }
        }

        return Array.from(
          map.values()
        ).sort(
          (a, b) => {
            if (
              a.is_shared !==
              b.is_shared
            ) {
              return a.is_shared
                ? 1
                : -1;
            }

            return (
              a.expert_id -
              b.expert_id
            );
          }
        );
      },
      [weights]
    );


  useEffect(() => {
    if (
      experts.length === 0
    ) {
      setExpertKey("");
      return;
    }

    const exists =
      experts.some(
        (expert) =>
          expert.key ===
          expertKey
      );

    if (!exists) {
      setExpertKey(
        experts[0].key
      );
    }
  }, [experts, expertKey]);


  const selectedExpert =
    useMemo(
      () =>
        experts.find(
          (expert) =>
            expert.key ===
            expertKey
        ) ?? null,
      [experts, expertKey]
    );


  const matrixOptions =
    useMemo(
      () => {
        if (!selectedExpert) {
          return [];
        }

        const names =
          new Set();

        for (const weight of weights) {
          if (
            weight.expert_id ===
              selectedExpert.expert_id
            &&
            Boolean(weight.is_shared) ===
              selectedExpert.is_shared
          ) {
            names.add(
              normalizeMatrixName(
                weight.matrix_name
              )
            );
          }
        }

        const order = [
          "gate",
          "up",
          "down",
        ];

        return Array.from(
          names
        ).sort(
          (a, b) =>
            order.indexOf(a) -
            order.indexOf(b)
        );
      },
      [weights, selectedExpert]
    );


  useEffect(() => {
    if (
      matrixOptions.length === 0
    ) {
      return;
    }

    if (
      !matrixOptions.includes(
        matrixName
      )
    ) {
      setMatrixName(
        matrixOptions[0]
      );
    }
  }, [matrixOptions, matrixName]);


  const matches =
    useMemo(
      () => {
        if (!selectedExpert) {
          return [];
        }

        return weights.filter(
          (weight) =>
            weight.expert_id ===
              selectedExpert.expert_id
            &&
            Boolean(weight.is_shared) ===
              selectedExpert.is_shared
            &&
            normalizeMatrixName(
              weight.matrix_name
            ) === matrixName
        ).sort(
          (a, b) =>
            (a.cube_id ?? 0) -
            (b.cube_id ?? 0)
        );
      },
      [
        weights,
        selectedExpert,
        matrixName,
      ]
    );


  return (
    <section className="mapping-locator">
      <div className="mapping-locator-title">
        <div>
          <strong>
            权重快速定位
          </strong>
          <span>
            Mapping Locator
          </span>
        </div>

        <small>
          按层、Expert 和矩阵直接定位到物理 Plane
        </small>
      </div>


      <div className="mapping-locator-controls">
        <label>
          <span>
            模型层 / Layer
          </span>

          <select
            value={layerId}
            onChange={
              (event) =>
                setLayerId(
                  Number(
                    event.target.value
                  )
                )
            }
          >
            {Array.from(
              {
                length:
                  Math.max(
                    Number(layerCount) || 58,
                    1
                  ),
              },
              (_, index) => (
                <option
                  key={index}
                  value={index}
                >
                  Layer {index}
                </option>
              )
            )}
          </select>
        </label>


        <label>
          <span>
            专家 / Expert
          </span>

          <select
            value={expertKey}
            disabled={
              loading ||
              experts.length === 0
            }
            onChange={
              (event) =>
                setExpertKey(
                  event.target.value
                )
            }
          >
            {experts.map(
              (expert) => (
                <option
                  key={expert.key}
                  value={expert.key}
                >
                  {expert.is_shared
                    ? `Shared Expert ${expert.expert_id}`
                    : `Expert ${expert.expert_id}`
                  }
                </option>
              )
            )}
          </select>
        </label>


        <label>
          <span>
            矩阵 / Matrix
          </span>

          <select
            value={matrixName}
            disabled={
              loading ||
              matrixOptions.length === 0
            }
            onChange={
              (event) =>
                setMatrixName(
                  event.target.value
                )
            }
          >
            {matrixOptions.map(
              (name) => (
                <option
                  key={name}
                  value={name}
                >
                  {matrixLabel(name)}
                </option>
              )
            )}
          </select>
        </label>
      </div>


      <div className="mapping-locator-result">
        {loading && (
          <div className="locator-state">
            正在读取 Layer {layerId}...
          </div>
        )}

        {!loading && error && (
          <div className="locator-state error">
            {error}
          </div>
        )}

        {!loading && !error &&
          matches.length === 0 && (
          <div className="locator-state">
            当前条件下没有找到 Weight-Cube。
          </div>
        )}

        {!loading && !error &&
          matches.map(
            (weight, index) => (
              <div
                className="locator-hit"
                key={weight.cube_id}
              >
                <div className="locator-hit-main">
                  <strong>
                    {matches.length > 1
                      ? `块 ${index + 1}`
                      : "目标权重"
                    }
                  </strong>

                  <span>
                    SC-{weight.subcube_id}
                    {" · "}
                    z={weight.z}
                    {" · "}
                    Plane {weight.physical_plane_id ?? "--"}
                    {" · "}
                    Slot {weight.slot_id ?? "--"}
                  </span>
                </div>

                <button
                  onClick={() =>
                    onLocate?.(weight)
                  }
                >
                  定位并打开
                </button>
              </div>
            )
          )
        }
      </div>

      <Style />
    </section>
  );
}


function Style() {
  return (
    <style>
      {`
        .mapping-locator {
          margin: 8px 0 12px;
          padding: 10px 12px;
          display: grid;
          grid-template-columns: 190px minmax(430px, 1fr) minmax(330px, 0.9fr);
          gap: 12px;
          align-items: center;
          border: 1px solid #dce3e9;
          border-radius: 7px;
          background: #ffffff;
        }

        .mapping-locator-title strong {
          display: block;
          color: #000000;
          font-size: 16px;
          line-height: 1.25;
        }

        .mapping-locator-title span {
          display: block;
          margin-top: 2px;
          color: #8794a1;
          font-size: 15px;
          font-weight: 600;
        }

        .mapping-locator-title small {
          display: block;
          margin-top: 6px;
          color: #87929d;
          font-size: 15px;
          line-height: 1.4;
        }

        .mapping-locator-controls {
          display: grid;
          grid-template-columns: repeat(3, minmax(120px, 1fr));
          gap: 8px;
        }

        .mapping-locator-controls label {
          min-width: 0;
        }

        .mapping-locator-controls label > span {
          display: block;
          margin-bottom: 5px;
          color: #6f7b88;
          font-size: 16px;
          font-weight: 650;
        }

        .mapping-locator-controls select {
          width: 100%;
          height: 36px;
          padding: 0 9px;
          border: 1px solid #d4dce4;
          border-radius: 5px;
          background: #fbfcfd;
          color: #000000;
          font-size: 16px;
          font-weight: 550;
          outline: none;
        }

        .mapping-locator-controls select:focus {
          border-color: #000000;
          box-shadow: 0 0 0 2px rgba(92, 130, 166, 0.12);
        }

        .mapping-locator-result {
          min-width: 0;
          max-height: 92px;
          overflow-y: auto;
        }

        .locator-state {
          min-height: 42px;
          display: flex;
          align-items: center;
          padding: 0 10px;
          border: 1px dashed #d8e0e7;
          border-radius: 5px;
          background: #fafbfd;
          color: #73808d;
          font-size: 16px;
        }

        .locator-state.error {
          color: #9b5656;
          border-color: #e2c1c1;
          background: #fff7f7;
        }

        .locator-hit {
          min-height: 42px;
          padding: 5px 6px 5px 9px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          border: 1px solid #dce3e9;
          border-radius: 5px;
          background: #f8fafc;
        }

        .locator-hit + .locator-hit {
          margin-top: 5px;
        }

        .locator-hit-main {
          min-width: 0;
        }

        .locator-hit-main strong {
          display: block;
          color: #000000;
          font-size: 16px;
        }

        .locator-hit-main span {
          display: block;
          margin-top: 2px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          color: #73808c;
          font-size: 15px;
        }

        .locator-hit button {
          flex: 0 0 auto;
          height: 32px;
          padding: 0 11px;
          border: 1px solid #668aaa;
          border-radius: 5px;
          background: #7196b8;
          color: #ffffff;
          font-size: 16px;
          font-weight: 650;
          cursor: pointer;
        }

        .locator-hit button:hover {
          background: #6288aa;
        }

        @media (max-width: 1250px) {
          .mapping-locator {
            grid-template-columns: 170px minmax(390px, 1fr);
          }

          .mapping-locator-result {
            grid-column: 1 / -1;
            max-height: 80px;
          }
        }
      `}
    </style>
  );
}


export default MappingLocator;
