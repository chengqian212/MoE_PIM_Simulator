import { useState } from "react";

import PrefillRequestPanel from "./PrefillRequestPanel";
import DecodeRequestPanel from "./DecodeRequestPanel";


function RequestSimulator() {
  const [mode, setMode] = useState("prefill");

  return (
    <div className="request-simulator">
      <div className="request-header">
        <div>
          <div className="request-kicker">真实请求模拟 / REQUEST SIMULATOR</div>
          <h2>Prefill / Decode 分阶段模拟</h2>
          <p>
            使用同一套 Mapping，分别查看多 Token Prefill 与单 Token Decode 的 MoE Expert 执行结果。
          </p>
        </div>

        <div className="scope-badge">
          MoE Expert Only
          <span>非完整 TTFT / TPOT</span>
        </div>
      </div>

      <div className="mode-switch">
        <button
          className={mode === "prefill" ? "active" : ""}
          onClick={() => setMode("prefill")}
        >
          <strong>Prefill</strong>
          <span>多 Token · Segment 0</span>
        </button>

        <button
          className={mode === "decode" ? "active" : ""}
          onClick={() => setMode("decode")}
        >
          <strong>Decode</strong>
          <span>单 Token · Segment 1+</span>
        </button>
      </div>

      {mode === "prefill" ? (
        <PrefillRequestPanel />
      ) : (
        <DecodeRequestPanel />
      )}

      <Style />
    </div>
  );
}


function Style() {
  return (
    <style>{`
      .request-simulator {
        width: 100%;
      }

      .request-header {
        min-height: 62px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 20px;
      }

      .request-kicker {
        margin-bottom: 4px;
        color: #7f8995;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
      }

      .request-header h2 {
        margin: 0 0 5px;
        color: #2d3946;
        font-size: 24px;
        font-weight: 700;
      }

      .request-header p {
        margin: 0;
        color: #74808d;
        font-size: 15px;
      }

      .scope-badge {
        min-width: 180px;
        padding: 8px 12px;
        border: 1px solid #d5dde5;
        border-radius: 6px;
        background: #ffffff;
        color: #536476;
        font-size: 15px;
        font-weight: 700;
        text-align: center;
      }

      .scope-badge span {
        display: block;
        margin-top: 2px;
        color: #5f7083;
        font-size: 15px;
        font-weight: 500;
      }

      .mode-switch {
        margin-bottom: 10px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .mode-switch button {
        min-height: 58px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border: 1px solid #d9e0e7;
        border-radius: 6px;
        background: #ffffff;
        color: #687482;
        cursor: pointer;
      }

      .mode-switch button:hover {
        background: #f6f8fa;
      }

      .mode-switch button.active {
        border-color: #3b82f6;
        background: #dbeafe;
        color: #123f70;
        box-shadow: inset 4px 0 0 #4F7195;
      }

      .mode-switch strong {
        font-size: 19px;
      }

      .mode-switch span {
        font-size: 15px;
      }

      @media (max-width: 980px) {
        .request-header {
          flex-direction: column;
        }

        .scope-badge {
          width: 100%;
        }
      }
    `}</style>
  );
}


export default RequestSimulator;
