import {
  useCallback,
} from "react";

import FullTokenRunner
  from "./FullTokenRunner";


// ============================================================
// FullTokenPlaybackPanel
//
// 这里只负责把 FullTokenRunner 的播放帧继续向上转发。
// 不再重复显示 Global Cycle / Layer / Active SC，
// 因为 FullTokenRunner 自己已经显示这些信息。
// ============================================================


function FullTokenPlaybackPanel({
  token,
  onSelectLayer,
  onPlaybackFrame,
}) {

  const handlePlaybackFrame =
    useCallback(
      (frame) => {

        if (onPlaybackFrame) {

          onPlaybackFrame(
            frame
          );
        }
      },
      [
        onPlaybackFrame
      ]
    );


  return (
    <div className="full-token-playback-panel">

      <FullTokenRunner
        token={
          token
        }

        onSelectLayer={
          onSelectLayer
        }

        onPlaybackFrame={
          handlePlaybackFrame
        }
      />


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

        .full-token-playback-panel {
          width: 100%;
        }

      `}
    </style>
  );
}


export default FullTokenPlaybackPanel;
