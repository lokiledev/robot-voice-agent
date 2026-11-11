import threading

import numpy as np
import numpy.typing as npt

from rva_voice_audio.replay import ReplayedMicrophone


async def test_a_recording_is_replayed_in_whole_frames_then_its_tail_at_its_own_pace() -> None:
    started = threading.Event()
    started.set()
    heard: list[float] = []

    def on_frame(time: float, frame: npt.NDArray[np.float32]) -> None:
        heard.append(time)

    # 1100 samples: two frames of 512, the rest filled by the 512 samples of tail.
    samples = np.arange(1100, dtype=np.float32)
    microphone = ReplayedMicrophone(samples, on_frame, started, tail_s=0.032)

    frames = [frame async for frame in microphone.frames()]

    assert [len(frame) for frame in frames] == [512, 512, 512]
    assert frames[1][0] == 512.0
    assert list(frames[2][-3:]) == [0.0, 0.0, 0.0]
    # One frame every 32 ms.
    assert np.allclose(np.diff(heard), 0.032)
