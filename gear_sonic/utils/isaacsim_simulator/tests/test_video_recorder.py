import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from gear_sonic.utils.isaacsim_simulator.video_recorder import AsyncVideoRecorder


class AsyncVideoRecorderTests(unittest.TestCase):
    def test_worker_writes_mp4_without_blocking_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.mp4"
            recorder = AsyncVideoRecorder(output, fps=20, width=64, height=48)
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = 255
            self.assertTrue(recorder.submit(frame))
            # The single slot deliberately drops a frame while the worker is busy.
            recorder.submit(frame)
            time.sleep(0.1)
            recorder.submit(frame)
            recorder.close()
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertGreaterEqual(recorder.stats()["submitted"], 1)


if __name__ == "__main__":
    unittest.main()
