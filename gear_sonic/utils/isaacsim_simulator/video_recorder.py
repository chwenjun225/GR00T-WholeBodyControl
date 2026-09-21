"""Non-blocking MP4 writer for the Isaac Sim control process."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import queue

import numpy as np


def _video_writer_worker(
    shm_name: str,
    shape: tuple[int, int, int],
    output_path: str,
    fps: float,
    fourcc: str,
    lock,
    frame_ready,
    stop_event,
    status_queue,
) -> None:
    """Copy frames from shared memory and encode them outside Isaac Sim."""
    shm = None
    writer = None
    try:
        import cv2

        shm = shared_memory.SharedMemory(name=shm_name)
        shared_frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (shape[1], shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open MP4 writer: {output_path}")
        status_queue.put(("ready", ""))

        while True:
            available = frame_ready.wait(timeout=0.1)
            if available:
                with lock:
                    rgb = shared_frame.copy()
                    frame_ready.clear()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            elif stop_event.is_set():
                break
    except BaseException as exc:
        status_queue.put(("error", repr(exc)))
    finally:
        if writer is not None:
            writer.release()
        if shm is not None:
            shm.close()


class AsyncVideoRecorder:
    """Single-slot producer/consumer recorder that drops frames instead of blocking."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        fourcc: str = "mp4v",
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.shape = (height, width, 3)
        self.submitted = 0
        self.dropped = 0
        self._closed = False

        context = mp.get_context("spawn")
        self._shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.shape)))
        self._shared_frame = np.ndarray(self.shape, dtype=np.uint8, buffer=self._shm.buf)
        self._lock = context.Lock()
        self._frame_ready = context.Event()
        self._stop_event = context.Event()
        self._status_queue = context.Queue(maxsize=2)
        self._process = context.Process(
            target=_video_writer_worker,
            args=(
                self._shm.name,
                self.shape,
                str(self.output_path),
                fps,
                fourcc,
                self._lock,
                self._frame_ready,
                self._stop_event,
                self._status_queue,
            ),
            name="isaacsim-video-recorder",
            daemon=True,
        )
        self._process.start()
        try:
            state, detail = self._status_queue.get(timeout=10.0)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("Video recorder process did not start within 10 seconds") from exc
        if state != "ready":
            self.close()
            raise RuntimeError(f"Video recorder failed to start: {detail}")

    def submit(self, rgb: np.ndarray) -> bool:
        """Submit one RGB frame without waiting; return False when it is dropped."""
        if self._closed or not self._process.is_alive():
            raise RuntimeError("Video recorder process is not running")
        frame = np.asarray(rgb)
        if frame.shape != self.shape:
            raise ValueError(f"Expected video frame {self.shape}, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if self._frame_ready.is_set() or not self._lock.acquire(block=False):
            self.dropped += 1
            return False
        try:
            np.copyto(self._shared_frame, frame)
            self.submitted += 1
            self._frame_ready.set()
            return True
        finally:
            self._lock.release()

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._process.is_alive():
            self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._status_queue.close()
        self._status_queue.join_thread()
        self._shm.close()
        self._shm.unlink()

    def stats(self) -> dict[str, int]:
        return {"submitted": self.submitted, "dropped": self.dropped}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
