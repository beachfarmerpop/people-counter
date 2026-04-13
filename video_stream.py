import os
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np

# Set RTSP/FFMPEG timeouts globally (microseconds) so cv2.VideoCapture
# never blocks for more than ~10 seconds on unreachable hosts.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;10000000|stimeout;10000000",
)


class VideoStream:
    """Threaded RTSP / USB / HTTP / file stream with non-blocking connect."""

    def __init__(
        self,
        source: Union[str, int],
        reconnect_interval: float = 2.0,
        max_retries: int = 0,
    ) -> None:
        self.source = source
        self.reconnect_interval = max(0.5, reconnect_interval)
        self.max_retries = max_retries

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._retry_count = 0
        self._last_reconnect_ts = 0.0
        self._connected = False
        self._connecting = False

    # ------ public properties ------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_connecting(self) -> bool:
        return self._connecting

    # ------ public API -------------------------------------------------------

    def start_stream(self) -> bool:
        """Start (or restart) the stream — returns immediately."""
        self.stop()
        self._running = True
        self._connecting = True
        self._connected = False
        self._retry_count = 0
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame or *None* (never blocks)."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._release_capture()
        with self._lock:
            self._frame = None
        self._connected = False
        self._connecting = False

    # ------ internals --------------------------------------------------------

    def _open_capture(self) -> bool:
        self._release_capture()
        try:
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        except Exception:
            self._retry_count += 1
            return False

        if cap is None or not cap.isOpened():
            self._retry_count += 1
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            return False

        with self._lock:
            self._cap = cap
        self._retry_count = 0
        self._connected = True
        return True

    def _reader_loop(self) -> None:
        """Background thread — connect then continuously grab frames."""
        self._open_capture()
        self._connecting = False

        while self._running:
            cap = None
            with self._lock:
                cap = self._cap

            # ---------- no capture — try reconnect --------------------------
            if cap is None or not cap.isOpened():
                self._connected = False
                now = time.time()
                if now - self._last_reconnect_ts >= self.reconnect_interval:
                    self._last_reconnect_ts = now
                    if self.max_retries == 0 or self._retry_count < self.max_retries:
                        self._connecting = True
                        self._open_capture()
                        self._connecting = False
                time.sleep(0.1)
                continue

            # ---------- read frame -----------------------------------------
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None

            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                # connection lost
                self._connected = False
                with self._lock:
                    self._cap = None
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.05)

    def _release_capture(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
