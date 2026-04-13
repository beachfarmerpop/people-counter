import time
from typing import Optional, Union

import cv2
import numpy as np


class VideoStream:
    """RTSP/video source wrapper with automatic reconnect."""

    def __init__(
        self,
        source: Union[str, int],
        reconnect_interval: float = 2.0,
        max_retries: int = 0,
        use_ffmpeg: bool = True,
    ) -> None:
        self.source = source
        self.reconnect_interval = max(0.2, reconnect_interval)
        self.max_retries = max_retries
        self.use_ffmpeg = use_ffmpeg
        self.cap: Optional[cv2.VideoCapture] = None
        self._retry_count = 0
        self._last_reconnect_ts = 0.0

    def start_stream(self) -> bool:
        return self._open_capture()

    def _open_capture(self) -> bool:
        self._release_capture()
        backend = cv2.CAP_FFMPEG if self.use_ffmpeg else cv2.CAP_ANY

        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source, backend)

        if self.cap is None or not self.cap.isOpened():
            self._retry_count += 1
            return False

        self._retry_count = 0
        return True

    def _release_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            now = time.time()
            if now - self._last_reconnect_ts >= self.reconnect_interval:
                self._last_reconnect_ts = now
                self._open_capture()
            return None

        ok, frame = self.cap.read()
        if ok and frame is not None:
            return frame

        now = time.time()
        if now - self._last_reconnect_ts >= self.reconnect_interval:
            self._last_reconnect_ts = now
            if self.max_retries == 0 or self._retry_count < self.max_retries:
                self._open_capture()
        return None

    def stop(self) -> None:
        self._release_capture()
