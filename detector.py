from typing import Dict, List

from ultralytics import YOLO


class PersonDetector:
    """YOLO person detector with confidence filtering."""

    def __init__(self, model_path: str, conf_threshold: float = 0.4, imgsz: int = 640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {model_path}. {exc}") from exc

    def detect(self, frame) -> List[list]:
        """Return detections as [x1, y1, x2, y2, confidence]."""
        results = self.model.predict(
            frame,
            classes=[0],
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        detections: List[list] = []
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
        return detections

    def track(self, frame) -> List[Dict]:
        """Return tracked persons with stable IDs from YOLO track()."""
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return []

        ids = boxes.id
        if ids is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        ids_np = ids.cpu().numpy()

        tracks: List[Dict] = []
        for i in range(len(xyxy)):
            track_id_raw = ids_np[i]
            if track_id_raw is None:
                continue

            try:
                track_id = int(track_id_raw)
            except (TypeError, ValueError):
                continue

            x1, y1, x2, y2 = xyxy[i]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            bot_cy = int(y2)

            tracks.append(
                {
                    "track_id": track_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": (cx, cy),
                    "bottom_center": (cx, bot_cy),
                }
            )

        return tracks
