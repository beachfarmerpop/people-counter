from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class PersonByteTracker:
    """ByteTrack wrapper returning stable IDs and centers."""

    def __init__(self, frame_rate: int = 25):
        args = SimpleNamespace(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
        self.tracker = BYTETracker(args=args, frame_rate=frame_rate)

    def update(self, detections: List[list], frame_shape) -> List[Dict]:
        h, w = frame_shape[:2]
        if detections:
            data = [[d[0], d[1], d[2], d[3], d[4], 0.0] for d in detections]
            tensor = torch.tensor(data, dtype=torch.float32)
        else:
            tensor = torch.empty((0, 6), dtype=torch.float32)

        boxes = Boxes(tensor, orig_shape=(h, w))
        tracked = self.tracker.update(boxes, img=np.zeros((h, w, 3), dtype=np.uint8))

        output: List[Dict] = []
        if tracked is None or len(tracked) == 0:
            return output

        for row in tracked:
            x1, y1, x2, y2, track_id = row[:5]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            output.append(
                {
                    "track_id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": (cx, cy),
                }
            )
        return output
