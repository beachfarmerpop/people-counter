from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Point = Tuple[int, int]


@dataclass
class CounterState:
    count_in: int = 0
    count_out: int = 0


class LineCounter:
    """Robust line crossing counter.  Tracks min/max signed distance
    for each track_id — if both sides of the line were observed, the
    person definitively crossed.  Works even with unstable IDs."""

    CROSS_MARGIN = 5  # noise margin in pixels

    def __init__(
        self,
        p1: Point,
        p2: Point,
        counting_region: int = 120,
        direction_mode: str = "right_in",
    ):
        self.p1 = p1
        self.p2 = p2
        self.counting_region = counting_region
        self.direction_mode = direction_mode if direction_mode in {"right_in", "right_out"} else "right_in"
        self.state = CounterState()
        self.counted_ids: Set[int] = set()
        # min/max signed distance per track_id: {id: [min_side, max_side]}
        self._side_range: Dict[int, list] = {}
        # last side and x-motion history per track_id
        self._last_side: Dict[int, float] = {}
        self._x_motion: Dict[int, list] = {}
        self._update_line_vectors()

    def toggle_direction_mode(self) -> str:
        self.direction_mode = "right_out" if self.direction_mode == "right_in" else "right_in"
        return self.direction_mode

    def set_line(self, p1: Point, p2: Point) -> None:
        self.p1 = p1
        self.p2 = p2
        self._update_line_vectors()
        # Reset ranges — line moved, old ranges are meaningless
        self._side_range.clear()
        self._last_side.clear()
        self._x_motion.clear()

    def _update_line_vectors(self) -> None:
        """Pre-compute numpy vectors for distance calculations."""
        self._start = np.array(self.p1, dtype=np.float64)
        self._end = np.array(self.p2, dtype=np.float64)
        line_vec = self._end - self._start
        length = np.linalg.norm(line_vec)
        if length < 1e-6:
            length = 1.0
        self._unit = line_vec / length
        self._normal = np.array([-self._unit[1], self._unit[0]])
        self._length = length

    def _signed_distance(self, point: Point) -> float:
        """Signed distance from *point* to the counting line."""
        return float(np.dot(np.array(point, dtype=np.float64) - self._start, self._normal))

    def _is_near_line(self, point: Point) -> bool:
        """Check if *point* lies within the counting region around the line."""
        pt = np.array(point, dtype=np.float64)
        proj = np.dot(pt - self._start, self._unit)
        if proj < 0 or proj > self._length:
            return False
        return abs(self._signed_distance(point)) <= self.counting_region

    def update(self, tracks: List[Dict]) -> bool:
        counts_changed = False
        active_ids = set()

        for track in tracks:
            track_id = track["track_id"]
            point = track.get("bottom_center", track["center"])
            active_ids.add(track_id)

            side = self._signed_distance(point)
            self._last_side[track_id] = side
            x = int(point[0])

            if track_id not in self._x_motion:
                self._x_motion[track_id] = [x, x]
            else:
                self._x_motion[track_id][1] = x

            if track_id in self.counted_ids:
                continue

            # Update min/max range for this ID
            if track_id not in self._side_range:
                self._side_range[track_id] = [side, side]
            else:
                rng = self._side_range[track_id]
                if side < rng[0]:
                    rng[0] = side
                if side > rng[1]:
                    rng[1] = side

            min_s, max_s = self._side_range[track_id]

            # Crossing: we've seen this track on BOTH sides of the line
            if min_s < -self.CROSS_MARGIN and max_s > self.CROSS_MARGIN:
                first_x, last_x = self._x_motion.get(track_id, [x, x])
                moving_right = (last_x - first_x) > 0

                if self.direction_mode == "right_in":
                    is_in = moving_right
                else:
                    is_in = not moving_right

                if is_in:
                    self.state.count_in += 1
                else:
                    self.state.count_out += 1
                self.counted_ids.add(track_id)
                counts_changed = True

        # Clean up stale IDs
        stale = set(self._side_range.keys()) - active_ids
        for sid in stale:
            self._side_range.pop(sid, None)
            self._last_side.pop(sid, None)
            self._x_motion.pop(sid, None)

        return counts_changed

    def snapshot(self) -> Dict[str, int]:
        return {"count_in": self.state.count_in, "count_out": self.state.count_out}


class LineDragController:
    """Simple drag handler to move full counting line with mouse."""

    def __init__(self) -> None:
        self.dragging = False
        self.last_point: Optional[Point] = None

    @staticmethod
    def _distance_to_line(point: Point, p1: Point, p2: Point) -> float:
        x0, y0 = point
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        px = x1 + t * dx
        py = y1 + t * dy
        return ((x0 - px) ** 2 + (y0 - py) ** 2) ** 0.5

    def on_mouse(self, event, x, y, flags, counter: LineCounter):
        del flags
        current = (x, y)
        if event == 1:  # cv2.EVENT_LBUTTONDOWN
            dist = self._distance_to_line(current, counter.p1, counter.p2)
            if dist <= 18:
                self.dragging = True
                self.last_point = current
        elif event == 4:  # cv2.EVENT_LBUTTONUP
            self.dragging = False
            self.last_point = None
        elif event == 0 and self.dragging and self.last_point is not None:  # cv2.EVENT_MOUSEMOVE
            dx = current[0] - self.last_point[0]
            dy = current[1] - self.last_point[1]
            p1 = (counter.p1[0] + dx, counter.p1[1] + dy)
            p2 = (counter.p2[0] + dx, counter.p2[1] + dy)
            counter.set_line(p1, p2)
            self.last_point = current
