import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass
class FaceParams:
    bbox_xywh: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    landmarks_norm: List[List[float]]
    contour_norm: List[List[float]]
    radial_distances: List[float]
    orientation_ypr: List[float]
    skin_rgb_norm: List[float]

    def to_dict(self) -> Dict:
        return {
            "bbox_xywh": list(self.bbox_xywh),
            "centroid_xy": [float(self.centroid_xy[0]), float(self.centroid_xy[1])],
            "landmarks_norm": self.landmarks_norm,
            "contour_norm": self.contour_norm,
            "radial_distances": self.radial_distances,
            "orientation_ypr": self.orientation_ypr,
            "skin_rgb_norm": self.skin_rgb_norm,
        }


class PassengerIdentifier:
    """Day 2 prototype: extracts compact face/profile parameters without storing face images."""

    def __init__(self, standard_size: int = 100, contour_points: int = 32, radial_angles: Sequence[int] = (0, 45, 90, 135, 180, 225, 270, 315)):
        self.standard_size = int(standard_size)
        self.contour_points = int(contour_points)
        self.radial_angles = tuple(int(a) for a in radial_angles)

        self._mp_face_mesh = None
        self._mp_face_detection = None
        if mp is not None:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.45,
            )
            self._mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )

        self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def close(self) -> None:
        if self._mp_face_mesh is not None:
            self._mp_face_mesh.close()
        if self._mp_face_detection is not None:
            self._mp_face_detection.close()

    def extract_face_params(self, frame_bgr: np.ndarray, bbox_xyxy: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        bbox = self._resolve_bbox(frame_bgr, bbox_xyxy)
        if bbox is None:
            return None

        x, y, bw, bh = bbox
        roi = frame_bgr[y : y + bh, x : x + bw]
        if roi.size == 0:
            return None

        landmarks_px = self._extract_landmarks(roi)
        if landmarks_px is None or len(landmarks_px) < 8:
            return None

        centroid = np.mean(landmarks_px, axis=0)
        contour = self._build_contour(landmarks_px)
        radial_distances = self._radial_features(contour, roi.shape[1], roi.shape[0], centroid)
        orientation = self._estimate_orientation(landmarks_px, roi.shape[1], roi.shape[0], centroid)
        skin_rgb = self._estimate_skin_rgb(roi, contour)

        norm_landmarks = self._normalize_points(landmarks_px, roi.shape[1], roi.shape[0])
        norm_contour = self._normalize_points(contour, roi.shape[1], roi.shape[0])

        params = FaceParams(
            bbox_xywh=(x, y, bw, bh),
            centroid_xy=(float(centroid[0] + x), float(centroid[1] + y)),
            landmarks_norm=norm_landmarks,
            contour_norm=norm_contour,
            radial_distances=[float(v) for v in radial_distances],
            orientation_ypr=[float(v) for v in orientation],
            skin_rgb_norm=[float(v) for v in skin_rgb],
        )
        return params.to_dict()

    def compute_descriptor(self, face_params: Dict) -> np.ndarray:
        lm = np.asarray(face_params["landmarks_norm"], dtype=np.float32).reshape(-1)
        ct = np.asarray(face_params["contour_norm"], dtype=np.float32).reshape(-1)
        rd = np.asarray(face_params["radial_distances"], dtype=np.float32).reshape(-1)
        ori = np.asarray(face_params["orientation_ypr"], dtype=np.float32).reshape(-1)
        clr = np.asarray(face_params["skin_rgb_norm"], dtype=np.float32).reshape(-1)
        return np.concatenate([lm, ct, rd, ori, clr], axis=0)

    def descriptor_distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        n = min(len(d1), len(d2))
        if n == 0:
            return 1.0
        a = d1[:n].astype(np.float32)
        b = d2[:n].astype(np.float32)
        return float(np.linalg.norm(a - b) / max(1.0, math.sqrt(n)))

    def serialize_descriptor(self, descriptor: np.ndarray) -> str:
        return json.dumps([float(x) for x in descriptor.tolist()], ensure_ascii=False)

    def _resolve_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        fh, fw = frame_bgr.shape[:2]

        if bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            x1 = max(0, min(x1, fw - 1))
            y1 = max(0, min(y1, fh - 1))
            x2 = max(x1 + 1, min(x2, fw))
            y2 = max(y1 + 1, min(y2, fh))
            return x1, y1, x2 - x1, y2 - y1

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._mp_face_detection is not None:
            det = self._mp_face_detection.process(rgb)
            if det.detections:
                bb = det.detections[0].location_data.relative_bounding_box
                x = int(bb.xmin * fw)
                y = int(bb.ymin * fh)
                bw = int(bb.width * fw)
                bh = int(bb.height * fh)
                x = max(0, x)
                y = max(0, y)
                bw = max(1, min(bw, fw - x))
                bh = max(1, min(bh, fh - y))
                return x, y, bw, bh

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
        if len(faces) > 0:
            x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3])
            return int(x), int(y), int(bw), int(bh)

        # Fallback для Day 2: центральный ROI, чтобы пайплайн параметров
        # не останавливался на кадрах со сложным ракурсом/светом.
        bw = max(64, int(fw * 0.45))
        bh = max(64, int(fh * 0.60))
        x = max(0, (fw - bw) // 2)
        y = max(0, (fh - bh) // 2)
        return x, y, bw, bh

    def _extract_landmarks(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        rh, rw = roi_bgr.shape[:2]
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        if self._mp_face_mesh is not None:
            res = self._mp_face_mesh.process(rgb)
            if res.multi_face_landmarks:
                pts = []
                lm = res.multi_face_landmarks[0].landmark
                if len(lm) >= 68:
                    idx = np.linspace(0, len(lm) - 1, 68).astype(np.int32)
                    for i in idx:
                        p = lm[int(i)]
                        pts.append((float(p.x * rw), float(p.y * rh)))
                if len(pts) >= 8:
                    return np.asarray(pts, dtype=np.float32)

        # Fallback: generate pseudo-landmarks from ellipse within ROI.
        cx, cy = rw / 2.0, rh / 2.0
        rx, ry = rw * 0.32, rh * 0.42
        pts = []
        for a in np.linspace(0, 2 * np.pi, 68, endpoint=False):
            pts.append((cx + rx * np.cos(a), cy + ry * np.sin(a)))
        return np.asarray(pts, dtype=np.float32)

    def _build_contour(self, landmarks: np.ndarray) -> np.ndarray:
        hull = cv2.convexHull(landmarks.astype(np.float32))
        hull = hull.reshape(-1, 2)
        if len(hull) == 0:
            return landmarks

        # Uniformly sample contour_points points over hull.
        pts = []
        for i in range(self.contour_points):
            t = i / float(self.contour_points)
            idx = int(t * len(hull)) % len(hull)
            pts.append(hull[idx])
        return np.asarray(pts, dtype=np.float32)

    def _radial_features(self, contour: np.ndarray, w: int, h: int, centroid: np.ndarray) -> List[float]:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 255)

        max_r = int(max(w, h))
        cx, cy = float(centroid[0]), float(centroid[1])
        values = []
        for ang in self.radial_angles:
            a = math.radians(ang)
            ux, uy = math.cos(a), math.sin(a)
            last_inside = 0
            for r in range(1, max_r):
                px = int(round(cx + ux * r))
                py = int(round(cy + uy * r))
                if px < 0 or py < 0 or px >= w or py >= h:
                    break
                if mask[py, px] > 0:
                    last_inside = r
                else:
                    break
            values.append(float(last_inside) / float(max(1, max_r)))
        return values

    def _estimate_orientation(self, landmarks: np.ndarray, w: int, h: int, centroid: np.ndarray) -> List[float]:
        left = landmarks[np.argmin(landmarks[:, 0])]
        right = landmarks[np.argmax(landmarks[:, 0])]

        yaw = float((centroid[0] - (w / 2.0)) / max(1.0, w / 2.0))
        pitch = float((centroid[1] - (h / 2.0)) / max(1.0, h / 2.0))
        roll = float(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])) / math.pi)
        return [yaw, pitch, roll]

    def _estimate_skin_rgb(self, roi_bgr: np.ndarray, contour: np.ndarray) -> List[float]:
        h, w = roi_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
        mean_bgr = cv2.mean(roi_bgr, mask=mask)[:3]
        r = float(mean_bgr[2]) / 255.0
        g = float(mean_bgr[1]) / 255.0
        b = float(mean_bgr[0]) / 255.0
        return [r, g, b]

    def _normalize_points(self, points_xy: np.ndarray, w: int, h: int) -> List[List[float]]:
        out = np.zeros_like(points_xy, dtype=np.float32)
        out[:, 0] = np.clip(points_xy[:, 0] / max(1.0, float(w)), 0.0, 1.0)
        out[:, 1] = np.clip(points_xy[:, 1] / max(1.0, float(h)), 0.0, 1.0)
        return out.tolist()


class PassengerDB:
    """Хранит дескрипторы пассажиров в SQLite и сопоставляет новые лица с известными ID.

    День 7: мультимодальный матчинг — face descriptor + appearance descriptor.
    Когда лицо не видно (боковая камера), сопоставление идёт по силуэту/одежде.
    """

    FACE_WEIGHT = 0.7      # вес лицевого дескриптора при комбинированном матчинге
    APPEAR_WEIGHT = 0.3    # вес дескриптора внешности
    APPEAR_THRESHOLD = 0.18  # порог матчинга только по внешности (без лица)

    def __init__(self, db_path: str, identifier: PassengerIdentifier, threshold: float = 0.22):
        self._db_path = db_path
        self._ident = identifier
        self._threshold = threshold
        self._cache: Dict[str, np.ndarray] = {}  # pid -> face descriptor
        self._app_cache: Dict[str, np.ndarray] = {}  # pid -> appearance descriptor
        self._init_db()
        self._load_cache()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS passengers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                passenger_id    TEXT UNIQUE NOT NULL,
                face_descriptor TEXT NOT NULL,
                appear_descriptor TEXT,
                first_seen      TEXT NOT NULL,
                last_seen       TEXT NOT NULL,
                encounter_count INTEGER DEFAULT 1,
                from_stop       TEXT,
                to_stop         TEXT
            )
        ''')
        # Таблица траекторий: полный путь пассажира по остановкам маршрута
        # (от остановки входа до остановки выхода, включая промежуточные)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS passenger_trajectories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                passenger_id    TEXT NOT NULL,
                route           TEXT,
                entry_stop      TEXT,
                exit_stop       TEXT,
                stops_visited   TEXT,
                entry_time      TEXT,
                exit_time       TEXT
            )
        ''')
        # Таблица параметров внешности: рост, цвета одежды/волос, тип одежды
        conn.execute('''
            CREATE TABLE IF NOT EXISTS passenger_appearance (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                passenger_id        TEXT NOT NULL,
                timestamp           TEXT,
                estimated_height_cm REAL,
                upper_body_color    TEXT,
                lower_body_color    TEXT,
                hair_color          TEXT,
                upper_color_name    TEXT,
                lower_color_name    TEXT,
                hair_color_name     TEXT,
                clothing_type       TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _load_cache(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute('SELECT passenger_id, face_descriptor, appear_descriptor FROM passengers').fetchall()
            conn.close()
            for row in rows:
                pid, desc_json, app_json = row[0], row[1], row[2]
                try:
                    self._cache[pid] = np.asarray(json.loads(desc_json), dtype=np.float32)
                except Exception:
                    pass
                if app_json:
                    try:
                        self._app_cache[pid] = np.asarray(json.loads(app_json), dtype=np.float32)
                    except Exception:
                        pass
        except Exception:
            pass

    def _next_pid(self) -> str:
        n = len(self._cache) + 1
        while f'P{n:04d}' in self._cache:
            n += 1
        return f'P{n:04d}'

    def match_or_create(self, face_params: Optional[Dict], from_stop: str = '',
                        appear_desc: Optional[np.ndarray] = None) -> Tuple[str, bool]:
        """Мультимодальный матчинг: лицо + внешность.

        Режимы:
          1) face_params есть + appear_desc есть → комбинированный матч
          2) face_params есть, appear_desc нет → только лицо
          3) face_params нет, appear_desc есть → только по внешности (боковая камера)
        """
        face_desc = None
        if face_params is not None:
            face_desc = self._ident.compute_descriptor(face_params)

        best_pid, best_dist = None, None

        if face_desc is not None and appear_desc is not None:
            # Комбинированный: взвешенная сумма расстояний
            threshold = self._threshold
            for pid, stored_face in self._cache.items():
                d_face = self._ident.descriptor_distance(face_desc, stored_face)
                stored_app = self._app_cache.get(pid)
                if stored_app is not None and len(stored_app) == len(appear_desc):
                    d_app = float(np.linalg.norm(appear_desc - stored_app) / max(1.0, math.sqrt(len(appear_desc))))
                    d = self.FACE_WEIGHT * d_face + self.APPEAR_WEIGHT * d_app
                else:
                    d = d_face
                if d < threshold:
                    threshold = d
                    best_pid = pid
                    best_dist = d
        elif face_desc is not None:
            # Только лицо
            threshold = self._threshold
            for pid, stored_face in self._cache.items():
                d = self._ident.descriptor_distance(face_desc, stored_face)
                if d < threshold:
                    threshold = d
                    best_pid = pid
                    best_dist = d
        elif appear_desc is not None:
            # Только внешность (боковая камера, лицо не видно)
            threshold = self.APPEAR_THRESHOLD
            for pid, stored_app in self._app_cache.items():
                if len(stored_app) != len(appear_desc):
                    continue
                d = float(np.linalg.norm(appear_desc - stored_app) / max(1.0, math.sqrt(len(appear_desc))))
                if d < threshold:
                    threshold = d
                    best_pid = pid
                    best_dist = d
        else:
            return 'P0000', False

        if best_pid is not None:
            self._update_encounter(best_pid)
            # Обновить appearance descriptor, если есть
            if appear_desc is not None:
                self._update_appearance_desc(best_pid, appear_desc)
            return best_pid, False

        pid = self._next_pid()
        if face_desc is not None:
            self._cache[pid] = face_desc
        if appear_desc is not None:
            self._app_cache[pid] = appear_desc
        self._save_new(pid, face_desc, from_stop, appear_desc)
        return pid, True

    def _save_new(self, pid: str, desc: Optional[np.ndarray], from_stop: str,
                  appear_desc: Optional[np.ndarray] = None) -> None:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        desc_json = json.dumps([float(x) for x in desc.tolist()]) if desc is not None else '[]'
        app_json = json.dumps([float(x) for x in appear_desc.tolist()]) if appear_desc is not None else None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                'INSERT OR IGNORE INTO passengers '
                '(passenger_id, face_descriptor, appear_descriptor, first_seen, last_seen, from_stop) '
                'VALUES (?,?,?,?,?,?)',
                (pid, desc_json, app_json, ts, ts, from_stop)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _update_appearance_desc(self, pid: str, appear_desc: np.ndarray) -> None:
        """Обновляет appearance descriptor в кеше и БД (скользящее среднее)."""
        old = self._app_cache.get(pid)
        if old is not None and len(old) == len(appear_desc):
            # Экспоненциальное скользящее среднее (alpha=0.3)
            blended = 0.7 * old + 0.3 * appear_desc
            self._app_cache[pid] = blended
        else:
            blended = appear_desc
            self._app_cache[pid] = appear_desc
        app_json = json.dumps([float(x) for x in blended.tolist()])
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute('UPDATE passengers SET appear_descriptor=? WHERE passenger_id=?', (app_json, pid))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _update_encounter(self, pid: str) -> None:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                'UPDATE passengers SET last_seen=?, encounter_count=encounter_count+1 WHERE passenger_id=?',
                (ts, pid)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def update_exit_stop(self, pid: str, to_stop: str) -> None:
        """Фиксирует остановку выхода пассажира (для OD-матрицы)."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute('UPDATE passengers SET to_stop=? WHERE passenger_id=?', (to_stop, pid))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_unique_count(self) -> int:
        """Число уникальных пассажиров в текущей сессии."""
        return len(self._cache)

    def get_od_matrix(self) -> Dict:
        """Возвращает словарь {from_stop: {to_stop: count}}."""
        matrix: Dict = {}
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                'SELECT from_stop, to_stop, COUNT(*) FROM passengers '
                'WHERE from_stop IS NOT NULL AND to_stop IS NOT NULL '
                'GROUP BY from_stop, to_stop'
            ).fetchall()
            conn.close()
            for fr, to, cnt in rows:
                matrix.setdefault(str(fr), {})[str(to)] = int(cnt)
        except Exception:
            pass
        return matrix

    # ─── Методы для траекторий пассажиров ─────────────────────────────────────

    def start_trajectory(self, pid: str, entry_stop: str, route: str) -> None:
        """Начинает запись траектории пассажира при входе в транспорт.

        Создаёт новую строку в passenger_trajectories.
        stops_visited инициализируется JSON-списком с одной остановкой входа.

        Args:
            pid: идентификатор пассажира ('P0001')
            entry_stop: название остановки входа
            route: название маршрута
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stops_json = json.dumps([entry_stop], ensure_ascii=False)
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                'INSERT INTO passenger_trajectories '
                '(passenger_id, route, entry_stop, stops_visited, entry_time) '
                'VALUES (?, ?, ?, ?, ?)',
                (pid, route, entry_stop, stops_json, ts)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def add_stop_to_trajectory(self, pid: str, stop: str) -> None:
        """Добавляет промежуточную остановку в незавершённую траекторию пассажира.

        Находит запись с exit_stop IS NULL (пассажир ещё в транспорте)
        и дописывает новую остановку в JSON-список stops_visited.
        Дубликаты подряд не добавляются.

        Args:
            pid: идентификатор пассажира
            stop: название новой остановки
        """
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                'SELECT id, stops_visited FROM passenger_trajectories '
                'WHERE passenger_id = ? AND exit_stop IS NULL '
                'ORDER BY id DESC LIMIT 1',
                (pid,)
            ).fetchone()
            if row:
                tid, stops_json = row
                stops = json.loads(stops_json) if stops_json else []
                # Не дублируем одну и ту же остановку подряд
                if not stops or stops[-1] != stop:
                    stops.append(stop)
                conn.execute(
                    'UPDATE passenger_trajectories SET stops_visited = ? WHERE id = ?',
                    (json.dumps(stops, ensure_ascii=False), tid)
                )
                conn.commit()
            conn.close()
        except Exception:
            pass

    def finish_trajectory(self, pid: str, exit_stop: str) -> None:
        """Завершает траекторию пассажира при выходе из транспорта.

        Записывает exit_stop и exit_time, добавляет exit_stop в stops_visited.

        Args:
            pid: идентификатор пассажира
            exit_stop: название остановки выхода
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                'SELECT id, stops_visited FROM passenger_trajectories '
                'WHERE passenger_id = ? AND exit_stop IS NULL '
                'ORDER BY id DESC LIMIT 1',
                (pid,)
            ).fetchone()
            if row:
                tid, stops_json = row
                stops = json.loads(stops_json) if stops_json else []
                if not stops or stops[-1] != exit_stop:
                    stops.append(exit_stop)
                conn.execute(
                    'UPDATE passenger_trajectories '
                    'SET exit_stop = ?, exit_time = ?, stops_visited = ? '
                    'WHERE id = ?',
                    (exit_stop, ts, json.dumps(stops, ensure_ascii=False), tid)
                )
                conn.commit()
            conn.close()
        except Exception:
            pass

    # ─── Методы для параметров внешности ─────────────────────────────────────

    def save_appearance(self, pid: str, params: Dict) -> None:
        """Сохраняет параметры внешности пассажира в БД.

        Args:
            pid: идентификатор пассажира ('P0001')
            params: словарь с ключами:
                estimated_height_cm  — оценка роста (см)
                upper_body_color     — RGB верхней одежды [r,g,b]
                lower_body_color     — RGB нижней одежды [r,g,b]
                hair_color           — RGB волос [r,g,b]
                upper_color_name     — название цвета верха
                lower_color_name     — название цвета низа
                hair_color_name      — название цвета волос
                clothing_type        — тип одежды
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                'INSERT INTO passenger_appearance '
                '(passenger_id, timestamp, estimated_height_cm, '
                'upper_body_color, lower_body_color, hair_color, '
                'upper_color_name, lower_color_name, hair_color_name, clothing_type) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    pid, ts,
                    params.get('estimated_height_cm', 0),
                    json.dumps(params.get('upper_body_color', []), ensure_ascii=False),
                    json.dumps(params.get('lower_body_color', []), ensure_ascii=False),
                    json.dumps(params.get('hair_color', []), ensure_ascii=False),
                    params.get('upper_color_name', ''),
                    params.get('lower_color_name', ''),
                    params.get('hair_color_name', ''),
                    params.get('clothing_type', ''),
                )
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_all_trajectories(self) -> List[Dict]:
        """Возвращает все траектории пассажиров для формирования отчётов.

        Returns:
            Список словарей: passenger_id, route, entry_stop, exit_stop,
            stops_visited (список), stops_count, entry_time, exit_time
        """
        result = []
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                'SELECT passenger_id, route, entry_stop, exit_stop, '
                'stops_visited, entry_time, exit_time '
                'FROM passenger_trajectories ORDER BY entry_time'
            ).fetchall()
            conn.close()
            for r in rows:
                stops = []
                try:
                    stops = json.loads(r[4]) if r[4] else []
                except Exception:
                    pass
                result.append({
                    'passenger_id': r[0],
                    'route': r[1],
                    'entry_stop': r[2],
                    'exit_stop': r[3],
                    'stops_visited': ' → '.join(stops),
                    'stops_count': len(stops),
                    'entry_time': r[5],
                    'exit_time': r[6],
                })
        except Exception:
            pass
        return result

    def get_all_appearances(self) -> List[Dict]:
        """Возвращает все записи параметров внешности для отчётов.

        Returns:
            Список словарей: passenger_id, timestamp, estimated_height_cm,
            upper_color_name, lower_color_name, hair_color_name, clothing_type
        """
        result = []
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                'SELECT passenger_id, timestamp, estimated_height_cm, '
                'upper_body_color, lower_body_color, hair_color, '
                'upper_color_name, lower_color_name, hair_color_name, clothing_type '
                'FROM passenger_appearance ORDER BY timestamp'
            ).fetchall()
            conn.close()
            for r in rows:
                result.append({
                    'passenger_id': r[0],
                    'timestamp': r[1],
                    'estimated_height_cm': r[2],
                    'upper_body_color': r[3],
                    'lower_body_color': r[4],
                    'hair_color': r[5],
                    'upper_color_name': r[6],
                    'lower_color_name': r[7],
                    'hair_color_name': r[8],
                    'clothing_type': r[9],
                })
        except Exception:
            pass
        return result

    def get_passengers_param_table(self) -> List[Dict]:
        """Таблица параметров всех пассажиров (последняя запись внешности на каждого).

        Returns:
            Список словарей для отображения на дашборде.
        """
        result = []
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute('''
                SELECT p.passenger_id, p.first_seen, p.last_seen, p.encounter_count,
                       p.from_stop, p.to_stop,
                       a.estimated_height_cm, a.upper_color_name, a.lower_color_name,
                       a.hair_color_name, a.clothing_type,
                       a.upper_body_color, a.lower_body_color, a.hair_color
                FROM passengers p
                LEFT JOIN (
                    SELECT passenger_id,
                           estimated_height_cm, upper_color_name, lower_color_name,
                           hair_color_name, clothing_type,
                           upper_body_color, lower_body_color, hair_color,
                           MAX(id) as max_id
                    FROM passenger_appearance GROUP BY passenger_id
                ) a ON p.passenger_id = a.passenger_id
                ORDER BY p.id
            ''').fetchall()
            conn.close()
            for r in rows:
                upper_rgb = self._parse_rgb(r[11])
                lower_rgb = self._parse_rgb(r[12])
                hair_rgb = self._parse_rgb(r[13])
                result.append({
                    'pid': r[0],
                    'first_seen': r[1] or '',
                    'last_seen': r[2] or '',
                    'encounters': r[3] or 1,
                    'from_stop': r[4] or '—',
                    'to_stop': r[5] or '—',
                    'height_cm': round(r[6], 1) if r[6] else '—',
                    'upper_color': r[7] or '—',
                    'lower_color': r[8] or '—',
                    'hair_color': r[9] or '—',
                    'clothing': r[10] or '—',
                    'upper_rgb': upper_rgb,
                    'lower_rgb': lower_rgb,
                    'hair_rgb': hair_rgb,
                })
        except Exception:
            pass
        return result

    @staticmethod
    def _parse_rgb(raw) -> Tuple[int, int, int]:
        if raw is None:
            return (128, 128, 128)
        try:
            vals = json.loads(raw)
            return (int(vals[0]), int(vals[1]), int(vals[2]))
        except Exception:
            return (128, 128, 128)

    def get_passengers_html_table(self) -> str:
        """Генерирует HTML-таблицу параметров пассажиров для веб-дашборда."""
        rows = self.get_passengers_param_table()
        if not rows:
            return '<p style="color:#8b949e">Пассажиры ещё не обнаружены</p>'
        html = '<table style="width:100%;border-collapse:collapse;font-size:12px">'
        html += ('<tr style="background:#21262d;color:#8b949e;text-align:left">'
                 '<th style="padding:6px">ID</th>'
                 '<th style="padding:6px">Рост</th>'
                 '<th style="padding:6px">Верх</th>'
                 '<th style="padding:6px">Низ</th>'
                 '<th style="padding:6px">Волосы</th>'
                 '<th style="padding:6px">Одежда</th>'
                 '<th style="padding:6px">Откуда</th>'
                 '<th style="padding:6px">Куда</th>'
                 '<th style="padding:6px">Встреч</th>'
                 '</tr>')
        for r in rows:
            u_rgb = r['upper_rgb']
            l_rgb = r['lower_rgb']
            h_rgb = r['hair_rgb']
            html += (
                f'<tr style="border-bottom:1px solid #21262d">'
                f'<td style="padding:5px;color:#58a6ff;font-weight:600">{r["pid"]}</td>'
                f'<td style="padding:5px">{r["height_cm"]}</td>'
                f'<td style="padding:5px">'
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
                f'background:rgb({u_rgb[0]},{u_rgb[1]},{u_rgb[2]});vertical-align:middle;margin-right:4px"></span>'
                f'{r["upper_color"]}</td>'
                f'<td style="padding:5px">'
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
                f'background:rgb({l_rgb[0]},{l_rgb[1]},{l_rgb[2]});vertical-align:middle;margin-right:4px"></span>'
                f'{r["lower_color"]}</td>'
                f'<td style="padding:5px">'
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
                f'background:rgb({h_rgb[0]},{h_rgb[1]},{h_rgb[2]});vertical-align:middle;margin-right:4px"></span>'
                f'{r["hair_color"]}</td>'
                f'<td style="padding:5px">{r["clothing"]}</td>'
                f'<td style="padding:5px">{r["from_stop"]}</td>'
                f'<td style="padding:5px">{r["to_stop"]}</td>'
                f'<td style="padding:5px;text-align:center">{r["encounters"]}</td>'
                f'</tr>'
            )
        html += '</table>'
        return html


# ═══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции: определение цвета, доминантный цвет (K-means)
# ═══════════════════════════════════════════════════════════════════════════════

# Таблица основных цветов (RGB) и их русских названий для классификации
COLOR_NAMES_RU = [
    ((0, 0, 0),       'чёрный'),
    ((255, 255, 255), 'белый'),
    ((128, 128, 128), 'серый'),
    ((192, 192, 192), 'светло-серый'),
    ((64, 64, 64),    'тёмно-серый'),
    ((255, 0, 0),     'красный'),
    ((0, 128, 0),     'зелёный'),
    ((0, 0, 255),     'синий'),
    ((255, 255, 0),   'жёлтый'),
    ((255, 165, 0),   'оранжевый'),
    ((128, 0, 128),   'фиолетовый'),
    ((165, 42, 42),   'коричневый'),
    ((255, 192, 203), 'розовый'),
    ((0, 255, 255),   'голубой'),
    ((0, 0, 128),     'тёмно-синий'),
    ((0, 100, 0),     'тёмно-зелёный'),
    ((139, 0, 0),     'бордовый'),
    ((245, 245, 220), 'бежевый'),
    ((75, 0, 130),    'индиго'),
]


def color_name_ru(rgb: Tuple[int, int, int]) -> str:
    """Определяет ближайшее название цвета на русском по RGB-значению.

    Использует евклидово расстояние в RGB-пространстве.

    Args:
        rgb: кортеж (R, G, B) в диапазоне 0–255

    Returns:
        Название цвета на русском языке
    """
    best_name = 'неопределённый'
    best_dist = float('inf')
    for ref_rgb, name in COLOR_NAMES_RU:
        # Евклидово расстояние между цветами
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb)))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def dominant_color_kmeans(roi_bgr: np.ndarray, k: int = 2) -> Tuple[int, int, int]:
    """Находит доминантный цвет в ROI через K-means кластеризацию.

    Пиксели области группируются в k кластеров.
    Возвращается центр самого большого кластера (= самый частый цвет).

    Args:
        roi_bgr: фрагмент изображения BGR (OpenCV)
        k: число кластеров (2–3 оптимально)

    Returns:
        Кортеж (R, G, B) доминантного цвета
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return (128, 128, 128)
    # Преобразуем ROI в плоский массив пикселей
    pixels = roi_bgr.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        mean = pixels.mean(axis=0).astype(int)
        return (int(mean[2]), int(mean[1]), int(mean[0]))  # BGR → RGB
    # K-means кластеризация OpenCV
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # Выбираем кластер с наибольшим числом пикселей
    counts = np.bincount(labels.flatten())
    dominant_idx = counts.argmax()
    bgr = centers[dominant_idx].astype(int)
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR → RGB


# ═══════════════════════════════════════════════════════════════════════════════
# ProfileAnalyzer — анализ силуэта/профиля пассажира (камера сбоку или фронт)
# ═══════════════════════════════════════════════════════════════════════════════

class ProfileAnalyzer:
    """Анализатор параметров пассажира по силуэту/профилю.

    Работает когда камера расположена сбоку (профильный вид) или когда
    лицо не распознано. Извлекает параметры без распознавания лица:
      - Оценку роста (см) по пропорции бокса к кадру
      - Доминантные цвета верхней и нижней одежды
      - Цвет волос
      - Грубый тип одежды (куртка, пальто, платье и т.д.)

    Пример использования:
        analyzer = ProfileAnalyzer(camera_height_m=2.5)
        params = analyzer.extract_full_appearance(frame, (x, y, w, h), frame.shape[0])
    """

    def __init__(self, camera_height_m: float = 2.5, visible_height_m: float = 2.2):
        """Инициализация анализатора профиля.

        Args:
            camera_height_m: высота установки камеры в метрах
            visible_height_m: реальная высота видимой зоны кадра (м)
        """
        self.camera_height_m = camera_height_m
        self.visible_height_m = visible_height_m

    def estimate_height_cm(self, bbox_h: int, frame_h: int) -> float:
        """Оценка роста пассажира по высоте бокса в кадре.

        Пропорциональная модель:
            рост_м ≈ (высота_бокса / высота_кадра) × видимая_высота_зоны

        Args:
            bbox_h: высота бокса человека (пиксели)
            frame_h: высота кадра (пиксели)

        Returns:
            Рост в сантиметрах (ограничен 50–230 см)
        """
        if frame_h <= 0 or bbox_h <= 0:
            return 170.0  # Значение по умолчанию
        ratio = float(bbox_h) / float(frame_h)
        height_m = ratio * self.visible_height_m
        # Ограничиваем разумным диапазоном роста человека
        height_cm = max(50.0, min(230.0, height_m * 100.0))
        return round(height_cm, 1)

    def extract_clothing_colors(self, frame_bgr: np.ndarray,
                                 bbox_xywh: Tuple[int, int, int, int]) -> Dict:
        """Извлекает доминантные цвета одежды и волос из бокса.

        Бокс делится на 3 вертикальные зоны:
          - Верхние 15%  → голова/волосы
          - Средние 40%  → верхняя одежда (куртка, рубашка)
          - Нижние 45%   → нижняя одежда (штаны, юбка)
        Горизонтально берётся центральная полоса (исключая фон по краям).

        Args:
            frame_bgr: полный кадр BGR
            bbox_xywh: бокс (x, y, ширина, высота)

        Returns:
            Словарь: hair_rgb, hair_color_name, upper_rgb, upper_color_name,
                     lower_rgb, lower_color_name
        """
        x, y, w, h = bbox_xywh
        fh, fw = frame_bgr.shape[:2]
        # Безопасные границы
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        # Границы зон по вертикали
        hair_y1, hair_y2 = y, y + int(h * 0.15)
        upper_y1, upper_y2 = hair_y2, y + int(h * 0.55)
        lower_y1, lower_y2 = upper_y2, y + h
        # Центральная полоса — исключаем фон по краям бокса
        margin_x = int(w * 0.15)
        cx1, cx2 = x + margin_x, x + w - margin_x
        if cx2 <= cx1:
            cx1, cx2 = x, x + w
        # Вырезаем ROI для каждой зоны
        roi_hair = frame_bgr[hair_y1:hair_y2, cx1:cx2]
        roi_upper = frame_bgr[upper_y1:upper_y2, cx1:cx2]
        roi_lower = frame_bgr[lower_y1:lower_y2, cx1:cx2]
        # Определяем доминантный цвет (RGB) в каждой зоне
        hair_rgb = dominant_color_kmeans(roi_hair, k=2)
        upper_rgb = dominant_color_kmeans(roi_upper, k=3)
        lower_rgb = dominant_color_kmeans(roi_lower, k=3)
        return {
            'hair_rgb': list(hair_rgb),
            'hair_color_name': color_name_ru(hair_rgb),
            'upper_rgb': list(upper_rgb),
            'upper_color_name': color_name_ru(upper_rgb),
            'lower_rgb': list(lower_rgb),
            'lower_color_name': color_name_ru(lower_rgb),
        }

    def classify_clothing(self, frame_bgr: np.ndarray,
                           bbox_xywh: Tuple[int, int, int, int]) -> str:
        """Грубая классификация типа одежды по силуэту и цвету.

        Анализирует соотношение сторон бокса и разницу цветов верх/низ.
        Результаты:
          - «пальто/плащ»       — однотонный, вытянутый силуэт
          - «платье/комбинезон» — однотонный, обычные пропорции
          - «куртка/пуховик»    — широкий верх
          - «куртка+штаны»      — разноцветные верх и низ

        Args:
            frame_bgr: кадр BGR
            bbox_xywh: бокс (x, y, w, h)

        Returns:
            Тип одежды на русском
        """
        x, y, w, h = bbox_xywh
        fh, fw = frame_bgr.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        # Соотношение сторон: стоящий человек ~2.0–3.0
        aspect = float(h) / float(w) if w > 0 else 2.0
        # Расстояние между цветами верха и низа
        colors = self.extract_clothing_colors(frame_bgr, bbox_xywh)
        upper_c = np.array(colors['upper_rgb'], dtype=float)
        lower_c = np.array(colors['lower_rgb'], dtype=float)
        color_diff = float(np.linalg.norm(upper_c - lower_c))
        # Классификация по комбинации признаков
        if color_diff < 40 and aspect > 2.5:
            return 'пальто/плащ'
        elif color_diff < 40 and aspect <= 2.5:
            return 'платье/комбинезон'
        elif aspect < 1.8:
            return 'куртка/пуховик'
        else:
            return 'куртка+штаны'

    def extract_full_appearance(self, frame_bgr: np.ndarray,
                                 bbox_xywh: Tuple[int, int, int, int],
                                 frame_h: int) -> Optional[Dict]:
        """Полный анализ внешности: рост + цвета + тип одежды.

        Основной метод вызова — комбинирует все анализаторы.

        Args:
            frame_bgr: кадр BGR
            bbox_xywh: бокс (x, y, w, h)
            frame_h: высота кадра (пиксели)

        Returns:
            Словарь параметров внешности или None при ошибке:
              estimated_height_cm, upper_body_color, lower_body_color,
              upper_color_name, lower_color_name, hair_color,
              hair_color_name, clothing_type
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        x, y, w, h = bbox_xywh
        # Оценка роста по высоте бокса
        height_cm = self.estimate_height_cm(h, frame_h)
        # Извлечение цветов одежды и волос
        colors = self.extract_clothing_colors(frame_bgr, bbox_xywh)
        # Определение типа одежды
        clothing_type = self.classify_clothing(frame_bgr, bbox_xywh)
        return {
            'estimated_height_cm': height_cm,
            'upper_body_color': colors['upper_rgb'],
            'lower_body_color': colors['lower_rgb'],
            'upper_color_name': colors['upper_color_name'],
            'lower_color_name': colors['lower_color_name'],
            'hair_color': colors['hair_rgb'],
            'hair_color_name': colors['hair_color_name'],
            'clothing_type': clothing_type,
        }

    # ── День 7: Дескриптор внешности для мультимодального матчинга ────

    def compute_appearance_descriptor(self, frame_bgr: np.ndarray,
                                       bbox_xywh: Tuple[int, int, int, int],
                                       frame_h: int) -> Optional[np.ndarray]:
        """Строит числовой вектор внешности (профиль+одежда) для сопоставления.

        Вектор из 10 компонент:
          [0]    — нормированный рост (height_cm / 200)
          [1-3]  — RGB верхней одежды (0..1)
          [4-6]  — RGB нижней одежды (0..1)
          [7-9]  — RGB волос (0..1)

        Позволяет идентифицировать пассажира с боковой камеры,
        когда лицо не видно.
        """
        app = self.extract_full_appearance(frame_bgr, bbox_xywh, frame_h)
        if app is None:
            return None
        parts = [
            app['estimated_height_cm'] / 200.0,
            app['upper_body_color'][0] / 255.0,
            app['upper_body_color'][1] / 255.0,
            app['upper_body_color'][2] / 255.0,
            app['lower_body_color'][0] / 255.0,
            app['lower_body_color'][1] / 255.0,
            app['lower_body_color'][2] / 255.0,
            app['hair_color'][0] / 255.0,
            app['hair_color'][1] / 255.0,
            app['hair_color'][2] / 255.0,
        ]
        return np.asarray(parts, dtype=np.float32)
