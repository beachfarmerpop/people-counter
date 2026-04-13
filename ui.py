from datetime import datetime
from typing import Dict, List

import cv2

from text_render import draw_text


def init_window(name: str, width: int = 960, height: int = 540) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)


def draw_scene(frame, tracks: List[Dict], line_p1, line_p2, count_in: int, count_out: int, fps: float, header: str):
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        track_id = tr["track_id"]
        cx, cy = tr["center"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 210, 70), 2)
        draw_text(frame, f"ID {track_id}", (x1, max(6, y1 - 24)), (0, 255, 255), 20)
        cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)

    cv2.line(frame, line_p1, line_p2, (20, 180, 255), 3)

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw_text(frame, header, (10, 6), (255, 255, 255), 23, shadow=True)
    draw_text(frame, f"ВХОД: {count_in}   ВЫХОД: {count_out}", (10, 36), (0, 230, 255), 25, shadow=True)
    draw_text(frame, f"FPS: {fps:.1f}", (10, 68), (255, 220, 120), 22, shadow=True)
    draw_text(frame, now_text, (10, 98), (180, 255, 180), 21, shadow=True)

    return frame
