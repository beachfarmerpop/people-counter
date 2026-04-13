from datetime import datetime
from typing import Dict, List

import cv2

from text_render import draw_texts_batch


def init_window(name: str, width: int = 960, height: int = 540) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)


def draw_scene(frame, tracks: List[Dict], line_p1, line_p2, count_in: int, count_out: int, fps: float, header: str):
    # Draw boxes and circles first (pure OpenCV — fast)
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        cx, cy = tr["center"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 210, 70), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)

    cv2.line(frame, line_p1, line_p2, (20, 180, 255), 3)

    # Collect ALL text items and draw in ONE batch (1 PIL conversion max)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_items = [
        # (text, (x,y), (b,g,r), size, shadow)
        (header, (10, 6), (255, 255, 255), 23, True),
        (f"IN: {count_in}   OUT: {count_out}", (10, 36), (0, 230, 255), 25, True),
        (f"FPS: {fps:.1f}", (10, 68), (255, 220, 120), 22, True),
        (now_text, (10, 98), (180, 255, 180), 21, True),
    ]

    # Add track ID labels (ASCII only — very fast, no PIL)
    for tr in tracks:
        x1, y1 = tr["bbox"][0], tr["bbox"][1]
        track_id = tr["track_id"]
        text_items.append((f"ID {track_id}", (x1, max(6, y1 - 24)), (0, 255, 255), 20, False))

    draw_texts_batch(frame, text_items)
    return frame
