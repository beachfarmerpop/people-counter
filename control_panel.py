from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from text_render import draw_text


@dataclass
class AppState:
    bus_id: str
    stop_id: str
    conf_threshold: float
    process_every_n_frames: int


class ControlPanel:
    def __init__(self, app_state: AppState, contexts: List):
        self.app_state = app_state
        self.contexts = contexts
        self.window_name = "Program12 Panel"
        self.selected_idx = 0
        self._pending_action: Optional[str] = None
        self._buttons: List[Tuple[str, Tuple[int, int, int, int], str]] = []

        self.width = 900
        self.height = 550

        self.input_active = False
        self.input_action: Optional[str] = None
        self.input_label = ""
        self.input_buffer = ""
        self._submitted_input: Optional[Tuple[str, str]] = None
        self._build_buttons()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _build_buttons(self) -> None:
        x = 10
        y = 140
        w = 136
        h = 38
        g = 10

        def add(label: str, action: str):
            nonlocal x, y
            self._buttons.append((label, (x, y, w, h), action))
            x += w + g
            if x + w > self.width - 10:
                x = 10
                y += h + g

        add("Остановка -", "stop_dec")
        add("Остановка +", "stop_inc")
        add("Автобус -", "bus_dec")
        add("Автобус +", "bus_inc")
        add("Дверь <-", "door_prev")
        add("Дверь ->", "door_next")
        add("Источник <-", "source_prev")
        add("Источник ->", "source_next")
        add("USB 0", "usb0")
        add("USB 1", "usb1")
        add("USB 2", "usb2")
        add("RTSP URL", "rtsp_input")
        add("IP Webcam", "ipwebcam")
        add("DroidCam", "droidcam")
        add("Линия выше", "line_up")
        add("Линия ниже", "line_down")
        add("Порог -", "conf_dec")
        add("Порог +", "conf_inc")
        add("Скорость -", "proc_dec")
        add("Скорость +", "proc_inc")
        add("Направление", "direction_toggle")
        add("Сохранить", "save")
        add("Выход", "quit")

    @staticmethod
    def _rounded_rect(img, x, y, w, h, r, color_fill, color_border):
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color_fill, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color_fill, -1)
        cv2.circle(img, (x + r, y + r), r, color_fill, -1)
        cv2.circle(img, (x + w - r, y + r), r, color_fill, -1)
        cv2.circle(img, (x + r, y + h - r), r, color_fill, -1)
        cv2.circle(img, (x + w - r, y + h - r), r, color_fill, -1)
        cv2.rectangle(img, (x + r, y), (x + w - r, y + 1), color_border, 1)
        cv2.rectangle(img, (x + r, y + h - 1), (x + w - r, y + h), color_border, 1)
        cv2.rectangle(img, (x, y + r), (x + 1, y + h - r), color_border, 1)
        cv2.rectangle(img, (x + w - 1, y + r), (x + w, y + h - r), color_border, 1)
        cv2.circle(img, (x + r, y + r), r, color_border, 1)
        cv2.circle(img, (x + w - r, y + r), r, color_border, 1)
        cv2.circle(img, (x + r, y + h - r), r, color_border, 1)
        cv2.circle(img, (x + w - r, y + h - r), r, color_border, 1)

    def _on_mouse(self, event, x, y, flags, userdata):
        del flags, userdata
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for _label, (bx, by, bw, bh), action in self._buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._pending_action = action
                return

    def pull_action(self) -> Optional[str]:
        action = self._pending_action
        self._pending_action = None
        return action

    def start_input(self, action: str, label: str, initial: str = "") -> None:
        self.input_active = True
        self.input_action = action
        self.input_label = label
        self.input_buffer = initial

    def handle_key(self, key: int) -> None:
        if not self.input_active:
            return

        # Enter
        if key in (10, 13):
            self._submitted_input = (self.input_action or "", self.input_buffer.strip())
            self.input_active = False
            self.input_action = None
            return

        # Esc
        if key == 27:
            self.input_active = False
            self.input_action = None
            return

        # Backspace
        if key in (8, 127):
            self.input_buffer = self.input_buffer[:-1]
            return

        if 32 <= key <= 126:
            self.input_buffer += chr(key)

    def pull_input_submit(self) -> Optional[Tuple[str, str]]:
        out = self._submitted_input
        self._submitted_input = None
        return out

    def selected_context(self):
        if not self.contexts:
            return None
        self.selected_idx = max(0, min(self.selected_idx, len(self.contexts) - 1))
        return self.contexts[self.selected_idx]

    def render(self) -> None:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (28, 32, 40)

        # Accent bands for cleaner dashboard look.
        cv2.rectangle(canvas, (0, 0), (self.width, 115), (40, 60, 85), -1)
        cv2.rectangle(canvas, (0, 115), (self.width, 118), (84, 140, 220), -1)

        draw_text(canvas, "Program12 - Панель управления", (14, 10), (245, 245, 245), 31, shadow=True)
        draw_text(
            canvas,
            f"Автобус: {self.app_state.bus_id}   Остановка: {self.app_state.stop_id}   Порог: {self.app_state.conf_threshold:.2f}   Скорость(N): {self.app_state.process_every_n_frames}",
            (14, 48),
            (230, 238, 255),
            22,
            shadow=True,
        )

        ctx = self.selected_context()
        if ctx is not None:
            src = getattr(ctx.stream, "source", "-")
            mode = getattr(ctx.counter, "direction_mode", "right_in")
            mode_text = "Вправо=IN, Влево=OUT" if mode == "right_in" else "Вправо=OUT, Влево=IN"
            draw_text(
                canvas,
                f"Выбрана дверь: {ctx.door_id} ({ctx.name})   Источник: {src}",
                (14, 82),
                (205, 255, 205),
                21,
                shadow=True,
            )
            draw_text(canvas, f"Режим направления: {mode_text}", (14, 110), (255, 225, 170), 19, shadow=True)

        for label, (x, y, w, h), _action in self._buttons:
            self._rounded_rect(
                canvas,
                x,
                y,
                w,
                h,
                10,
                (62, 86, 124),
                (150, 198, 255),
            )
            draw_text(canvas, label, (x + 12, y + 8), (250, 250, 250), 21)

        if self.input_active:
            x, y, w, h = 14, self.height - 100, self.width - 28, 54
            self._rounded_rect(canvas, x, y, w, h, 10, (48, 58, 74), (180, 210, 255))
            draw_text(canvas, f"{self.input_label}: {self.input_buffer}_", (x + 12, y + 12), (245, 245, 245), 22)
            draw_text(canvas, "Enter - применить, Esc - отмена", (x + 12, y - 30), (180, 190, 205), 18)

        draw_text(canvas, "Подсказка: линию можно перетаскивать мышью в окне каждой двери", (14, self.height - 30), (188, 188, 188), 19)

        cv2.imshow(self.window_name, canvas)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
