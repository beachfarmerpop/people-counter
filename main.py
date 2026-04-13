import json
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from control_panel import AppState, ControlPanel
from counter import LineCounter, LineDragController
from csv_logger import CounterCsvLogger
from detector import PersonDetector
from splash import show_splash
from tracker import PersonByteTracker
from text_render import draw_text
from ui import draw_scene, init_window
from video_stream import VideoStream


DEFAULT_CONFIG: Dict = {
    "model_path": "yolov8n.pt",
    "conf_threshold": 0.3,
    "imgsz": 640,
    "frame_rate": 25,
    "process_every_n_frames": 1,
    "direction_mode": "right_in",
    "resize_width": 640,
    "periodic_log_sec": 10,
    "reconnect_interval": 2.0,
    "max_retries": 0,
    "bus_id": "bus-1",
    "stop_id": "1",
    "streams": [
        {
            "name": "Door 1",
            "door_id": "1",
            "enabled": True,
            "source": 0,
            "line": [[50, 260], [900, 260]],
            "csv_path": "reports/door_1_counts.csv",
        }
    ],
}


@dataclass
class StreamContext:
    name: str
    door_id: str
    stream: VideoStream
    tracker: PersonByteTracker
    counter: LineCounter
    dragger: LineDragController
    logger: CounterCsvLogger
    process_every_n_frames: int
    resize_width: int
    source_options: List = None
    source_index: int = 0
    frame_index: int = 0
    last_tracks: List[Dict] = None
    last_log_ts: float = 0.0
    last_fps_ts: float = 0.0
    fps: float = 0.0

    def __post_init__(self):
        if self.last_tracks is None:
            self.last_tracks = []


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _bundle_dir() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle)
    return _app_dir()


def _resolve_existing_file(path_value: str) -> Path:
    p = Path(path_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            _app_dir() / p,
            _bundle_dir() / p,
            Path.cwd() / p,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"File not found: {path_value}. Checked: {[str(c) for c in candidates]}")


def _load_config(path: Path) -> Dict:
    config_candidates = [
        _app_dir() / path,
        _bundle_dir() / path,
        Path.cwd() / path,
    ]
    for cfg_path in config_candidates:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Fallback config for first launch from packaged exe.
    app_cfg = _app_dir() / "config.json"
    try:
        with open(app_cfg, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return dict(DEFAULT_CONFIG)


def _prompt_if_missing(cfg: Dict, key: str, prompt: str, default_value: str) -> str:
    val = str(cfg.get(key, "")).strip()
    if val:
        return val
    typed = input(f"{prompt} [{default_value}]: ").strip()
    return typed or default_value


def _build_contexts(cfg: Dict, detector: PersonDetector) -> List[StreamContext]:
    del detector
    streams_cfg = cfg.get("streams", [])
    contexts: List[StreamContext] = []

    for idx, s in enumerate(streams_cfg):
        if not bool(s.get("enabled", True)):
            continue

        name = s.get("name", f"Door {idx + 1}")
        door_id = str(s.get("door_id", idx + 1))
        source = s.get("source", 0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        source_options = s.get("source_options", [source])
        if not source_options:
            source_options = [source]
        source_index = 0
        for opt_idx, opt in enumerate(source_options):
            if opt == source:
                source_index = opt_idx
                break

        resize_width = int(s.get("resize_width", cfg.get("resize_width", 960)))
        process_every_n = int(s.get("process_every_n_frames", cfg.get("process_every_n_frames", 2)))
        line = s.get("line", [
            [50, int(cfg.get("line_y", 240))],
            [900, int(cfg.get("line_y", 240))],
        ])
        p1 = (int(line[0][0]), int(line[0][1]))
        p2 = (int(line[1][0]), int(line[1][1]))

        stream = VideoStream(
            source=source,
            reconnect_interval=float(cfg.get("reconnect_interval", 2.0)),
            max_retries=int(cfg.get("max_retries", 0)),
        )
        tracker = PersonByteTracker(frame_rate=int(cfg.get("frame_rate", 25)))
        direction_mode = str(s.get("direction_mode", cfg.get("direction_mode", "right_in")))
        counter = LineCounter(p1=p1, p2=p2, direction_mode=direction_mode)
        dragger = LineDragController()
        logger = CounterCsvLogger(csv_path=str(s.get("csv_path", f"reports/door_{door_id}_counts.csv")))

        contexts.append(
            StreamContext(
                name=name,
                door_id=door_id,
                stream=stream,
                tracker=tracker,
                counter=counter,
                dragger=dragger,
                logger=logger,
                process_every_n_frames=max(1, process_every_n),
                resize_width=max(320, resize_width),
                source_options=source_options,
                source_index=source_index,
            )
        )

    return contexts


def _mouse_callback_factory(ctx: StreamContext):
    def _callback(event, x, y, flags, userdata):
        del userdata
        ctx.dragger.on_mouse(event, x, y, flags, ctx.counter)

    return _callback


def _resize_keep_aspect(frame, width: int):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    ratio = width / float(w)
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def run(config_path: str = "config.json") -> None:
    cfg = _load_config(Path(config_path))

    show_splash(_app_dir())
    bus_id = str(cfg.get("bus_id", "1"))
    stop_id = str(cfg.get("stop_id", "1"))

    model_path = str(cfg.get("model_path", "yolov8n.pt"))
    resolved_model_path = _resolve_existing_file(model_path)

    detector = PersonDetector(
        model_path=str(resolved_model_path),
        conf_threshold=float(cfg.get("conf_threshold", 0.4)),
        imgsz=int(cfg.get("imgsz", 640)),
    )

    contexts = _build_contexts(cfg, detector)
    if not contexts:
        raise RuntimeError("No streams configured. Add at least one stream to config.json")

    app_state = AppState(
        bus_id=bus_id,
        stop_id=stop_id,
        conf_threshold=float(cfg.get("conf_threshold", 0.4)),
        process_every_n_frames=int(cfg.get("process_every_n_frames", 2)),
    )
    panel = ControlPanel(app_state, contexts)

    for ctx in contexts:
        ok = ctx.stream.start_stream()
        init_window(ctx.name)
        cv2.setMouseCallback(ctx.name, _mouse_callback_factory(ctx))
        if not ok:
            print(f"[WARN] Поток {ctx.name} пока не открыт. Включен авто-переподключатель.")

    periodic_log_sec = float(cfg.get("periodic_log_sec", 10.0))

    print("Управление: q - выход, кнопки в панели, Enter/Esc для RTSP ввода")

    def _bump_number(value: str, delta: int) -> str:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return str(max(0, int(value) + delta))
        m = re.match(r"^(.*?)(\d+)$", value)
        if m:
            prefix, num = m.groups()
            return f"{prefix}{max(0, int(num) + delta)}"
        return value

    def _save_runtime_config() -> None:
        cfg["bus_id"] = app_state.bus_id
        cfg["stop_id"] = app_state.stop_id
        cfg["conf_threshold"] = app_state.conf_threshold
        cfg["process_every_n_frames"] = app_state.process_every_n_frames
        streams_out = []
        for c in contexts:
            streams_out.append(
                {
                    "name": c.name,
                    "door_id": c.door_id,
                    "enabled": True,
                    "source": c.stream.source,
                    "source_options": c.source_options,
                    "direction_mode": getattr(c.counter, "direction_mode", "right_in"),
                    "line": [[int(c.counter.p1[0]), int(c.counter.p1[1])], [int(c.counter.p2[0]), int(c.counter.p2[1])]],
                    "csv_path": f"reports/door_{c.door_id}_counts.csv",
                }
            )
        cfg["streams"] = streams_out
        (_app_dir() / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Настройки сохранены")

    while True:
        global_now = time.time()
        panel.render()

        action = panel.pull_action()
        if action == "quit":
            break
        elif action == "stop_dec":
            app_state.stop_id = _bump_number(app_state.stop_id, -1)
        elif action == "stop_inc":
            app_state.stop_id = _bump_number(app_state.stop_id, 1)
        elif action == "bus_dec":
            app_state.bus_id = _bump_number(app_state.bus_id, -1)
        elif action == "bus_inc":
            app_state.bus_id = _bump_number(app_state.bus_id, 1)
        elif action == "door_prev" and panel.contexts:
            panel.selected_idx = max(0, panel.selected_idx - 1)
        elif action == "door_next" and panel.contexts:
            panel.selected_idx = min(len(panel.contexts) - 1, panel.selected_idx + 1)
        elif action in ("source_prev", "source_next"):
            selected = panel.selected_context()
            if selected is not None and selected.source_options:
                step = -1 if action == "source_prev" else 1
                selected.source_index = (selected.source_index + step) % len(selected.source_options)
                selected.stream.source = selected.source_options[selected.source_index]
                selected.stream.start_stream()
        elif action == "usb0":
            selected = panel.selected_context()
            if selected is not None:
                selected.stream.source = 0
                selected.stream.start_stream()
        elif action == "usb1":
            selected = panel.selected_context()
            if selected is not None:
                selected.stream.source = 1
                selected.stream.start_stream()
        elif action == "usb2":
            selected = panel.selected_context()
            if selected is not None:
                selected.stream.source = 2
                selected.stream.start_stream()
        elif action == "rtsp_input":
            selected = panel.selected_context()
            if selected is not None:
                current = selected.stream.source if isinstance(selected.stream.source, str) else "rtsp://"
                panel.start_input("rtsp_apply", "RTSP URL", str(current))
        elif action == "ipwebcam":
            selected = panel.selected_context()
            if selected is not None:
                panel.start_input("rtsp_apply", "IP Webcam (http://IP:8080/video)", "http://192.168.1.:8080/video")
        elif action == "droidcam":
            selected = panel.selected_context()
            if selected is not None:
                panel.start_input("rtsp_apply", "DroidCam (http://IP:4747/video)", "http://192.168.1.:4747/video")
        elif action == "line_up":
            selected = panel.selected_context()
            if selected is not None:
                selected.counter.set_line((selected.counter.p1[0], selected.counter.p1[1] - 10), (selected.counter.p2[0], selected.counter.p2[1] - 10))
        elif action == "line_down":
            selected = panel.selected_context()
            if selected is not None:
                selected.counter.set_line((selected.counter.p1[0], selected.counter.p1[1] + 10), (selected.counter.p2[0], selected.counter.p2[1] + 10))
        elif action == "conf_dec":
            app_state.conf_threshold = max(0.1, round(app_state.conf_threshold - 0.05, 2))
            detector.conf_threshold = app_state.conf_threshold
        elif action == "conf_inc":
            app_state.conf_threshold = min(0.9, round(app_state.conf_threshold + 0.05, 2))
            detector.conf_threshold = app_state.conf_threshold
        elif action == "proc_dec":
            app_state.process_every_n_frames = max(1, app_state.process_every_n_frames - 1)
            for c in contexts:
                c.process_every_n_frames = app_state.process_every_n_frames
        elif action == "proc_inc":
            app_state.process_every_n_frames = min(10, app_state.process_every_n_frames + 1)
            for c in contexts:
                c.process_every_n_frames = app_state.process_every_n_frames
        elif action == "direction_toggle":
            selected = panel.selected_context()
            if selected is not None:
                selected.counter.toggle_direction_mode()
        elif action == "save":
            _save_runtime_config()

        for ctx in contexts:
            frame = ctx.stream.read()
            if frame is None:
                placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
                if ctx.stream.is_connecting:
                    status_msg = f"{ctx.name}: подключение к {ctx.stream.source}..."
                    status_color = (100, 255, 255)
                elif ctx.stream.connected:
                    status_msg = f"{ctx.name}: ожидание кадра..."
                    status_color = (100, 255, 180)
                else:
                    status_msg = f"{ctx.name}: нет соединения — переподключение..."
                    status_color = (100, 200, 255)
                draw_text(placeholder, status_msg, (30, 200), status_color, 24, shadow=True)
                draw_text(placeholder, f"Источник: {ctx.stream.source}", (30, 240), (180, 180, 180), 20)
                draw_text(placeholder, "Телефон: установите IP Webcam или DroidCam", (30, 280), (150, 150, 220), 18)
                cv2.imshow(ctx.name, placeholder)
                continue

            frame = _resize_keep_aspect(frame, ctx.resize_width)
            ctx.frame_index += 1
            should_process = (ctx.frame_index % ctx.process_every_n_frames) == 0

            if should_process:
                detections = detector.detect(frame)
                tracks = ctx.tracker.update(detections, frame.shape)
                ctx.last_tracks = tracks

                changed = ctx.counter.update(tracks)
                snapshot = ctx.counter.snapshot()
                if changed:
                    ctx.logger.log(
                        bus_id=app_state.bus_id,
                        door_id=ctx.door_id,
                        stop_id=app_state.stop_id,
                        count_in=snapshot["count_in"],
                        count_out=snapshot["count_out"],
                        event="counter_changed",
                    )

            if global_now - ctx.last_log_ts >= periodic_log_sec:
                snap = ctx.counter.snapshot()
                ctx.logger.log(
                    bus_id=app_state.bus_id,
                    door_id=ctx.door_id,
                    stop_id=app_state.stop_id,
                    count_in=snap["count_in"],
                    count_out=snap["count_out"],
                    event="periodic",
                )
                ctx.last_log_ts = global_now

            now = time.time()
            dt = max(1e-6, now - ctx.last_fps_ts) if ctx.last_fps_ts > 0 else 0.0
            if dt > 0:
                ctx.fps = 1.0 / dt
            ctx.last_fps_ts = now

            snap = ctx.counter.snapshot()
            header = f"Автобус: {app_state.bus_id}  Дверь: {ctx.door_id}  Остановка: {app_state.stop_id}"
            rendered = draw_scene(
                frame=frame,
                tracks=ctx.last_tracks,
                line_p1=ctx.counter.p1,
                line_p2=ctx.counter.p2,
                count_in=snap["count_in"],
                count_out=snap["count_out"],
                fps=ctx.fps,
                header=header,
            )
            cv2.imshow(ctx.name, rendered)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        panel.handle_key(key)
        submit = panel.pull_input_submit()
        if submit is not None:
            submit_action, submit_value = submit
            if submit_action == "rtsp_apply" and submit_value:
                selected = panel.selected_context()
                if selected is not None:
                    selected.stream.source = submit_value
                    if submit_value not in selected.source_options:
                        selected.source_options.insert(0, submit_value)
                        selected.source_index = 0
                    selected.stream.start_stream()

    for ctx in contexts:
        ctx.stream.stop()
    panel.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run("config.json")
    except Exception as exc:
        crash_file = _app_dir() / "crash.log"
        with open(crash_file, "a", encoding="utf-8") as f:
            f.write("\n=== Program12_Counter Crash ===\n")
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(str(exc) + "\n")
            f.write(traceback.format_exc() + "\n")
        print(f"Fatal error: {exc}")
        print(f"Details saved to: {crash_file}")
        input("Press Enter to exit...")
