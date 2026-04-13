from pathlib import Path
from typing import Dict, List


def _ask(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _to_int(value: str, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _edit_streams(cfg: Dict) -> None:
    streams: List[Dict] = cfg.get("streams", [])
    if not streams:
        streams = [{"name": "Door 1", "door_id": "1", "source": 0, "enabled": True}]

    print("\n--- Stream Setup (RTSP/USB/File) ---")
    count = _to_int(_ask("Number of doors/streams", str(len(streams))), len(streams))
    count = max(1, min(6, count))

    new_streams: List[Dict] = []
    for i in range(count):
        prev = streams[i] if i < len(streams) else {}
        name = _ask(f"Door {i+1} name", str(prev.get("name", f"Door {i+1}")))
        door_id = _ask(f"Door {i+1} id", str(prev.get("door_id", i + 1)))
        enabled = _ask(f"Door {i+1} enabled (y/n)", "y").lower().startswith("y")

        print("Source type: 1=USB camera, 2=RTSP URL, 3=Video file")
        source_type = _ask("Choose source type", "1")
        if source_type == "1":
            src_default = str(prev.get("source", 0)) if isinstance(prev.get("source", 0), int) else "0"
            source = _to_int(_ask("USB camera index", src_default), 0)
        elif source_type == "2":
            source = _ask("RTSP URL", str(prev.get("source", "rtsp://user:pass@ip:554/stream")))
        else:
            source = _ask("Video file path", str(prev.get("source", "Video.mp4")))

        line_default = prev.get("line", [[50, 260], [900, 260]])
        x1 = _to_int(_ask("Line x1", str(line_default[0][0])), 50)
        y1 = _to_int(_ask("Line y1", str(line_default[0][1])), 260)
        x2 = _to_int(_ask("Line x2", str(line_default[1][0])), 900)
        y2 = _to_int(_ask("Line y2", str(line_default[1][1])), 260)

        stream = {
            "name": name,
            "door_id": door_id,
            "enabled": enabled,
            "source": source,
            "line": [[x1, y1], [x2, y2]],
            "csv_path": f"reports/door_{door_id}_counts.csv",
        }
        new_streams.append(stream)

    cfg["streams"] = new_streams


def _edit_common(cfg: Dict) -> None:
    print("\n--- Common Settings ---")
    cfg["bus_id"] = _ask("Bus ID", str(cfg.get("bus_id", "bus-1")))
    cfg["stop_id"] = _ask("Start stop ID", str(cfg.get("stop_id", "1")))
    cfg["conf_threshold"] = float(_ask("Confidence threshold", str(cfg.get("conf_threshold", 0.4))))
    cfg["process_every_n_frames"] = _to_int(
        _ask("Process every N frames", str(cfg.get("process_every_n_frames", 2))),
        2,
    )
    cfg["resize_width"] = _to_int(_ask("Resize width", str(cfg.get("resize_width", 960))), 960)
    cfg["periodic_log_sec"] = _to_int(_ask("Periodic log sec", str(cfg.get("periodic_log_sec", 10))), 10)


def configure_before_start(cfg: Dict, app_dir: Path) -> Dict:
    print("\nProgram12 Counter - Startup Menu")
    print("1) Quick start (current settings)")
    print("2) Setup RTSP/USB sources")
    print("3) Setup common settings")
    print("4) Full setup")
    print("5) Exit")

    choice = _ask("Choose", "1")
    if choice == "5":
        raise KeyboardInterrupt("User canceled")
    if choice in ("2", "4"):
        _edit_streams(cfg)
    if choice in ("3", "4"):
        _edit_common(cfg)

    save_now = _ask("Save settings to config.json (y/n)", "y").lower().startswith("y")
    if save_now:
        cfg_path = app_dir / "config.json"
        cfg_path.write_text(__import__("json").dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {cfg_path}")

    return cfg
