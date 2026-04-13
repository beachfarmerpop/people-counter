# Program12 Counter (Production Rewrite)

This repository contains the updated Program12 people counter with a modular pipeline and a button-based control panel.

## Pipeline
- RTSP/USB/File stream -> YOLO person detection -> ByteTrack IDs -> line crossing counter -> CSV logging -> UI

## Main modules
- `main.py` - app entrypoint and runtime flow
- `video_stream.py` - stream open/reconnect logic
- `detector.py` - YOLO person detector
- `tracker.py` - ByteTrack wrapper (`track_id`, bbox, center)
- `counter.py` - line crossing logic and one-count-per-id protection
- `ui.py` - per-door window rendering
- `control_panel.py` - button-based control UI
- `text_render.py` - Unicode/Cyrillic text rendering via Pillow
- `csv_logger.py` - periodic + event-based CSV logging

## Control panel features
- Change bus/stop values with buttons
- Select active door
- Switch source (previous/next)
- Quick USB buttons (`USB 0/1/2`)
- Enter RTSP URL directly in panel (`RTSP URL` button, then Enter)
- Move line up/down and drag line with mouse in door window
- Change confidence and processing speed
- Save runtime settings to `config.json`

## Build output
- Local build artifacts are generated under `dist/` and excluded from git.
