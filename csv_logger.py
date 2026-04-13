import csv
import os
from datetime import datetime
from typing import Optional


class CounterCsvLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "bus_id", "door_id", "stop_id", "count_in", "count_out", "event"])

    def log(self, bus_id: str, door_id: str, stop_id: str, count_in: int, count_out: int, event: Optional[str] = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ts, bus_id, door_id, stop_id, count_in, count_out, event or "periodic"])
