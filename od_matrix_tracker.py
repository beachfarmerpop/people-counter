import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ODMatrixTracker:
    """День 4–5: Явное отслеживание OD-потоков (откуда-куда) с полной телеметрией."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Создаёст таблицы для OD-логирования и мониторинга потоков."""
        conn = sqlite3.connect(self._db_path)

        # Логирование отдельных переходов
        conn.execute('''
            CREATE TABLE IF NOT EXISTS od_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                passenger_id    TEXT NOT NULL,
                from_stop       TEXT NOT NULL,
                to_stop         TEXT,
                entry_time      TEXT,
                exit_time       TEXT,
                time_on_board_sec REAL,
                FOREIGN KEY (passenger_id) REFERENCES passengers(passenger_id)
            )
        ''')

        # Сводная матрица (кеш)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS od_matrix (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                from_stop       TEXT NOT NULL,
                to_stop         TEXT NOT NULL,
                passenger_count INTEGER DEFAULT 0,
                UNIQUE(from_stop, to_stop)
            )
        ''')

        conn.commit()
        conn.close()

    def log_entry(self, passenger_id: str, from_stop: str) -> None:
        """Логирует вход пассажира на остановку."""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            conn = sqlite3.connect(self._db_path)
            # Ищем открытую запись (to_stop == NULL)
            existing = conn.execute(
                'SELECT id FROM od_log WHERE passenger_id=? AND to_stop IS NULL ORDER BY id DESC LIMIT 1',
                (passenger_id,)
            ).fetchone()
            if existing:
                # Обновляем предыдущую запись: закрываем на той же остановке (если не уехал)
                conn.execute(
                    'UPDATE od_log SET to_stop=from_stop, exit_time=? WHERE id=?',
                    (ts, existing[0])
                )
            # Вставляем новую запись входа
            conn.execute(
                'INSERT INTO od_log (passenger_id, from_stop, entry_time, timestamp) VALUES (?,?,?,?)',
                (passenger_id, from_stop, ts, ts)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def log_exit(self, passenger_id: str, to_stop: str) -> None:
        """Логирует выход пассажира на остановку."""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            conn = sqlite3.connect(self._db_path)
            # Найдём открытую запись (to_stop == NULL)
            row = conn.execute(
                'SELECT id, entry_time FROM od_log WHERE passenger_id=? AND to_stop IS NULL ORDER BY id DESC LIMIT 1',
                (passenger_id,)
            ).fetchone()
            if row:
                rid, entry_ts = row
                # Ставим to_stop и считаем время в пути
                time_on_board = None
                if entry_ts:
                    try:
                        et = datetime.strptime(entry_ts, '%Y-%m-%d %H:%M:%S')
                        xt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        time_on_board = (xt - et).total_seconds()
                    except Exception:
                        pass
                conn.execute(
                    'UPDATE od_log SET to_stop=?, exit_time=?, time_on_board_sec=? WHERE id=?',
                    (to_stop, ts, time_on_board, rid)
                )
                # Обновляем матрицу
                from_stop = conn.execute(
                    'SELECT from_stop FROM od_log WHERE id=?', (rid,)
                ).fetchone()[0]
                self._update_matrix(from_stop, to_stop)
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _update_matrix(self, from_stop: str, to_stop: str) -> None:
        """Обновляет строку матрицы."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute('''
                INSERT OR IGNORE INTO od_matrix (from_stop, to_stop, passenger_count)
                VALUES (?, ?, 0)
            ''', (from_stop, to_stop))
            conn.execute(
                'UPDATE od_matrix SET passenger_count = passenger_count + 1 WHERE from_stop=? AND to_stop=?',
                (from_stop, to_stop)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_od_matrix(self) -> Dict[str, Dict[str, int]]:
        """Возвращает OD-матрицу как {from_stop: {to_stop: count}}."""
        matrix = defaultdict(dict)
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute('SELECT from_stop, to_stop, passenger_count FROM od_matrix').fetchall()
            conn.close()
            for fr, to, cnt in rows:
                matrix[fr][to] = cnt
        except Exception:
            pass
        return dict(matrix)

    def export_od_json(self, filepath: str) -> bool:
        """День 5: Экспортирует OD-матрицу в JSON."""
        try:
            matrix = self.get_od_matrix()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(matrix, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def export_od_csv(self, filepath: str) -> bool:
        """День 5: Экспортирует OD-матрицу в CSV."""
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                'SELECT from_stop, to_stop, passenger_count FROM od_matrix ORDER BY from_stop, to_stop'
            ).fetchall()
            conn.close()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('От остановки,На остановку,Пассажиров\n')
                for fr, to, cnt in rows:
                    f.write(f'{fr},{to},{cnt}\n')
            return True
        except Exception:
            return False

    def get_od_html_table(self) -> str:
        """День 6: Возвращает OD-матрицу в виде HTML таблицы для веб-дашборда."""
        try:
            conn = sqlite3.connect(self._db_path)
            stops = sorted(set(
                list(conn.execute('SELECT DISTINCT from_stop FROM od_log WHERE to_stop IS NOT NULL').fetchall()) +
                list(conn.execute('SELECT DISTINCT to_stop FROM od_log WHERE to_stop IS NOT NULL').fetchall())
            ))
            stops = [s[0] if isinstance(s, tuple) else s for s in stops if s]

            if not stops:
                conn.close()
                return '<p>Нет данных для матрицы.</p>'

            matrix = self.get_od_matrix()
            html = '<table style="width:100%; border-collapse:collapse"><tr><th style="border:1px solid #ccc">От</th>'
            for to in stops:
                html += f'<th style="border:1px solid #ccc">{to}</th>'
            html += '</tr>'
            for fr in stops:
                html += f'<tr><td style="border:1px solid #ccc;font-weight:bold">{fr}</td>'
                for to in stops:
                    cnt = matrix.get(fr, {}).get(to, 0)
                    color = '#e8f5e9' if cnt > 0 else '#f5f5f5'
                    html += f'<td style="border:1px solid #ccc;background:{color};text-align:center">{cnt}</td>'
                html += '</tr>'
            html += '</table>'
            conn.close()
            return html
        except Exception:
            return '<p>Ошибка при формировании таблицы.</p>'

    def get_duration_stats(self) -> Dict:
        """День 6: Статистика по времени в дороге."""
        stats = {'avg_sec': 0, 'min_sec': None, 'max_sec': None, 'total_trips': 0}
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                'SELECT time_on_board_sec FROM od_log WHERE time_on_board_sec IS NOT NULL'
            ).fetchall()
            conn.close()
            if rows:
                times = [float(r[0]) for r in rows if r[0] is not None]
                if times:
                    stats['avg_sec'] = sum(times) / len(times)
                    stats['min_sec'] = min(times)
                    stats['max_sec'] = max(times)
                    stats['total_trips'] = len(times)
        except Exception:
            pass
        return stats
