"""
cross_route_stitcher.py — Ночная сшивка траекторий пассажиров через маршруты.

Задача: объединить поездки одного и того же человека на разных маршрутах
в единую цепочку перемещений по транспортной сети.

Пример:
    Маршрут 5: P0003 сел «Центр» → вышел «Вокзал»  (09:15–09:40)
    Маршрут 12: P0017 сел «Вокзал» → вышел «Парк»   (09:52–10:10)
    → если P0003 и P0017 — один человек → цепочка: Центр → Вокзал → Парк

Запуск:
    python cross_route_stitcher.py                   # из текущей папки, passenger_flow.db
    python cross_route_stitcher.py db1.db db2.db     # несколько БД с разных бортов
    python cross_route_stitcher.py --all *.db         # все БД в папке
"""

import glob
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Настройки ─────────────────────────────────────────────────────────────────
DESCRIPTOR_MATCH_THRESHOLD = 0.25   # порог схожести дескрипторов (лицо)
APPEAR_MATCH_THRESHOLD = 0.20       # порог схожести по внешности
COMBINED_WEIGHT_FACE = 0.65         # вес лицевого дескриптора
COMBINED_WEIGHT_APPEAR = 0.35       # вес дескриптора внешности
MAX_TRANSFER_MINUTES = 90           # макс. время пересадки (минут)
MIN_TRANSFER_MINUTES = 1            # мин. время пересадки (минут)

OUTPUT_FOLDER = 'reports'


def load_passengers(db_path: str) -> List[Dict]:
    """Загружает пассажиров из одной БД."""
    result = []
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            'SELECT passenger_id, face_descriptor, appear_descriptor, '
            'first_seen, last_seen, from_stop, to_stop '
            'FROM passengers'
        ).fetchall()
        conn.close()
    except Exception:
        return result

    for r in rows:
        face_desc = None
        appear_desc = None
        try:
            face_desc = np.asarray(json.loads(r[1]), dtype=np.float32) if r[1] else None
        except Exception:
            pass
        try:
            appear_desc = np.asarray(json.loads(r[2]), dtype=np.float32) if r[2] else None
        except Exception:
            pass

        result.append({
            'pid': r[0],
            'db': db_path,
            'face_desc': face_desc,
            'appear_desc': appear_desc,
            'first_seen': r[3],
            'last_seen': r[4],
            'from_stop': r[5] or '',
            'to_stop': r[6] or '',
        })
    return result


def load_trajectories(db_path: str) -> List[Dict]:
    """Загружает траектории из одной БД."""
    result = []
    try:
        conn = sqlite3.connect(db_path)
        # Проверяем наличие таблицы
        has = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='passenger_trajectories'"
        ).fetchone()
        if not has:
            conn.close()
            return result
        rows = conn.execute(
            'SELECT passenger_id, route, entry_stop, exit_stop, '
            'stops_visited, entry_time, exit_time '
            'FROM passenger_trajectories ORDER BY entry_time'
        ).fetchall()
        conn.close()
    except Exception:
        return result

    for r in rows:
        stops = []
        try:
            stops = json.loads(r[4]) if r[4] else []
        except Exception:
            pass
        result.append({
            'pid': r[0],
            'db': db_path,
            'route': r[1] or '',
            'entry_stop': r[2] or '',
            'exit_stop': r[3] or '',
            'stops_visited': stops,
            'entry_time': r[5] or '',
            'exit_time': r[6] or '',
        })
    return result


def descriptor_distance(d1: Optional[np.ndarray], d2: Optional[np.ndarray]) -> float:
    """Евклидовое расстояние между нормализованными дескрипторами."""
    if d1 is None or d2 is None:
        return 1.0
    n = min(len(d1), len(d2))
    if n == 0:
        return 1.0
    return float(np.linalg.norm(d1[:n] - d2[:n]) / max(1.0, math.sqrt(n)))


def combined_distance(p1: Dict, p2: Dict) -> float:
    """Комбинированное расстояние (лицо + внешность)."""
    d_face = descriptor_distance(p1.get('face_desc'), p2.get('face_desc'))
    d_app = descriptor_distance(p1.get('appear_desc'), p2.get('appear_desc'))

    has_face = (p1.get('face_desc') is not None and p2.get('face_desc') is not None
                and len(p1['face_desc']) > 0 and len(p2['face_desc']) > 0)
    has_app = (p1.get('appear_desc') is not None and p2.get('appear_desc') is not None
               and len(p1['appear_desc']) > 0 and len(p2['appear_desc']) > 0)

    if has_face and has_app:
        return COMBINED_WEIGHT_FACE * d_face + COMBINED_WEIGHT_APPEAR * d_app
    elif has_face:
        return d_face
    elif has_app:
        return d_app
    return 1.0


def match_passengers_across_dbs(all_passengers: List[Dict]) -> Dict[str, str]:
    """Сопоставляет пассажиров из разных БД/маршрутов И внутри одной БД.

    Сравнивает каждую пару по комбинированному дескриптору.
    Возвращает словарь {(db, pid) → global_id}: унификация номеров.
    """
    n = len(all_passengers)
    # Union-Find для группировки одинаковых пассажиров
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Сравниваем каждую пару
    for i in range(n):
        for j in range(i + 1, n):
            # Не сравниваем пассажира с самим собой (одинаковый pid+db)
            if (all_passengers[i]['db'] == all_passengers[j]['db'] and
                    all_passengers[i]['pid'] == all_passengers[j]['pid']):
                continue
            d = combined_distance(all_passengers[i], all_passengers[j])
            if d < DESCRIPTOR_MATCH_THRESHOLD:
                union(i, j)

    # Присваиваем глобальные ID по группам
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    mapping = {}
    gid_counter = 0
    for root, members in groups.items():
        gid_counter += 1
        gid = f'G{gid_counter:04d}'
        for idx in members:
            p = all_passengers[idx]
            mapping[(p['db'], p['pid'])] = gid

    return mapping


def parse_dt(s: str) -> Optional[datetime]:
    """Парсит строку с датой/временем."""
    if not s:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%H:%M:%S  %d.%m.%Y', '%H:%M:%S %d.%m.%Y'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def stitch_trajectories(all_trajectories: List[Dict],
                        mapping: Dict[Tuple[str, str], str]) -> List[Dict]:
    """Сшивает траектории одного пассажира через маршруты в цепочки.

    Логика:
      1. Группируем все поездки по global_id
      2. Сортируем по времени
      3. Если exit_stop одной поездки == entry_stop следующей
         и разрыв по времени < MAX_TRANSFER_MINUTES → сшиваем
    """
    # Привязываем global_id
    for t in all_trajectories:
        key = (t['db'], t['pid'])
        t['gid'] = mapping.get(key, t['pid'])

    # Группируем по gid
    by_gid = defaultdict(list)
    for t in all_trajectories:
        by_gid[t['gid']].append(t)

    chains = []
    for gid, trips in by_gid.items():
        # Сортировка по entry_time
        trips.sort(key=lambda x: x.get('entry_time', '') or '')

        # Сшиваем последовательные поездки
        current_chain = [trips[0]] if trips else []
        for i in range(1, len(trips)):
            prev = current_chain[-1]
            curr = trips[i]

            prev_exit_dt = parse_dt(prev.get('exit_time', ''))
            curr_entry_dt = parse_dt(curr.get('entry_time', ''))

            can_stitch = False
            if prev_exit_dt and curr_entry_dt:
                gap = (curr_entry_dt - prev_exit_dt).total_seconds() / 60.0
                if MIN_TRANSFER_MINUTES <= gap <= MAX_TRANSFER_MINUTES:
                    # Остановка выхода совпадает с остановкой входа (пересадка)
                    if prev.get('exit_stop') and prev['exit_stop'] == curr.get('entry_stop'):
                        can_stitch = True
                    # Или просто временное окно подходит (разные остановки — пешком?)
                    elif gap <= 30:
                        can_stitch = True

            if can_stitch:
                current_chain.append(curr)
            else:
                chains.append(_build_chain_record(gid, current_chain))
                current_chain = [curr]

        if current_chain:
            chains.append(_build_chain_record(gid, current_chain))

    return chains


def _build_chain_record(gid: str, trips: List[Dict]) -> Dict:
    """Формирует запись о цепочке поездок."""
    all_stops = []
    routes = []
    for t in trips:
        route = t.get('route', '?')
        if route not in routes:
            routes.append(route)
        stops = t.get('stops_visited', [])
        if isinstance(stops, str):
            stops = [s.strip() for s in stops.split('→') if s.strip()]
        for s in stops:
            if not all_stops or all_stops[-1] != s:
                all_stops.append(s)

    first = trips[0]
    last = trips[-1]

    entry_dt = parse_dt(first.get('entry_time', ''))
    exit_dt = parse_dt(last.get('exit_time', ''))
    duration_min = None
    if entry_dt and exit_dt:
        duration_min = round((exit_dt - entry_dt).total_seconds() / 60.0, 1)

    return {
        'global_id': gid,
        'legs': len(trips),
        'routes': ' → '.join(routes),
        'origin': first.get('entry_stop', '?'),
        'destination': last.get('exit_stop', '?'),
        'full_path': ' → '.join(all_stops) if all_stops else '?',
        'start_time': first.get('entry_time', ''),
        'end_time': last.get('exit_time', ''),
        'duration_min': duration_min,
        'transfers': len(trips) - 1,
    }


def export_chains_json(chains: List[Dict], path: str) -> None:
    """Экспорт сшитых цепочек в JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chains, f, ensure_ascii=False, indent=2)


def export_chains_csv(chains: List[Dict], path: str) -> None:
    """Экспорт сшитых цепочек в CSV."""
    import csv
    if not chains:
        return
    keys = ['global_id', 'legs', 'routes', 'origin', 'destination',
            'full_path', 'start_time', 'end_time', 'duration_min', 'transfers']
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=';')
        w.writeheader()
        for c in chains:
            w.writerow({k: c.get(k, '') for k in keys})


def print_summary(chains: List[Dict]) -> None:
    """Выводит сводку по сшитым маршрутам."""
    total = len(chains)
    multi_leg = sum(1 for c in chains if c['legs'] > 1)
    max_legs = max((c['legs'] for c in chains), default=0)
    avg_dur = 0
    dur_count = 0
    for c in chains:
        if c['duration_min'] is not None:
            avg_dur += c['duration_min']
            dur_count += 1
    avg_dur = round(avg_dur / max(1, dur_count), 1)

    print('\n' + '=' * 60)
    print('  СШИВКА ТРАЕКТОРИЙ — РЕЗУЛЬТАТЫ')
    print('=' * 60)
    print(f'  Всего цепочек поездок:    {total}')
    print(f'  Мультимаршрутных (≥2):    {multi_leg}')
    print(f'  Макс. пересадок:          {max_legs - 1}')
    print(f'  Средняя длительность:     {avg_dur} мин')
    print()

    if multi_leg > 0:
        print('  Примеры мультимаршрутных поездок:')
        shown = 0
        for c in chains:
            if c['legs'] > 1 and shown < 5:
                print(f'    {c["global_id"]}: {c["full_path"]}')
                print(f'      маршруты: {c["routes"]}, пересадок: {c["transfers"]}, '
                      f'длительность: {c["duration_min"]} мин')
                shown += 1
        print()


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    db_paths = []

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '--all':
                continue
            for p in glob.glob(arg):
                if p.endswith('.db') and os.path.isfile(p):
                    db_paths.append(p)
    else:
        # По умолчанию — passenger_flow.db в текущей папке
        default = 'passenger_flow.db'
        if os.path.isfile(default):
            db_paths.append(default)

    if not db_paths:
        print('Не найдено файлов БД. Укажите пути:')
        print('  python cross_route_stitcher.py passenger_flow.db')
        print('  python cross_route_stitcher.py db1.db db2.db')
        print('  python cross_route_stitcher.py --all *.db')
        return

    print(f'Загрузка из {len(db_paths)} БД: {", ".join(db_paths)}')

    # 1. Загружаем пассажиров и траектории
    all_passengers = []
    all_trajectories = []
    for db in db_paths:
        pax = load_passengers(db)
        traj = load_trajectories(db)
        print(f'  {db}: {len(pax)} пассажиров, {len(traj)} траекторий')
        all_passengers.extend(pax)
        all_trajectories.extend(traj)

    if not all_passengers:
        print('Нет данных о пассажирах.')
        return

    # 2. Межмаршрутное сопоставление
    print('\nСопоставление пассажиров по дескрипторам...')
    mapping = match_passengers_across_dbs(all_passengers)
    unique_globals = len(set(mapping.values()))
    print(f'  Локальных ID: {len(mapping)} → Глобальных: {unique_globals}')

    # 3. Сшивка траекторий
    if all_trajectories:
        print('Сшивка траекторий...')
        chains = stitch_trajectories(all_trajectories, mapping)
        print_summary(chains)

        # 4. Экспорт
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        json_path = os.path.join(OUTPUT_FOLDER, 'cross_route_chains.json')
        csv_path = os.path.join(OUTPUT_FOLDER, 'cross_route_chains.csv')
        export_chains_json(chains, json_path)
        export_chains_csv(chains, csv_path)
        print(f'Экспорт: {json_path}, {csv_path}')
    else:
        print('Нет траекторий для сшивки.')

    # 5. Сохраняем результат сопоставления в БД (первую)
    try:
        conn = sqlite3.connect(db_paths[0])
        conn.execute('''
            CREATE TABLE IF NOT EXISTS global_passenger_map (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db   TEXT,
                local_pid   TEXT,
                global_pid  TEXT,
                stitched_at TEXT
            )
        ''')
        conn.execute('DELETE FROM global_passenger_map')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for (db, pid), gid in mapping.items():
            conn.execute(
                'INSERT INTO global_passenger_map (source_db, local_pid, global_pid, stitched_at) '
                'VALUES (?,?,?,?)',
                (db, pid, gid, ts)
            )
        conn.commit()
        conn.close()
        print(f'Маппинг сохранён в {db_paths[0]} (таблица global_passenger_map)')
    except Exception as e:
        print(f'Ошибка сохранения маппинга: {e}')

    print('\nГотово!')


if __name__ == '__main__':
    main()
