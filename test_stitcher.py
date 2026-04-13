"""Тест сшивки траекторий через маршруты."""
import json
import os
import sqlite3
import tempfile

from cross_route_stitcher import (
    load_passengers, load_trajectories,
    match_passengers_across_dbs, stitch_trajectories
)

db = tempfile.mktemp(suffix='.db')
conn = sqlite3.connect(db)
conn.execute(
    'CREATE TABLE passengers ('
    'id INTEGER PRIMARY KEY, passenger_id TEXT UNIQUE, '
    'face_descriptor TEXT, appear_descriptor TEXT, '
    'first_seen TEXT, last_seen TEXT, '
    'encounter_count INTEGER DEFAULT 1, from_stop TEXT, to_stop TEXT)'
)
conn.execute(
    'CREATE TABLE passenger_trajectories ('
    'id INTEGER PRIMARY KEY, passenger_id TEXT, route TEXT, '
    'entry_stop TEXT, exit_stop TEXT, stops_visited TEXT, '
    'entry_time TEXT, exit_time TEXT)'
)

# Один пассажир с одинаковым дескриптором — два маршрута
desc_same = json.dumps([0.5] * 10)
conn.execute(
    'INSERT INTO passengers VALUES (1, ?, ?, ?, ?, ?, 2, ?, ?)',
    ('P0001', '[]', desc_same, '2026-04-09 09:15', '2026-04-09 09:40', 'Центр', 'Вокзал')
)
conn.execute(
    'INSERT INTO passengers VALUES (2, ?, ?, ?, ?, ?, 1, ?, ?)',
    ('P0002', '[]', desc_same, '2026-04-09 09:52', '2026-04-09 10:10', 'Вокзал', 'Парк')
)

# Другой пассажир — другой дескриптор
desc_diff = json.dumps([0.9] * 10)
conn.execute(
    'INSERT INTO passengers VALUES (3, ?, ?, ?, ?, ?, 1, ?, ?)',
    ('P0003', '[]', desc_diff, '2026-04-09 08:00', '2026-04-09 08:30', 'Школа', 'Больница')
)

# Траектории
conn.execute(
    'INSERT INTO passenger_trajectories VALUES (1, ?, ?, ?, ?, ?, ?, ?)',
    ('P0001', 'Маршрут 5', 'Центр', 'Вокзал', json.dumps(['Центр', 'Вокзал']),
     '2026-04-09 09:15', '2026-04-09 09:40')
)
conn.execute(
    'INSERT INTO passenger_trajectories VALUES (2, ?, ?, ?, ?, ?, ?, ?)',
    ('P0002', 'Маршрут 12', 'Вокзал', 'Парк', json.dumps(['Вокзал', 'Парк']),
     '2026-04-09 09:52', '2026-04-09 10:10')
)
conn.execute(
    'INSERT INTO passenger_trajectories VALUES (3, ?, ?, ?, ?, ?, ?, ?)',
    ('P0003', 'Маршрут 3', 'Школа', 'Больница', json.dumps(['Школа', 'Больница']),
     '2026-04-09 08:00', '2026-04-09 08:30')
)
conn.commit()
conn.close()

# Тест
pax = load_passengers(db)
traj = load_trajectories(db)
print(f'Пассажиров: {len(pax)}, Траекторий: {len(traj)}')

mapping = match_passengers_across_dbs(pax)
unique = len(set(mapping.values()))
print(f'Локальных ID: {len(mapping)} -> Глобальных: {unique}')

chains = stitch_trajectories(traj, mapping)
print(f'Цепочек поездок: {len(chains)}')
for c in chains:
    print(f'  {c["global_id"]}: {c["full_path"]} (маршруты: {c["routes"]}, пересадок: {c["transfers"]})')

os.unlink(db)
print('\nТест пройден!')
