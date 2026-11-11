"""SQLite storage for bioacoustic classification results."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bioacoustics.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS classifications (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    recording     TEXT NOT NULL,
    duration_s    REAL,
    sample_rate   INTEGER,
    segment       INTEGER NOT NULL,
    top_class     TEXT NOT NULL,
    probability   REAL NOT NULL,
    alternatives  TEXT,
    model_path    TEXT
);
CREATE INDEX IF NOT EXISTS idx_class_top_class ON classifications (top_class);
CREATE INDEX IF NOT EXISTS idx_class_created   ON classifications (created_at);
"""

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as con:
        con.executescript(_SCHEMA)

def save_classification(*, recording: str, segment: int, top_class: str,
                        probability: float, alternatives: list | None = None,
                        duration_s: float | None = None,
                        sample_rate: int | None = None,
                        model_path: str | None = None,
                        db_path: Path = DB_PATH) -> int:
    with get_connection(db_path) as con:
        cur = con.execute(
            """INSERT INTO classifications
               (created_at, recording, duration_s, sample_rate, segment,
                top_class, probability, alternatives, model_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             recording, duration_s, sample_rate, segment,
             top_class, float(probability),
             json.dumps(alternatives or [], ensure_ascii=False), model_path))
        return int(cur.lastrowid)

def get_history(limit: int = 100, species: str | None = None,
                db_path: Path = DB_PATH) -> pd.DataFrame:
    query = """SELECT created_at, recording, segment, top_class,
                      probability, duration_s, sample_rate
               FROM classifications"""
    params: list = []
    if species:
        query += " WHERE top_class = ?"
        params.append(species)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_connection(db_path) as con:
        return pd.read_sql_query(query, con, params=params)

def get_species_summary(db_path: Path = DB_PATH) -> pd.DataFrame:
    query = """SELECT top_class AS species,
                      COUNT(*) AS detections,
                      ROUND(AVG(probability), 3) AS mean_score,
                      ROUND(MAX(probability), 3) AS max_score,
                      MIN(created_at) AS first_seen,
                      MAX(created_at) AS last_seen
               FROM classifications
               GROUP BY top_class ORDER BY detections DESC"""
    with get_connection(db_path) as con:
        return pd.read_sql_query(query, con)

def get_recordings(db_path: Path = DB_PATH) -> pd.DataFrame:
    query = """SELECT recording, COUNT(*) AS segments,
                      MIN(created_at) AS analyzed_at
               FROM classifications GROUP BY recording
               ORDER BY analyzed_at DESC"""
    with get_connection(db_path) as con:
        return pd.read_sql_query(query, con)

def clear_history(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as con:
        con.execute("DELETE FROM classifications")