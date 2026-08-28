import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .config import HISTORY_DB_PATH


def init_db(db_path: Path = HISTORY_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                uncertainty TEXT,
                top_3_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def record_prediction(
    filename: str,
    predicted_class: str,
    confidence: float,
    confidence_level: str,
    uncertainty: Optional[str],
    top_3_predictions: Iterable[dict],
    db_path: Path = HISTORY_DB_PATH,
) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                timestamp, filename, predicted_class, confidence,
                confidence_level, uncertainty, top_3_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                filename,
                predicted_class,
                confidence,
                confidence_level,
                uncertainty,
                json.dumps(list(top_3_predictions)),
            ),
        )
        connection.commit()


def load_history(db_path: Path = HISTORY_DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as connection:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY timestamp DESC",
            connection,
        )


def clear_history(db_path: Path = HISTORY_DB_PATH) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("DELETE FROM predictions")
        connection.commit()
