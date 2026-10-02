"""Prediction-history public API (implemented by the local SQLite service)."""

from .analytics import clear_history, init_db, load_history, record_prediction

__all__ = ["clear_history", "init_db", "load_history", "record_prediction"]
