from src.history import clear_history, load_history, record_prediction


def test_history_round_trip(tmp_path):
    db_path = tmp_path / "history.db"
    record_prediction(
        "bottle.jpg", "plastic", 0.8, "High", None,
        [{"class_name": "plastic", "probability": 0.8}], db_path,
    )
    history = load_history(db_path)
    assert len(history) == 1
    assert history.iloc[0]["predicted_class"] == "plastic"
    clear_history(db_path)
    assert load_history(db_path).empty
