import sqlite3 as sqlite
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from TSDB.preserve_feature import preprocess, preserve_features


def test_preprocess_basic():
    df = pd.DataFrame({
        "stock_id": ["A"] * 6 + ["B"] * 5,
        "price_date": pd.to_datetime([
            "2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-06",
            "2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05",
        ]),
        "open_price":  [10,11,12,13,14,15, 20,21,22,23,24],
        "max_price":   [11,12,13,14,15,16, 21,22,23,24,25],
        "min_price":   [9,10,11,12,13,14, 19,20,21,22,23],
        "close_price": [10,11,12,13,14,15, 20,21,22,23,24],
        "trading_volume": [100]*6 + [200]*5,
        "spread": [1]*11,
        "trading_turnover": [1000]*11,
    })

    result_df, features = preprocess(df)

    # zero close_price removed
    assert (result_df["close_price"] == 0).sum() == 0

    # sorted by stock_id, price_date
    assert result_df.equals(
        result_df.sort_values(["stock_id", "price_date"])
    )

    # required features returned
    assert features == [
        "open_price", "max_price", "min_price", "close_price",
        "trading_volume", "spread", "trading_turnover", "ma5", "returns"
    ]

    # new columns exist
    assert "ma5" in result_df.columns
    assert "returns" in result_df.columns

    # no NaN after fill
    assert result_df.isna().sum().sum() == 0



def test_preserve_features_sqlite(tmp_path):
    db = tmp_path / "test.db"
    table = "features"

    df = pd.DataFrame({
        "stock_id": ["A", "A", "B"],
        "price_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "close_price": [10.0, 11.0, 20.0],
    })

    with sqlite.connect(db) as conn:
        conn.execute("""
            create table features (
                stock_id text,
                price_date text,
                close_price real
            )
        """)
        conn.execute("""
            insert into features values ('A','2024-01-01',99.0)
        """)

    preserve_features(df, str(db), table)

    with sqlite.connect(db) as conn:
        rows = conn.execute(
            "select * from features order by price_date, stock_id"
        ).fetchall()

    assert rows == [
        ("A", "2024-01-01", 10.0),
        ("A", "2024-01-01", 11.0),
        ("B", "2024-01-02", 20.0),
    ]

    # explicit cleanup
    db.unlink()