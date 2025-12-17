import pytest
import pandas as pd

from datetime import datetime


@pytest.fixture
def fake_txn_dates():
# Sample trading dates
    df = pd.DataFrame({
        "date": [
            "2025-12-01",
            "2025-12-02",
            "2025-12-03",
            "2025-12-04",
            "2025-12-05",
            "2025-12-08",
            "2025-12-09",
            "2025-12-10",
            "2025-12-11",
            "2025-12-12",
            "2025-12-15",
        ]
    })
    return df

@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "stock_id": ["2330", "2330"],
        "price_date": ["2025-12-01", "2025-12-02"],
        "close": [600, 610]
    })
    return df

@pytest.fixture
def stock_ids():
    return ["2330", "2317"]

@pytest.fixture
def date_range():
    start = datetime(2025, 12, 1)
    end = datetime(2025, 12, 2)
    return start, end