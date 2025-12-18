import pytest
import pandas as pd
from unittest.mock import patch, Mock
from datetime import datetime

from TSDB.get_stock_raw import (
    get_data_as_df, RespNot200Error, get_recent_txn_range,
    insert_txn_summary_data
)


@patch("TSDB.get_stock_raw.requests.get")
def test_get_data_as_df_success(mock_get):
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
        ]
    }
    mock_get.return_value = mock_resp

    df = get_data_as_df({"q": "test"})

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["id", "name"]


@patch("TSDB.get_stock_raw.requests.get")
def test_get_data_as_df_error(mock_get):
    mock_resp = Mock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "detail": [{"msg": "bad request"}]
    }
    mock_get.return_value = mock_resp
    
    with pytest.raises(RespNot200Error) as exc:
        get_data_as_df({"q": "bad"})
    
    assert "bad request" in str(exc.value)


@patch("TSDB.get_stock_raw.get_data_as_df")
def test_get_recent_txn_range(mock_get_data, fake_txn_dates):
    # return the fake dataframe
    mock_get_data.return_value = fake_txn_dates

    base_date = datetime(2025, 12, 12)
    
    # test default n=-14
    start_date, end_date = get_recent_txn_range(base_date=base_date)
    
    # check that end_date is base_date
    assert end_date == "2025-12-12"
    # check that start_date is 14 trading days before (depending on your fake data)
    # since we only have 11 days in fake, it will pick the earliest
    assert start_date == "2025-12-01"


@patch("TSDB.get_stock_raw.get_data_as_df")
def test_get_recent_txn_range_positive_n(mock_get_data, fake_txn_dates):
    mock_get_data.return_value = fake_txn_dates
    base_date = datetime(2025, 12, 1)
    
    start_date, end_date = get_recent_txn_range(base_date=base_date, n=3)
    
    # When n > 0, start and end are swapped
    assert start_date == "2025-12-01"
    assert end_date == "2025-12-04"


@patch("TSDB.get_stock_raw.get_data_as_df")
@patch("TSDB.get_stock_raw.sqlite.connect")
def test_insert_txn_summary_data(mock_connect, mock_get_data, stock_ids, date_range, sample_df):
    mock_get_data.return_value = sample_df

    # mock sqlite connection and cursor
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value.__enter__.return_value = mock_conn

    start_date, end_date = date_range

    # Call function
    insert_txn_summary_data(start_date, end_date, stock_ids)

    # Assertions

    # get_data_as_df called for each stock_id
    assert mock_get_data.call_count == len(stock_ids)

    # SQLite connection called twice per stock_id (delete + insert)
    assert mock_connect.call_count == 2 * len(stock_ids)

    # Cursor executed delete and executemany
    assert mock_cursor.execute.call_count == len(stock_ids)
    assert mock_cursor.executemany.call_count == len(stock_ids)



@patch("TSDB.get_stock_raw.get_data_as_df")
@patch("TSDB.get_stock_raw.sqlite.connect")
def test_insert_txn_summary_data_empty_list(mock_connect, mock_get_data, date_range):
    mock_get_data.return_value = pd.DataFrame(columns=["stock_id", "price_date", "close"])
    start_date, end_date = date_range

    insert_txn_summary_data(start_date, end_date, [])

    # get_data_as_df and sqlite.connect should NOT be called
    mock_get_data.assert_not_called()
    mock_connect.assert_not_called()