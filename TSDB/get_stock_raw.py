import argparse
import requests
import pandas as pd
import sqlite3 as sqlite

from datetime import datetime, timedelta
from logging import getLogger

from .config import url, sqlite_db_name, table_name, TOKEN


logger = getLogger(__name__)

class LoggedError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        logger.error(msg)

class RespNot200Error(LoggedError):
    pass

class URLnotProvidedError(LoggedError):
    pass

class BadBaseDateError(LoggedError):
    pass

if not url:
    raise URLnotProvidedError("NO STOCK API URL ENDPOINT PROVIDED IN CONFIG!")


def get_data_as_df(params: dict):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    resp = requests.get(url, headers=headers, params=params)
    resp_code = resp.status_code
    if  resp_code == 200:
        data = resp.json()
        df = pd.DataFrame(data['data'])
        logger.info(f"API Call succeed, get {df.shape[1]} records")
    else:
        if "detail" in resp.json():
            resp_msg = resp.json()["detail"][0]['msg']
        else:
            resp_msg = resp.json()["msg"]
        err_msg = (f"API Call to {url} respond with code {resp_code}"
            f"params: {params}"
            f"error: {resp_msg}")
        raise RespNot200Error(err_msg)
    return df


def get_recent_txn_range(base_date: datetime = datetime.today(), n: int = -14):
    parameter = {
        "dataset": "TaiwanStockTradingDate",
    }
    txn_dates = get_data_as_df(parameter) # early -> old
    dt_fmt = "%Y-%m-%d"
    base_date_str = base_date.strftime(dt_fmt)
    end_ind = txn_dates[txn_dates.date == base_date_str].index
    
    if end_ind.empty:
        err_msg = "Bad Base Date, choose another one"
        raise BadBaseDateError(err_msg)
    
    if n == 0:
        return base_date_str, base_date_str
    
    start_ind = end_ind.item() + n
    # avoid lower or over limit
    start_ind = max(min(start_ind, txn_dates.shape[0]-1), 0)
    end_date = base_date_str
    start_date = txn_dates.loc[start_ind].values[0]

    if n > 0:
        start_date, end_date = end_date, start_date
    return start_date, end_date


def insert_txn_summary_data(start_date: datetime, end_date: datetime, stock_id_list: list[str]):
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": "",
        "start_date": start_date,
        "end_date": end_date
    }
    logger.info(f"insert data with parameter: {parameter}")
    if not stock_id_list:
        logger.warning("No stock_id_list provided")
    
    for stock_id in stock_id_list:
        parameter.update({"data_id": stock_id})
        data = get_data_as_df(parameter)

        if data.shape[0] == 0:
            logger.warning(f"no data found for {stock_id}")
        else:
            logger.info(f"got {data.shape[0]} records for {stock_id}")
            try:
                with sqlite.connect(sqlite_db_name) as conn:
                    # auto commit using with statement
                    cursor = conn.cursor()
                    delete_q = (f"delete from {table_name} where stock_id = {stock_id} "
                                   f"and price_date>= '{start_date}' "
                                   f"and price_date <= '{end_date}' ")
                    cursor.execute(delete_q)
                
                with sqlite.connect(sqlite_db_name) as conn:
                    # auto commit using with statement
                    cursor = conn.cursor()
                    cursor.executemany(f"""insert into {table_name} values (
                                    {",".join(data.shape[1]*"?")})""", list(data.values))
            except Exception as e:
                logger.error(f"Error while insert {stock_id} data into {table_name}"
                             f"error msg: {e}")
                raise
    logger.info("data succesefully insert")


if __name__ == '__main__':
    # get this stock txn dates
    parser = argparse.ArgumentParser(
                    prog='get_stock_raw',
                    description='download stock data')
    
    parser.add_argument('ind_cat', help="industry category for stock")
    
    parser.add_argument('--base_date', help="base date to download file"
                        "should be dates have stock transactions, default to today"
                        "format: yyyy-mm-dd")
    
    parser.add_argument('-n', help="int, how many days to download from base_date, "
                        "positive or negtive, default 0", default=0)

    args = parser.parse_args()

    if args.base_date:
        base_date = datetime.strptime(args.base_date, "%Y-%m-%d")
    else:
        base_date = datetime.today()
    n = args.n
    ind_cat = args.ind_cat

    start_date, end_date = get_recent_txn_range(base_date=base_date, n=n)

    # get todays stock list
    parameter = {
        "dataset": "TaiwanStockInfo",
    }
    stock_list = get_data_as_df(parameter)
    SOI = stock_list[stock_list["industry_category"]==ind_cat].stock_id.to_list()
    
    # insert raw data into db
    insert_txn_summary_data(start_date, end_date, SOI)
