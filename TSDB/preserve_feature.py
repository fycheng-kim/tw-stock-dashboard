import argparse
import sqlite3  as sqlite
import pandas as pd
import logging

from .config import (
    sqlite_db_name_raw, table_name_price, 
    sqlite_db_name_feature, table_name_feature
)

logger = logging.getLogger(__name__)


class NoDataError(Exception):
    pass


def preprocess(df: pd.DataFrame):
    df = df.sort_values(['stock_id', 'price_date'])
    df = df[df['close_price']!=0]
    # simple features
    df['ma5'] = df.groupby('stock_id')['close_price'].transform(lambda x: x.rolling(5).mean())
    df['returns'] = df.groupby('stock_id')['close_price'].pct_change()
    df = df.fillna(method='bfill').fillna(method='ffill')

    features = [ 'open_price', 'max_price', 'min_price', 'close_price', 
                'trading_volume', 'spread', 'trading_turnover', 'ma5', 'returns']
    return df, features


def preserve_features(df: pd.DataFrame, sqlite_db_name: str, sqlite_table_name: str):
    cols = df.columns.values
    for price_date, g in df.groupby('price_date'):
        with sqlite.connect(sqlite_db_name) as conn:
            # auto commit using with statement
            delete_q = f"""delete from {sqlite_table_name} 
                where price_date = '{price_date}'"""
            cursor = conn.cursor()
            cursor.execute(delete_q)

        with sqlite.connect(sqlite_db_name) as conn:
            # auto commit using with statement
            cursor = conn.cursor()
            cursor.executemany(f"""insert into {sqlite_table_name} ({",".join(cols)})
                values ({",".join(g.shape[1]*"?")})""", list(g.values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='preserve_feature',
                    description='preserve feature to db')

    parser.add_argument('--price_date', help="date of the price, format: yyyy-mm-dd")
    parser.add_argument('--stock_id', help="the stock id")
    args = parser.parse_args()


    # Load data
    read_sql_query = f"select * from {table_name_price} "
    condition = []
    for k, v in vars(args).items():
        if v:
            condition.append(f" {k} = '{v}' ")
    if condition:
        read_sql_query += f" where {'and'.join(condition)}"

    with sqlite.connect(sqlite_db_name_raw) as conn:
        df = pd.read_sql_query(read_sql_query, conn)
    if df.shape[0] == 0:
        raise NoDataError("No Data loaded from sqlitedb")

    df, features = preprocess(df)
    preserve_features(df, sqlite_db_name_feature, table_name_feature)
