# create sqlite table
import logging
import sqlite3 as sqlite
from sqlite3 import OperationalError

from .config import (
    sqlite_db_name_raw, table_name_price, 
    sqlite_db_name_feature, table_name_feature
)

logger = logging.getLogger(__name__)


create_table_str = f"""create table {table_name_price} (
        price_date date, 
        stock_id string, 
        trading_volume int, 
        trading_money int, 
        open_price float, 
        max_price float, 
        min_price float,
        close_price float,
        spread float,
        trading_turnover int)"""
try:
    with sqlite.connect(sqlite_db_name_raw) as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_str)
except OperationalError:
    logger.warning("WARN: table exist")


create_table_str = f"""create table {table_name_feature} (
        price_date date, 
        stock_id string, 
        trading_volume int, 
        trading_money int, 
        open_price float,
        max_price float,
        min_price float,
        close_price float,
        spread float,
        trading_turnover int,
        ma5 float,
        returns float)"""
try:
    with sqlite.connect(sqlite_db_name_feature) as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_str)
except OperationalError:
    logger.warning("WARN: table exist")
