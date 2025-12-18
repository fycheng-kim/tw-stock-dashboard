# create sqlite table
import logging
import sqlite3 as sqlite
from sqlite3 import OperationalError

from .config import sqlite_db_name, table_name

logger = logging.getLogger(__name__)


create_table_str = f"""create table {table_name} (
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
    with sqlite.connect(sqlite_db_name) as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_str)
except OperationalError:
    logger.warning("WARN: table exist")
