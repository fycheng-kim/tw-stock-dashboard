from dotenv import dotenv_values

class CriticalVairableNotSet(Exception):
    pass

config = dotenv_values("./config")

sqlite_db_name = config.get("SQLITE_DB_NAME", "stock_raw")
table_name = config.get("TABLE_NAME", "stock_price")
url = config.get("URL", "")
TOKEN = config.get("TOKEN", "")

if not sqlite_db_name :
    raise CriticalVairableNotSet("sqlite_db_name")

if not table_name :
    raise CriticalVairableNotSet("table_name")
