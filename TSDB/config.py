from dotenv import dotenv_values

class CriticalVairableNotSet(Exception):
    pass

config = dotenv_values("./config")

sqlite_db_name_raw = config.get("SQLITE_DB_NAME_RAW", "stock_raw")
table_name_price = config.get("TABLE_NAME_PRICE", "stock_price")

sqlite_db_name_feature = config.get("SQLITE_DB_NAME_FEATURE", "stock_feature") 
table_name_feature = config.get("TABLE_NAME_FEATURE", "price_feature")

url = config.get("URL", "")
TOKEN = config.get("TOKEN", "")

if not sqlite_db_name_raw :
    raise CriticalVairableNotSet("sqlite_db_name_raw")

if not table_name_price :
    raise CriticalVairableNotSet("table_name_price")

if not sqlite_db_name_feature :
    raise CriticalVairableNotSet("sqlite_db_name_feature")

if not table_name_feature :
    raise CriticalVairableNotSet("table_name_feature")