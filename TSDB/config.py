from dotenv import dotenv_values


config = dotenv_values("./config")

sqlite_db_name = config.get("SQLITE_DB_NAME", "stock_raw")
table_name = config.get("TABLE_NAME", "stock_price")
url = config.get("URL", "")
TOKEN = config.get("URL", "")