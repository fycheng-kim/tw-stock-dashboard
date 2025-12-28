# Taiwan Stock Price Predict

This project aims to predict if any stock’s price will rise over 5% within three days.

## Data source

- findmind: https://finmind.github.io/quickstart/

### Dataset Used

- TaiwanStockTradingDate: https://finmind.github.io/tutor/TaiwanMarket/Technical/#taiwanstocktradingdate

- TaiwanStockPrice: https://finmind.github.io/tutor/TaiwanMarket/Technical/#taiwanstockprice

- TaiwanStockInfo: https://finmind.github.io/tutor/TaiwanMarket/Technical/#taiwanstockinfo

## Project Structure

```
    TSDB
    ├── __init__.py
    ├── __main__.py
    ├── config.py                   # proejct config
    ├── create_table.py             # create sqlite table
    ├── get_stock_raw.py            # get raw data
    └── train.py                    # model training
```

## Usage

1. Install Packages
    ```bash
    pip install requirements.txt
    ```

2. update config file
    
    - SQLITE_DB_NAME: sqlite db name
    - TABLE_NAME: sqlite table name
    - URL=https://api.finmindtrade.com/api/v4/data
    - TOKEN= token from findmind, can be empty, see findmind web for detail.

3. Create sqlite table
    
    ```bash
    python -m TSDB.create_table
    ```

4. Download Raw Data

    download data from findmind, store into sqlite db: <SQLITE_DB_NAME_RAW>.
    
    ```bash
    python -m TSDB.get_stock_raw -h
    # positional arguments:
    # ind_cat               industry category for stock

    # options:
    # -h, --help            show this help message and exit
    # --base_date BASE_DATE
    #                 base date to download fileshould be dates have stock transactions, default to today, format: yyyy-mm-dd
    # -n N            int, how many days to download from base_date, positive or negtive, default 0

    ```

5. Preserver Model Features

    generate feature from raw data and store into sqlite db: <SQLITE_DB_NAME_FEATURE>.

    ```bash
    python -m TSDB.preserve_feature

    # options:
    # -h, --help            show this help message and exit
    # --price_date PRICE_DATE
    #                       date of the price, format: yyyy-mm-dd
    # --stock_id STOCK_ID   the stock id
    ```

6. Update Model

    train model base on features.

    ```bash
    python -m TSDB.train
    ```

7. predict
    
    predict from features, price_date is must

    ```bash
    python -m TSDB.infer <price_date>
    ```
