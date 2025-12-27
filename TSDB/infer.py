import argparse
import sqlite3  as sqlite
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging

from datetime import datetime
from .train import features, StockLSTM
from .config import (
    sqlite_db_name_feature, table_name_feature
)


logger = logging.getLogger(__name__)


class NoDataError(Exception):
    pass


def load_model_and_scaler(model_path: str, scaler_path: str,
                          input_size: int, device="cpu"):
    m = StockLSTM(input_size=input_size)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()
    sc = joblib.load(scaler_path)
    return m, sc


def create_sequences_inference(group_df, features: list, window=10):
    """
    Returns:
      X: np.array shape (n_seq, window, n_features)
      meta: list of (stock_id, price_date of last in seq)
    """
    data = group_df[features].values
    dates = group_df['price_date'].values
    stock_id = group_df['stock_id'].iat[0]
    X_seq = []
    meta = []
    for i in range(len(data) - window):
        X_seq.append(data[i:i + window])
        meta.append((stock_id, dates[i + window]))   # prediction corresponds to that end-date

    if len(X_seq) == 0:
        return np.empty((0, window, len(features))), []
    return np.stack(X_seq), meta


def predict_on_new(df_new, model, scaler, features, window_size=10, threshold=0.7, device="cpu"):
    """
    df_new: must contain ['stock_id','price_date'] + features. price_date is aligned to rows.
    returns: DataFrame with columns ['stock_id','price_date','prob','pred']
    """
    # copy to avoid modifying original
    df = df_new.copy()
    # ensure correct sorting    
    df = df.sort_values(['stock_id', 'price_date']).reset_index(drop=True)

    # scale features with the *trained* scaler (do NOT fit here)
    # if scaler was fit on a superset of features, apply transform directly:
    df[features] = scaler.transform(df[features])

    all_rows = []
    for stock_id, g in df.groupby('stock_id'):
        if g.shape[0] < window_size:
            continue
        X_seq, meta = create_sequences_inference(g, features, window=window_size)
        if X_seq.size == 0:
            continue
        # to tensor
        Xt = torch.tensor(X_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = model(Xt).squeeze().cpu().numpy()
        # make sure shape is (n,)
        probs = probs.reshape(-1)
        preds = (probs > threshold).astype(int)
        for (sid, date), p, pr in zip(meta, probs, preds):
            all_rows.append((sid, date, float(p), int(pr)))

    out = pd.DataFrame(all_rows, columns=['stock_id', 'price_date', 'prob', 'pred'])
    # sort useful for downstream
    out = out.sort_values(['stock_id', 'price_date']).reset_index(drop=True)
    return out


def get_data_within_window(window_size: int, db_name: str, price_date: str):
    read_sql_query = f"""select *
        from (
            select *,
                row_number() over (
                    partition by stock_id
                    order by price_date desc
                ) as rn
            from {table_name_feature}
            where price_date <= '{price_date}'
        )
        where rn <= {window_size+1} """

    # predict
    with sqlite.connect(db_name) as conn:
        df = pd.read_sql_query(read_sql_query, conn)
    if df.shape[0] <= window_size:
        raise NoDataError("No Data loaded from sqlitedb")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='preserve_feature',
                    description='preserve feature to db')

    parser.add_argument('price_date', help="date of the price, format: yyyy-mm-dd")
    args = parser.parse_args()

    model_version = '20251227_172022_626731'
    results_path = f'./results/{model_version}'
    model, scaler = load_model_and_scaler(
        f'{results_path}/stock_lstm_{model_version}.pth',
        f'{results_path}/scaler_{model_version}.pkl',
        len(features))
    window_size = 10
    
    df = get_data_within_window(window_size, sqlite_db_name_feature, args.price_date)

    predict = predict_on_new(df,
                             model, scaler, features,
                             window_size=window_size,
                             threshold=0.7)
    predict = predict[predict.price_date == args.price_date]
    if predict.shape[0]==0:
        logger.warning("No prediction add")
    predict['predict_ts'] = datetime.now().strftime("%Y%m%d %H%M%S")
    predict['model_version'] = model_version
    result_table_name = 'prediction'

    with sqlite.connect(sqlite_db_name_feature) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""delete from {result_table_name} where
            price_date = '{args.price_date}' and model_version = '{model_version}'""")
        cursor.executemany(f"""insert into {result_table_name} values (
            {",".join(predict.shape[1]*"?")})""", list(predict.values))
