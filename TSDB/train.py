import sqlite3  as sqlite
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import logging
import joblib
import json
import os

from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from .config import (
    sqlite_db_name_feature, table_name_feature
)

logger = logging.getLogger(__name__)

features = [ 'open_price', 'max_price', 'min_price', 'close_price', 
            'trading_volume', 'spread', 'trading_turnover', 'ma5', 'returns' ]


class NoDataError(Exception):
    pass


def create_sequences(group_df, window=10):
    X_seq, y_seq = [], []
    data = group_df[features].values
    targets = group_df['target'].values
    for i in range(len(data) - window):
        X_seq.append(data[i:i+window])
        y_seq.append(targets[i+window])
    return np.array(X_seq), np.array(y_seq)


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return torch.sigmoid(out)


def train(df, features, window_size=10):
    scaler = StandardScaler()
    
    # Create target: +5% within 3 days
    df['future_close'] = df.groupby('stock_id')['close_price'].shift(-3)
    df['target'] = ((df['future_close'] - df['close_price']) / df['close_price'] >= 0.05) \
        .astype(int)
    # drop date without close date
    df = df.dropna(subset=['future_close'])
    df = df[df['close_price']!=0]
    
    df[features] = scaler.fit_transform(df[features])
    
    logger.info("preparing sequence data")
    
    X_all, y_all = [], []
    for stock_id, g in df.groupby('stock_id')[features + ['target']]:
        if g.shape[0] > window_size:
            X_seq, y_seq = create_sequences(g, window=window_size)
            X_all.append(X_seq)
            y_all.append(y_seq)

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    logger.info("converting to tensors")
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    model = StockLSTM(input_size=len(features))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info("training")
    for epoch in range(22):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")


    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).squeeze().numpy()
    y_pred = (preds > 0.7).astype(int)
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)

    logger.info("\n=== Evaluation ===")
    logger.info(report)
    
    logger.info("\n=== Evaluation ===")
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = f"./results/{version_tag}"
    os.mkdir(result_dir)
    torch.save(model.state_dict(), f"{result_dir}/stock_lstm_{version_tag}.pth")
    joblib.dump(scaler, f"{result_dir}/scaler_{version_tag}.pkl")
    
    with open(f"{result_dir}/report_{version_tag}.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    read_sql_query = f"""select * from {table_name_feature}"""
    
    with sqlite.connect(sqlite_db_name_feature) as conn:
        df = pd.read_sql_query(read_sql_query, conn)
    if df.shape[0] <= 10:
        raise NoDataError("No Data loaded from sqlitedb")

    train(df, features, window_size=10)
