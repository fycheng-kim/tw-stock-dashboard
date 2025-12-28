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

FEATURES = [ 'open_price', 'max_price', 'min_price', 'close_price', 
            'trading_volume', 'spread', 'trading_turnover', 'ma5', 'returns' ]


class NoDataError(Exception):
    pass


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return torch.sigmoid(out)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df.groupby("stock_id")["close_price"].shift(-3)
    df["target"] = (
        (df["future_close"] - df["close_price"]) / df["close_price"] >= 0.05
    ).astype(int)
    return df.dropna(subset=["future_close"])


def scale_features(df: pd.DataFrame, features: list):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler


def create_sequences(group_df, features, window=10):
    X, y = [], []
    data = group_df[features].values
    target = group_df["target"].values

    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(target[i + window])

    return np.array(X), np.array(y)


def build_dataset(df, features, window_size):
    X_all, y_all = [], []

    for _, g in df.groupby("stock_id"):
        if len(g) > window_size:
            X, y = create_sequences(g, features, window_size)
            X_all.append(X)
            y_all.append(y)

    return np.concatenate(X_all), np.concatenate(y_all)


def train_model(X, y, input_size, epochs=22):
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )

    model = StockLSTM(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            loss = criterion(model(xb).squeeze(), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

    report = classification_report(
        y_test, (preds > 0.7).astype(int), output_dict=True
    )

    return model, report


def main():
    with sqlite.connect(sqlite_db_name_feature) as conn:
        df = pd.read_sql_query(f"select * from {table_name_feature}", conn)

    df = df[df["close_price"] != 0]
    df = build_target(df)
    df, scaler = scale_features(df, FEATURES)

    X, y = build_dataset(df, FEATURES, window_size=10)
    model, report = train_model(X, y, input_size=len(FEATURES))

    version = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = f"./results/{version}"
    os.mkdir(out)

    torch.save(model.state_dict(), f"{out}/model.pth")
    joblib.dump(scaler, f"{out}/scaler.pkl")
    with open(f"{out}/report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
