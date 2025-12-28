import numpy as np
import pandas as pd
import torch
from TSDB.infer import create_sequences_inference, predict_on_new


def test_create_sequences_inference():
    df = pd.DataFrame({
        "stock_id": ["A"] * 6,
        "price_date": range(6),
        "f1": range(6),
        "f2": range(6),
    })

    X, meta = create_sequences_inference(df, ["f1", "f2"], window=3)

    assert X.shape == (3, 3, 2)
    assert meta[0] == ("A", 3)


class DummyScaler:
    def transform(self, X):
        return X


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ones((x.shape[0], 1)) * 0.8


def test_predict_on_new():
    df = pd.DataFrame({
        "stock_id": ["A"] * 6,
        "price_date": range(6),
        "f1": np.arange(6),
        "f2": np.arange(6),
    })

    out = predict_on_new(
        df,
        model=DummyModel(),
        scaler=DummyScaler(),
        features=["f1", "f2"],
        window_size=3,
        threshold=0.7,
    )

    assert out.shape[0] == 3
    assert out["pred"].eq(1).all()
