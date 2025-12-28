
import pandas as pd
import numpy as np
import torch

from TSDB.train import (
    build_target,
    scale_features,
    create_sequences,
    build_dataset,
    train_model,
    StockLSTM
)

FEATURES = ["f1", "f2"]


def test_build_target():
    df = pd.DataFrame({
        "stock_id": ["A"] * 6,
        "close_price": [100, 102, 104, 110, 115, 120],
    })

    out = build_target(df)

    assert "target" in out.columns
    assert out["target"].iloc[0] == 1


def test_scale_features():
    df = pd.DataFrame({
        "f1": [1, 2, 3],
        "f2": [10, 20, 30],
    })

    out, scaler = scale_features(df, FEATURES)

    assert np.isclose(out[FEATURES].mean().values, 0).all()
    assert scaler is not None


def test_create_sequences():
    df = pd.DataFrame({
        "f1": range(10),
        "f2": range(10),
        "target": [0, 1] * 5,
    })

    X, y = create_sequences(df, FEATURES, window=5)

    assert X.shape == (5, 5, 2)
    assert y.shape == (5,)


def test_build_dataset():
    df = pd.DataFrame({
        "stock_id": ["A"] * 10,
        "f1": range(10),
        "f2": range(10),
        "target": [0, 1] * 5,
    })

    X, y = build_dataset(df, FEATURES, window_size=5)

    assert X.shape[1:] == (5, 2)
    assert len(X) == len(y)


def test_lstm_forward():
    model = StockLSTM(input_size=3)
    x = torch.randn(4, 10, 3)

    y = model(x)

    assert y.shape == (4, 1)
    assert torch.all((y >= 0) & (y <= 1))


def test_train_smoke():
    X = np.random.rand(50, 5, 3)
    y = np.random.randint(0, 2, size=50)

    model, report = train_model(
        X, y, input_size=3, epochs=1
    )

    assert "accuracy" in report
    assert model is not None
