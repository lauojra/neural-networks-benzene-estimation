import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Callable, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)

# =========================
# 0) KONFIGURACJA EKSPERYMENTÓW
# =========================

@dataclass
class DataConfig:
    csv_path: str = "AirQualityUCI.csv"
    sep: str = ";"
    decimal: str = ","
    target: str = "C6H6(GT)"
    features: List[str] = None
    train_ratio: float = 0.70
    val_ratio: float = 0.15  # test = reszta
    time_col_date: str = "Date"
    time_col_time: str = "Time"


@dataclass
class TrainConfig:
    repeats: int = 5
    epochs: int = 150
    batch_size: int = 4096
    patience: int = 15
    restore_best: bool = True
    base_seed: int = 123


DEFAULT_FEATURES = [
    'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
    'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH'
]

data_cfg = DataConfig(features=DEFAULT_FEATURES)
train_cfg = TrainConfig(
    repeats=5, epochs=150, batch_size=32, patience=15, restore_best=True, base_seed=123
)

# -------------------------
# LISTY PARAMETRÓW 
# -------------------------
PARAM_GRID = {
    # liczba warstw ukrytych (MLP)
    "mlp_layers": [2, 3, 4],

    # liczba neuronów w warstwie (dla MLP: base_units; dla CNN/RNN: units/filters bazowe)
    "units": [3, 5, 10, 20],

    # learning rate
    "lr": [0.001, 0.005, 0.01, 0.05],

    # optymalizator
    "optimizer": ["adam", "sgd"],

    # momentum (dotyczy tylko SGD; dla pozostałych ignorujemy)
    "momentum": [0.0, 0.5, 0.9, 0.99],

    # liczba epok (maksymalna; EarlyStopping może zakończyć wcześniej)
    "epochs": [10, 50, 100, 150],

    # dla RNN: długość okna (ile godzin wstecz)
    "window": [3, 6, 12, 24],
}

# =========================
# 1) UTYLITY: SEED + METRYKI
# =========================

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2":  float(r2_score(y_true, y_pred)),
    }

def append_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Dopisuje DataFrame do CSV.
    Jeśli plik nie istnieje – zapisuje z nagłówkiem.
    Jeśli istnieje – dopisuje bez nagłówka.
    """
    file_exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not file_exists, index=False)


# =========================
# 2) WCZYTANIE + PREPROCESSING 
# =========================

def load_and_preprocess(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path, sep=cfg.sep, decimal=cfg.decimal)

    # usuń puste wiersze i ostatnie 2 kolumny jak u Ciebie
    df.dropna(how="all", inplace=True)
    df = df.iloc[:, :-2]

    # -200 => NaN
    df.replace(-200, np.nan, inplace=True)

    # target: usuń NaN (brak sensownej interpolacji targetu)
    before = len(df)
    df.dropna(subset=[cfg.target], inplace=True)
    after = len(df)
    print(f"[INFO] Usunięto {before-after} wierszy z NaN w target ({cfg.target}). Pozostało: {after}")

    # interpolacja liniowa w numeric
    cols_numeric = df.select_dtypes(include=[np.number]).columns
    df[cols_numeric] = df[cols_numeric].interpolate(method="linear", limit_direction="both")

    # Datetime i sortowanie czasowe (żeby split był chronologiczny)
    temp_time = df[cfg.time_col_time].astype(str).str.replace(".", ":", regex=False)
    df["Datetime"] = pd.to_datetime(df[cfg.time_col_date] + " " + temp_time, dayfirst=True)
    df.sort_values(by="Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # sprawdź, czy po interpolacji zostały NaN
    n_nan_rows = df.isna().any(axis=1).sum()
    print(f"[INFO] Liczba wierszy z NaN po interpolacji: {n_nan_rows}")

    return df


def split_scale(df: pd.DataFrame, cfg: DataConfig) -> Dict[str, Any]:
    n = len(df)
    train_end = int(cfg.train_ratio * n)
    val_end = int((cfg.train_ratio + cfg.val_ratio) * n)

    df = df.copy()
    df["Set"] = "test"
    df.loc[:train_end-1, "Set"] = "train"
    df.loc[train_end:val_end-1, "Set"] = "val"

    print("[INFO] Podział Set:")
    print(df["Set"].value_counts(sort=False))

    X_train = df[df["Set"] == "train"][cfg.features].values
    y_train = df[df["Set"] == "train"][cfg.target].values

    X_val = df[df["Set"] == "val"][cfg.features].values
    y_val = df[df["Set"] == "val"][cfg.target].values

    X_test = df[df["Set"] == "test"][cfg.features].values
    y_test = df[df["Set"] == "test"][cfg.target].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "df": df,
        "scaler": scaler,
        "X_train": X_train_scaled, "y_train": y_train,
        "X_val": X_val_scaled, "y_val": y_val,
        "X_test": X_test_scaled, "y_test": y_test,
    }

# =========================
# 3) PRZYGOTOWANIE DANYCH POD CNN i RNN
# =========================

def as_cnn_1d(X: np.ndarray) -> np.ndarray:
    # (N, 7) -> (N, 7, 1)
    return X[..., np.newaxis]

def make_sequences(X: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    # X: (N, F), y: (N,)
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window, :])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

def sequences_by_split(data: Dict[str, Any], window: int) -> Dict[str, Any]:
    # robimy sekwencje osobno w train/val/test, żeby nie mieszać granic czasowych
    X_tr, y_tr = make_sequences(data["X_train"], data["y_train"], window)
    X_v, y_v = make_sequences(data["X_val"], data["y_val"], window)
    X_te, y_te = make_sequences(data["X_test"], data["y_test"], window)
    return {"X_train": X_tr, "y_train": y_tr, "X_val": X_v, "y_val": y_v, "X_test": X_te, "y_test": y_te}

# =========================
# 4) BUDOWANIE MODELI (MLP / CNN / RNN)
# =========================

def build_optimizer(name: str, lr: float, momentum: float) -> tf.keras.optimizers.Optimizer:
    name = name.lower()
    if name == "adam":
        return Adam(learning_rate=lr)
    if name == "rmsprop":
        return RMSprop(learning_rate=lr)
    if name == "sgd":
        return SGD(learning_rate=lr, momentum=momentum)
    raise ValueError(f"Nieznany optimizer: {name}")

def build_mlp(input_dim: int, layers: int, units: int, lr: float, optimizer: str, momentum: float) -> tf.keras.Model:
    model = Sequential()
    # pierwsza warstwa
    model.add(Dense(units, activation="tanh", input_shape=(input_dim,)))
    # kolejne warstwy (jeśli layers > 1)
    for _ in range(layers - 1):
        model.add(Dense(units, activation="tanh"))
    # wyjście regresyjne
    model.add(Dense(1, activation="linear"))

    opt = build_optimizer(optimizer, lr, momentum)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def build_cnn(input_len: int, layers: int, units: int, lr: float, optimizer: str, momentum: float) -> tf.keras.Model:
    # input: (features=7, channels=1) -> (7,1)
    # "layers" tutaj interpretujemy jako liczbę warstw Conv1D
    model = Sequential()
    model.add(Conv1D(filters=units, kernel_size=3, activation="tanh", input_shape=(input_len, 1)))
    for _ in range(layers - 1):
        model.add(Conv1D(filters=units, kernel_size=3, activation="tanh", padding="same"))
    model.add(Flatten())
    model.add(Dense(1, activation="linear"))

    opt = build_optimizer(optimizer, lr, momentum)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def build_rnn(kind: str, window: int, n_features: int, units: int, lr: float, optimizer: str, momentum: float) -> tf.keras.Model:
    kind = kind.lower()
    model = Sequential()
    if kind == "lstm":
        model.add(LSTM(units, input_shape=(window, n_features)))
    elif kind == "gru":
        model.add(GRU(units, input_shape=(window, n_features)))
    else:
        raise ValueError("RNN kind must be 'lstm' or 'gru'")

    model.add(Dense(1, activation="linear"))

    opt = build_optimizer(optimizer, lr, momentum)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


# =========================
# 5) URUCHAMIANIE EKSPERYMENTU 
# =========================

def train_one_model(
    model: tf.keras.Model,
    X_train, y_train,
    X_val, y_val,
    epochs: int,
    batch_size: int,
    patience: int,
    restore_best: bool
) -> tf.keras.callbacks.History:

    early = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=restore_best
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=0
    )
    return history

def evaluate_model(model: tf.keras.Model, X, y) -> Dict[str, float]:
    y_pred = model.predict(X, verbose=0).flatten()
    return compute_metrics(y, y_pred)

def run_repeated(
    build_fn: Callable[[], tf.keras.Model],
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    train_cfg: TrainConfig,
    epochs_override: Optional[int] = None
) -> Dict[str, Any]:
    all_runs = []

    epochs = epochs_override if epochs_override is not None else train_cfg.epochs

    for r in range(train_cfg.repeats):
        seed = train_cfg.base_seed + r * 1000 + random.randint(0, 999)
        set_all_seeds(seed)

        model = build_fn()
        _ = train_one_model(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=train_cfg.batch_size,
            patience=train_cfg.patience,
            restore_best=train_cfg.restore_best
        )

        metrics_train = evaluate_model(model, X_train, y_train)
        metrics_val = evaluate_model(model, X_val, y_val)
        metrics_test = evaluate_model(model, X_test, y_test)

        all_runs.append({
            # "seed": seed,
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test
        })

    # agregacja: mean/std + best (po MAE test)
    df_runs = pd.json_normalize(all_runs)
    # columns będą typu: train.MSE, val.MAE itd.
    agg = {}

    for split in ["train", "val", "test"]:
        for metric in ["MSE", "MAE", "R2"]:
            col = f"{split}.{metric}"
            agg[f"{col}_mean"] = df_runs[col].mean()
            agg[f"{col}_std"] = df_runs[col].std(ddof=1)

    best_idx = df_runs["test.MAE"].idxmin()
    best_row = df_runs.loc[best_idx]

    best = {f"best_{k}": best_row[k] for k in df_runs.columns if k != "seed"}
    return {"runs": df_runs, "agg": agg, "best": best}


# =========================
# 6) EKSPERYMENTY: MLP / CNN / RNN (LSTM i GRU)
# =========================

def experiments_mlp(data: Dict[str, Any], grid: Dict[str, List], train_cfg: TrainConfig) -> pd.DataFrame:
    rows = []
    input_dim = data["X_train"].shape[1]

    for layers in grid["mlp_layers"]:
        for units in grid["units"]:
            for lr in grid["lr"]:
                for opt_name in grid["optimizer"]:
                    for mom in grid["momentum"]:
                        for epochs in grid["epochs"]:
                            # momentum sensowne tylko dla SGD; dla innych ustaw na 0.0 (żeby nie mnożyć bez sensu)
                            if opt_name != "sgd" and mom != grid["momentum"][0]:
                                continue

                            def build_fn():
                                return build_mlp(
                                    input_dim=input_dim,
                                    layers=layers,
                                    units=units,
                                    lr=lr,
                                    optimizer=opt_name,
                                    momentum=mom
                                )

                            out = run_repeated(
                                build_fn,
                                data["X_train"], data["y_train"],
                                data["X_val"], data["y_val"],
                                data["X_test"], data["y_test"],
                                train_cfg=train_cfg,
                                epochs_override=epochs
                            )

                            row = {
                                "model_type": "MLP",
                                "layers": layers,
                                "units": units,
                                "lr": lr,
                                "optimizer": opt_name,
                                "momentum": mom,
                                "epochs": epochs,
                            }
                            row.update(out["agg"])
                            row.update(out["best"])
                            rows.append(row)

                            # --- APPEND DO CSV PO KAŻDEJ KONFIGURACJI ---
                            append_to_csv(pd.DataFrame([row]), "results_mlp_full.csv")

    return pd.DataFrame(rows)


def experiments_cnn(data: Dict[str, Any], grid: Dict[str, List], train_cfg: TrainConfig) -> pd.DataFrame:
    rows = []
    X_train = as_cnn_1d(data["X_train"])
    X_val   = as_cnn_1d(data["X_val"])
    X_test  = as_cnn_1d(data["X_test"])
    input_len = X_train.shape[1]  

    for layers in grid["mlp_layers"]:  
        for units in grid["units"]:  
            for lr in grid["lr"]:
                for opt_name in grid["optimizer"]:
                    for mom in grid["momentum"]:
                        for epochs in grid["epochs"]:
                            if opt_name != "sgd" and mom != grid["momentum"][0]:
                                continue

                            def build_fn():
                                return build_cnn(
                                    input_len=input_len,
                                    layers=layers,
                                    units=units,
                                    lr=lr,
                                    optimizer=opt_name,
                                    momentum=mom
                                )

                            out = run_repeated(
                                build_fn,
                                X_train, data["y_train"],
                                X_val, data["y_val"],
                                X_test, data["y_test"],
                                train_cfg=train_cfg,
                                epochs_override=epochs
                            )

                            row = {
                                "model_type": "CNN1D",
                                "layers": layers,
                                "units": units,
                                "lr": lr,
                                "optimizer": opt_name,
                                "momentum": mom,
                                "epochs": epochs,
                            }
                            row.update(out["agg"])
                            row.update(out["best"])
                            rows.append(row)
                            append_to_csv(pd.DataFrame([row]), "results_cnn_full.csv")

    return pd.DataFrame(rows)


def experiments_rnn(data: Dict[str, Any], grid: Dict[str, List], train_cfg: TrainConfig, kind: str) -> pd.DataFrame:
    rows = []
    n_features = data["X_train"].shape[1]

    for window in grid["window"]:
        seq = sequences_by_split(data, window)
        # jeśli po oknie zostało za mało danych, pomiń (zwykle nie powinno, ale zabezpieczenie)
        if len(seq["X_train"]) < 100 or len(seq["X_val"]) < 50 or len(seq["X_test"]) < 50:
            continue

        for units in grid["units"]:
            for lr in grid["lr"]:
                for opt_name in grid["optimizer"]:
                    for mom in grid["momentum"]:
                        for epochs in grid["epochs"]:
                            if opt_name != "sgd" and mom != grid["momentum"][0]:
                                continue

                            def build_fn():
                                return build_rnn(
                                    kind=kind,
                                    window=window,
                                    n_features=n_features,
                                    units=units,
                                    lr=lr,
                                    optimizer=opt_name,
                                    momentum=mom
                                )

                            out = run_repeated(
                                build_fn,
                                seq["X_train"], seq["y_train"],
                                seq["X_val"], seq["y_val"],
                                seq["X_test"], seq["y_test"],
                                train_cfg=train_cfg,
                                epochs_override=epochs
                            )

                            row = {
                                "model_type": kind.upper(),
                                "window": window,
                                "units": units,
                                "lr": lr,
                                "optimizer": opt_name,
                                "momentum": mom,
                                "epochs": epochs,
                            }
                            row.update(out["agg"])
                            row.update(out["best"])
                            rows.append(row)
                            append_to_csv(pd.DataFrame([row]), "results_rnn_full.csv")

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Wczytanie i preprocessing danych
    df = load_and_preprocess(data_cfg)
    data = split_scale(df, data_cfg)

    # Definicja pełnego gridu
    grid = PARAM_GRID
    
    # 2a) MLP - pełny grid
    results_mlp = experiments_mlp(data, grid, train_cfg)
    print("[INFO] MLP results:", results_mlp.shape)

    # 2b) CNN - pełny grid
    results_cnn = experiments_cnn(data, grid, train_cfg)
    print("[INFO] CNN results:", results_cnn.shape)

    # 2c) RNN (LSTM) - pełny grid
    results_lstm = experiments_rnn(data, grid, train_cfg, kind="lstm")
    print("[INFO] LSTM results:", results_lstm.shape)
