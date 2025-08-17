import pandas as pd
import numpy as np
import os, json
import xgboost as xgb

LAGS = [1, 2, 4, 8, 16, 32, 96]
ROLL_WINDOWS = [4, 16, 96, 192]
TARGET_HORIZON = 1

FEATURE_COLS = (
    ["return_1", "return_4", "return_16"] +
    [f"lag_{l}" for l in LAGS] +
    sum(([f"roll_mean_{w}", f"roll_std_{w}", f"vol_mean_{w}", f"vol_std_{w}"] for w in ROLL_WINDOWS), []) +
    ["hl_range", "hour", "dayofweek", "month"]
)
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
API_URL = os.getenv("API_URL", "https://api.binance.com/api/v3/klines")

model_path = os.getenv("XGBOOST_MODEL_PATH")

with open(f"{model_path}/xgb_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
FEATURES = meta["features"]
BEST_IT = meta.get("best_iteration")

MODEL_FILE = os.path.join(model_path, "xgb_model.ubj")

bst = xgb.Booster()
bst.load_model(MODEL_FILE)



def make_features(df: pd.DataFrame, with_target=True) -> pd.DataFrame:
    out = df.copy()
    out["return_1"]  = out["close"].pct_change(1).astype("float32")
    out["return_4"]  = out["close"].pct_change(4).astype("float32")
    out["return_16"] = out["close"].pct_change(16).astype("float32")

    for l in LAGS:
        out[f"lag_{l}"] = out["close"].shift(l).astype("float32")

    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = out["close"].rolling(w, min_periods=w).mean().astype("float32")
        out[f"roll_std_{w}"]  = out["close"].rolling(w, min_periods=w).std().astype("float32")
        if "Volume" in out.columns:
            out[f"vol_mean_{w}"] = out["volume"].rolling(w, min_periods=w).mean().astype("float32")
            out[f"vol_std_{w}"]  = out["volume"].rolling(w, min_periods=w).std().astype("float32")

    if "High" in out.columns and "Low" in out.columns:
        out["hl_range"] = (out["high"] - out["=low"]).astype("float32")

    dt = pd.to_datetime(out["timestamp"], utc=True)
    out["hour"]      = dt.dt.hour.astype("int16")
    out["dayofweek"] = dt.dt.dayofweek.astype("int16")
    out["month"]     = dt.dt.month.astype("int16")

    if with_target:
        out["fwd_ret_1"] = (out["close"].shift(-TARGET_HORIZON) / out["close"] - 1.0).astype("float32")
        out["target"] = out["fwd_ret_1"]

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = out.dropna().reset_index(drop=True)
    return out


def xgboost_predict_next(X_row):
    # x_input = build_input_from_candles(candle_buf)
    # pred = model.predict(features, verbose=0)[0]
    # return pred
    return float(bst.predict(X_row)[0])
