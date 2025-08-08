from django.core.management.base import BaseCommand
from pyexpat import features

from predict.models import *
from django.utils.timezone import make_aware
import datetime, time, os
import requests
import numpy as np
import pickle
from collections import deque
from tensorflow.keras.models import load_model
from keras.layers import Layer
from keras import backend as K
import pandas as pd
import pandas_ta as ta
import traceback


class SimpleAttention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1],1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],1),
                                 initializer='zeros',
                                 trainable=True)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)


# 配置
WINDOW = int(os.getenv("WINDOW", 30))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
API_URL = os.getenv("API_URL", "https://api.binance.com/api/v3/klines")

model_path = os.getenv("MODEL_PATH")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"model file not found: {model_path}")
model = load_model(model_path, custom_objects={"Attention": SimpleAttention})


SCALER_PATH = os.getenv("SCALER_PATH")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"scaler not found: {SCALER_PATH}")
scaler = pickle.load(open(SCALER_PATH, "rb"))

def align_to_minute(ts: int) -> int:
    return ts - (ts % 60)

def fetch_klines(start_ts: int, end_ts: int, symbol="BTCUSDT", interval="1m"):
    # 注意 Binance API 限制：每次最多 1000 条
    url = "https://api.binance.com/api/v3/klines"
    start_ts = align_to_minute(start_ts)
    end_ts = align_to_minute(end_ts)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts * 1000,
        "endTime": end_ts * 1000,
        "limit": 1000
    }
    response = requests.get(url, params=params)
    ks = response.json()
    return [(int(k[6] // 1000), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in ks]
def get_or_fetch_recent_candles(n=60) -> deque:
    latest = MarketCandle.objects.order_by('-timestamp')[:n * 2]
    df = pd.DataFrame.from_records(
        latest.values("timestamp", "close", "high", "low", "volume")
    )
    df = df.sort_values("timestamp")

    # 构建 time index
    all_timestamps = pd.date_range(
        end=df['timestamp'].iloc[-1],
        periods=n,
        freq='1min'
    )

    df = df.set_index('timestamp').reindex(all_timestamps)

    missing_ts = df[df['close'].isna()].index.to_list()

    if missing_ts:
        print(f"⚠️ 缺失数据条数: {len(missing_ts)}，开始补全")

        start_ts = int(missing_ts[0].timestamp())
        end_ts   = int(missing_ts[-1].timestamp()) + 60

        fetched = fetch_klines(start_ts, end_ts)
        for ts, high, low, close, volume in fetched:
            MarketCandle.objects.update_or_create(
                timestamp=datetime.utcfromtimestamp(ts),
                defaults={
                    "close": close,  # ✅ 改为 MarketCandle 的字段名
                    "high": high,
                    "low": low,
                    "volume": volume
                }
            )

        return get_or_fetch_recent_candles(n)

    candle_buf = deque(maxlen=n)
    for ts, row in df.iterrows():
        candle_buf.append({
            "timestamp": ts,
            "close": row["close"],
            "high": row["high"],
            "low": row["low"],
            "volume": row["volume"]
        })
    return candle_buf

def fetch_closed_klines(limit=1):
    params = {"symbol": SYMBOL, "interval": "1m", "limit": limit}
    ks = requests.get(API_URL, params=params, timeout=5).json()
    return [(k[6]//1000, float(k[4]), float(k[5])) for k in ks]  # (timestamp, close, volume)

def predict_next(candle_buf):
    x_input = build_input_from_candles(candle_buf)
    pred_scaled = model.predict(x_input, verbose=0)[0, 0]

    feature_names = ["close", "volume", "hl_spread", "return", "rsi"]
    last_features = x_input[0, -1, 1:]  # (4,)
    full_input = np.concatenate([[pred_scaled], last_features])  # (5,)

    # inverse_transform 并提取 close 的值
    close_inverse = scaler.inverse_transform([full_input])[0][0]
    return close_inverse

def build_input_from_candles(candle_buf, window_size=30):

    if len(candle_buf) < 90:
        raise ValueError(" Need to have more than 60 data to get all feature")

    df = pd.DataFrame(candle_buf)


    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"] = df["high"] - df["low"]
    df["price_delta"] = df["close"] - df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["vol_change"] = df["volume"].pct_change()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["std_60min"] = df["close"].rolling(window=60).std()

    df.dropna(inplace=True)

    features = df[["close", "volume", "hl_spread", "return", "rsi"]].values

    if len(features) < window_size:
        raise ValueError(f"Not enough available feature, current feature: {len(features)}")

    features_scaled = scaler.transform(features)  # shape: (N, 5)

    # 滑窗提取最新一段
    x = features_scaled[-window_size:]  # shape: (30, 5)
    x = x.reshape(1, window_size, x.shape[1])   # shape: (1, 30, 5)
    return x