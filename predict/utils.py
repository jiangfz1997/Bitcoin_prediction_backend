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
from django.utils import timezone
from .models import MarketCandle, ModelPrediction
from datetime import timezone as dt_timezone
import tensorflow as tf

# ignore FutureWarnings from sklearn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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
model = load_model(model_path, custom_objects={"Attention": SimpleAttention}, compile=False)
y_mean, y_std = None, None
if os.getenv("NPZ_PATH", None):
    stats = np.load(os.getenv("NPZ_PATH"))
    y_mean, y_std = stats["y_mean"], stats["y_std"]

SCALER_PATH = os.getenv("SCALER_PATH")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"scaler not found: {SCALER_PATH}")
scaler = pickle.load(open(SCALER_PATH, "rb"))
def last_step_mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true[:, -1], y_pred[:, -1])

def mean_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def make_weighted_huber_and_diff(weights=None, delta=1.0, lambda_diff=0.5,
                                 w_start=1.0, w_end=2.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # per-element Huber (shape: [B, H])
        err = y_pred - y_true
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, delta)
        linear = abs_err - quadratic
        per_elem = 0.5 * tf.square(quadratic) + delta * linear  # [B, H]

        # horizon weights w: [H] or [1, H]
        H = tf.shape(y_true)[1]
        if weights is None:
            w = tf.linspace(tf.cast(w_start, tf.float32),
                            tf.cast(w_end,   tf.float32),
                            tf.cast(H, tf.int32))           # [H]
        else:
            w = tf.convert_to_tensor(weights, dtype=tf.float32)  # [H]
        w = tf.reshape(w, (1, -1))                               # [1, H] for broadcast

        huber_term = tf.reduce_mean(tf.reduce_sum(per_elem * w, axis=1))  # scalar

        # Δ-regularization on horizon axis
        # when H==1, make diff_term=0 to avoid empty slice
        def diff():
            dy_t = y_true[:, 1:] - y_true[:, :-1]
            dy_p = y_pred[:, 1:] - y_pred[:, :-1]
            return tf.reduce_mean(tf.square(dy_p - dy_t))
        diff_term = tf.cond(tf.greater(tf.shape(y_true)[1], 1),
                            true_fn=diff, false_fn=lambda: tf.constant(0.0, tf.float32))

        return huber_term + lambda_diff * diff_term
    return loss

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
def get_or_fetch_recent_candles(n=60, symbol="BTCUSDT", end_dt: datetime.datetime | None = None, return_type="deque"):
    """
    拉取最近 n 根 1min K 线，自动补齐缺口并返回 deque 供模型输入。
    - 按 symbol 过滤，匹配唯一键 (symbol, timestamp)
    - 去重：同一 timestamp 留最后一条
    - 统一 tz：确保 index 和 date_range 一致
    - 缺口用远端 fetch_klines(start_ts, end_ts) 补齐；写库用 tz-aware 时间
    """


    # 1) 先读一批数据（多取一点，给补数留空间）
    qs = (MarketCandle.objects
          .filter(symbol=symbol)
          .order_by('-timestamp')
          .values('timestamp', 'close', 'high', 'low', 'volume')[:max(n*3, n+10)])
    df = pd.DataFrame.from_records(qs)
    if df.empty:
        return deque(maxlen=n)

    # 2) 排序、去重（防止重复索引导致 reindex 报错）
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset='timestamp', keep='last')

    # 3) 构造完整时间序列（与数据同 tz）
    tz = df['timestamp'].dt.tz
    end_ts = df['timestamp'].iloc[-1]
    all_ts = pd.date_range(end=end_ts, periods=n, freq='1min', tz=tz)

    # 4) 对齐索引并标出缺口
    df = df.set_index('timestamp').reindex(all_ts)
    missing_ts = df.index[df['close'].isna()].to_list()

    # 5) 如有缺口，调用远端接口补数后再重新取一次
    if missing_ts:
        # 合并连续缺口为一个段，减少请求次数（简单分段）
        ranges = []
        if missing_ts:
            start = missing_ts[0]
            prev = missing_ts[0]
            for t in missing_ts[1:]:
                if (t - prev) == pd.Timedelta(minutes=1):
                    prev = t
                else:
                    ranges.append((start, prev))
                    start, prev = t, t
            ranges.append((start, prev))

        for st, et in ranges:
            # fetch_klines 习惯是 [start, end) 秒；右边 +60 覆盖末尾分钟
            start_sec = int(st.timestamp())
            end_sec = int(et.timestamp()) + 60
            fetched = fetch_klines(start_sec, end_sec, symbol=symbol)

            # 写库时一定带 symbol，timestamp 用 tz-aware
            for ts, high, low, close, volume in fetched:
                MarketCandle.objects.update_or_create(
                    symbol=symbol,
                    timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
                    defaults={
                        'close': close,
                        'high': high,
                        'low': low,
                        'volume': volume
                    }
                )

        # 补完再重新拉一次，走统一流程
        qs = (MarketCandle.objects
              .filter(symbol=symbol)
              .order_by('-timestamp')
              .values('timestamp', 'close', 'high', 'low', 'volume')[:max(n*3, n+10)])
        df = pd.DataFrame.from_records(qs).sort_values('timestamp').drop_duplicates('timestamp', keep='last')
        tz = df['timestamp'].dt.tz
        end_ts = df['timestamp'].iloc[-1]
        all_ts = pd.date_range(end=end_ts, periods=n, freq='1min', tz=tz)
        df = df.set_index('timestamp').reindex(all_ts).ffill()

    else:
        # 没缺口就前向填充一下（个别指标字段可能有空）
        df = df.ffill()

    # 6) 仅保留尾部 n 条，转为 deque
    df = df.tail(n)
    if return_type == "deque":
        out = deque(maxlen=n)
        for ts, row in df.iterrows():
            out.append({
                'timestamp': ts,            # tz-aware
                'close': float(row['close']),
                'high': float(row['high']),
                'low': float(row['low']),
                'volume': float(row['volume']),
            })
        return out
    else:
        # 返回 DataFrame 形式
        df.reset_index().rename(columns={'index': 'timestamp'}, inplace=True)
        df = df.astype({
            'close': 'float32',
            'high': 'float32',
            'low': 'float32',
            'volume': 'float32'
        })
        return df

def fetch_closed_klines(limit=1):
    params = {"symbol": SYMBOL, "interval": "1m", "limit": limit}
    ks = requests.get(API_URL, params=params, timeout=5).json()
    return [(k[6]//1000, float(k[4]), float(k[5])) for k in ks]  # (timestamp, close, volume)

def predict_next(candle_buf,predict_for=1):
    x_input = build_input_from_candles(candle_buf)
    pred_scaled = model.predict(x_input, verbose=0)
    pred_scaled = pred_scaled[0,0]
    feature_names = ["close", "volume", "hl_spread", "return", "rsi"]
    last_features = x_input[0, -1, 1:]  # (4,)
    full_input = np.concatenate([[pred_scaled], last_features])  # (5,)

    close_inverse = scaler.inverse_transform([full_input])[predict_for-1][0]
    # close_inverse = scaler.inverse_transform([full_input])[0, :, 0]
    return close_inverse

def predict_next_v2(candle_buf,predict_for=1):
    x_input = build_input_from_candles_v2(candle_buf)
    # y_pred_z = model.predict(x_input, verbose=0)[0]
    # y_pred_mean = y_pred_z * y_std.squeeze() + y_mean.squeeze()
    #
    # H = y_pred_mean.shape[0]
    # horizons = np.arange(1, H + 1, dtype=np.float32)  # [1,2,3,4,5]
    # cum_logret = y_pred_mean * horizons
    # # 当前价格（从 candle_buf 最后一行取 close）
    # last_price = candle_buf[-1]['close'] # 假设第 0 列是 close
    # price_path = last_price * np.exp(cum_logret)   # (H,)
    # future_price = float(price_path[predict_for - 1])
    # 用 return 换算回 price
    step_logret = model.predict(x_input, verbose=0)[0]  # shape: (H,) 或 (H,1)
    step_logret = np.asarray(step_logret, dtype=np.float32).reshape(-1)  # (H,)

    # 3) 逐步 → 累计 log-return（可加性，英文：additivity of log-returns）
    cum_logret = np.cumsum(step_logret)  # shape: (H,)

    # 4) 还原价格路径（复利，英文：compounding）
    if predict_for < 1 or predict_for > len(cum_logret):
        raise ValueError(f"predict_for 超界：1..{len(cum_logret)}")
    last_price = float(candle_buf[-1]['close'])
    price_path = last_price * np.exp(cum_logret)  # shape: (H,)

    return float(price_path[predict_for - 1])

def build_input_from_candles(candle_buf, window_size=30):

    if len(candle_buf) < 90:
        raise ValueError(" Need to have more than 60 data to get all feature")

    df = pd.DataFrame(candle_buf)

    # def log_na(step):
    #     print(f"[DEBUG] After {step}: NaN counts:\n{df.isna().sum()}\n")
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"] = df["high"] - df["low"]
    # df["price_delta"] = df["close"] - df["close"].shift(1)
    # df["tr"] = np.maximum(
    #     df["high"] - df["low"],
    #     np.maximum(
    #         abs(df["high"] - df["close"].shift(1)),
    #         abs(df["low"] - df["close"].shift(1))
    #     )
    # )
    # df["vol_change"] = df["volume"].pct_change()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["std_60min"] = df["close"].rolling(window=60).std()
    df["volatility"] = df["return"].rolling(window=window_size).std()

    df.dropna(inplace=True)

    features = df[["close", "volume", "hl_spread", "return", "rsi"]].values

    if len(features) < window_size:
        raise ValueError(f"Not enough available feature, current feature: {len(features)}")

    features_scaled = scaler.transform(features)  # shape: (N, 5)

    # 滑窗提取最新一段
    x = features_scaled[-window_size:]  # shape: (30, 5)
    x = x.reshape(1, window_size, x.shape[1])   # shape: (1, 30, 5)
    return x


def build_input_from_candles_v2(candle_buf, window_size=30):

    if len(candle_buf) < 90:
        raise ValueError(" Need to have more than 60 data to get all feature")

    df = pd.DataFrame(candle_buf)

    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['price_delta'] = df['close'] - df['close'].shift(1)
    df['hl_spread'] = df['high'] - df['low']
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['vol_change'] = df['volume'].pct_change()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['std_60min'] = df['close'].rolling(window=30).std()
    df["volatility"] = df["return"].rolling(window=WINDOW).std()

    df.dropna(inplace=True)
    features = df[['close', 'return', 'volume', 'hl_spread', 'rsi', 'volatility', 'ema20', 'std_60min']].values

    if len(features) < window_size:
        raise ValueError(f"Not enough available feature, current feature: {len(features)}")

    features_scaled = scaler.transform(features)  # shape: (N, 5)

    # 滑窗提取最新一段
    x = features_scaled[-window_size:]  # shape: (30, 5)
    x = x.reshape(1, window_size, x.shape[1])   # shape: (1, 30, 5)
    return x

def add_time_and_tech_features(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    assume_tz: str = "UTC",       # 当时间戳为naive时假定的时区
    local_tz: str | None = None,  # 若需要按本地时区取hour/dow，传入如 "America/Toronto"
    window_size: int = 60,
    keep_time_ints: bool = False  # 是否保留整数版 hour/dayofweek/minute
) -> pd.DataFrame:
    dt = pd.to_datetime(df[ts_col], errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(assume_tz)
    if local_tz:
        dt = dt.dt.tz_convert(local_tz)

    hour    = dt.dt.hour.astype(int)
    minute  = dt.dt.minute.astype(int)
    dow     = dt.dt.dayofweek.astype(int)   # Monday=0

    df["hour_sin"]      = np.sin(2 * np.pi * hour   / 24.0)
    df["hour_cos"]      = np.cos(2 * np.pi * hour   / 24.0)
    df["minute_sin"]    = np.sin(2 * np.pi * minute / 60.0)
    df["minute_cos"]    = np.cos(2 * np.pi * minute / 60.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow    / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow    / 7.0)

    if keep_time_ints:
        df["hour"] = hour
        df["minute"] = minute
        df["dayofweek"] = dow

    # 4) 你现有的技术特征(technical features)
    df["return"]     = np.log(df["close"]).diff()        # log-return
    df["hl_spread"]  = df["high"] - df["low"]            # high-low spread
    df["rsi"]        = ta.rsi(df["close"], length=14)    # RSI(14)
    df["volatility"] = df["return"].rolling(window=window_size).std()

    return df


def build_gru_input_from_candles(candle_buf, window_size=30):

    if len(candle_buf) < 90:
        raise ValueError(" Need to have more than 60 data to get all feature")

    df = pd.DataFrame(candle_buf)

    # def log_na(step):
    #     print(f"[DEBUG] After {step}: NaN counts:\n{df.isna().sum()}\n")
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"] = df["high"] - df["low"]
    # df["price_delta"] = df["close"] - df["close"].shift(1)
    # df["tr"] = np.maximum(
    #     df["high"] - df["low"],
    #     np.maximum(
    #         abs(df["high"] - df["close"].shift(1)),
    #         abs(df["low"] - df["close"].shift(1))
    #     )
    # )
    # df["vol_change"] = df["volume"].pct_change()
    df["rsi"] = ta.rsi(df["close"], length=14)
    # df["ema20"] = ta.ema(df["close"], length=20)
    # df["std_60min"] = df["close"].rolling(window=60).std()
    df["volatility"] = df["return"].rolling(window=window_size).std()

    df.dropna(inplace=True)

    features = df[["close", "volume", "hl_spread", "return", "rsi"]].values

    if len(features) < window_size:
        raise ValueError(f"Not enough available feature, current feature: {len(features)}")

    features_scaled = scaler.transform(features)  # shape: (N, 5)

    # 滑窗提取最新一段
    x = features_scaled[-window_size:]  # shape: (30, 5)
    x = x.reshape(1, window_size, x.shape[1])   # shape: (1, 30, 5)
    return x

def _to_utc(dt_in):
    if dt_in is None:
        return None
    if dt_in.tzinfo is None:
        return timezone.make_aware(dt_in, dt_timezone.utc)
    return dt_in.astimezone(dt_timezone.utc)

def update_bias_and_smooth(
    symbol: str,
    model_name: str,
    step_index: int = 1,
    window: int = 20,                 # 取最近 N 个点
    alpha: float = 0.2,               # EWMA 学习率
    pred_raw: float | None = None,    # 本次模型原始预测（必传才会返回平滑后的预测）
    bias_prev: float = 0.0,           # 旧的 bias（从你的校准表取）
    end_dt: datetime.datetime | None = None # 截止时间，默认用“上一根收盘”
):
    """
    返回:
      {
        "timestamps": [...],                # 对齐后的时间戳（:59）
        "close_series": np.array,           # 实际收盘
        "pred_series": np.array,            # 历史已校正预测 pred_corr
        "mean_err": float,                  # 窗口平均误差 (close - pred)
        "bias_new": float,                  # 用 mean_err 做 EWMA 后的新 bias
        "sigma_real": float,                # 实际波动率（log-return 的 std）
        "sigma_pred": float,                # 预测波动率（log-return 的 std）
        "smooth_factor": float,             # 平滑系数 = min(1, sigma_real/sigma_pred)
        "pred_smoothed": float | None,      # 用平滑系数+新bias 得到的本次校正预测
        "last_pred_corr": float | None      # 最近一个已存的校正预测（用于平滑的起点）
      }
    """
    # —— 1) 取最近 window 条预测，拿到它们的 predicted_for 作为对齐锚（:59） —— #
    pred_qs = (ModelPrediction.objects
               .filter(symbol=symbol, model_name=model_name, step_index=step_index)
               .order_by("-predicted_for"))

    if end_dt:
        end_dt = _to_utc(end_dt)
        pred_qs = pred_qs.filter(predicted_for__lte=end_dt)

    preds = list(pred_qs[:window][::-1])  # 时间升序
    if not preds:
        return {
            "timestamps": [],
            "close_series": np.array([]),
            "pred_series": np.array([]),
            "mean_err": 0.0,
            "bias_new": bias_prev,
            "sigma_real": 0.0,
            "sigma_pred": 0.0,
            "smooth_factor": 1.0,
            "pred_smoothed": None,
            "last_pred_corr": None,
        }

    ts_list = [p.predicted_for for p in preds]

    # —— 2) 取同时间戳的真实 close（你的库里 timestamp 也是 :59） —— #
    candles = (MarketCandle.objects
               .filter(symbol=symbol, timestamp__in=ts_list)
               .values("timestamp", "close"))
    close_map = {c["timestamp"]: float(c["close"]) for c in candles}

    # 对齐：只保留两边都有的数据点
    aligned = [(t, close_map[t], float(p.pred_corr) if p.pred_corr is not None else None)
               for t, p in zip(ts_list, preds) if t in close_map and p.pred_corr is not None]

    if len(aligned) < max(5, int(window/3)):
        # 点太少，避免数值不稳
        close_series = np.array([a[1] for a in aligned], dtype=float)
        pred_series  = np.array([a[2] for a in aligned], dtype=float)
    else:
        close_series = np.array([a[1] for a in aligned], dtype=float)
        pred_series  = np.array([a[2] for a in aligned], dtype=float)

    # —— 3) 误差均值 + EWMA 更新 bias —— #
    if len(close_series) == 0:
        mean_err = 0.0
        bias_new = bias_prev
    else:
        mean_err = float(np.mean(close_series - pred_series))
        bias_new = (1 - alpha) * bias_prev + alpha * mean_err

    # —— 4) 用 log-return 的 std 当“窗口波动率” —— #
    def _ret_std(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        r = np.diff(np.log(x))
        if len(r) == 0:
            return 0.0
        return float(np.std(r))

    sigma_real = _ret_std(close_series)
    sigma_pred = _ret_std(pred_series)
    smooth_factor = 1.0 if sigma_pred <= 0 else min(1.0, sigma_real / (sigma_pred + 1e-8))

    # —— 5) 生成“平滑后的本次预测” —— #
    last_pred_corr = float(pred_series[-1]) if len(pred_series) > 0 else None
    pred_smoothed = None
    if pred_raw is not None and last_pred_corr is not None:
        # 先做 bias 纠偏，再做振幅平滑（以 last_pred_corr 为锚，避免跳变）
        pred_bias_corrected = float(pred_raw) + bias_new  # 你的 err 定义是 actual - pred
        pred_smoothed = last_pred_corr + (pred_bias_corrected - last_pred_corr) * smooth_factor

    return {
        "timestamps": [a[0] for a in aligned],
        "close_series": close_series,
        "pred_series": pred_series,
        "mean_err": mean_err,
        "bias_new": bias_new,
        "sigma_real": sigma_real,
        "sigma_pred": sigma_pred,
        "smooth_factor": smooth_factor,
        "pred_smoothed": pred_smoothed,
        "last_pred_corr": last_pred_corr,
    }

# pip install pmdarima
from pmdarima import auto_arima

def arima_residual_correction(symbol, model_name, step_index, end_ts,
                              yhat_lstm_corr, lookback=200, seasonal=False,
                              max_pq=3, alpha=None, min_len=40, min_var=1e-6):
    """
    返回 (yhat_final, err_forecast, order, aic)
    yhat_lstm_corr / close 都是价格量纲（price）
    """

    # 1) 拉取历史并对齐（与你现有一致）
    preds = (ModelPrediction.objects
             .filter(symbol=symbol, model_name=model_name, step_index=step_index,
                     predicted_for__lte=end_ts)
             .order_by("-predicted_for")
             .values("predicted_for", "pred_corr")[:lookback])
    preds = list(preds)[::-1]
    if not preds:
        return yhat_lstm_corr, 0.0, None, None

    ts_list  = [p["predicted_for"] for p in preds]
    pred_map = {p["predicted_for"]: float(p["pred_corr"]) for p in preds if p["pred_corr"] is not None}
    close_map = {c["timestamp"]: float(c["close"]) for c in
                 MarketCandle.objects.filter(symbol=symbol, timestamp__in=ts_list).values("timestamp","close")}

    pairs = [(t, close_map[t], pred_map[t]) for t in ts_list if t in close_map and t in pred_map]
    if len(pairs) < min_len:
        return yhat_lstm_corr, 0.0, None, None

    closes     = np.array([x[1] for x in pairs], dtype=float)
    preds_corr = np.array([x[2] for x in pairs], dtype=float)
    resid      = closes - preds_corr

    # 2) 如果残差几乎“白噪声”，就别修了（避免过拟合抖动）
    if np.var(resid) < min_var:
        return yhat_lstm_corr, 0.0, None, None

    # 3) 拟合 ARIMA（或缓存/增量更新，见后述）
    try:
        arima = auto_arima(
            resid, start_p=0, start_q=0, max_p=max_pq, max_q=max_pq,
            seasonal=seasonal, m=1, information_criterion="aic",
            stepwise=True, suppress_warnings=True, error_action="ignore"
        )
        err_forecast = float(arima.predict(n_periods=1)[0])
        order = arima.order
        aic = float(arima.aic())
    except Exception:
        return yhat_lstm_corr, 0.0, None, None

    # 4) 混合系数 alpha：可固定，也可自适应（推荐离线调一个 per-horizon α）
    if alpha is None:
        # 简单经验：步长越远，alpha 可稍大；也可从验证集回归/网格选
        alpha = 0.6 if step_index >= 4 else 0.4

    yhat_final = yhat_lstm_corr + alpha * err_forecast
    return yhat_final, err_forecast, order, aic


