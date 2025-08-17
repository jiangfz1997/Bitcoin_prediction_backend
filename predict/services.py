# app_name/services.py
import time
import datetime
import traceback
from django.utils.timezone import make_aware


from .models import *
from .utils import *
from .xgboost_feature import *

ALPHA_DEFAULT = 0.25
MODEL = os.environ.get("MODEL", "LSTM-v1")
def minute_anchor(ts_dt, offset=1):
    return ts_dt.replace(second=0, microsecond=0) + datetime.timedelta(minutes=offset)

def minute_start(dt):  # 该分钟起始 :00
    return dt.replace(second=0, microsecond=0)

def next_minute_close(dt):  # 下一分钟的收盘 :59
    return minute_start(dt) + datetime.timedelta(minutes=1, seconds=59)

def next_k_minute_close(dt, k=1):
    return minute_start(dt) + datetime.timedelta(minutes=k, seconds=59)

def ewma_update(bias, error, alpha):
    return (1 - alpha) * bias + alpha * error

def load_calibration(model_name, symbol):
    cal, _ = ModelCalibration.objects.get_or_create(
        model_name=model_name, symbol=symbol,
        defaults={"alpha": ALPHA_DEFAULT, "bias": 0.0}
    )
    return cal


def start_fetch_loop():


    WINDOW = 30
    SYMBOL = "BTCUSDT"

    print("🔍 初始化过去 1 小时市场数据…")
    end_ts = int(time.time())
    start_ts = end_ts - 3600
    candles = fetch_klines(start_ts, end_ts, symbol=SYMBOL)
    for ts, high, low, close, volume in candles:
        MarketCandle.objects.update_or_create(
            timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
            defaults={"high": high, "low": low, "close": close, "volume": volume}
        )
    print(f"✅ 数据回补完成，条数：{len(candles)}")

    print(f"Start real-time prediction using model {MODEL} loop every 60 seconds...")
    last_ts = end_ts
    while True:
        try:
            candles = fetch_klines(last_ts + 1, last_ts + 120)
            if not candles:
                print("Fetch failed or no new data, retrying in 5 seconds...")
                time.sleep(5)
                continue

            new_candle = candles[-1]
            ts, high, low, close, volume = new_candle
            last_ts = ts

            MarketCandle.objects.update_or_create(
                symbol=SYMBOL,
                timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
                defaults={
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                }
            )

            ts_dt = make_aware(datetime.datetime.utcfromtimestamp(ts))
            cal = load_calibration(MODEL, SYMBOL)
            prev_pred = ModelPrediction.objects.filter(
                symbol=SYMBOL, model_name=MODEL, predicted_for=ts_dt, step_index=1
            ).order_by("-id").first()

            if prev_pred and prev_pred.pred_corr is not None:
                err = close - float(prev_pred.pred_corr) # single step error

                cal.bias = ewma_update(cal.bias, err, cal.alpha)
                cal.save(update_fields=["bias", "updated_at"])

            candle_buf = get_or_fetch_recent_candles(n=90)
            yhat_raw = predict_next(candle_buf)
            # yhat_corr = yhat_raw + cal.bias

            step_index =  1  # 步骤索引从 1 开始
            stats = update_bias_and_smooth(
                symbol=SYMBOL,
                model_name=MODEL,
                step_index=1,
                window=20,
                alpha=0.2,
                pred_raw=yhat_raw,
                bias_prev=cal.bias,  # 从你的校准表取
                end_dt=ts_dt  # 这根收盘（:59）
            )
            cal.bias = stats["bias_new"]
            cal.save(update_fields=["bias", "updated_at"])

            # 这次用于入库的校正预测：
            yhat_corr = stats["pred_smoothed"] if stats["pred_smoothed"] is not None else (yhat_raw + cal.bias)

            yhat_arima, err_hat, arima_order, aic = arima_residual_correction(
                symbol=SYMBOL,
                model_name=MODEL,
                step_index=1,
                end_ts=ts_dt,
                yhat_lstm_corr=yhat_raw,
                lookback=200,
                seasonal=False,
                max_pq=3
            )

            anchor_for = minute_anchor(ts_dt)  # 下一根K线的开始时间
            predicted_for = next_k_minute_close(ts_dt, step_index)
            horizon_sec = 60

            ModelPrediction.objects.update_or_create(
                symbol=SYMBOL,
                model_name=MODEL,
                anchor_for=anchor_for,
                step_index=step_index,
                defaults=dict(
                    predicted_for=predicted_for,
                    horizon_sec=horizon_sec* step_index,
                    pred_raw=yhat_raw,
                    pred_corr=yhat_corr,
                    pred_arima=yhat_arima,
                    bias_used=cal.bias,
                    alpha_used=cal.alpha,
                    extra_meta=f"arima_order={arima_order},aic={aic},err_hat={err_hat:.6f}"
                ),
            )

            print(f"[{datetime.datetime.fromtimestamp(ts)}] close={close:.2f} pred={yhat_raw:.2f} EWMA_fixed={yhat_corr:.2f} arima_fixed={yhat_arima:.2f} bias={cal.bias:.4f}")
            sleep_sec = 60 - datetime.datetime.utcnow().second
            time.sleep(max(sleep_sec, 1))

        except Exception as e:
            print(f"❌ 错误: {e}")
            traceback.print_exc()
            time.sleep(5)


def start_fetch_loop_v2():
    step_index = 5
    WINDOW = 30
    SYMBOL = "BTCUSDT"
    MODEL = "LSTM-v2"
    print("🔍 初始化过去 1 小时市场数据…")
    end_ts = int(time.time())
    start_ts = end_ts - 3600
    candles = fetch_klines(start_ts, end_ts, symbol=SYMBOL)
    for ts, high, low, close, volume in candles:
        MarketCandle.objects.update_or_create(
            timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
            defaults={"high": high, "low": low, "close": close, "volume": volume}
        )
    print(f"✅ 数据回补完成，条数：{len(candles)}")

    print(f"Start real-time prediction using model {MODEL} loop every 60 seconds...")
    last_ts = end_ts
    while True:
        try:
            candles = fetch_klines(last_ts + 1, last_ts + 120)
            if not candles:
                print("Fetch failed or no new data, retrying in 5 seconds...")
                time.sleep(5)
                continue

            new_candle = candles[-1]
            ts, high, low, close, volume = new_candle
            last_ts = ts

            MarketCandle.objects.update_or_create(
                symbol=SYMBOL,
                timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
                defaults={
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                }
            )

            ts_dt = make_aware(datetime.datetime.utcfromtimestamp(ts))
            cal = load_calibration(MODEL, SYMBOL)
            prev_pred = ModelPrediction.objects.filter(
                symbol=SYMBOL, model_name=MODEL, predicted_for=ts_dt, step_index=1
            ).order_by("-id").first()

            if prev_pred and prev_pred.pred_corr is not None:
                err = close - float(prev_pred.pred_corr) # single step error

                cal.bias = ewma_update(cal.bias, err, cal.alpha)
                cal.save(update_fields=["bias", "updated_at"])

            candle_buf = get_or_fetch_recent_candles(n=90)
            yhat_raw = predict_next_v2(candle_buf, predict_for=step_index)
            # yhat_corr = yhat_raw + cal.bias


            stats = update_bias_and_smooth(
                symbol=SYMBOL,
                model_name=MODEL,
                step_index=1,
                window=20,
                alpha=0.2,
                pred_raw=yhat_raw,
                bias_prev=cal.bias,  # 从你的校准表取
                end_dt=ts_dt  # 这根收盘（:59）
            )
            cal.bias = stats["bias_new"]
            cal.save(update_fields=["bias", "updated_at"])

            # 这次用于入库的校正预测：
            yhat_corr = stats["pred_smoothed"] if stats["pred_smoothed"] is not None else (yhat_raw + cal.bias)

            yhat_arima, err_hat, arima_order, aic = arima_residual_correction(
                symbol=SYMBOL,
                model_name=MODEL,
                step_index=1,
                end_ts=ts_dt,
                yhat_lstm_corr=yhat_raw,
                lookback=200,
                seasonal=False,
                max_pq=3
            )


            anchor_for = minute_anchor(ts_dt)
            predicted_for = next_k_minute_close(ts_dt, k=step_index)
            horizon_sec = 60*step_index

            ModelPrediction.objects.update_or_create(
                symbol=SYMBOL,
                model_name=MODEL,
                anchor_for=anchor_for,
                step_index=step_index,
                defaults=dict(
                    predicted_for=predicted_for,
                    horizon_sec=horizon_sec,
                    pred_raw=yhat_raw,
                    pred_corr=yhat_corr,
                    pred_arima=yhat_arima,
                    bias_used=cal.bias,
                    alpha_used=cal.alpha,
                    extra_meta=f"arima_order={arima_order},aic={aic},err_hat={err_hat:.6f}"
                ),
            )

            print(f"[{datetime.datetime.fromtimestamp(ts)}] predict_for={predicted_for} close={close:.2f} pred={yhat_raw:.2f} EWMA_fixed={yhat_corr:.2f} arima_fixed={yhat_arima:.2f} bias={cal.bias:.4f}")
            sleep_sec = 60 - datetime.datetime.utcnow().second
            time.sleep(max(sleep_sec, 1))

        except Exception as e:
            print(f"❌ 错误: {e}")
            traceback.print_exc()
            time.sleep(5)


def start_fetch_xgboost_loop():
    SYMBOL = "BTCUSDT"
    WINDOW = 300
    end_ts = int(time.time())
    start_ts = end_ts - 3600
    candles = fetch_klines(start_ts, end_ts, symbol=SYMBOL)
    for ts, high, low, close, volume in candles:
        MarketCandle.objects.update_or_create(
            timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
            defaults={"high": high, "low": low, "close": close, "volume": volume}
        )
    print(f"✅ 数据回补完成，条数：{len(candles)}")

    print(f"Start real-time prediction using model {MODEL} loop every 60 seconds...")
    last_ts = end_ts
    while True:
        try:
            candles = fetch_klines(last_ts + 1, last_ts + 120)
            if not candles:
                print("Fetch failed or no new data, retrying in 5 seconds...")
                time.sleep(5)
                continue

            new_candle = candles[-1]
            ts, high, low, close, volume = new_candle
            last_ts = ts

            MarketCandle.objects.update_or_create(
                symbol=SYMBOL,
                timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
                defaults={
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                }
            )

            ts_dt = make_aware(datetime.datetime.utcfromtimestamp(ts))
            cal = load_calibration(MODEL, SYMBOL)
            prev_pred = ModelPrediction.objects.filter(
                symbol=SYMBOL, model_name=MODEL, predicted_for=ts_dt, step_index=1
            ).order_by("-id").first()

            if prev_pred and prev_pred.pred_corr is not None:
                err = close - float(prev_pred.pred_corr)  # single step error

                cal.bias = ewma_update(cal.bias, err, cal.alpha)
                cal.save(update_fields=["bias", "updated_at"])

            candle_buf = get_or_fetch_recent_candles(n=WINDOW, return_type="dataframe")
            feat_df  = make_features(candle_buf)
            if feat_df.empty:
                return {"ok": False, "reason": "features_empty"}

                # c) 取最后一行，组装 x_test（列顺序必须与训练一致）

            def _build_FEATURES(columns):
                return [c for c in columns if c not in ("datetime", "target", "fwd_ret_1")]

            FEATURES = _build_FEATURES(feat_df.columns)
            last = feat_df.iloc[-1]
            if last[FEATURES].isna().any():
                return {"ok": False, "reason": "nan_in_features"}

            X_row = last[FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)

            pred_ret = xgboost_predict_next(X_row)
            close_t = float(last["Close"])
            pred_close = close_t * (1.0 + pred_ret)

            t = pd.to_datetime(last["datetime"])
            predicted_at = pd.Timestamp(t).isoformat()
            predicted_for = (pd.Timestamp(t) + pd.Timedelta(minutes=1)).isoformat()




            anchor_for = minute_anchor(ts_dt)  # 下一根K线的开始时间
            predicted_for = next_minute_close(ts_dt)
            horizon_sec = 60
            step_index = 1

            ModelPrediction.objects.update_or_create(
                symbol=SYMBOL,
                model_name="XGBoost-v1",
                anchor_for=anchor_for,
                step_index=step_index,
                defaults=dict(
                    predicted_for=predicted_for,
                    horizon_sec=horizon_sec,
                    pred_raw=pred_close,
                    pred_corr=pred_close,
                    bias_used=cal.bias,
                    alpha_used=cal.alpha,
                    extra_meta=f"xgboost_features={FEATURES}",
                ),
            )

            print(
                f"[{datetime.datetime.fromtimestamp(ts)}] close={close:.2f} pred={pred_close:.2f}")
            sleep_sec = 60 - datetime.datetime.utcnow().second
            time.sleep(max(sleep_sec, 1))

        except Exception as e:
            print(f"❌ 错误: {e}")
            traceback.print_exc()
            time.sleep(5)
