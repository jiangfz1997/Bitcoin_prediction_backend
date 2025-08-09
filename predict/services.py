# app_name/services.py
import time
import datetime
import traceback
from django.utils.timezone import make_aware


from .models import *
from .utils import *
ALPHA_DEFAULT = 0.10
MODEL = os.environ.get("MODEL", "LSTM-v1")
def minute_anchor(ts_dt, offset=1):
    return ts_dt.replace(second=0, microsecond=0) + datetime.timedelta(minutes=offset)

def minute_start(dt):  # 该分钟起始 :00
    return dt.replace(second=0, microsecond=0)

def next_minute_close(dt):  # 下一分钟的收盘 :59
    return minute_start(dt) + datetime.timedelta(minutes=1, seconds=59)

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
            predicted_for = next_minute_close(ts_dt)
            horizon_sec = 60
            step_index = 1

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

            print(f"[{datetime.datetime.fromtimestamp(ts)}] close={close:.2f} pred={yhat_raw:.2f} EWMA_fixed={yhat_corr:.2f} arima_fixed={yhat_arima:.2f} bias={cal.bias:.4f}")
            sleep_sec = 60 - datetime.datetime.utcnow().second
            time.sleep(max(sleep_sec, 1))

        except Exception as e:
            print(f"❌ 错误: {e}")
            traceback.print_exc()
            time.sleep(5)
