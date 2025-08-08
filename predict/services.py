# app_name/services.py
import time
import datetime
import traceback
from django.utils.timezone import make_aware
from .models import MarketCandle, ModelPrediction
from .utils import *

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

    print("🚀 开始实时预测…")
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
                timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
                defaults={"high": high, "low": low, "close": close, "volume": volume}
            )

            candle_buf = get_or_fetch_recent_candles(n=90)
            pred = predict_next(candle_buf)

            ModelPrediction.objects.update_or_create(
                timestamp=make_aware(datetime.datetime.utcfromtimestamp(ts)),
                model_name="LSTM-v1",
                defaults={"pred": pred}
            )

            print(f"[{datetime.datetime.fromtimestamp(ts)}] close={close:.2f} pred={pred:.2f}")
            sleep_sec = 60 - datetime.datetime.utcnow().second
            time.sleep(max(sleep_sec, 1))

        except Exception as e:
            print(f"❌ 错误: {e}")
            traceback.print_exc()
            time.sleep(5)
