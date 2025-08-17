# predict/management/commands/backfill_data.py
import datetime as dt
import time
from django.core.management.base import BaseCommand, CommandError
from django.utils.timezone import make_aware
from django.db import transaction

from predict.models import MarketCandle, ModelPrediction
from predict.services import *
from predict.utils import *

SYMBOL_DEFAULT = "BTCUSDT"
MODEL_DEFAULT  = "LSTM-v1"

def _utc_floor_minute(ts: dt.datetime) -> dt.datetime:
    return ts.replace(second=0, microsecond=0, tzinfo=ts.tzinfo)

def _utc_from_iso(s: str) -> dt.datetime:
    if s.endswith("Z"):
        s = s[:-1]
    try:
        dt_naive = dt.datetime.fromisoformat(s)
    except Exception as e:
        raise CommandError(f"Invalid datetime: {s}") from e
    if dt_naive.tzinfo is None:
        return make_aware(dt_naive)
    return dt_naive

def backfill_candles(symbol: str, start: dt.datetime, end: dt.datetime, batch_minutes: int = 1000):

    start_epoch = int(start.timestamp())
    end_epoch   = int(end.timestamp())

    cur = start_epoch
    inserted = 0
    while cur < end_epoch:
        nxt = min(cur + batch_minutes * 60, end_epoch)
        batch = fetch_klines(cur, nxt, symbol=symbol)
        for ts, high, low, close, volume in batch:
            MarketCandle.objects.update_or_create(
                symbol=symbol,
                timestamp=make_aware(dt.datetime.utcfromtimestamp(ts)),
                defaults={"high": high, "low": low, "close": close, "volume": volume},
            )
            inserted += 1
        print(f"  Klines {dt.datetime.utcfromtimestamp(cur)} ~ {dt.datetime.utcfromtimestamp(nxt)} -> {len(batch)} rows")
        cur = nxt
        time.sleep(0.1)
    print(f"✅ Candle backfill done. total rows touched: {inserted}")

def backfill_predictions(symbol: str, model_name: str, start: dt.datetime, end: dt.datetime,
                         window_for_bias: int = 20, alpha: float = 0.2, lookback_arima: int = 200):

    cal = load_calibration(model_name, symbol)

    t = start
    total = 0
    while t < end:
        ts_dt = t.replace(second=59, microsecond=0)

        if not MarketCandle.objects.filter(symbol=symbol, timestamp=ts_dt).exists():
            fetched = fetch_klines(int(t.timestamp()), int((t + dt.timedelta(minutes=1)).timestamp()), symbol=symbol)
            for ts, high, low, close, volume in fetched:
                MarketCandle.objects.update_or_create(
                    symbol=symbol,
                    timestamp=make_aware(dt.datetime.utcfromtimestamp(ts)),
                    defaults={"high": high, "low": low, "close": close, "volume": volume},
                )

        candle = MarketCandle.objects.filter(symbol=symbol, timestamp=ts_dt).values("close").first()
        if not candle:
            t += dt.timedelta(minutes=1)
            continue
        close = float(candle["close"])

        prev_pred = ModelPrediction.objects.filter(
            symbol=symbol, model_name=model_name, predicted_for=ts_dt, step_index=1
        ).order_by("-id").first()
        if prev_pred and prev_pred.pred_corr is not None:
            err = close - float(prev_pred.pred_corr)
            cal.bias = (1 - alpha) * cal.bias + alpha * err  # ewma_update
            cal.save(update_fields=["bias", "updated_at"])

        candle_buf = get_or_fetch_recent_candles(n=90)
        yhat_raw = predict_next(candle_buf)

        stats = update_bias_and_smooth(
            symbol=symbol,
            model_name=model_name,
            step_index=1,
            window=window_for_bias,
            alpha=alpha,
            pred_raw=yhat_raw,
            bias_prev=cal.bias,
            end_dt=ts_dt
        )
        cal.bias = stats["bias_new"]
        cal.save(update_fields=["bias", "updated_at"])

        yhat_corr = stats["pred_smoothed"] if stats["pred_smoothed"] is not None else (yhat_raw + cal.bias)

        yhat_arima, err_hat, arima_order, aic = arima_residual_correction(
            symbol=symbol,
            model_name=model_name,
            step_index=1,
            end_ts=ts_dt,
            yhat_lstm_corr=yhat_corr,
            lookback=lookback_arima,
            seasonal=False,
            max_pq=3
        )

        anchor_for    = minute_anchor(ts_dt)
        predicted_for = next_minute_close(ts_dt)
        horizon_sec   = 60
        step_index    = 1

        # 幂等：以 (symbol, model_name, anchor_for, step_index) 为键回放
        ModelPrediction.objects.update_or_create(
            symbol=symbol, model_name=model_name,
            anchor_for=anchor_for, step_index=step_index,
            defaults=dict(
                predicted_for=predicted_for,
                horizon_sec=horizon_sec,
                pred_raw=yhat_raw,
                pred_corr=yhat_corr,
                pred_arima=yhat_arima,
                bias_used=cal.bias,
                alpha_used=alpha,
                extra_meta=(
                    f"backfill=1, arima_order={arima_order}, aic={aic}, "
                    f"err_hat={(err_hat or 0):.6f}, smooth_factor={(stats.get('smooth_factor',1.0)):.4f}"
                )
            )
        )

        total += 1
        if total % 200 == 0:
            print(f"  Pred backfill @ {ts_dt} … processed {total} mins")
        t += dt.timedelta(minutes=1)

    print(f"✅ Prediction backfill done. minutes processed: {total}")

class Command(BaseCommand):
    help = "Backfill candles and/or predictions from a start datetime to an end datetime."

    def add_arguments(self, parser):
        parser.add_argument("--from", dest="from_dt", required=True,
                            help="Backfill start datetime, e.g. 2025-08-02T00:00:00Z")
        parser.add_argument("--to", dest="to_dt", default=None,
                            help="Backfill end datetime (exclusive). Default=now")
        parser.add_argument("--symbol", default=SYMBOL_DEFAULT)
        parser.add_argument("--model",  default=MODEL_DEFAULT)
        parser.add_argument("--only",   choices=["candles", "preds", "both"], default="both")
        parser.add_argument("--bias_window", type=int, default=20)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--arima_lookback", type=int, default=200)

    @transaction.atomic
    def handle(self, *args, **opts):
        symbol = opts["symbol"]
        model  = opts["model"]
        start  = _utc_floor_minute(_utc_from_iso(opts["from_dt"]))
        end    = _utc_floor_minute(_utc_from_iso(opts["to_dt"])) if opts["to_dt"] else _utc_floor_minute(make_aware(dt.datetime.utcnow()))

        if end <= start:
            raise CommandError("`to` must be greater than `from`.")

        print(f"Backfill range: {start}  ->  {end}  (symbol={symbol}, model={model})")

        mode = opts["only"]
        if mode in ("candles", "both"):
            print("▶ Backfilling candles …")
            backfill_candles(symbol, start, end)

        if mode in ("preds", "both"):
            print("▶ Backfilling predictions …")
            backfill_predictions(
                symbol=symbol,
                model_name=model,
                start=start,
                end=end,
                window_for_bias=opts["bias_window"],
                alpha=opts["alpha"],
                lookback_arima=opts["arima_lookback"],
            )

        print("All done.")
