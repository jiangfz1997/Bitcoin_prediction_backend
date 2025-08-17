# views.py
import os

from rest_framework import viewsets
from django.http import JsonResponse
from django.utils.dateparse import parse_datetime

from .serializers import ModelPredictionSerializer
from .models import ModelPrediction, MarketCandle

MODEL = os.environ.get("MODEL", "LSTM-v1")
class ModelPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelPrediction.objects.all()
    serializer_class = ModelPredictionSerializer

    def get_queryset(self):
        qs = ModelPrediction.objects.all()

        symbol     = self.request.query_params.get("symbol")
        model_name = self.request.query_params.get("model_name") or self.request.query_params.get("model")
        step       = self.request.query_params.get("step_index")

        if symbol:
            qs = qs.filter(symbol=symbol)
        if model_name:
            qs = qs.filter(model_name=model_name)
        if step:
            qs = qs.filter(step_index=int(step))
        else:
            qs = qs.filter(step_index=1)  # 默认只看第1步（t+1）

        return qs.order_by("-predicted_for")[:200]


def get_predictions(request):
    start_time = request.GET.get("from")
    end_time   = request.GET.get("to")
    limit      = request.GET.get("limit")
    symbol     = request.GET.get("symbol")
    model_name = request.GET.get("model_name")
    step       = request.GET.get("step_index") or "1"

    qs = ModelPrediction.objects.all()

    if symbol:
        qs = qs.filter(symbol=symbol)
    if model_name:
        qs = qs.filter(model_name=model_name)
    if step:
        qs = qs.filter(step_index=int(step))

    if start_time:
        dt = parse_datetime(start_time)
        if dt:
            qs = qs.filter(predicted_for__gte=dt)
    if end_time:
        dt = parse_datetime(end_time)
        if dt:
            qs = qs.filter(predicted_for__lte=dt)

    qs = qs.order_by("predicted_for")
    if limit:
        qs = qs[:int(limit)]

    data = [
        {
            "timestamp": obj.predicted_for,
            "model_name": obj.model_name,
            "pred": obj.pred_corr,
            # "pred_raw": obj.pred_raw, "bias_used": obj.bias_used
        }
        for obj in qs
    ]
    return JsonResponse(data, safe=False)


def get_market_candles(request):
    start_time = request.GET.get("from")
    end_time   = request.GET.get("to")
    limit      = request.GET.get("limit")
    symbol     = request.GET.get("symbol")

    qs = MarketCandle.objects.all()
    if symbol:
        qs = qs.filter(symbol=symbol)

    if start_time:
        dt = parse_datetime(start_time)
        if dt:
            qs = qs.filter(timestamp__gte=dt)
    if end_time:
        dt = parse_datetime(end_time)
        if dt:
            qs = qs.filter(timestamp__lte=dt)

    qs = qs.order_by("timestamp")
    if limit:
        qs = qs[:int(limit)]

    data = list(qs.values("timestamp", "close", "high", "low", "volume"))
    return JsonResponse(data, safe=False)


def get_pred_and_actual(request):
    start_time = request.GET.get("from")
    end_time   = request.GET.get("to")
    model_name = request.GET.get("model_name")
    symbol     = request.GET.get("symbol")
    step       = request.GET.get("step_index") or "1"  # 默认第1步

    start_dt = parse_datetime(start_time) if start_time else None
    end_dt   = parse_datetime(end_time) if end_time else None

    candle_qs = MarketCandle.objects.all()
    if symbol:
        candle_qs = candle_qs.filter(symbol=symbol)
    if start_dt:
        candle_qs = candle_qs.filter(timestamp__gte=start_dt)
    if end_dt:
        candle_qs = candle_qs.filter(timestamp__lte=end_dt)
    candle_qs = candle_qs.order_by("timestamp")

    pred_qs = ModelPrediction.objects.all()
    if symbol:
        pred_qs = pred_qs.filter(symbol=symbol)
    if model_name:
        pred_qs = pred_qs.filter(model_name=model_name)
    if step:
        pred_qs = pred_qs.filter(step_index=int(step))
    if start_dt:
        pred_qs = pred_qs.filter(predicted_for__gte=start_dt)
    if end_dt:
        pred_qs = pred_qs.filter(predicted_for__lte=end_dt)
    pred_qs = pred_qs.order_by("predicted_for")

    pred_dict = {
        p.predicted_for: {
            "pred_corr": p.pred_corr,
            "pred_raw": p.pred_raw,
            "pred_arima": p.pred_arima,
        }
        for p in pred_qs
    }
    result = []
    for c in candle_qs:
        pred_info = pred_dict.get(c.timestamp, {})

        result.append({
            "timestamp": c.timestamp,
            "close": c.close,
            "pred_corr": pred_info.get("pred_corr"),
            "pred_raw": pred_info.get("pred_raw"),
            "pred_arima": pred_info.get("pred_arima"),
        })
    return JsonResponse(result, safe=False)
def list_models(request):
    qs = ModelPrediction.objects.all()

    symbol = request.GET.get("symbol")
    if symbol:
        qs = qs.filter(symbol=symbol)

    since = request.GET.get("since")
    if since:
        dt = parse_datetime(since)
        if dt:
            qs = qs.filter(predicted_at__gte=dt)

    names = list(qs.values_list("model_name", flat=True).distinct().order_by("model_name"))
    return JsonResponse({"models": names})