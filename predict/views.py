# views.py
import os

from rest_framework import viewsets
from django.http import JsonResponse
from django.utils.dateparse import parse_datetime

from .serializers import ModelPredictionSerializer
from .models import ModelPrediction, MarketCandle

MODEL = os.environ.get("MODEL", "LSTM-v1")
class ModelPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelPrediction.objects.all()  # 供 DRF 推断 basename
    serializer_class = ModelPredictionSerializer

    def get_queryset(self):
        qs = ModelPrediction.objects.all()

        # 可选过滤参数：?symbol=BTCUSDT&model_name=LSTM-v1&step_index=1
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

        # 用新字段排序；取最近200条
        return qs.order_by("-predicted_for")[:200]


def get_predictions(request):
    """按时间范围返回模型预测数据（兼容老字段名：timestamp/pred）"""
    start_time = request.GET.get("from")
    end_time   = request.GET.get("to")
    limit      = request.GET.get("limit")
    symbol     = request.GET.get("symbol")
    model_name = request.GET.get("model_name")
    step       = request.GET.get("step_index") or "1"   # 默认只看第1步

    qs = ModelPrediction.objects.all()

    if symbol:
        qs = qs.filter(symbol=symbol)
    if model_name:
        qs = qs.filter(model_name=model_name)
    if step:
        qs = qs.filter(step_index=int(step))

    # 时间过滤基于 predicted_for
    if start_time:
        dt = parse_datetime(start_time)
        if dt:
            qs = qs.filter(predicted_for__gte=dt)
    if end_time:
        dt = parse_datetime(end_time)
        if dt:
            qs = qs.filter(predicted_for__lte=dt)

    qs = qs.order_by("predicted_for")  # 时间升序方便画图
    if limit:
        qs = qs[:int(limit)]

    # 输出映射：predicted_for -> timestamp，pred_corr -> pred
    data = [
        {
            "timestamp": obj.predicted_for,
            "model_name": obj.model_name,
            "pred": obj.pred_corr,
            # 如需调试也可额外返回 raw/bias：
            # "pred_raw": obj.pred_raw, "bias_used": obj.bias_used
        }
        for obj in qs
    ]
    return JsonResponse(data, safe=False)


def get_market_candles(request):
    """按时间范围返回市场K线数据"""
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

    # 用 predicted_for 对齐到 candle.timestamp
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
    """
    返回可用的模型名列表。
    可选参数：
      - symbol: 只统计该交易对下出现过的模型名
      - since: 只统计该时间之后产生过预测的模型（ISO时间，UTC）
    """
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