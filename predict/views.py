
# views.py
from rest_framework import viewsets
from .serializers import ModelPredictionSerializer
from django.http import JsonResponse
from .models import ModelPrediction, MarketCandle


class ModelPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelPrediction.objects.all().order_by('-timestamp')[:200]
    serializer_class = ModelPredictionSerializer

from django.http import JsonResponse
from django.utils.dateparse import parse_datetime
from .models import ModelPrediction, MarketCandle

def get_predictions(request):
    """按时间范围返回模型预测数据"""
    start_time = request.GET.get("from")
    end_time   = request.GET.get("to")
    limit      = request.GET.get("limit")

    qs = ModelPrediction.objects.all()

    # 时间过滤
    if start_time:
        start_dt = parse_datetime(start_time)
        if start_dt:
            qs = qs.filter(timestamp__gte=start_dt)
    if end_time:
        end_dt = parse_datetime(end_time)
        if end_dt:
            qs = qs.filter(timestamp__lte=end_dt)

    qs = qs.order_by("timestamp")  # 时间升序方便画图
    if limit:
        qs = qs[:int(limit)]

    data = list(qs.values("timestamp", "model_name", "pred"))
    return JsonResponse(data, safe=False)


def get_market_candles(request):
    """按时间范围返回市场K线数据"""
    start_time = request.GET.get("from")
    end_time   = request.GET.get("to")
    limit      = request.GET.get("limit")

    qs = MarketCandle.objects.all()

    if start_time:
        start_dt = parse_datetime(start_time)
        if start_dt:
            qs = qs.filter(timestamp__gte=start_dt)
    if end_time:
        end_dt = parse_datetime(end_time)
        if end_dt:
            qs = qs.filter(timestamp__lte=end_dt)

    qs = qs.order_by("timestamp")
    if limit:
        qs = qs[:int(limit)]

    data = list(qs.values("timestamp", "close", "high", "low", "volume"))
    return JsonResponse(data, safe=False)


def get_pred_and_actual(request):
    """返回预测值 + 实际值（按时间范围 & 模型名过滤）"""
    start_time = request.GET.get("from")
    end_time   = request.GET.get("to")
    model_name = request.GET.get("model_name")

    # 解析时间
    start_dt = parse_datetime(start_time) if start_time else None
    end_dt   = parse_datetime(end_time) if end_time else None

    # 取实际行情数据
    candle_qs = MarketCandle.objects.all()
    if start_dt:
        candle_qs = candle_qs.filter(timestamp__gte=start_dt)
    if end_dt:
        candle_qs = candle_qs.filter(timestamp__lte=end_dt)
    candle_qs = candle_qs.order_by("timestamp")

    # 取预测数据
    pred_qs = ModelPrediction.objects.all()
    # if model_name:
    #     pred_qs = pred_qs.filter(model_name=model_name)
    if start_dt:
        pred_qs = pred_qs.filter(timestamp__gte=start_dt)
    if end_dt:
        pred_qs = pred_qs.filter(timestamp__lte=end_dt)
    pred_qs = pred_qs.order_by("timestamp")

    # 转成 dict 方便合并
    pred_dict = {p.timestamp: p.pred for p in pred_qs}

    # 合并
    result = []
    for c in candle_qs:
        result.append({
            "timestamp": c.timestamp,
            "close": c.close,
            "pred": pred_dict.get(c.timestamp, None)  # 没预测就返回 null
        })

    return JsonResponse(result, safe=False)