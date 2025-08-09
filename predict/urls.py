from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter()
router.register(r'predictions', ModelPredictionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path("predictions/", get_predictions, name="get_predictions"),
    path("candles/", get_market_candles, name="get_market_candles"),
    path("pred_and_actual/", get_pred_and_actual, name="get_pred_and_actual"),
    path("models/", list_models, name="list_models"),

]
