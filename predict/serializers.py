from rest_framework import serializers
from .models import *

class ModelPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelPrediction
        fields = '__all__'
class MarketCandleSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketCandle
        fields = '__all__'