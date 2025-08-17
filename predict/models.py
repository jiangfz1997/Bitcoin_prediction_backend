# models.py
from django.contrib.postgres.fields import ArrayField
from django.db import models

class MarketCandle(models.Model):
    symbol    = models.CharField(max_length=32, default="BTCUSDT")  # 或 BTCUSDT
    timestamp = models.DateTimeField()  # 实际K线时间
    close     = models.FloatField()
    high      = models.FloatField()
    low       = models.FloatField()
    volume    = models.FloatField()

    class Meta:
        unique_together = ("symbol", "timestamp")
        indexes = [models.Index(fields=["symbol", "timestamp"])]

    def __str__(self):
        return f"[{self.symbol} {self.timestamp}] close={self.close}"

class ModelCalibration(models.Model):
    model_name = models.CharField(max_length=100)
    symbol     = models.CharField(max_length=32, default="BTCUSDT")
    alpha      = models.FloatField(default=0.10)  # EWMA smoothing factor
    bias       = models.FloatField(default=0.0)   # current EWMA bias (b_t)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("model_name", "symbol")

class ModelPrediction(models.Model):
    symbol        = models.CharField(max_length=32, default="BTCUSDT")
    model_name    = models.CharField(max_length=100)

    anchor_for    = models.DateTimeField(db_index=True)

    step_index    = models.PositiveSmallIntegerField(default=1)

    predicted_for = models.DateTimeField(db_index=True)

    horizon_sec   = models.IntegerField(default=60)

    predicted_at  = models.DateTimeField(auto_now_add=True)

    pred_raw      = models.FloatField(null=True, blank=True)
    pred_corr     = models.FloatField()
    pred_arima    = models.FloatField(null=True, blank=True)  # ARIMA residual correction

    bias_used     = models.FloatField(default=0.0)
    alpha_used    = models.FloatField(default=0.10)

    pred_agg      = models.FloatField(null=True, blank=True)
    agg_weights   = models.JSONField(null=True, blank=True)

    created_at    = models.DateTimeField(auto_now_add=True)
    extra_meta = models.JSONField(null=True, blank=True)

    class Meta:
        unique_together = ("symbol", "model_name", "anchor_for", "step_index")
        indexes = [
            models.Index(fields=["symbol", "model_name", "predicted_for"]),
            models.Index(fields=["predicted_at"]),
        ]

    def __str__(self):
        return f"{self.symbol} {self.predicted_for} | {self.model_name} [k={self.step_index}] -> {self.pred_corr:.2f}"

class MinuteFeature(models.Model):
    price = models.OneToOneField(MarketCandle, on_delete=models.CASCADE,
                                 primary_key=True, related_name="feature")
    vec = ArrayField(models.FloatField())
    base_close = models.FloatField()

    class Meta:
        db_table = "prices_minute_feature"