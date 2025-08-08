from django.db import models

class ModelPrediction(models.Model):
    timestamp  = models.DateTimeField()
    model_name = models.CharField(max_length=100)
    pred       = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('timestamp', 'model_name')

    def __str__(self):
        return f"{self.timestamp} | {self.model_name} ➜ {self.pred:.2f}"

class MarketCandle(models.Model):
    timestamp = models.DateTimeField(unique=True)
    close     = models.FloatField()
    high      = models.FloatField()
    low       = models.FloatField()
    volume    = models.FloatField()

    def __str__(self):
        return f"[{self.timestamp}] close={self.close}"