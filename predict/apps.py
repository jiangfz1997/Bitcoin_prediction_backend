from django.apps import AppConfig
import threading
from django.db import connection
class PredictConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predict'

    def ready(self):
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA journal_mode=WAL;")
        from .services import start_fetch_loop
        # 避免在多进程模式（比如 gunicorn）重复启动
        if not threading.main_thread().is_alive():
            return
        t = threading.Thread(target=start_fetch_loop, daemon=True)
        t.start()