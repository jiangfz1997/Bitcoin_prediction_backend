from django.apps import AppConfig
import threading
from django.db import connection
class PredictConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predict'

    def ready(self):
        pass
        # with connection.cursor() as cursor:
        #     cursor.execute("PRAGMA journal_mode=WAL;")
        # from .services import start_fetch_loop
        # if not threading.main_thread().is_alive():
        #     return
        # t = threading.Thread(target=start_fetch_loop, daemon=True)
        # t.start()