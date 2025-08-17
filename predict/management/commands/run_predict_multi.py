from django.core.management.base import BaseCommand
from predict.services import start_fetch_loop_v2

class Command(BaseCommand):
    help = "Run realtime prediction fetch loop (single instance)"

    def handle(self, *args, **options):
        start_fetch_loop_v2()
