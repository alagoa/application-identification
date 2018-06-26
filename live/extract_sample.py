from celery import Celery
from tasks import *

app = Celery()

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Calls extract_sample() every 30 seconds.
    sender.add_periodic_task(30.0, extract_sample.s(), name='add every 30')