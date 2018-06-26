# Adding rabbitMQ info for broker settings.
# amqp://<username>:<password>@localhost:5672/<virtual_host>
BROKER_URL = 'amqp://guest:guest@localhost:5672/'

# Using the database to store task state and results
CELERY_RESULT_BACKEND = 'redis://localhost:6379/3'

# List of modules to import when celery starts.
CELERY_IMPORTS = ('tasks', )