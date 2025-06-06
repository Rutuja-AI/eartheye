# gunicorn.conf.py (create this file in your project root)
bind = "0.0.0.0:10000"
timeout = 120
worker_class = "gthread"
threads = 2
max_requests = 100
max_requests_jitter = 20