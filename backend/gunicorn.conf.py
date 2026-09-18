# Gunicorn configuration for production deployment
# Usage: gunicorn -c gunicorn.conf.py src.app:app

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5001')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', 2))
worker_class = 'sync'
worker_connections = 1000
timeout = 120  # Increased timeout for ML model inference
keepalive = 5

# Restart workers after this many requests (prevent memory leaks)
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'sundanese-translator-backend'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# Pre-load the app (shares model memory across workers)
preload_app = True

def on_starting(server):
    """Called just before the master process is initialized."""
    print("=" * 80)
    print("Sundanese Translator Backend - Production Server")
    print("=" * 80)
    print(f"Workers: {workers}")
    print(f"Bind: {bind}")
    print(f"Timeout: {timeout}s")
    print("=" * 80)

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    print(f"Worker {worker.pid} was interrupted")

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    print(f"Worker {worker.pid} aborted")
