# ============================================================
#  Gunicorn configuration for Mumbai121
#  Run with: gunicorn -c gunicorn.conf.py main:app
# ============================================================

import os

# ── WORKER CONFIG ─────────────────────────────────────────────
worker_class = "uvicorn.workers.UvicornWorker"

# Using 1 worker to prevent duplicate change stream processing
# and duplicate email sending across multiple workers
workers = 1
threads = 2

# ── NETWORK ───────────────────────────────────────────────────
port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

# ── TIMEOUTS ──────────────────────────────────────────────────
timeout          = 120
keepalive        = 5
graceful_timeout = 30

# ── LOGGING ───────────────────────────────────────────────────
loglevel  = "info"
accesslog = "-"
errorlog  = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── PROCESS NAMING ────────────────────────────────────────────
proc_name = "mumbai121"

# ── WORKER LIFECYCLE ──────────────────────────────────────────
max_requests        = 500
max_requests_jitter = 50