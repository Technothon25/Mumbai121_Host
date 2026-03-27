# ============================================================
#  Gunicorn configuration for Mumbai121
#  Run with: gunicorn -c gunicorn.conf.py main:app
# ============================================================

import os
import multiprocessing

# ── WORKER CONFIG ─────────────────────────────────────────────
worker_class = "uvicorn.workers.UvicornWorker"
workers = 4
threads = 2

# ── NETWORK ───────────────────────────────────────────────────
# Railway injects PORT env variable — must bind to 0.0.0.0
port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

# ── TIMEOUTS ──────────────────────────────────────────────────
timeout          = 120
keepalive        = 5
graceful_timeout = 30

# ── LOGGING — stdout/stderr for cloud hosting ─────────────────
# Railway captures stdout/stderr automatically — no log files needed
loglevel  = "info"
accesslog = "-"   # stdout
errorlog  = "-"   # stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── PROCESS NAMING ────────────────────────────────────────────
proc_name = "mumbai121"

# ── WORKER LIFECYCLE ──────────────────────────────────────────
max_requests        = 500
max_requests_jitter = 50