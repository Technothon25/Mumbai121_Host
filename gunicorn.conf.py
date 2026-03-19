# ============================================================
#  Gunicorn configuration for Mumbai121
#  Run with: gunicorn -c gunicorn.conf.py main:app
# ============================================================

import multiprocessing

# ── WORKER CONFIG ─────────────────────────────────────────────
# Use Uvicorn workers so FastAPI's ASGI benefits are preserved
worker_class = "uvicorn.workers.UvicornWorker"

# 4 workers is right for a VPS with 2–4 CPU cores.
# Formula: (2 × CPU cores) + 1 — cap at 4 for your current scale.
workers = 4

# Each worker handles up to 2 concurrent connections (sync PyMongo
# means true parallelism comes from multiple workers, not threads)
threads = 2

# ── NETWORK ───────────────────────────────────────────────────
# Bind to localhost only — Nginx will proxy to this port
bind = "127.0.0.1:8000"

# ── TIMEOUTS ──────────────────────────────────────────────────
# 120s for email sending + PDF upload + ML ranking which can be slow
timeout          = 120
keepalive        = 5
graceful_timeout = 30

# ── LOGGING ───────────────────────────────────────────────────
loglevel      = "info"
accesslog = "logs/access.log"
errorlog  = "logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs'

# ── PROCESS NAMING ────────────────────────────────────────────
proc_name = "mumbai121"

# ── WORKER LIFECYCLE ──────────────────────────────────────────
# Restart workers after this many requests to prevent memory leaks
# from ML model inference accumulating over time
max_requests          = 500
max_requests_jitter   = 50   # randomise so all workers don't restart at once

# ── SECURITY ──────────────────────────────────────────────────
# Limit request line and headers to protect against oversized requests
limit_request_line        = 8190
limit_request_fields      = 100
limit_request_field_size  = 8190