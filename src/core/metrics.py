"""Prometheus metrics for request tracking and ML inference monitoring."""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# HTTP request metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    name="http_requests_total",
    documentation="Total number of HTTP requests received.",
    labelnames=["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    name="http_request_duration_seconds",
    documentation="HTTP request latency in seconds.",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ACTIVE_REQUESTS = Gauge(
    name="http_active_requests",
    documentation="Number of HTTP requests currently being processed.",
)

# ---------------------------------------------------------------------------
# ML inference metrics
# ---------------------------------------------------------------------------

INFERENCE_TIME = Histogram(
    name="ml_inference_duration_seconds",
    documentation="Time spent on model inference in seconds.",
    labelnames=["model_name"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

INFERENCE_COUNT = Counter(
    name="ml_inference_total",
    documentation="Total number of inference requests processed.",
    labelnames=["model_name", "status"],
)

MODEL_LOAD_TIME = Gauge(
    name="ml_model_load_duration_seconds",
    documentation="Time taken to load the model in seconds (last load).",
    labelnames=["model_name"],
)
