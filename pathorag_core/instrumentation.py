# instrumentation.py
"""
OpenTelemetry tracing + Prometheus metrics for PathoRAG.

All dependencies are optional. When not installed, every call degrades to a
safe no-op so the core application runs unchanged.

Quick start:
    from pathorag_core.instrumentation import init_instrumentation

    init_instrumentation(
        service_name="pathorag",
        console_export=True,        # print spans to stdout (dev)
        otlp_endpoint=None,         # or "http://localhost:4317" (prod)
        prometheus_port=None,       # or 9090 to expose /metrics
    )
"""

from __future__ import annotations

import os
import time
import logging
from functools import wraps
from typing import Optional

logger = logging.getLogger("pathorag_core.instrumentation")

# ---------------------------------------------------------------------------
# Global state — lazy-initialised
# ---------------------------------------------------------------------------
_initialised: bool = False
_tracer = None
_registry = None

# Populated by _create_metrics() once prometheus_client is available
METRICS: dict = {}


# ---------------------------------------------------------------------------
# No-op fallbacks (used when OTel deps are missing)
# ---------------------------------------------------------------------------
class _NoOpSpan:
    """A span that does nothing — safe fallback."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    def set_attribute(self, *a, **kw):
        pass

    def set_status(self, *a, **kw):
        pass

    def record_exception(self, *a, **kw):
        pass

    def add_event(self, *a, **kw):
        pass


class _NoOpTracer:
    """A tracer that always returns no-op spans."""

    def start_as_current_span(self, *a, **kw):
        return _NoOpSpan()

    def start_span(self, *a, **kw):
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Public: get a tracer (safe)
# ---------------------------------------------------------------------------
def get_tracer(name: str = "pathorag_core"):
    """Return an OTel tracer, or a no-op fallback."""
    global _tracer, _initialised
    if _initialised and _tracer is not None:
        from opentelemetry import trace

        return trace.get_tracer(name)
    return _NoOpTracer()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_instrumentation(
    service_name: str = "pathorag",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = True,
    prometheus_port: Optional[int] = None,
) -> None:
    """Initialise tracing (OpenTelemetry) and metrics (Prometheus).

    Args:
        service_name: Logical service name attached to every span.
        otlp_endpoint: gRPC collector endpoint (overrides env var).
        console_export: Print spans to stdout (great for dev).
        prometheus_port: If set, start a Prometheus HTTP endpoint on this port.
    """
    global _tracer, _registry, _initialised

    otlp = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    # ---- Tracing (OpenTelemetry) ----
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if console_export:
            provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
        if otlp:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp))
            )

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        logger.info(
            "Tracing OK (console=%s, otlp=%s)",
            console_export,
            "yes" if otlp else "no",
        )
    except ImportError:
        logger.warning(
            "opentelemetry SDK not installed — tracing disabled. "
            "Fix: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-grpc"
        )
    except Exception as e:
        logger.error("Tracing init failed: %s", e)

    # ---- Metrics (Prometheus) ----
    try:
        from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

        _registry = CollectorRegistry()

        METRICS.update(
            {
                # --- Counters ---
                "aquery_total": Counter(
                    "pathorag_aquery_total", "Total queries processed", registry=_registry
                ),
                "aquery_errors_total": Counter(
                    "pathorag_aquery_errors_total",
                    "Total failed queries",
                    registry=_registry,
                ),
                "llm_calls_total": Counter(
                    "pathorag_llm_calls_total",
                    "LLM API calls",
                    ["provider"],
                    registry=_registry,
                ),
                "llm_errors_total": Counter(
                    "pathorag_llm_errors_total",
                    "LLM API errors",
                    ["provider"],
                    registry=_registry,
                ),
                "embedding_calls_total": Counter(
                    "pathorag_embedding_calls_total",
                    "Embedding API calls",
                    ["provider"],
                    registry=_registry,
                ),
                "embedding_errors_total": Counter(
                    "pathorag_embedding_errors_total",
                    "Embedding API errors",
                    ["provider"],
                    registry=_registry,
                ),
                "neo4j_operations_total": Counter(
                    "pathorag_neo4j_operations_total",
                    "Neo4j operations",
                    ["operation"],
                    registry=_registry,
                ),
                "neo4j_errors_total": Counter(
                    "pathorag_neo4j_errors_total",
                    "Neo4j errors",
                    ["operation"],
                    registry=_registry,
                ),
                "cache_hits_total": Counter(
                    "pathorag_cache_hits_total",
                    "Cache hits",
                    ["mode"],
                    registry=_registry,
                ),
                "cache_misses_total": Counter(
                    "pathorag_cache_misses_total",
                    "Cache misses",
                    ["mode"],
                    registry=_registry,
                ),
                "retry_total": Counter(
                    "pathorag_retry_total",
                    "Retry attempts",
                    ["component"],
                    registry=_registry,
                ),
                "keyword_fallback_total": Counter(
                    "pathorag_keyword_fallback_total",
                    "LLM→keyword fallback events",
                    registry=_registry,
                ),
                # --- Histograms ---
                "aquery_duration_seconds": Histogram(
                    "pathorag_aquery_duration_seconds",
                    "Query latency in seconds",
                    registry=_registry,
                ),
                "llm_call_duration_seconds": Histogram(
                    "pathorag_llm_call_duration_seconds",
                    "LLM call latency",
                    ["provider"],
                    registry=_registry,
                ),
                "embedding_duration_seconds": Histogram(
                    "pathorag_embedding_duration_seconds",
                    "Embedding call latency",
                    ["provider"],
                    registry=_registry,
                ),
                "neo4j_operation_duration_seconds": Histogram(
                    "pathorag_neo4j_operation_duration_seconds",
                    "Neo4j operation latency",
                    ["operation"],
                    registry=_registry,
                ),
                # --- Gauges ---
                "active_queries": Gauge(
                    "pathorag_active_queries",
                    "In-flight query count",
                    registry=_registry,
                ),
            }
        )

        if prometheus_port:
            from prometheus_client import start_http_server

            start_http_server(prometheus_port, registry=_registry)
            logger.info("Prometheus /metrics on :%s", prometheus_port)

        logger.info("Metrics OK (%s instruments)", len(METRICS))
    except ImportError:
        logger.warning(
            "prometheus_client not installed — metrics disabled. "
            "Fix: pip install prometheus-client"
        )
    except Exception as e:
        logger.error("Metrics init failed: %s", e)

    _initialised = True


# ---------------------------------------------------------------------------
# Tracing helpers
# ---------------------------------------------------------------------------
def traced(name: str, attrs: Optional[dict] = None):
    """Decorator: wrap an async function in an OTel span.

    Usage:
        @traced("my_operation")
        async def my_func(): ...
    """

    def deco(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                if attrs:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper

    return deco


def trace_span(name: str):
    """Context manager for a manual span (works in both sync and async).

    Usage:
        with trace_span("entity_retrieval"):
            result = await do_something()
    """
    tracer = get_tracer()
    return tracer.start_as_current_span(name)


# ---------------------------------------------------------------------------
# Metrics helpers (safe — no-op when metrics not initialised)
# ---------------------------------------------------------------------------
def _m(name: str):
    """Get a metric instrument by name, or None."""
    return METRICS.get(name)


def inc_counter(name: str, amount: int = 1, labels: Optional[dict] = None):
    m = _m(name)
    if m is not None:
        if labels:
            m.labels(**labels).inc(amount)
        else:
            m.inc(amount)


def observe_histogram(name: str, value: float, labels: Optional[dict] = None):
    m = _m(name)
    if m is not None:
        if labels:
            m.labels(**labels).observe(value)
        else:
            m.observe(value)


def set_gauge(name: str, value: float):
    m = _m(name)
    if m is not None:
        m.set(value)


def inc_gauge(name: str, amount: int = 1):
    m = _m(name)
    if m is not None:
        m.inc(amount)


def dec_gauge(name: str, amount: int = 1):
    m = _m(name)
    if m is not None:
        m.dec(amount)
