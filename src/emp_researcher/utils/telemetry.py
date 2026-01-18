"""Logging and telemetry utilities."""

import logging
import logging.handlers
import os
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_logging(config: dict[str, Any]) -> None:
    """Setup structured logging.

    Args:
        config: Logging configuration dictionary
    """
    level = config.get("level", "INFO")
    log_path = config.get("file_path", "logs/app.log")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=config.get("max_bytes", 10485760),
                backupCount=config.get("backup_count", 5),
            ),
        ],
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def setup_telemetry(config: dict[str, Any]) -> trace.Tracer:
    """Setup OpenTelemetry tracing.

    Args:
        config: Observability configuration

    Returns:
        Tracer instance
    """
    otel_config = config.get("otel", {})
    enabled = otel_config.get("enabled", False)

    if not enabled:
        return trace.get_tracer(__name__)

    exporter = OTLPSpanExporter(
        endpoint=otel_config.get("endpoint", "http://localhost:4318"),
        insecure=True,
    )

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            schedule_delay_millis=1000,
            max_export_batch_size=32,
        ),
    )

    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(
        otel_config.get("service_name", "emp-researcher"),
    )

    return tracer


def instrument_fastapi(app: Any) -> None:
    """Instrument FastAPI application for telemetry.

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)
