"""Observability service for OpenTelemetry metrics and tracing."""

from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


class ObservabilityService:
    """Service for managing observability metrics and tracing."""

    def __init__(self, app_name: str = "emp-researcher"):
        """Initialize observability service.

        Args:
            app_name: Application name
        """
        self.app_name = app_name
        self.tracer = trace.get_tracer(app_name)

    def instrument_fastapi(self, app: Any) -> None:
        """Instrument FastAPI application.

        Args:
            app: FastAPI application
        """
        FastAPIInstrumentor.instrument_app(app)

    def get_tracer(self):
        """Get OpenTelemetry tracer.

        Returns:
            Tracer instance
        """
        return self.tracer

    def record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a custom metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Metric tags
        """
        from prometheus_client import Counter, Histogram

        metrics = {
            "counter": Counter,
            "histogram": Histogram,
        }

        metric_type = tags.get("type", "counter") if tags else "counter"

        if metric_type in metrics:
            if metric_type == "counter":
                metrics[metric_type].labels(**tags).inc(value)
            elif metric_type == "histogram":
                metrics[metric_type].labels(**tags).observe(value)

    def create_span(self, name: str, parent_span_id: str | None = None) -> trace.Span:
        """Create a new tracing span.

        Args:
            name: Span name
            parent_span_id: Parent span ID

        Returns:
            Span instance
        """
        return self.tracer.start_span(
            name=name,
            parent=parent_span_id,
        )

    def record_event(
        self, span: trace.Span, event_name: str, attributes: dict[str, Any] | None = None
    ) -> None:
        """Record an event on a span.

        Args:
            span: Span instance
            event_name: Event name
            attributes: Event attributes
        """
        span.add_event(name=event_name, attributes=attributes or {})

    def set_attributes(self, span: trace.Span, attributes: dict[str, Any]) -> None:
        """Set span attributes.

        Args:
            span: Span instance
            attributes: Attributes dictionary
        """
        span.set_attributes(attributes)

    def record_exception(
        self, span: trace.Span, exception: Exception, attributes: dict[str, Any] | None = None
    ) -> None:
        """Record exception on a span.

        Args:
            span: Span instance
            exception: Exception to record
            attributes: Additional attributes
        """
        span.record_exception(exception, attributes=attributes or {})

    async def with_span(
        self,
        name: str,
        parent_span_id: str | None = None,
    ):
        """Context manager for span creation.

        Args:
            name: Span name
            parent_span_id: Parent span ID

        Yields:
            Span instance
        """
        span = self.create_span(name, parent_span_id)

        try:
            yield span
        finally:
            span.end()
