"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.fastapi import metrics

from ..agents import CoordinatorAgent
from ..models import (
    ErrorResponse,
    HealthResponse,
    ResearchTask,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
)
from ..services import LLMRouter
from ..utils import ConfigManager, get_config, instrument_fastapi, setup_logging, setup_telemetry


metrics_counter = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
metrics_histogram = Histogram("api_request_duration_seconds", "API request duration")

_global_config: ConfigManager | None = None
_global_llm_router: LLMRouter | None = None
_global_coordinator: CoordinatorAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Args:
        app: FastAPI application instance
    """
    global _global_config, _global_llm_router

    config = get_config()
    _global_config = config

    app_config = config.config.get("app", {})

    setup_logging(config.config.get("logging", {}))
    setup_telemetry(config.config.get("observability", {}))
    instrument_fastapi(app)

    llm_config = config.get_llm_config()
    _global_llm_router = LLMRouter(llm_config)
    _global_coordinator = CoordinatorAgent(_global_llm_router)

    yield

    await _global_llm_router.health_check_all()


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    config = get_config()
    app_config = config.config.get("app", {})

    app = FastAPI(
        title=app_config.get("name", "emp-researcher"),
        version=app_config.get("version", "0.1.0"),
        description="Enterprise Deep Research Agent System",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    cors_config = app_config.get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=True,
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
    )

    metrics(app)

    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(tasks_router, prefix="/api/v1/tasks", tags=["Tasks"])

    return app


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return generate_latest()


from fastapi import APIRouter

health_router = APIRouter()
tasks_router = APIRouter()

_global_tasks: dict[str, CoordinatorAgent] = {}


@health_router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status of all components
    """
    global _global_llm_router

    if _global_llm_router:
        provider_health = await _global_llm_router.health_check_all()
    else:
        provider_health = {}

    return HealthResponse(
        status="healthy" if all(provider_health.values()) else "degraded",
        version="0.1.0",
        components={
            "llm_providers": {
                k: {"status": "ok" if v else "error"} for k, v in provider_health.items()
            }
        },
    )


@tasks_router.post("", response_model=TaskResponse, status_code=201)
async def create_task(task: ResearchTask) -> TaskResponse:
    """Create a new research task.

    Args:
        task: Research task details

    Returns:
        Task creation response with task ID
    """
    global _global_coordinator

    task_id = await _global_coordinator.create_task(
        task.query,
        task.depth,
    )

    metrics_counter.labels(method="POST", endpoint="/tasks", status="created").inc()

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        query=task.query,
        created_at="",
    )


@tasks_router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get task status.

    Args:
        task_id: Task identifier

    Returns:
        Current task status
    """
    global _global_coordinator

    status = _global_coordinator.get_status()

    metrics_counter.labels(method="GET", endpoint="/tasks/{id}/status", status="success").inc()

    return TaskStatusResponse(
        task_id=task_id,
        status=status["status"],
        progress=status["progress"],
        current_step=status.get("current_step"),
        total_steps=status["total_steps"],
        completed_steps=status["completed_steps"],
        sources_found=status.get("sources_found", 0),
        created_at="",
        updated_at="",
    )


@tasks_router.get("/{task_id}/report")
async def get_report(task_id: str):
    """Get research report.

    Args:
        task_id: Task identifier

    Returns:
        Research report
    """
    global _global_coordinator

    report = await _global_coordinator.generate_report()

    metrics_counter.labels(method="GET", endpoint="/tasks/{id}/report", status="success").inc()

    return report


app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    app_config = config.config.get("app", {})

    uvicorn.run(
        "emp_researcher.api.main:app",
        host=app_config.get("host", "0.0.0.0"),
        port=app_config.get("port", 8000),
        reload=app_config.get("env", "development") == "development",
    )
