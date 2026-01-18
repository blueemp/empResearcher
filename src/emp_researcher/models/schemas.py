"""Data models for emp-researcher."""

from pydantic import BaseModel, Field
from typing import Any
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REFINE = "refine"


class TaskDepth(str, Enum):
    """Research depth level."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class OutputFormat(str, Enum):
    """Output format types."""

    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class ResearchTask(BaseModel):
    """Research task request model."""

    query: str = Field(..., description="Research query or question")
    depth: TaskDepth = Field(default=TaskDepth.STANDARD, description="Research depth")
    output_format: OutputFormat = Field(default=OutputFormat.MARKDOWN, description="Output format")
    language: str | None = Field(default=None, description="Language preference (zh/en)")
    max_sources: int = Field(default=50, description="Maximum number of sources")
    timeout: int = Field(default=3600, description="Timeout in seconds")


class TaskResponse(BaseModel):
    """Task creation response."""

    task_id: str
    status: TaskStatus
    query: str
    created_at: str


class TaskStatusResponse(BaseModel):
    """Task status response."""

    task_id: str
    status: TaskStatus
    progress: float = Field(default=0.0, description="Progress 0-100")
    current_step: str | None = None
    total_steps: int = 0
    completed_steps: int = 0
    sources_found: int = 0
    error_message: str | None = None
    created_at: str
    updated_at: str


class Source(BaseModel):
    """Information source."""

    id: str
    title: str
    url: str
    content: str
    language: str
    source_type: str
    relevance_score: float
    trust_score: float
    date: str | None = None


class ReportSection(BaseModel):
    """Report section."""

    title: str
    content: str
    sources: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Final research report."""

    task_id: str
    title: str
    summary: str
    sections: list[ReportSection]
    sources: list[Source]
    created_at: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    message: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: dict[str, dict[str, Any]]
