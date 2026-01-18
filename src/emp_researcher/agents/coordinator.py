"""Coordinator Agent for orchestrating research tasks."""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from ..models import TaskStatus
from ..services import LLMRouter


class Step:
    """Research workflow step."""

    def __init__(
        self,
        name: str,
        description: str,
        agent: str,
        status: str = "pending",
    ):
        """Initialize workflow step.

        Args:
            name: Step name
            description: Step description
            agent: Responsible agent
            status: Step status (pending, running, completed, failed)
        """
        self.name = name
        self.description = description
        self.agent = agent
        self.status = status
        self.result: Any = None
        self.error: str | None = None
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

    def start(self) -> None:
        """Mark step as started."""
        self.status = "running"
        self.started_at = datetime.now()

    def complete(self, result: Any) -> None:
        """Mark step as completed.

        Args:
            result: Step result
        """
        self.status = "completed"
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark step as failed.

        Args:
            error: Error message
        """
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.now()


class TodoItem:
    """Research todo item."""

    def __init__(
        self,
        description: str,
        priority: int = 0,
    ):
        """Initialize todo item.

        Args:
            description: Todo description
            priority: Priority (higher = more important)
        """
        self.id = str(uuid.uuid4())
        self.description = description
        self.priority = priority
        self.status = "pending"
        self.created_at = datetime.now()

    def mark_in_progress(self) -> None:
        """Mark todo as in progress."""
        self.status = "in_progress"

    def complete(self) -> None:
        """Mark todo as completed."""
        self.status = "completed"

    def fail(self, error: str) -> None:
        """Mark todo as failed.

        Args:
            error: Error message
        """
        self.status = "failed"
        self.error = error


class CoordinatorAgent:
    """Coordinates multi-agent research workflow."""

    def __init__(self, llm_router: LLMRouter):
        """Initialize coordinator agent.

        Args:
            llm_router: LLM routing service
        """
        self.llm_router = llm_router
        self.task_id: str | None = None
        self.query: str | None = None
        self.todolist: list[TodoItem] = []
        self.steps: list[Step] = []
        self.status = TaskStatus.PENDING

    async def create_task(
        self,
        query: str,
        depth: str = "standard",
    ) -> str:
        """Create a new research task.

        Args:
            query: Research question
            depth: Research depth (quick, standard, deep)

        Returns:
            Task ID
        """
        self.task_id = str(uuid.uuid4())
        self.query = query
        self.status = TaskStatus.PENDING

        await self._generate_todolist(query, depth)

        return self.task_id

    async def _generate_todolist(
        self,
        query: str,
        depth: str,
    ) -> None:
        """Generate initial todolist using LLM.

        Args:
            query: Research query
            depth: Research depth level
        """
        messages = [
            {
                "role": "system",
                "content": "You are a research planner. Break down research queries into a list of specific todo items.",
            },
            {
                "role": "user",
                "content": f"Create a todolist for researching: {query}\n\nDepth: {depth}\n\nReturn a JSON list of todo items with 'description' and 'priority' (0-10).",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="coordinator_planning",
        )

        try:
            import json

            todos = json.loads(response.choices[0].message.content)
            self.todolist = [
                TodoItem(
                    description=item.get("description", ""),
                    priority=item.get("priority", 5),
                )
                for item in todos
            ]
        except Exception as e:
            self.todolist = [
                TodoItem(
                    description=f"Research: {query}",
                    priority=10,
                )
            ]

    async def execute_workflow(self) -> dict[str, Any]:
        """Execute research workflow.

        Returns:
            Workflow execution results
        """
        self.status = TaskStatus.RUNNING

        results = {
            "task_id": self.task_id,
            "query": self.query,
            "steps_completed": 0,
            "total_steps": len(self.todolist),
            "sources_found": 0,
            "progress": 0.0,
        }

        for i, todo in enumerate(self.todolist):
            todo.mark_in_progress()

            step = Step(
                name=f"Step {i + 1}",
                description=todo.description,
                agent="coordinator",
            )
            step.start()

            try:
                result = await self._execute_todo(todo)
                step.complete(result)
                todo.complete()
            except Exception as e:
                step.fail(str(e))
                todo.fail(str(e))

            self.steps.append(step)
            results["steps_completed"] = i + 1
            results["progress"] = (i + 1) / len(self.todolist) * 100

        if all(todo.status == "completed" for todo in self.todolist):
            self.status = TaskStatus.COMPLETED
        else:
            self.status = TaskStatus.FAILED

        return results

    async def _execute_todo(self, todo: TodoItem) -> Any:
        """Execute a single todo item.

        Args:
            todo: Todo item to execute

        Returns:
            Execution result
        """
        messages = [
            {
                "role": "system",
                "content": "You are a research assistant. Answer questions comprehensively.",
            },
            {
                "role": "user",
                "content": f"Research task: {todo.description}\n\nProvide a comprehensive answer with sources.",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="coordinator_planning",
        )

        return {
            "description": todo.description,
            "answer": response.choices[0].message.content,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current task status.

        Returns:
            Status dictionary
        """
        completed = sum(1 for todo in self.todolist if todo.status == "completed")
        total = len(self.todolist)

        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": (completed / total * 100) if total > 0 else 0,
            "current_step": self.todolist[completed].description if completed < total else None,
            "total_steps": total,
            "completed_steps": completed,
        }

    async def generate_report(self) -> dict[str, Any]:
        """Generate final research report.

        Returns:
            Report dictionary
        """
        messages = [
            {
                "role": "system",
                "content": "You are a report writer. Synthesize research into a structured report.",
            },
            {
                "role": "user",
                "content": f"Based on the following research results, generate a comprehensive report:\n\n{self._format_results()}\n\nInclude sections: Summary, Key Findings, Sources.",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="final_report_generation",
        )

        return {
            "task_id": self.task_id,
            "title": f"Research Report: {self.query}",
            "content": response.choices[0].message.content,
            "status": self.status,
        }

    def _format_results(self) -> str:
        """Format execution results for LLM.

        Returns:
            Formatted results string
        """
        lines = []
        for step in self.steps:
            lines.append(f"- {step.description}")
            if step.result:
                lines.append(f"  Result: {step.result.get('answer', '')[:200]}...")

        return "\n".join(lines)
