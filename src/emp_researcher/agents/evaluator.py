"""Evaluator Agent for quality assessment and reflection."""

import asyncio
from typing import Any

from ..services import LLMRouter


class EvaluatorAgent:
    """Agent for evaluating research quality and triggering reflection."""

    def __init__(self, llm_router: LLMRouter):
        """Initialize evaluator agent.

        Args:
            llm_router: LLM routing service
        """
        self.llm_router = llm_router

    async def evaluate_sources(
        self,
        sources: list[dict[str, Any]],
        query: str,
    ) -> dict[str, Any]:
        """Evaluate source quality and coverage.

        Args:
            sources: List of information sources
            query: Original research query

        Returns:
            Evaluation results
        """
        if not sources:
            return {
                "status": "insufficient",
                "message": "No sources found",
                "score": 0.0,
            }

        metrics = {
            "total_sources": len(sources),
            "avg_relevance": sum(s.get("relevance_score", 0) for s in sources) / len(sources),
            "avg_trust": sum(s.get("trust_score", 0) for s in sources) / len(sources),
            "languages": list(set(s.get("language", "unknown") for s in sources)),
            "source_types": list(set(s.get("source_type", "unknown") for s in sources)),
        }

        coverage_score = min(len(sources) / 10, 1.0)

        overall_score = (
            metrics["avg_relevance"] * 0.4 + metrics["avg_trust"] * 0.3 + coverage_score * 0.3
        )

        return {
            "status": "evaluated",
            "metrics": metrics,
            "overall_score": overall_score,
            "is_sufficient": overall_score >= 0.6,
        }

    async def check_consistency(
        self,
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check consistency across multiple sources.

        Args:
            sources: List of information sources

        Returns:
            Consistency analysis
        """
        messages = [
            {
                "role": "system",
                "content": "You are a fact-checking specialist. Identify inconsistencies and conflicts in information from multiple sources.",
            },
            {
                "role": "user",
                "content": f"""Check consistency across these {len(sources)} sources:

{chr(10).join([f"{i + 1}. {s.get('title', '')[:100]}..." for i, s in enumerate(sources)])}

Identify:
1. Contradictions: Direct conflicts between sources
2. Gaps: Important information missing
3. Divergent claims: Different versions of the same fact

Return JSON with keys: contradictions (list), gaps (list), divergent_claims (list), consistency_score (float 0-1).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="coordinator_planning",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return {
                "status": "checked",
                "contradictions": result.get("contradictions", []),
                "gaps": result.get("gaps", []),
                "divergent_claims": result.get("divergent_claims", []),
                "consistency_score": result.get("consistency_score", 0.8),
            }
        except Exception:
            return {
                "status": "checked",
                "contradictions": [],
                "gaps": [],
                "divergent_claims": [],
                "consistency_score": 0.8,
            }

    async def should_expand_search(
        self,
        evaluation: dict[str, Any],
        max_iterations: int = 10,
    ) -> tuple[bool, str | None]:
        """Determine if search should be expanded.

        Args:
            evaluation: Evaluation results
            max_iterations: Maximum search iterations

        Returns:
            Tuple of (should_expand, reason)
        """
        if not evaluation.get("is_sufficient", False):
            return True, "Source quality or coverage below threshold"

        score = evaluation.get("overall_score", 0)
        if score < 0.7:
            return True, f"Overall score {score:.2f} below threshold 0.7"

        return False, None

    async def generate_reflection(
        self,
        current_state: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate reflection and next actions.

        Args:
            current_state: Current workflow state
            evaluation: Evaluation results

        Returns:
            Reflection with recommendations
        """
        messages = [
            {
                "role": "system",
                "content": "You are a research coordinator. Reflect on current progress and recommend next actions.",
            },
            {
                "role": "user",
                "content": f"""Current progress:
- Query: {current_state.get("query", "")}
- Sources found: {evaluation.get("metrics", {}).get("total_sources", 0)}
- Overall score: {evaluation.get("overall_score", 0):.2f}
- Status: {evaluation.get("status", "")}

Recommend next steps:
1. Should we do more searches? (new queries, new engines)
2. Should we refine the research focus? (narrow down, pivot)
3. Should we proceed to synthesis? (sufficient information)

Return JSON with keys: action (search/refine/proceed), next_queries (list), reason (string).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="coordinator_planning",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return {
                "action": result.get("action", "proceed"),
                "next_queries": result.get("next_queries", []),
                "reason": result.get("reason", "Information sufficient"),
            }
        except Exception:
            return {
                "action": "proceed",
                "next_queries": [],
                "reason": "Information appears sufficient",
            }
