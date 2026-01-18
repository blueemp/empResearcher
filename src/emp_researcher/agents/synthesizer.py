"""Synthesizer Agent for information aggregation."""

from typing import Any

from ..services import LLMRouter


class SynthesizerAgent:
    """Agent for aggregating and synthesizing information."""

    def __init__(self, llm_router: LLMRouter):
        """Initialize synthesizer agent.

        Args:
            llm_router: LLM routing service
        """
        self.llm_router = llm_router

    async def synthesize_findings(
        self,
        findings: list[dict[str, Any]],
        query: str,
    ) -> dict[str, Any]:
        """Synthesize findings from multiple sources.

        Args:
            findings: List of research findings
            query: Original research query

        Returns:
            Synthesized knowledge structure
        """
        if not findings:
            return {"status": "empty", "message": "No findings to synthesize"}

        themes = await self._identify_themes(findings, query)

        clustered = await self._cluster_findings(findings, themes)

        synthesized = {
            "query": query,
            "themes": themes,
            "clusters": clustered,
            "total_findings": len(findings),
            "confidence": self._calculate_confidence(clustered),
        }

        return synthesized

    async def _identify_themes(
        self,
        findings: list[dict[str, Any]],
        query: str,
    ) -> list[str]:
        """Identify key themes from findings.

        Args:
            findings: List of research findings
            query: Research query

        Returns:
            List of theme names
        """
        messages = [
            {
                "role": "system",
                "content": "You are a research analyst. Identify 3-5 key themes from research findings.",
            },
            {
                "role": "user",
                "content": f"""Research query: {query}

Findings ({len(findings)}):
{chr(10).join([f"{i + 1}. {f.get('content', '')[:100]}..." for i, f in enumerate(findings)])}

Identify 3-5 key themes that emerge from these findings.
Return JSON array of theme names.""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="coordinator_planning",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return result if isinstance(result, list) else ["General Research"]
        except Exception:
            return ["General Research"]

    async def _cluster_findings(
        self,
        findings: list[dict[str, Any]],
        themes: list[str],
    ) -> list[dict[str, Any]]:
        """Cluster findings by themes.

        Args:
            findings: List of research findings
            themes: List of identified themes

        Returns:
            List of theme clusters
        """
        messages = [
            {
                "role": "system",
                "content": "You are a data analyst. Cluster findings into themes.",
            },
            {
                "role": "user",
                "content": f"""Themes: {", ".join(themes)}

Findings to cluster ({len(findings)}):
{chr(10).join([f"{i + 1}. {f.get('content', '')[:150]}..." for i, f in enumerate(findings)])}

For each finding, assign to most relevant theme.
Return JSON array of objects with keys: theme, findings (list of indices).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="coordinator_planning",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return result if isinstance(result, list) else []
        except Exception:
            return []

    def _calculate_confidence(self, clusters: list[dict[str, Any]]) -> float:
        """Calculate overall confidence score.

        Args:
            clusters: List of theme clusters

        Returns:
            Confidence score 0-1
        """
        if not clusters:
            return 0.0

        total_findings = sum(len(c.get("findings", [])) for c in clusters)
        avg_findings_per_theme = total_findings / len(clusters)

        confidence = min(avg_findings_per_theme / 5, 1.0)
        return round(confidence, 2)

    async def create_timeline(
        self,
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create chronological timeline from findings.

        Args:
            findings: List of research findings

        Returns:
            Timeline of events/milestones
        """
        timeline = []

        for finding in findings:
            if finding.get("date"):
                timeline.append(
                    {
                        "date": finding["date"],
                        "event": finding.get("content", "")[:200],
                        "source": finding.get("source", "unknown"),
                    }
                )

        return sorted(timeline, key=lambda x: x["date"])

    async def create_comparison_table(
        self,
        findings: list[dict[str, Any]],
        dimensions: list[str],
    ) -> dict[str, Any]:
        """Create comparison table for findings.

        Args:
            findings: List of research findings
            dimensions: List of comparison dimensions

        Returns:
            Comparison table structure
        """
        messages = [
            {
                "role": "system",
                "content": "You are a comparative analyst. Create comparison tables.",
            },
            {
                "role": "user",
                "content": f"""Dimensions: {", ".join(dimensions)}

Findings to compare ({len(findings)}):
{chr(10).join([f"{i + 1}. {f.get('content', '')[:150]}..." for i, f in enumerate(findings)])}

Create a comparison table in markdown format with:
- Rows for each finding
- Columns for each dimension
- Brief comparison notes

Return markdown table.""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="document_summarization",
        )

        return {
            "dimensions": dimensions,
            "content": response.choices[0].message.content,
            "findings_count": len(findings),
        }
