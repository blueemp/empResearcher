"""Reporter Agent for report generation."""

from datetime import datetime
from typing import Any

from ..services import LLMRouter


class ReporterAgent:
    """Agent for generating final research reports."""

    def __init__(self, llm_router: LLMRouter):
        """Initialize reporter agent.

        Args:
            llm_router: LLM routing service
        """
        self.llm_router = llm_router

    async def generate_report(
        self,
        task_id: str,
        query: str,
        sections: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        output_format: str = "markdown",
    ) -> dict[str, Any]:
        """Generate final research report.

        Args:
            task_id: Task identifier
            query: Original research query
            sections: List of report sections
            sources: List of information sources
            output_format: Output format (markdown, html, pdf, json)

        Returns:
            Complete report with content and metadata
        """
        synthesized_sections = []

        for i, section in enumerate(sections):
            messages = [
                {
                    "role": "system",
                    "content": "You are a technical report writer. Write comprehensive research sections.",
                },
                {
                    "role": "user",
                    "content": f"""Section: {section['title']}
Description: {section.get('description', '')}

All sources ({len(sources)} total):
{chr(10).join([f'{i+1}. [{s.get("source_type", "unknown")}] {s.get("title", "")[:50]}...' for i, s in enumerate(sources)])}

Write a detailed section (400-600 words).""",
                },
            ]

            response = await self.llm_router.route_chat(
                messages=messages,
                task_type="final_report_generation",
            )

            synthesized_sections.append(
                {
                    "title": section["title"],
                    "content": response.choices[0].message.content,
                    "order": section.get("order", i),
                }
            )

        summary = await self._generate_summary(query, synthesized_sections, sources)

        formatted_sources = self._format_sources(sources)

        report_content = await self._format_report(
            query, summary, synthesized_sections, formatted_sources, output_format
        )

        return {
            "task_id": task_id,
            "title": f"Research Report: {query}",
            "query": query,
            "format": output_format,
            "content": report_content,
            "metadata": {
                "sections_count": len(sections),
                "sources_count": len(sources),
                "generated_at": datetime.now().isoformat(),
                "languages": list(set(s.get("language", "unknown") for s in sources)),
            },
        }

    async def _generate_summary(
        self,
        query: str,
        sections: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> str:
        """Generate report summary.

        Args:
            query: Research query
            sections: Synthesized sections
            sources: Information sources

        Returns:
            Summary text
        """
        messages = [
            {
                "role": "system",
                "content": "You are an executive summary writer. Create concise, actionable summaries.",
            },
            {
                "role": "user",
                "content": f"""Research query: {query}

Sections ({len(sections)}):
{chr(10).join([f'- {s.get("title", "")}' for s in sections])}

Create a 200-300 word executive summary with:
- Research objective
- Key findings (2-3 bullets)
- Main conclusion
- Confidence level""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="document_summarization",
        )

        return response.choices[0].message.content

    def _format_sources(
        self,
        sources: list[dict[str, Any]],
    ) -> str:
        """Format sources for report.

        Args:
            sources: List of sources

        Returns:
            Formatted sources string
        """
        lines = ["## References\n"]
        for i, source in enumerate(sources, 1):
            lines.append(
                f"{i}. **{source.get('title', 'Untitled')}**\n"
                f"   - URL: {source.get('url', '')}\n"
                f"   - Source: {source.get('engine', source.get('source_type', 'unknown'))}\n"
                f"   - Language: {source.get('language', 'unknown')}\n"
                f"   - Relevance: {source.get('relevance_score', 0):.2f}\n"
            )

        return "\n".join(lines)

    async def _format_report(
        self,
        query: str,
        summary: str,
        sections: list[dict[str, Any]],
        sources: str,
        output_format: str,
    ) -> str:
        """Format complete report.

        Args:
            query: Research query
            summary: Executive summary
            sections: Synthesized sections
            sources: Formatted sources
            output_format: Output format

        Returns:
            Formatted report content
        """
        sorted_sections = sorted(sections, key=lambda x: x.get("order", 0))

        if output_format == "markdown":
            return f"# {query}\n\n{summary}\n\n{''.join([f'## {s[\"title\"]}\n\n{s[\"content\"]}\n\n' for s in sorted_sections])}{sources}"

        messages = [
            {
                "role": "system",
                "content": "You are a document formatter. Convert structured content to the specified format.",
            },
            {
                "role": "user",
                "content": f"""Convert this markdown report to {output_format}:
# {query}
{summary}
{''.join([f'## {s[\"title\"]}\\n{s[\"content\"]}\\n' for s in sorted_sections])}
{sources}""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="document_summarization",
        )

        return response.choices[0].message.content
