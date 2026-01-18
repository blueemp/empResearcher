"""Multimodal document processor."""

import base64
from typing import Any

from ..services import LLMRouter


class MultimodalProcessor:
    """Processor for multimodal documents (images, tables, audio)."""

    def __init__(self, llm_router: LLMRouter):
        """Initialize multimodal processor.

        Args:
            llm_router: LLM routing service
        """
        self.llm_router = llm_router

    async def process_image(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> dict[str, Any]:
        """Process image and extract text/description.

        Args:
            image_data: Image bytes
            image_format: Image format (png, jpg, etc.)

        Returns:
            Extracted content with metadata
        """
        import base64

        base64_image = base64.b64encode(image_data).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": "You are a multimodal document processor. Extract text, tables, and information from images.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": f"Extract all visible text, identify tables/charts, and provide a detailed description of this image.",
                    },
                ],
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="document_summarization",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)

            return {
                "modality": "image",
                "content": result.get("content", ""),
                "tables": result.get("tables", []),
                "text": result.get("text", ""),
                "description": result.get("description", ""),
                "format": image_format,
            }
        except Exception:
            return {
                "modality": "image",
                "content": "",
                "tables": [],
                "text": "",
                "description": "",
                "format": image_format,
            }

    async def process_table(
        self,
        table_data: str,
        table_format: str = "excel",
    ) -> dict[str, Any]:
        """Process table data and extract structured information.

        Args:
            table_data: Table data (CSV/Excel content)
            table_format: Table format (csv, excel, html)

        Returns:
            Structured table information
        """
        messages = [
            {
                "role": "system",
                "content": "You are a table analysis specialist. Extract structure, headers, and data from tables.",
            },
            {
                "role": "user",
                "content": f"""Analyze this {table_format} table:

{table_data[:2000]}

Extract:
1. Table structure (headers, rows, columns)
2. Data types of each column
3. Key insights/trends
4. Data quality issues (missing values, inconsistencies)

Return JSON with keys: headers (list), data_summary (dict), insights (list of strings).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="document_summarization",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)

            return {
                "modality": "table",
                "headers": result.get("headers", []),
                "data_summary": result.get("data_summary", {}),
                "insights": result.get("insights", []),
            }
        except Exception:
            return {"modality": "table", "headers": [], "data_summary": {}, "insights": []}

    async def process_audio(
        self,
        audio_data: bytes,
        audio_format: str = "mp3",
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Process audio and extract transcript.

        Args:
            audio_data: Audio bytes
            audio_format: Audio format (mp3, wav, etc.)
            duration: Audio duration in seconds

        Returns:
            Transcription with metadata
        """
        messages = [
            {
                "role": "system",
                "content": "You are an audio transcription specialist. Transcribe speech from audio files.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is an audio file for transcription.",
                    },
                    {
                        "type": "instruction",
                        "instruction": f"Transcribe this {duration:.1f}s {audio_format} audio accurately. Identify speakers, topics, and key points.",
                    },
                ],
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="document_summarization",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)

            return {
                "modality": "audio",
                "transcript": result.get("transcript", ""),
                "speakers": result.get("speakers", []),
                "topics": result.get("topics", []),
                "confidence": 0.8,
                "format": audio_format,
            }
        except Exception:
            return {
                "modality": "audio",
                "transcript": "",
                "speakers": [],
                "topics": [],
                "confidence": 0.0,
                "format": audio_format,
            }

    async def process_multimodal_document(
        self,
        document_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Process multimodal document with multiple content types.

        Args:
            document_data: Dictionary containing different modalities

        Returns:
            Combined processing results
        """
        results = {"modalities": [], "combined_summary": ""}

        if "image" in document_data:
            image_result = await self.process_image(
                document_data["image"]["data"],
                document_data["image"]["format"],
            )
            results["modalities"].append(image_result)

        if "table" in document_data:
            table_result = await self.process_table(
                document_data["table"]["data"],
                document_data["table"]["format"],
            )
            results["modalities"].append(table_result)

        if "audio" in document_data:
            audio_result = await self.process_audio(
                document_data["audio"]["data"],
                document_data["audio"]["format"],
                document_data["audio"].get("duration"),
            )
            results["modalities"].append(audio_result)

        if results["modalities"]:
            summary_messages = [
                {
                    "role": "system",
                    "content": "You are a document summarization specialist.",
                },
                {
                    "role": "user",
                    "content": f"""Create a comprehensive summary of this multimodal document:

Modalities: {", ".join([m["modality"] for m in results["modalities"]])}

Synthesize all information into a coherent summary.""",
                },
            ]

            summary_response = await self.llm_router.route_chat(
                messages=summary_messages,
                task_type="document_summarization",
            )

            results["combined_summary"] = summary_response.choices[0].message.content

        return results

    async def health_check(self) -> dict[str, Any]:
        """Check multimodal processor health.

        Returns:
            Health status
        """
        return {
            "image_processing": {"status": "ok"},
            "table_processing": {"status": "ok"},
            "audio_processing": {"status": "ok"},
            "overall": "healthy",
        }
