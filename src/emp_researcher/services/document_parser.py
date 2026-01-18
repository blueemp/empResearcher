"""Document parser service."""

import asyncio
from pathlib import Path
from typing import Any

from pypdf import PdfReader


class DocumentChunk:
    """Represents a chunk of document text."""

    def __init__(
        self,
        content: str,
        metadata: dict[str, Any],
        chunk_id: str,
    ):
        """Initialize document chunk.

        Args:
            content: Chunk text content
            metadata: Chunk metadata (source, page, etc.)
            chunk_id: Unique chunk identifier
        """
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
        }


class DocumentParser:
    """Parser for multiple document formats."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """Initialize document parser.

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def parse_file(self, file_path: str | Path) -> list[DocumentChunk]:
        """Parse document and create chunks.

        Args:
            file_path: Path to document file

        Returns:
            List of document chunks
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        parsers = {
            ".pdf": self._parse_pdf,
            ".txt": self._parse_text,
            ".md": self._parse_text,
        }

        parser = parsers.get(file_ext)
        if not parser:
            raise ValueError(f"Unsupported file format: {file_ext}")

        content = await parser(file_path)
        return self._chunk_content(content, str(file_path))

    async def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text
        """
        reader = PdfReader(str(file_path))
        text_parts = []

        for page in reader.pages:
            text_parts.append(page.extract_text())

        return "\n".join(text_parts)

    async def _parse_text(self, file_path: Path) -> str:
        """Parse plain text or markdown file.

        Args:
            file_path: Path to text file

        Returns:
            File content
        """
        return file_path.read_text(encoding="utf-8")

    def _chunk_content(
        self,
        content: str,
        source_path: str,
    ) -> list[DocumentChunk]:
        """Split content into chunks.

        Args:
            content: Full document text
            source_path: Original file path

        Returns:
            List of document chunks
        """
        chunks = []
        words = content.split()
        current_chunk = []
        chunk_id = 0

        for word in words:
            current_chunk.append(word)

            if len(current_chunk) >= self.chunk_size:
                chunk_text = " ".join(current_chunk)
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source_path": source_path,
                        "chunk_index": chunk_id,
                    },
                    chunk_id=f"{source_path}_{chunk_id}",
                )
                chunks.append(chunk)
                current_chunk = current_chunk[-self.chunk_overlap :]
                chunk_id += 1

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    "source_path": source_path,
                    "chunk_index": chunk_id,
                },
                chunk_id=f"{source_path}_{chunk_id}",
            )
            chunks.append(chunk)

        return chunks

    async def parse_multiple(
        self,
        file_paths: list[str | Path],
    ) -> dict[str, list[DocumentChunk]]:
        """Parse multiple documents.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file paths to chunks
        """
        tasks = [self.parse_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunks_by_file = {}
        for path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                print(f"Error parsing {path}: {result}")
                continue
            chunks_by_file[str(path)] = result

        return chunks_by_file
