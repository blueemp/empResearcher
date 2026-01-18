"""Vector store service using Milvus."""

import uuid
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
)


class VectorStore:
    """Milvus vector store abstraction."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "emp_researcher",
        dimension: int = 768,
    ):
        """Initialize vector store.

        Args:
            host: Milvus host
            port: Milvus port
            collection_name: Collection name
            dimension: Embedding dimension
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = None
        self.collection = None

    async def connect(self) -> None:
        """Connect to Milvus."""
        self.client = MilvusClient(
            host=self.host,
            port=self.port,
        )
        connections.connect(host=self.host, port=self.port)
        await self._get_or_create_collection()

    async def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self.client:
            connections.disconnect(host=self.host, port=self.port)

    async def _get_or_create_collection(self) -> None:
        """Get or create collection."""
        try:
            self.collection = Collection(name=self.collection_name)
            if not self.collection.has_index():
                await self._create_collection()
        except Exception:
            await self._create_collection()

    async def _create_collection(self) -> None:
        """Create new collection."""
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=64,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dimension,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON,
                ),
            ],
            description="Document embeddings",
            enable_dynamic_field=True,
        )

        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
        )
        await asyncio.to_thread(
            self.collection.create_index, field_name="embedding", index_type="IVF_FLAT"
        )

    async def insert(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Insert chunks and embeddings.

        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors

        Returns:
            List of inserted IDs
        """
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        data = [
            {
                "id": chunk_id,
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": chunk["metadata"],
            }
            for chunk_id, chunk, embedding in zip(ids, chunks, embeddings)
        ]

        await asyncio.to_thread(self.collection.insert, data)
        return ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            filter_expr: Optional filter expression

        Returns:
            List of search results with scores
        """
        self.collection.load()

        results = await asyncio.to_thread(
            self.collection.search,
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "content", "metadata"],
        )

        formatted_results = []
        for result in results[0]:
            formatted_results.append(
                {
                    "id": result.id,
                    "content": result.entity.get("content"),
                    "metadata": result.entity.get("metadata"),
                    "score": result.distance,
                }
            )

        return formatted_results

    async def delete(self, ids: list[str]) -> None:
        """Delete chunks by IDs.

        Args:
            ids: List of chunk IDs to delete
        """
        await asyncio.to_thread(self.collection.delete, ids)

    async def get_count(self) -> int:
        """Get total number of chunks.

        Returns:
            Chunk count
        """
        return await asyncio.to_thread(self.collection.num_entities)
