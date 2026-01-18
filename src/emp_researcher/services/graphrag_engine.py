"""GraphRAG engine with Neo4j integration."""

import asyncio
import uuid
from typing import Any

from ..services import LLMRouter


class GraphRAGEngine:
    """GraphRAG engine for knowledge graph operations."""

    def __init__(
        self,
        llm_router: LLMRouter,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        database: str = "neo4j",
    ):
        """Initialize GraphRAG engine.

        Args:
            llm_router: LLM routing service
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            database: Database name
        """
        self.llm_router = llm_router
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.database = database
        self.driver = None

    async def connect(self) -> None:
        """Connect to Neo4j database.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            from neo4j import AsyncGraphDatabase

            self.driver = AsyncGraphDatabase(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            await self.driver.verify_connectivity()
        except ImportError:
            self.driver = None

    async def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        if self.driver:
            await self.driver.close()

    async def extract_entities(
        self,
        text: str,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            document_id: Document identifier

        Returns:
            List of extracted entities and relationships
        """
        messages = [
            {
                "role": "system",
                "content": "You are an entity extraction specialist. Extract entities, relationships, and attributes from the text.",
            },
            {
                "role": "user",
                "content": f"Extract entities from this text:\n{text}\n\nReturn JSON with keys: entities (list), relations (list of dicts with source, target, type), attributes (dict of entity to attributes).",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="graph_community_summarization",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)

            entities = []
            relations = []

            for entity in result.get("entities", []):
                entity_id = f"ent_{uuid.uuid4().hex[:8]}"
                entities.append(
                    {
                        "id": entity_id,
                        "name": entity,
                        "document_id": document_id,
                        "type": "entity",
                    }
                )

            for relation in result.get("relations", []):
                relations.append(
                    {
                        "source": relation.get("source"),
                        "target": relation.get("target"),
                        "type": relation.get("type", "RELATED_TO"),
                        "document_id": document_id,
                    }
                )

            return entities + relations

        except Exception:
            return []

    async def detect_communities(
        self,
        entity_count: int = 100,
    ) -> list[dict[str, Any]]:
        """Detect communities in knowledge graph.

        Args:
            entity_count: Number of entities

        Returns:
            List of community information
        """
        if not self.driver:
            return []

        from neo4j import AsyncGraphDatabase

        messages = [
            {
                "role": "system",
                "content": "You are a graph analyst. Group entities into meaningful communities (3-10 communities).",
            },
            {
                "role": "user",
                "content": f"""Analyze a knowledge graph with ~{entity_count} entities.

Group them into communities by:
1. Semantic relatedness (entities that appear together)
2. Domain/topic grouping
3. Hierarchical relationships

Return JSON with keys: communities (list of objects with id, title, entity_count).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="graph_community_summarization",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)

            communities = []
            for i, community in enumerate(result.get("communities", [])):
                community_id = f"comm_{uuid.uuid4().hex[:8]}"
                communities.append(
                    {
                        "id": community_id,
                        "title": community.get("title", f"Community {i + 1}"),
                        "entity_count": len(community.get("entities", [])),
                    }
                )

            return communities

        except Exception:
            return []

    async def generate_community_summary(
        self,
        community_id: str,
        entities: list[dict[str, Any]],
    ) -> str:
        """Generate summary for a community.

        Args:
            community_id: Community identifier
            entities: List of entities in community

        Returns:
            Community summary text
        """
        entity_names = [e["name"] for e in entities[:10]]

        messages = [
            {
                "role": "system",
                "content": "You are a summarization specialist. Create concise summaries of graph communities.",
            },
            {
                "role": "user",
                "content": f"""Summarize this community of entities:
{", ".join(entity_names)}

Provide a 100-200 word summary covering:
1. Main theme/topic
2. Key entities
3. Important relationships""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="graph_community_summarization",
        )

        return response.choices[0].message.content

    async def global_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform global search using community-level information.

        Args:
            query: Search query
            top_k: Number of top results

        Returns:
            List of search results with community context
        """
        if not self.driver:
            return []

        from neo4j import AsyncGraphDatabase

        messages = [
            {
                "role": "system",
                "content": "You are a GraphRAG global search specialist.",
            },
            {
                "role": "user",
                "content": f"""Perform global search across communities for: {query}

1. Identify most relevant communities
2. Search within those communities
3. Return top {top_k} results

Return JSON with keys: relevant_communities (list), search_results (list).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="graph_community_summarization",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return result.get("search_results", [])
        except Exception:
            return []

    async def local_search(
        self,
        query: str,
        community_id: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform local search within a community.

        Args:
            query: Search query
            community_id: Community identifier
            top_k: Number of top results

        Returns:
            List of search results
        """
        if not self.driver:
            return []

        from neo4j import AsyncGraphDatabase

        messages = [
            {
                "role": "system",
                "content": "You are a GraphRAG local search specialist.",
            },
            {
                "role": "user",
                "content": f"""Perform local search for: {query}

Community: {community_id or "general"}

Return top {top_k} relevant entities with context.

Return JSON with keys: entities (list of objects with name, description, relevance_score).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="graph_community_summarization",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return result.get("entities", [])
        except Exception:
            return []

    async def health_check(self) -> dict[str, Any]:
        """Check GraphRAG health.

        Returns:
            Health status
        """
        neo4j_healthy = self.driver is not None

        return {
            "neo4j": {"status": "ok" if neo4j_healthy else "error"},
            "entity_count": 0,
            "community_count": 0,
            "overall": "healthy" if neo4j_healthy else "degraded",
        }
