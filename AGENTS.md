# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-18
**Status:** Design Phase (Greenfield)

## OVERVIEW

Enterprise Deep Research Agent System (深度研究智能体) - Multi-agent platform using LLM + RAG + Web Search for automated research workflows (question → information collection → analysis → structured report).

**Target:** Enterprise-deployable alternative to Perplexity Deep Research/MiroFlow with local private deployment support.

## STRUCTURE

```
empResearcher/
├── PRD.md        # Product Requirements (319 lines)
└── HLD.md        # High-Level Design (373 lines)
```

**Planned Structure** (from HLD):
```
├── src/
│   ├── agents/          # 7 specialized agents (Coordinator, QueryRewriter, Retriever, WebSearcher, Evaluator, Synthesizer, Reporter)
│   ├── services/        # LLM Router, GraphRAG, Vector Store, Document Parser, Search Orchestrator, Rerank
│   ├── api/             # FastAPI entry point
│   ├── models/          # Data models
│   └── utils/           # Logging, telemetry
├── config/              # YAML configs (LLM providers, search engines)
├── tests/               # Unit/integration tests
├── frontend/            # Web UI (React-based)
└── deployment/          # Docker/K8s manifests
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Requirements | PRD.md | System scope, core features, implementation phases |
| Architecture | HLD.md | 4-layer design, model routing, bilingual pipeline |
| LLM Provider Design | HLD.md §2.1-2.3 | small-fast/stronger/rerank model abstraction |
| Agent Roles | PRD.md §2.4.1 | Coordinator, QueryRewriter, WebSearcher, etc. |
| GraphRAG Integration | HLD.md §4.1 | Community detection, global/local search |
| Bilingual Search | HLD.md §3 | zh/en parallel queries, fusion strategy |

## CONVENTIONS

**Documentation Language:** All specs in Chinese (PRD.md, HLD.md)

**Implementation Language:** Python (inferred from code examples in HLD)

**Model Categorization:** All LLM calls classified as small-fast / stronger / rerank (routing layer handles provider selection)

**Bilingual Constraint:** ALL retrieval steps MUST query BOTH Chinese and English sources (hard constraint per HLD)

**Multi-Agent Pattern:** ReAct + Step-based Reasoning (MiroFlow style) with explicit observation/action/plan cycles

## ANTI-PATTERNS (THIS PROJECT)

None yet - greenfield project.

## COMMANDS

**None yet** - design phase only.

## NOTES

**Implementation Status:** Phase 0 - Documentation only (no code)

**Tech Stack:** Python + LangGraph + FastAPI + Neo4j (graph) + Milvus/Weaviate (vector) + PostgreSQL + Docker/K8s

**LLM Providers:** OpenAI-compatible, SiliconFlow (primary cloud), Ollama (local fallback)

**Search Stack:** SearXNG (meta-search) + Firecrawl (deep crawling)

**Observability:** Prometheus + Grafana + OpenTelemetry + Jaeger

**Phase Strategy:**
- Phase 1 (4-6w): MVP - LLM abstraction + basic search + single Coordinator Agent
- Phase 2 (6-8w): Multi-agent + TodoList + reflection + rerank + observability
- Phase 3 (8-10w): GraphRAG + multimodal + productionization
