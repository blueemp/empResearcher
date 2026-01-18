# empResearcher

Enterprise Deep Research Agent System (深度研究智能体) - A multi-agent platform using LLM + RAG + Web Search for automated research workflows.

## Overview

empResearcher is an enterprise-grade AI-powered research platform that automates the complete workflow from question understanding → information collection (local + web) → multi-round analysis → structured report generation.

**Target:** Enterprise-deployable alternative to Perplexity Deep Research/MiroFlow with local private deployment support.

## Features

### Core Capabilities
- **Multi-Agent Orchestration**: 7 specialized agents (Coordinator, QueryRewriter, DocumentRetriever, WebSearcher, Evaluator, Synthesizer, Reporter)
- **GraphRAG Integration**: Knowledge graph with community detection (Neo4j) for global + local search
- **Bilingual Search**: Parallel Chinese/English queries with language-aware result fusion
- **Multi-LLM Support**: OpenAI, SiliconFlow, Ollama with smart routing (small-fast/stronger/rerank models)
- **Multi-modal Knowledge Base**: Support for PDF, Word, Excel, PPT, images, code, and audio
- **Web Search**: SearXNG meta-search + Firecrawl deep crawling
- **Observability**: OpenTelemetry, Prometheus, Grafana for production monitoring

### Phase Implementation
- **Phase 1 (MVP)**: LLM abstraction + basic search + single Coordinator Agent
- **Phase 2**: Multi-agent + TodoList + reflection + rerank + observability
- **Phase 3**: GraphRAG + multimodal + production deployment

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/blueemp/empResearcher.git
cd empResearcher
```

2. **Install dependencies**
```bash
pip install -e ".[dev,knowledge]"
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

4. **Start services**
```bash
docker-compose up -d
```

5. **Run the application**
```bash
python -m emp_researcher.api.main
```

Or use the CLI:
```bash
emp-researcher start
```

### Development Setup

```bash
# Install dev tools
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/emp_researcher --cov-report=html

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Project Structure

```
empResearcher/
├── src/emp_researcher/
│   ├── agents/          # Multi-agent implementations
│   ├── services/        # Core services (LLM, RAG, search)
│   ├── api/             # FastAPI application
│   ├── models/          # Data models
│   └── utils/           # Utilities (logging, telemetry)
├── config/              # YAML configurations
├── tests/               # Unit and integration tests
├── deployment/          # Docker/K8s manifests
└── docs/               # Additional documentation
```

## Configuration

### LLM Providers

Configure LLM routing in `config/llm_providers.yaml`:

```yaml
llm:
  providers:
    openai:
      type: "openai_compatible"
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
    siliconflow:
      type: "openai_compatible"
      base_url: "https://api.siliconflow.cn/v1"
      api_key: "${SILICONFLOW_API_KEY}"

  task_model_mapping:
    query_rewrite: "small-fast-model"
    graph_community_summarization: "stronger-model"
    final_report_generation: "stronger-model"
    rerank_documents: "rerank-model"
```

### Search Engines

Configure search in `config/search_config.yaml`:

```yaml
search:
  searxng:
    base_url: "http://localhost:8080"
    engines_zh: ["baidu", "google"]
    engines_en: ["google", "bing", "duckduckgo"]
  bilingual:
    enabled: true
    language_balance_weight: 0.5
```

## Usage Examples

### Create a Research Task

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze the impact of AI on healthcare",
    "depth": "deep",
    "output_format": "markdown"
  }'
```

### Get Task Status

```bash
curl http://localhost:8000/api/v1/tasks/{task_id}/status
```

### Get Research Report

```bash
curl http://localhost:8000/api/v1/tasks/{task_id}/report
```

## API Documentation

Once the application is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

### Four-Layer Design

1. **UI & Admin Layer**: Research task creation, real-time progress, observability dashboard
2. **Agent Orchestration Layer**: Multi-agent workflow with ReAct + Step-based Reasoning
3. **Research Services Layer**: LLM Router, GraphRAG, Vector Store, Document Parser, Search Orchestrator
4. **Storage & Infrastructure Layer**: Object storage, vector DB, graph DB, configuration, monitoring

### Model Types

- **small-fast-model**: Query rewrite, tool calls (e.g., gpt-4o-mini, qwen2.5-7b)
- **stronger-model**: Complex reasoning, summarization (e.g., gpt-4o, deepseek-r1)
- **rerank-model**: Document relevance scoring (e.g., bge-reranker-v2-m3)

## Tech Stack

- **Language**: Python 3.10+
- **Web Framework**: FastAPI
- **Agent Orchestration**: LangGraph
- **Vector Database**: Milvus
- **Graph Database**: Neo4j
- **Search**: SearXNG, Firecrawl
- **Observability**: OpenTelemetry, Prometheus, Grafana
- **Deployment**: Docker, Kubernetes

## Documentation

- [Product Requirements (PRD)](PRD.md) - Detailed requirements (Chinese)
- [High-Level Design (HLD)](HLD.md) - Architecture specifications (Chinese)
- [Agent Knowledge Base](AGENTS.md) - Project conventions and patterns

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: https://github.com/blueemp/empResearcher/issues
- **Documentation**: https://github.com/blueemp/empResearcher/blob/main/README.md

## Acknowledgments

This project is inspired by and incorporates concepts from:
- [GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/) (Microsoft)
- [RAGFlow](https://ragflow.io/) (Deep document understanding)
- [Perplexity Deep Research](https://www.perplexity.ai/) (Multi-round research)
- [MiroFlow](https://github.com/sgoodfriend/miroflow) (Multi-agent orchestration)
