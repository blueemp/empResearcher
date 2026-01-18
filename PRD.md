一、产品定位与目标

产品定位
企业级「深度研究智能体」系统，用大模型 + 多 Agent + RAG + Web 搜索，自动完成从问题理解 → 全网/本地信息收集 → 多轮分析反思 → 结构化报告输出的全过程。

主要目标

    构建端到端深度研究工作流：从用户研究需求到可交付报告，全流程自动化。
    支持本地多模态（文本 / PDF / 图片 / 表格 / 代码等）知识库建设与整理（参考 GraphRAG、RAGFlow 思路）。
    通过网络搜索（SearXNG + Firecrawl）弥补本地知识缺口，效果对标 Perplexity Deep Research、MiroFlow 这类深度研究 Agent。
    LLM 后端可配置：支持 OpenAI 兼容接口、硅基流动（SiliconFlow）、本地 Ollama。
    使用多 Agent + 工作流编排，支持复杂长链路任务。
    内置深度研究增强能力：查询重写、分步 TodoList、Rerank、多轮反思、自主判断「是否已经研究充分」。
    强调可观测性、可维护性与健壮性：有完善监控、日志、告警、容错与扩展能力。

二、核心功能与模块
2.1 本地多模态文档知识库（参考 GraphRAG / RAGFlow）
2.1.1 文档接入与解析

需求

    支持多种文档类型：
        文本：Markdown、TXT、HTML
        办公文档：PDF、Word、Excel、PPT
        图片：PNG/JPEG（扫描件、截图等）
        代码：常见语言源码、Jupyter Notebook
    文档上传方式：
        Web 页面上传
        文件夹批量导入
        定时同步（如挂接共享盘/对象存储）

实现要点

    统一「文档解析管线」：
        文本 / PDF：使用 OCR（必要时）+ 布局解析→段落/标题/表格结构化
        图片：通过 VLM（图文模型）提取文字与关键描述（参考 RagFlow 使用 DeepDoc + VLM 的模式）
        表格：保留结构，单元格内容拆分成结构化 JSON
        代码：抽取函数/类/模块级摘要及依赖关系

输出统一为：
json
{
  "doc_id": "xxx",
  "chunk_id": "xxx",
  "content": "文本内容",
  "modality": "text|image|table|code",
  "metadata": {
    "source_path": "...",
    "page": 3,
    "section": "1.2",
    "created_at": "...",
    "tags": [...]
  }
}
2.1.2 向量索引 + 知识图谱（GraphRAG 思路）

需求

    除传统向量检索外，还需构建「文档级知识图」，支持全局关联与社区检索。
    支持大规模文档下的「主题社区」聚类和总结。

实现要点（参考 GraphRAG）

    实体与关系抽取
        使用 LLM + NER 模型提取实体（人名、机构、技术名词等）。
        使用 OpenIE 或 LLM 提示抽取「实体-关系-实体」三元组。
    知识图构建
        将实体、文档段落作为节点，关系作为边存入图数据库（Neo4j 等）。
    社区检测
        使用 Leiden 等社区检测算法，对图做层级聚类，形成主题簇与子簇。
        每个社区由 LLM 自动生成摘要，为「全局检索」阶段服务。
    多模态检索
        文本与图片向量均写入向量库（Milvus / Pinecone / Weaviate），通过统一接口封装。
        支持「先社区 → 再文档 → 再段落」的层次化检索（与 GraphRAG 的 Global / Local 查询类似）。

2.2 网络搜索与抓取（SearXNG + Firecrawl）
2.2.1 搜索工具与本地化配置

需求

    使用 SearXNG 作为可自建元搜索引擎（聚合 Google/Bing 等），支持 base_url 配置。
    使用 Firecrawl 作为 Web 抓取 / 搜索 API，base_url 同样可配置为自建实例。
    支持在配置界面为每个环境（dev/stage/prod）设置不同 base_url、API Key、代理配置。

实现要点

    封装统一 SearchOrchestrator：
        配置项：启用引擎列表（如 searxng / firecrawl / 其他），优先级与重试策略。
        根据查询类型与目标（快速 vs 深度）动态选择：
            快速问答：优先 SearXNG 搜索结果的摘要。
            深度研究：先 SearXNG 找候选 URL，再调用 Firecrawl 做深度爬取和结构化抽取。
    对标 Perplexity Deep Research：
        一次研究任务中自动发起「几十次搜索 + 上百页面爬取」→ 形成高覆盖的信息池。
        过程中记录所有来源并打分，最终报告给出可点击引用。

2.3 LLM 后端可配置（OpenAI 兼容 / 硅基流动 / Ollama）
2.3.1 LLM Provider 抽象层

需求

    屏蔽不同厂商 API 差异（URL、认证、参数名、流式协议等）。
    动态配置：某类任务使用哪个模型（例如重写用小模型、总结用大模型）。

实现要点

设计统一的 LLMClient 接口，例如：
python
class LLMClient:
    async def chat(self, messages, model: str, **kwargs) -> str:
        ...
    async def embed(self, texts, model: str, **kwargs) -> List[List[float]]:
        ...

支持的 Provider

    OpenAI 兼容：标准 /v1/chat/completions + /v1/embeddings。
    硅基流动（SiliconFlow）：[1]
        统一、多模型 OpenAI 兼容 API，支持多种开源/商用模型，延迟低，适合作为主力云端推理。
    本地 Ollama：
        通过 HTTP API 接入，支持多模型（如 llama3、deepseek 等）。
        可配合本地 GPU 集群（参考类似 OllamaFlow 的模式）提高高可用性。

配置示例
yaml
llm:
  default_provider: "siliconflow"
  providers:
    openai_compatible:
      base_url: "https://api.xxx/v1"
      api_key: "..."
    siliconflow:
      base_url: "https://api.siliconflow.cn/v1"
      api_key: "..."
    ollama:
      base_url: "http://ollama:11434"
      models:
        - "llama3:70b"
        - "qwen2.5:32b"
  routing_rules:
    query_rewrite: "small-fast-model"
    global_summary: "stronger-model"
    rerank: "rerank-model"
2.4 多 Agent + 工作流架构（参考 MiroFlow、多 Agent 设计模式）
2.4.1 典型 Agent 角色

建议基础版本包含如下角色：

    Coordinator Agent（协调器）
        职责：解析用户目标、生成 TodoList、分解子任务、调度其他 Agent。
        能力：基于 LangGraph / FSM 实现有状态多轮流程，具备「任务完成度」判断。

    QueryRewriter Agent（查询重写）
        职责：对用户原始问题做意图理解、查询扩展、生成多个子查询。
        对标：Agentic RAG 中的 Query Rewrite 模式、Microsoft Query Rewriting 功能[2]。

    DocumentRetriever Agent（本地知识库检索）
        职责：针对子问题检索本地向量库 + 图数据库。
        策略：先查询知识图的相关社区，再在该社区范围做精细向量检索（GraphRAG Global+Local）。

    WebSearcher Agent（网络搜索）
        职责：调用 SearXNG + Firecrawl，针对知识缺口做外部补充。
        能力：可根据当前证据的「时间性 / 权威性」决定是否加强某个方向的搜索。

    Evaluator / Critic Agent（评估&反思）
        职责：对当前收集的信息做质量评估：
            相关性、完整性、一致性、冲突检测等。
        逻辑：类似 Reflexion 模式，对不满意结果返回修改建议给 Coordinator 或 QueryRewriter。

    Synthesizer Agent（信息聚合）
        职责：将不同来源（本地 + 网络）的多条证据统一成结构化知识（树状大纲、对比表、时间线等）。

    Reporter Agent（报告生成）
        职责：按指定模版输出报告（研究报告 / 市场分析 / 技术调研），保留脚注和引用链接。
        能力：多格式导出（Markdown、HTML、PDF）。

2.4.2 工作流模式（对标 MiroFlow / LangGraph 深度研究）

整体流程

    任务创建
        用户输入研究目标（可多轮说明），Coordinator 生成规范化任务描述与初始 TodoList。

    TodoList 驱动执行
        每一个 Todo 项由 Coordinator 选用合适的 Agent 处理。
        可以并行执行：如「整理本地文档」与「并发 Web 搜索」。

    每步反思
        每个子任务结束后，Evaluator 评估：
            本步是否完成？
            信息是否足够？
            是否需要重新检索 / 重写查询？
        若未达到标准，则自动：
            触发 QueryRewriter 生成新检索指令；
            或让 WebSearcher 进一步深挖。

    目标达成判断
        当全局 TodoList 已完成，且 Evaluator 对整体「覆盖度 / 自洽性 / 可信度」打分高于阈值时：
            进入最终报告生成；
        否则，自动追加新的 Todo（例如「补充更多最新数据」「增加某国市场对比」）。

    报告输出与用户确认
        Reporter 生成报告草稿，用户可查看大纲和引用源列表。
        用户可以请求「进一步加深某一章节」，系统生成新的子任务循环。

2.5 深度研究关键能力设计
2.5.1 查询重写（Query Rewriting）

思路

    结合：
        OpenAI / 硅基流动高质量 LLM 做语义理解；
        基于规则的 Pattern（如识别时间范围、地点、实体）做结构补全；
        在反思过程中动态重写失败查询（参考 Agentic RAG 实践[3]）。

输出
json
{
  "original_query": "...",
  "intent": "用户想了解的是 ...",
  "sub_queries": ["子问题1", "子问题2", ...],
  "keywords": ["核心关键词1", "同义词A", ...]
}
2.5.2 Rerank（多级重排）

实现建议

    第一层：向量相似度粗排（从 10k → 1k）。
    第二层：使用编码器 / 交叉编码模型（如 bge-reranker）做语义精排（1k → 100）。
    第三层：根据以下信号加权重排：
        来源类型（权威机构 > 个人博客）；
        时间（越新权重大，但可配置）；
        与用户上下文的概念重合度。

2.5.3 TodoList 与任务编排

    Coordinator 持有每个任务的状态机：
        PENDING → RUNNING → REFINE / DONE / FAILED
    与用户交互界面：
        显示当前研究进度（多少 Todo 完成、预计剩余时间）。
        可手动调整 / 插入新 Todo（例如「加一节关于竞争对手 C 的比较」）。

2.5.4 每一步反思（Reflection）

    每一步由 Evaluator Agent 生成：
        当前证据的优缺点；
        明确指出「缺了哪些角度」；
        建议下一步策略（更多搜索 / 换关键词 / 回到本地文档）。
    模式可参考 Reflexion / Reflection Agents[4]。

2.5.5 目标/结果达成判断

    通过以下维度打分：
        覆盖度（引用来源数量 + 主题覆盖广度）；
        可信度（权威来源比例）；
        一致性（内部是否自洽、矛盾点是否说明清楚）。
    低于阈值时不结束任务，而是自动回流到 QueryRewriter / WebSearcher。

三、非功能需求：可观测性、维护性、健壮性
3.1 可观测性（Observability）

指标监控

    系统层：
        请求 QPS、错误率、平均延迟、队列长度。
    Agent 层：
        每个 Agent 的调用次数 / 平均耗时 / 成功率。
    LLM 调用：
        各 Provider 的延迟 / 错误率 / Token 使用量。
    检索效果：
        平均召回文档数、各轮 Rerank 后保留比例。

实现建议

    指标：Prometheus + Grafana。
    链路追踪：OpenTelemetry + Jaeger。
    日志：JSON 结构化日志（含 trace_id / task_id / agent_name）。

3.2 可维护性

    模块化 / 微服务风格：
        LLM Router、Search Orchestrator、Indexer、Agent Orchestrator、UI Server 独立可替换。
    配置中心：
        LLM Provider、搜索引擎 base_url、模型路由规则通过配置文件或配置服务管理。
    清晰的扩展点：
        新增 Agent（例如「代码分析 Agent」「财务报表 Agent」）只需实现统一接口并在 Coordinator 注册即可。

3.3 健壮性

    超时与重试机制：
        对外部服务（SearXNG, Firecrawl, LLM API）统一封装：超时、退避重试、熔断。
    降级策略：
        某个 Provider 不可用时自动切换备用模型。
        网络搜索失效时至少保证本地知识库检索可用。
    任务恢复：
        所有任务状态落盘（PostgreSQL 等），支持重启后续跑。

四、落地建议与实施优先级

Phase 1：最小可用版本（4–6 周）

    实现：
        基础 LLM 抽象层（先接 OpenAI 兼容 + 硅基流动）。
        本地文档解析 + 向量库检索（不必一开始就上知识图）。
        SearXNG + Firecrawl 简单集成。
        单 Agent（Coordinator）+ 内嵌工具形式完成一个端到端研究 Demo。
    目标：完成简单问题的「本地+网络」研究报告。

Phase 2：多 Agent + 深度研究增强（6–8 周）

    拆出 QueryRewriter / WebSearcher / Evaluator / Reporter 等独立 Agent。
    引入 TodoList、反思、Rerank 机制。
    加入基本可观测性（Prometheus / Grafana / 日志系统）。

Phase 3：GraphRAG & 多模态增强 + 生产化（8–10 周）

    引入知识图谱、社区检测，支持 Global + Local 检索。
    接入 VLM，对图片类资料支持更好抽取。
    完善安全、权限、审计，准备生产部署。
