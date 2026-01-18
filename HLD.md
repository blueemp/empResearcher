一、系统总体架构概览
1.1 架构定位

系统定位为：“面向深度研究场景的多 Agent + GraphRAG + 双语 Web 搜索的一体化平台”，整体目标：

    对标 / 借鉴：GraphRAG（微软）、RAGFlow（深度文档理解）、Perplexity Deep Research（多轮深度搜索）、MiroFlow（多 Agent 推理编排），形成企业可私有化部署的解决方案；
    支持本地多模态知识库 + 外部网络信息双轮驱动；
    提供可配置的多 LLM 后端路由（small-fast / stronger / rerank 三类模型分工）；
    所有研究工作流统一通过多 Agent 流程与可观测基座支撑。

1.2 逻辑组件划分

从上到下分为四层：

    交互与配置层（UI & Admin）
        研究任务创建界面（输入研究主题、输出格式、深度要求）。
        实时进度与可观测界面（任务状态、Agent 执行情况、引用来源列表）。
        模型与搜索配置控制台：选择不同 Provider，配置 small-fast / stronger / rerank 模型；配置 SearXNG / Firecrawl baseUrl；设置中英文搜索策略等。

    Agent 编排与工作流层
        Coordinator Agent（协调器）+ 多角色 Agent（QueryRewriter、Retriever、本地/网络搜索、Evaluator、Reporter 等）。
        基于 MiroFlow 风格的 ReAct + Step-based Reasoning 模式：每一步都可以调用工具（搜索 / 检索 / 解析 / 评估），并产出中间思考和 ToDo。

    研究服务层
        LLM 路由服务：统一封装 OpenAI 兼容 / SiliconFlow / 本地 Ollama / 其他 Provider，按任务类型路由到 small-fast / stronger / rerank。
        检索服务：
            本地知识库：向量检索 + GraphRAG（知识图谱 + 社区搜索）。
            网络搜索：SearXNG 元搜索 + Firecrawl 深度爬取。
        解析与索引：RAGFlow 风格的深度文档理解与分块、多模态解析（文本、PDF、Office、表格、图片、音频等）。

    存储与基础设施层
        对象存储：原始文档与解析结果。
        向量库：Milvus/Weaviate/PGVector。
        图数据库：Neo4j（GraphRAG 知识图谱）。
        配置中心：模型路由配置、搜索源配置、Agent 策略参数。
        日志 & 监控：Prometheus/Grafana + OpenTelemetry。

二、模型配置与路由设计（重点）
2.1 模型来源与类型抽象
2.1.1 Provider 抽象

统一抽象接口：
python
class LLMProvider:
    async def chat(self, messages, model: str, **kwargs) -> str: ...
    async def embed(self, texts, model: str, **kwargs) -> list[list[float]]: ...
    async def rerank(self, query: str, docs: list[str], model: str, **kwargs) -> list[tuple[int, float]]: ...

典型 Provider 实现：

    OpenAI 兼容（任意符合 /v1/chat/completions 的云服务或自建代理）。
    SiliconFlow：适合作为高性价比主力云端推理，让 stronger-model、rerank-model 覆盖尽可能多的任务。
    Ollama：本地私有部署，适合作为兜底 small-fast-model 或特定场景 stronger-model（如本地 llama3 / qwen2.5 / deepseek 等）。

2.1.2 模型类型分层

为了统一配置体验，将模型划分为三大类型：

    small-fast-model
        主要场景：query rewrite、短文本分析、工具调用参数生成、轻量反思等。
        要求：低延迟、低成本。
    stronger-model
        主要场景：长文档总结、复杂多步推理、GraphRAG 社区摘要生成、跨多源综合报告生成。
        要求：高推理能力、大上下文。
    rerank-model
        主要场景：向量粗排后的精排，针对本地文档与 Web 结果做相关性、可信度、时效性综合排序。
        通常为 cross-encoder 或专用 Reranker，如 BGE-Reranker 系列等。

2.2 模型路由配置方案
2.2.1 配置结构（后端 YAML/JSON）
yaml
llm:
  providers:
    openai:
      type: "openai_compatible"
      base_url: "https://api.openai-proxy.example.com/v1"
      api_key: "${OPENAI_KEY}"
    siliconflow:
      type: "openai_compatible"
      base_url: "https://api.siliconflow.cn/v1"
      api_key: "${SILICONFLOW_KEY}"
    ollama:
      type: "ollama"
      base_url: "http://ollama:11434"

  # 为每个 Provider 定义其可选 small/stronger/rerank 模型列表
  provider_models:
    openai:
      small-fast-models:
        - "gpt-4o-mini"
      stronger-models:
        - "gpt-4o-2025"
      rerank-models: []
    siliconflow:
      small-fast-models:
        - "qwen2.5-7b-instruct"
        - "deepseek-r1-mini"
      stronger-models:
        - "deepseek-r1"
        - "qwen2.5-72b-instruct"
      rerank-models:
        - "bge-reranker-v2-m3"
    ollama:
      small-fast-models:
        - "llama3:8b"
      stronger-models:
        - "llama3:70b"
      rerank-models: []

  # 按任务类型 -> 模型类型映射
  task_model_mapping:
    query_rewrite:
      model_type: "small-fast-model"
    bilingual_translation:
      model_type: "small-fast-model"
    graph_community_summarization:
      model_type: "stronger-model"
    final_report_generation:
      model_type: "stronger-model"
    rerank_documents:
      model_type: "rerank-model"

  # 默认选中的 Provider（可被前端 UI 覆盖）
  default_provider:
    small-fast-model: "siliconflow"
    stronger-model: "openai"
    rerank-model: "siliconflow"
2.2.2 前端界面配置交互

管理端提供一个清晰的矩阵式配置界面：

    维度 1：Provider（OpenAI / SiliconFlow / Ollama / Custom）。
    维度 2：模型类型（small-fast / stronger / rerank）。
    维度 3：任务场景（query-rewrite、translation、rerank、summary、report 等）。

界面功能：

    为每个 Provider 列出当前可用模型（可通过配置或写死模板）。
    运维可在 UI 中为每个任务类型选择：
        使用哪个 Provider；
        使用其下哪一个具体模型。
    支持设定 primary / secondary provider，用于故障切换：
        如：small-fast-model 主用 SiliconFlow 的 qwen2.5-7b，备选为本地 Ollama 的 llama3:8b。

2.2.3 路由执行逻辑

执行时流程：

    Agent 声明任务类型（如 task_type="query_rewrite"）。
    LLM Router 根据 task_model_mapping 找出模型类型（small-fast）。
    根据当前用户/环境选择（前端配置）确定 Provider。
    从 provider_models[provider][model_type] 中选择默认模型（或按权重/轮询策略选）。
    若 Provider 健康检查失败，则自动切换到备用 Provider/模型。

三、双语查询与检索设计（重点）
3.1 核心目标与原则

    **默认策略：所有研究任务的每一个“检索相关”步骤，均应并行查询中文和英文资料；
    保证来自中文互联网生态（如中文新闻、技术博客、中文论文）与英文互联网（英文论文、GitHub、英文博客、白皮书）在同一工作流内都能被考虑；
    在信息融合时，避免某一语言结果“淹没”另一方，提供可配置的语言平衡策略。

3.2 双语查询流水线

    语言检测与规范化**
        检测用户输入语言（zh/en/混合）。
        使用 small-fast-model 做语义归一说明，例如“用户真正要研究的主题是：……”。

    双向翻译
        若原始为中文：
            zh_query = 原始问题
            en_query = small-fast-model 翻译结果
        若原始为英文则相反。

    双语 Query Rewriting
        对 zh_query 与 en_query 各自进行 Query Rewriting，得到：
            zh_sub_queries = [q_zh1, q_zh2, ...]
            en_sub_queries = [q_en1, q_en2, ...]

    双语 Web 搜索（SearXNG + Firecrawl）并行
        对所有 zh_sub_queries 调用 “中文引擎优先” 的 SearXNG 配置；
        对 en_sub_queries 调用 “英文引擎优先” 的 SearXNG 配置；
        Firecrawl 对两侧各自返回的 URL 集合，进行多线程抓取和结构化清洗。

    本地知识库检索（双语支持）
        向量库中对中文与英文文档分别或统一建索引；
        对 zh_query / en_query 都执行向量检索 + BM25 检索；
        利用 GraphRAG 全局/本地搜索，同样支持中英文实体、关系节点匹配。

    结果融合与 Rerank
        将“中文 Web + 英文 Web + 本地中文 + 本地英文”结果合并；
        使用 rerank-model 对其做统一重排，特征包括：
            与 zh_query、en_query 的相似度；
            源类型（学术、官方、新闻、博客、代码仓库）；
            时间新鲜度；
            是否被中英文双向查询同时命中（加权）。

    多轮反思与补充
        Evaluator 若发现：
            某语言侧信息明显不足（例如中文很多、英文来源稀少）；
            某关键子主题在某语言上缺失；
        则自动发起新一轮偏向弱势语言的补充搜索。

3.3 搜索配置示例
yaml
search:
  searxng:
    base_url: "https://searxng.local"
    engines_zh: ["baidu", "so", "google"]
    engines_en: ["google", "bing", "duckduckgo"]
  firecrawl:
    base_url: "https://firecrawl.local"
    max_concurrency: 4

  bilingual:
    enabled: true
    language_balance_weight: 0.5   # 0.5 表示中英权重相等
    zh_priority_sources: ["cn", "sohu.com", "csdn.net", "zhihu.com"]
    en_priority_sources: ["arxiv.org", "github.com", "acm.org", "medium.com"]
    max_results_per_lang: 30
四、参考项目特征映射与模块设计补充（重点）
4.1 GraphRAG 特征融入

从最新 GraphRAG 文章与白皮书中可以总结几个关键点：[大规模社区检测、动态社区选择、增量索引、DRIFT 动态检索/推理/过滤等][1]。

在本系统中的体现：

    图索引构建
        文档解析后，使用 LLM 抽取实体与关系，构建三元组；
        使用 Neo4j 存储实体节点、文本节点与关系；
        参考 GraphRAG“每个知识库一个全局图”的做法，而不是“每个文档一个图”，提升全局推理能力。

    社区检测与摘要
        使用 Leiden 等算法对整个图做社区划分；
        每个社区生成多层级摘要（stronger-model），方便 Global Search 时快速定位主题；
        结合 RAGFlow 新版对 GraphRAG 重构的经验，将实体抽取配置拆分为 Light/General 两档（轻量抽取 / 全量抽取），权衡成本与效果。

    动态社区选择（DCS） & DRIFT 思路
        在查询时，不是全量图搜索，而是：
            使用 small-fast-model（如 GPT-4o-mini 类）先做轻量社区选择；
            只对高相关社区做深度检索和生成；
        将 DRIFT 概念化为一个可配置模块：判断当前检索路径是否需要“漂移”到新的社区，以探索新信息。

4.2 RAGFlow 特征融入

从 RAGFlow 最新版本特性来看，重点包括：深度文档理解、多模态支持、模板化分块、多路召回+重排序、GraphRAG 模块重构优化等[2]。

在本系统中的体现：

    深度文档解析管线
        针对 PDF / Word / PPT / Excel / HTML / 图片 / 音频：
            OCR + 布局分析 + 结构化输出；
            表格保留结构化 JSON；
            图片、音频通过 VLM/ASR 做内容提取。
        提供多种“分块模板”：按段落、按语义单元、按标题层级等。

    多路召回与融合
        结合向量检索、BM25 检索、知识图谱邻居召回、RAPTOR 风格的树状摘要检索；
        结果进入 rerank-model 精排。

    **多模态 GraphRAG
        支持在图中同时存在“文本结点”和“图片/表格/音频结点”；
        对多模态结点也可以进行社区检测与摘要。

4.3 Perplexity Deep Research 特征融入

Perplexity Deep Research 的关键点在于：多轮、多步搜索 + 证据归并 + 结构化长文报告输出**[3]。

系统中的对应设计：

    深度研究模式
        Coordinator Agent 会：
            将用户需求拆解为多个子任务（TodoList）；
            针对每个子任务启动 10~几十次搜索，抓取上百页面；
            并记录所有来源、引用与置信度。

    研究进度控制
        在 UI 展示类似 Perplexity 的进度条和“已阅读 N 篇文档”的统计；
        Evaluator 会评估当前覆盖度、信息多样性与一致性，不足则自动追加搜索迭代。

    **报告形态
        输出 Markdown / PDF，包含：
            摘要、结论、分章节分析；
            引用列表（中英文来源混合，明确标注语言与出处 URL）；
            若有冲突信息，则单独列一节说明。

4.4 MiroFlow 多 Agent 编排特征融入

MiroFlow 对多 Agent 协作的特征在于：以 ReAct 推理 + Step-based Reasoning 为核心，强调工具链调用和多步决策**[4]。

在本系统：

    Agent 角色分工
        Coordinator、QueryRewriter、DocumentRetriever、WebSearcher、Evaluator、Synthesizer、Reporter。
        每个 Agent 都可以访问工具（LLM Router、本地检索、Web 搜索、GraphRAG、Rerank 等）。

    Step-based Reasoning
        每一步都有明确：
            当前目标；
            使用的工具；
            得到的观察（Observation）；
            生成的下一步计划。
        支持对某一步进行回滚和重试（如查询重写失败、搜索结果偏题等）。

    工具增强推理
        对标 MiroThinker 的“工具增强推理”思路，将 LLM 推理与外部工具调用（搜索、分析、可视化）充分结合，减少幻觉、提高事实准确度。

五、数据流与关键工作流
5.1 核心研究工作流（端到端）

    用户通过 UI 创建研究任务：
        主题 + 背景描述 + 期望输出形式 + 研究深度（简要/深入）。
    Coordinator 解析意图，生成结构化任务定义与 TodoList。
    QueryRewriter 执行双语查询规划：
        生成 zh/en 主查询 + 多个子查询。
    DocumentRetriever & WebSearcher 并行：
        本地知识库多路召回（向量+GraphRAG）；
        网络中英文并行搜索 + Firecrawl 深爬。
    Rerank-model 对各来源结果统一重排。
    Evaluator 对当前证据集进行质量和覆盖度评估：
        若不足，返回给 Coordinator 更新 TodoList；
        可能触发新的双语搜索或特殊场景（如仅对某国/某时间段强化搜索）。
    Synthesizer 聚合结构化知识：
        建立主题大纲、对比表格、时间线、关键结论。
    Reporter 根据模板生成报告：
        提供多格式导出；
        输出附带按语言/来源分类的引文列表。
    用户可在 UI 中发起“加深某一章节”的二次研究，重启从 Step 2 开始的子流程。

六、非功能设计（可观测性 / 维护性 / 健壮性）
6.1 可观测性

    指标维度：
        按模型类型：small-fast / stronger / rerank 的 QPS / 延迟 / 错误率 / Token 消耗；
        按语言：zh / en 搜索调用次数、覆盖度；
        按 Agent：各 Agent 的任务数、失败重试次数。
    链路追踪：
        每个研究任务一个 task_id；
        每个 Agent 步骤一个 step_id；
        所有外部调用日志都挂在该链路下面，方便事后分析。

6.2 可维护性与扩展性

    将 LLM Provider、Search Orchestrator、GraphRAG Engine、Document Parser、Workflow Engine 等拆为独立服务；
    新增模型 Provider 时，只需：
        实现 LLMProvider 接口；
        在配置中心注册 Provider 及其 small/stronger/rerank 模型列表；
        即可被前端配置界面捕获并选择。

6.3 健壮性

    对所有外部调用统一设定：
        超时；
        指数退避重试；
        熔断策略（降级使用本地模型或仅使用本地知识库）。
    对长任务：
        持久化任务状态，支持中断后恢复；
        队列化任务调度，支持限流与优先级。

七、总结与落地建议

在原有 PRD 的基础上，本 HLD 已：

    **明确 small-fast / stronger / rerank 三类模型在不同 Provider 下的配置方式与路由策略，并支持通过 UI 界面按任务类型与 Provider 维度灵活切换；
    将双语查询作为“硬约束”融入到整个检索链路，保证每个查询都同步利用中英文信息源，且在融合阶段有可配置的语言平衡策略；
    将 GraphRAG、RAGFlow、Perplexity Deep Research 与 MiroFlow 的关键特性具体落实到各模块（图索引、深度文档解析、多轮研究、Agent 编排等）。

在实施层面，可以采用分阶段交付：

    阶段 1（MVP）：完成基础 Agent 流程 + 模型路由 V1 + 简单双语搜索；
    阶段 2：接入 GraphRAG / RAGFlow 风格索引与解析，完善双语检索和 Rerank；
    阶段 3：强化多 Agent 编排、反思机制和可观测性，实现对标 Perplexity Deep Research 的体验。

