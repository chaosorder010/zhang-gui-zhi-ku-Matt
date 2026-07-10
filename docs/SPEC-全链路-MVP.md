# SPEC — 掌柜智库 全链路 MVP

> 项目: 掌柜智库(zhang-gui-zhi-ku) — 企业级智能知识库,RAG 驱动。
> 本档: 全链路最小可跑版本(MVP)范围、用户故事、实现/测试决策、缝对齐。
> 参照: `docs/已知信息.md`(技术栈 + 检索架构)。

---

## Problem Statement

垂直领域(电子产品手册、维修指南、技术文档等)的海量专业知识沉睡在 PDF 里。

维护人员、客服、用户想问一句话就能拿到某型号/某步骤的精准答案。现状:

- 关键词检索跨不了同义改写换了就丢
- 通用大模型不懂私有文档,瞎编(hallucinate)
- 已有 PDF 无人采 + 无回证(citation),信不过

需要一个"喂 PDF → 问一句 → 拿带来源引用的答"的最小闭环,中文场景开箱即用。

---

## Solution

一套 Web 应用:上传 PDF/或原生 MD → 自动解析 + 智能分块 + 入库 → 中文问 → 多路并行检索(Milvus 混合 + HyDE)+ RRF 融合 + rerank 精排 → LLM 生成带 citation 答。

一键起:`docker compose up`,浏览器开聊。解析/检索/生成全异步,pipeline 每个阶段可 LangGraph 状态机观测与重试。

---

## User Stories

### 文档入库

1. 作为 知识库管理员,我 想上传一本 PDF 产品手册,系统 自动把它拆成可检索的知识块,不用人工切分。
2. 作为 管理员,我 想同时上传多本 PDF,系统 并行处理且互不串 (= item_name 隔离)。
3. 作为 管理员,我 想上传原生 Markdown 维修指南,系统 把它和 PDF 文档走同一条入库管线。
4. 作为 管理员,我 想在上传时看到抽取出的主体名(item name,如"格力 KFR-35GW 空调"),确认编目正确。
5. 作为 管理员,我 想删除某本文档,Milvus 里所有关联 chunk 一并清掉,检索不再返回失效片段。
6. 作为 管理员,我 想看已入库文档清单(标题 + 上传时间 + chunk 数),感知库容量。
7. 作为 管理员,我 想系统对过短章节自动合并、对超长章节自动二次切,每块落到合理 token 窗口。
8. 作为 管理员,我 想让图表段落留在原文附近(邻 chunk 保留),回答配图问题有上下文。

### 检索

9. 作为 终端用户,我 用口语(不精确)问"这个空调不制冷怎么查",系统 也能召回对应维修段落。
10. 作为 终端用户,我 问模糊问题,系统 用 HyDE 生成假设文档辅助检索,提升召回。
11. 作为 终端用户,我问 的问题既有语义匹配又有必要关键词匹配,系统 dense + sparse 混合,两路兼顾。
12. 作为 终端用户,我 问的问题同时命中多本手册,RRF 融合把跨文档最优片段拿到 top。
13. 作为 终端用户,我 希望 top-K 候选经 BGE/Qwen3 rerank 精排后,送进 LLM 的上下文最相关,减少跑题。
14. 作为 开发者,我 想调 α(dense/sparse 权重)和 RRF 常数 k,网格搜索调参,检索命中率可量化。

### 问答与对话

15. 作为 终端用户,我 问中文问题,答 回中文,并 附 原文片段引用(来源文档 + 片段预览),我可核对。
16. 作为 终端用户,我 连续问多轮,系统 带前面轮次(history),不重复解释前提。
17. 作为 用户,问题超出知识库覆盖范围,系统 明确说"文档未覆盖",不乱编。
18. 作为 终端用户,我 想看本轮用了哪路检索、召回几段、rerank 后排位 — 调试模式透明可观测。
19. 作为 终端用户,我 希望生成快速给出(流式 respond),不用干等整段。

### 运维

20. 作为 运维,我 想 `docker compose up` 一键起 Milvus + MongoDB + MinIO + FastAPI + nginx,不要手工配。
21. 作为 运维,embedding + rerank 走 本机 NVIDIA GPU,LLM 调 云端 API,资源切分清晰。
22. 作为 运维,环境变量(模型名、API 端点、α/k、token 范围)全 收 `.env`,不硬编码。
23. 作为 运维,`/health` 返回后端 + 依赖(Milvus/Mongo/MinIO)健康,接监控。

### 边界与退化

24. 作为 用户,上传极小/纯图 PDF,系统 给清晰错误,不 crash。
25. 作为 用户,上传大 PDF(500 页+),处理不阻塞问答(异步入库)。
26. 作为 用户,API 输错字段,422 返回具体哪个字段错,500 不泄露堆栈。
27. 作为 开发者,MinerU 没装/没 GPU,本机脚本仍可走,MVP 不强制容器内闭环它。

---

## Implementation Decisions

### D1. 模块切分(`app/` 包)

| 模块 | 职责 |
| --- | --- |
| `app/ingestion/{parsers,chunker,item_name,embed,index}` | 解析 / 分块 / 主体识别 / embed / 入库 |
| `app/retrieval/{milvus_search,hyde,rrf,rerank}` | 混合检索 / HyDE / RRF / rerank |
| `app/graph/` | LangGraph 状态机: `receive → (hyde‖retrieve) → rrf → rerank → generate → respond` |
| `app/api/routes/{documents,query,health}` | FastAPI 端点 |
| `app/core/{config,logging,events}` | 配置加载(pydantic-settings)、日志、lifespan |
| `frontend/` | 原生 HTML/CSS/JS |
| `docker-compose.yml` + `docker/{backend.Dockerfile,frontend.nginx.conf,env.template}` | 部署 |

### D2. 入库管线契约

- Input:原始 PDF 字节 / MD 文本 + 文件名。
- Parser 统一产出 `ParsedDoc(pages_md: list[str], images: list[ImageRef])`。PDF 走 MinerU(本机脚本 wrapper,失败明报);MD 直接读。
- `chunker` 三段:标题正则切章节 → 超长(>`max_tokens`)按段落+滑动窗口再切 → 过短(<`min_tokens`)合入前/后邻。产出 `Chunk(text, doc_id, item_name, section, seq, token_len)`。
- `item_name`:LLM 首调用抽主体名,拼到每 chunk 文本头 `"<item_name> | "`,boost embedding 中实体语义;Milvus `item_name` 常量字段冗余存(便于过滤)。
- `embed`:本地中文 embedding 模型,至少 dense;sparse 向量也落(Milvus hybrid collection)。模型名从配置读,MVP 默认 `bge-m3`(双模)。
- `index`:建 Milvus collection(schema: id、doc_id、chunk_id、item_name、dense vec、sparse vec、text)。Mongo 同步写文档元数据(标题、上传时间、chunk_count、状态)。

### D3. 检索契约

- 单 query 触发三路并行:
  - 路1 原始 query → Milvus hybrid top-20(`score = α·dense + (1-α)·sparse`,α 配置读)
  - 路2 HyDE 生成假设答 → Milvus top-20
  - 路3 占位 stub,MVP 返回空列表 [TODO:MCP 网络搜索];RRF 公式仍三路求和,兼容后续接。
- RRF:`score(d) = Σ_routes 1/(k + rank_i(d))`,k 配置读,默认 60。
- Rerank:候选池(路1+路2 并集,上限 30)→ BGE-Reranker-Large 或 Qwen3-Reranker 精排 → 取 top-N(默认 5)送 LLM。
- 路由决策:MCP 路未接时只走 Milvus 两路,degrade 不报错。

### D4. LangGraph 编排

状态 `RAGState`:
```
question: str
history: list[Message]
route: Literal["local"] | "web" | "reject"
candidates: list[Candidate]
reranked: list[Candidate]
answer: str
citations: list[Citation]
retrieval_trace: dict   # 观测:各路召回数、RRF 分数、rerank 分数
```

节点:`retrieve` → `fuse` → `rerank` → `generate` → `respond`。`generate` 内:无高质量 rerank 命中 → `route="reject"`,输出"文档未覆盖";否则拼 context + 历史 → LLM → 挂 citation。异常节点带 1 次 retry。

### D5. API 契约

| Method | Path | 请求 | 响应 |
| --- | --- | --- | --- |
| POST | `/api/documents` | multipart `file` + `lang?` | `{doc_id, item_name, chunk_count, status}` |
| GET | `/api/documents` | — | `[{doc_id, title, uploaded_at, chunks}]` |
| DELETE | `/api/documents/{id}` | — | `{deleted_chunks}` |
| POST | `/api/query` | `{question, session_id?, trace?}` | `{answer, citations[], trace?}` |
| GET | `/api/health` | — | `{backend, milvus, mongo, minio}` |

错误 envelope 统一 `{detail: str, code: str}`;校验错 422,业务错 4xx,内部错 500(不泄栈)。

### D6. 前端

原生 HTML/CSS/JS,无构建。两视图:Chat(输入框 + 消息列 + 引文折叠)、Docs(上传 + 列表 + 删除)。API 调后端 `/api`,nginx 反代。

### D7. 部署

- `docker compose up` 起:`milvus`(standalone)+`mongodb`+`minio`+`backend`+`frontend`(nginx)。
- Embedding + rerank MVP 后端容器直连本机 GPU(nvidia-container-runtime)。MinerU 本机脚本直跑(资源重,不强行容器化),`POST /documents` 内部调本机 pipeline 或提示用户先跑本机脚本。
- LLM 走 云端 LangChain 封装,环境变量注入 provider + key。

### D8. 配置

全收 `.env`:`EMBED_MODEL`、`EMBED_DENSE_DIM`、`RERANK_MODEL`、`LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_KEY`、`MILVUS_URI`、`MONGO_URI`、`MINIO_*`、`ALPHA`(default 0.7)、`RRF_K`(default 60)、`MAX_TOKENS`(default 512)、`MIN_TOKENS`(default 64)、`TOP_K_RETRIEVE`(20)、`TOP_N_RERANK`(5)。`core/config.py` pydantic-settings 读,启动校验。

---

## Testing Decisions

### 测什么(外部行为,不盯内部 impl)

- **管线端到端**:上传小样例 PDF/MD(仓库 `tests/fixtures/`)→ 断言 Milvus 收到 chunk 数 > 0,且 chunk 文本带 item_name 前缀。
- **检索回召**:seed 已知 chunk → 起 QA 对(手写 ≥ 20 对)→ `POST /query` → 断言正确 chunk 在 rerank 后 top-N 命中。
- **融合算对**:mock 两路独立排序 → 断言 RRF 分数与手算 k=60 一致;α=0 全 sparse、α=1 全 dense,hybrid 分数边界对。
- **Rerank 单调**:相候选池仅换 rerank 模型,分数相对序一致(=rerank 不破坏相关性序)。
- **护栏**:未覆盖问题回"文档未覆盖"且 citations 空;大文件异步不阻塞别个 API;坏文件 4xx 明报;`/health` 依赖 down 时标 unhealthy。
- **单元**:parser 形状、chunker 切点、RRF 算术、样本 embedding 维度、LangGraph 状态流转(reject 分支)。

### 测试栈

pytest + httpx + FastAPI `TestClient`,Milvus 用 `pymilvus` 连 compose 起的实库;embedding/rerank mock 为小矩阵,跑得快不卡 GPU。fixtures 复用。

### 缝对齐

最高层缝 = HTTP API(`/api/documents`, `/api/query`, `/api/health`)。次层 = `ingestion.pipeline.run(doc_bytes)` 和 `retrieval.search(question, state)` 这两入口。不专门给 chunk 内 helper 开缝,通过上面两层够。

---

## Out of Scope (MVP 外)

- MCP 网络搜索路(路3):stub 空,留接口、issue 跟踪
- 完整鉴权/多租户
- MinerU GPU 容器化(本机脚本跑)
- 前端构建工具链(Vite/Webpack)
- 批量调度/异步 worker(Celery) — MVP 同步入库,文件大到阻塞再提
- 多语言简中↔其它

---

## Further Notes

- item_name 抽取 Prompt 入 `app/ingestion/item_name/prompt.md`,可独立调优rerank 模型切换不改管线。
- Milvus hybrid collection schema 写 `docs/milvus_schema.md`,作为演进 ADR 锚点。
- LLM provider 走 LangChain,MVP 默认一个 provider,多 provider 切 后续做。
- 后续 v1可做:会话(Mongo 已落)

- **测试**:新模块三件套(`test_unit + test_integration + test_contract`),复用 `tests/_kit`。
- **配置**:pydantic-settings + `.env`,校验启动期。

---

## Further Notes

- item_name 抽取 Prompt 入 `app/ingestion/item_name/prompt.md`,独立可调优,rerank 模型切换不影响管线。
- Milvus hybrid collection schema 写 `docs/milvus_schema.md`,当作演进 ADR 锚点。
- LLM provider 走 LangChain,MVP 默认一个 provider,多 provider 切 后续做。

> **本 spec「已知信息.md」同源,做实现前先锁仓同意。issue标签 `ready-for-agent`/triage 由 /to-spec 工序配,remote 通后补推。**
