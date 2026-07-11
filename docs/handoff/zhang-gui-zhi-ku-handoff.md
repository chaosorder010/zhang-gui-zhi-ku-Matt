# Handoff — 掌柜智库 RAG 全链路 MVP

> 跳到这会话的新 agent 先看本档。SPEC、ADR、commit、GitHub issue 不重述 — 下面给链接/路径。

## 1. 摘要

项目:**掌柜智库** — 垂直领域企业级智能知识库,RAG 驱动,中文优先。全链路 MVP 从零起。

范围刚锚定 + spec + ticket 已产线化。**没写过一行功能代码**。当前停在 issue #2 (frontier)待 `/implement`。用户口头确认全链路 MVP、PDF+MD 原生。见 `docs/SPEC-全链路-MVP.md`。

## 2. 关键事实

- 仓库:`\\wsl.localhost\Ubuntu\home\leon\projects\zhang-gui-zhi-ku-Matt\`(WSL + UNC 路径)
- GitHub:`chaosorder010/zhang-gui-zhi-ku-Matt`(remote=origin,branch=master,空 — push 仓)
- 账户:Windows 用户名 `18047`,WSL user `leon`,GitHub `chaosorder010`
- gh auth 已配(keyring,token `gho_*`,scopes repo/gist/read:org)
- 工具链:Docker v29.4.2、Compose v5.1.3、NVIDIA-SMI 581.15 CUDA 13.0;本机 python3 是商店 stub — 用 `.venv`(Python 3.12)

## 3. 已产工件

| 工件 | 路径/URL |
|---|---|
| 已知技术栈 + 检索架构 | `docs/已知信息.md` |
| 全链路 MVP 规格 | `docs/SPEC-全链路-MVP.md`(27 用户故事、8 ID、测试决策、OOS) |
| 本地 ticket 文件 | `.scratch/mvp-全链路/issues/{01..04}-*.md`(源,已推 GitHub) |
| Issue tracker | GitHub issues (repo `chaosorder010/zhang-gui-zhi-ku-Matt`) |

### GitHub issue 映射

| # | 标题 | State | Blocked by | URL |
|---|---|---|---|---|
| #1 | SPEC 父票 | OPEN | — | `…/issues/1` |
| #2 | 01 — 仓库骨架 · 容器编排 · 配置 | **OPEN, frontier** | なし(可即开) | `…/issues/2` |
| #3 | 02 — 核心 RAG 环路 | OPEN | #2 | `…/issues/3` |
| #4 | 03 — 多路检索(hybrid·HyDE·RRF·rerank) | OPEN | #3 | `…/issues/4` |
| #5 | 04 — MCP 网络搜索路·trace 面板 | OPEN | #4 | `…/issues/5` |

全 4 ticket 贴 `ready-for-agent` + `mvp` label。`ready-for-agent` / `spec` / `mvp` label 在本会话/gh label create 新建(bug/doc/duplicate 等默认本就存在)。

推 issue 诀窍:bash 单引号 heredoc 写标题+body 再 `gh issue create … --label ready-for-agent,mvp`。Shell 双引号解析 `||` 乱,改用文件。

### 本地项目文件现状(读即懂,全空骨架)

- `pyproject.toml` — 仅 `>=3.12`,空 deps
- `main.py` — `print("Hello from ...")` placeholder
- `.python-version`=3.12、`.venv/` 存在、`.gitignore` 标配
- 无 commit、无分支、无 test、无 module

## 4. 设计锚点(不改 — 已签)

- 检索三路并行(路1 原始 query Milvus hybrid、路2 HyDE Milvus、路3 MCP 外网 — MVP stub)
- Milvus 内融合 `score = α·dense + (1-α)·sparse`,α 网格搜索
- RRF 融合三路 `score(d) = Σ 1/(k + rank_i(d))`
- rerank 二阶:BGE-Reranker-Large 或 Qwen3-Reranker
- item_name 用 LLM 抽 + 拼 chunk 头 + Milvus 常量字段存
- 三层分块(标题切→超长二次切→过短合并)
- LLM 走云 API + LangChain;embed + rerank 走本机 GPU
- 最小可跑:`docker compose up` 一键起
- 不做(MCP 路3 stub、鉴权、MinerU 容器化、前端构建工具链)

MinerU 不容器化:GPU/内存重,本机脚本跑;`POST /documents` 内部调本机 pipeline,MVP 强制本机先处理。

## 5. 即做(下一会话)

1. **`/implement #2`** — 仓库骨架 · 容器编排 · 配置(frontier,P0)
   - 具体:立 `app/` 包结构、`{ingestion,retrieval,graph,api,frontend}` 模块骨架
   - `docker-compose.yml`(milvus standalone+mongodb+minio+backend+frontend)
   - `.env.example` 全 + `core/config.py` 用 pydantic-settings 校验启动
   - `/api/health` 端点 + 三依赖探测
   - `pyproject.toml` 填全依赖 + pytest 冒烟绿
   - 完成后开分支 `feat/01-scaffold`,push 仓
2. 然后顺序 frontier `#3 → #4 → #5`(每票 `/clear` 清上下文再 `/implement`)

push 仓细节:branch 起名 `feat/01-<slug>`;commit 走 Conventional Commits(caveman-commit 模式可用 `/caveman-commit`)。当前 master 空 — 首次 commit 前确保全要文件 staged。

## 6. 建议启用的 skills

| Skill | 何时用 |
|---|---|
| `/implement` | 每 ticket 的主实现循环(必用,frontier = #2 先) |
| `/caveman-commit` | 每次提交写 Conventional Commit 简洁信息 |
| `/code-review` | 每票提 PR 前,review diff |
| `/tdd` | T2 环路单测 + T3 RRF/rerank 算术 — 单元级 TDD 合适 |
| `/security-review` | T4 MCP 外网路接入前后 — 外网回证注入风险低但需扫 |
| `/verify` | T1 docker compose 端到端冒烟、T2 上传→查询闭环 |

`/grill-me`、`/to-spec`、`/to-tickets`、`/handoff` 本会话已用过,下一会话不需重来。`/setup-matt-pocock-skills` 已跑(本机会话建 label)。

## 7. Conventions(接着来)

- Caveman 模式 active(中文压缩体)— 贯穿
- 中文主要语言,技术术语/API 名精确不动(import/named 不改)
- Module 顶部 `#:` 注释;public 给 Google-style docstring
- 单 public 单文件,private 前缀 `_`
- 测试:单测/集成/合同三件套,fixtures 进 `tests/fixtures/`
- embed/rerank 测试 mock 小矩阵,不卡 GPU;docker compose 实库连 Milvus/Mongo

## 8. 已知陷阱

- 本机 python3 是商店 stub — 永远显式用 `.venv/bin/python` 或 wsl python
- UNC 路径用双引号括,否则空格/`$` 解析乱
- shell heredoc 写 ticket body 用单引号 `'EOF'`,不解析 `$()`/`||`
- gh issue create 不原生支持 `--label`(真实用 `--label` 复数参数;本次 OK)

## 9. 不做 / 等下一会话议

- MCP 路3 真接(T4 才动)
- 鉴权、多租户、前端构建、批量 worker
- remote push 分支 + CI(GitHub Actions) — 本会话未配,下一会话 T1 顺带配可

> 状态:**spec + ticket 齐,frontier #2 待 `/implement`,零功能代码。**下一会话开干 `/implement #2`。
