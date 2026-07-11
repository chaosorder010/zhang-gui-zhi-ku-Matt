# Handoff — 掌柜智库 · 接 #3 RAG 环路

> 新 agent 跳这会话先看本档。SPEC / ADR / issue 在下面路径/链接,不重述。

## 1. 摘要

项目:**掌柜智库** — 垂直领域企业级 RAG 知识库,中文优先。MVP 按 SPEC(`docs/spec/SPEC-全链路-MVP.md`)拆 4 issue。

- #2 (`/implement 仓库骨架 · 容器编排 · 配置`)已完成 → commit `7f31750`,branch `feat/01-scaffold` 已 push origin,PR <https://github.com/chaosorder010/zhang-gui-zhi-ku-Matt/pull/6>。
- PR #6 `/code-review` 未跑,issue #2 仍 OPEN,需走 review 关票。
- Issue #3 (02 — 核心 RAG 环路)是下一 frontir,关 #2 后立即开。

## 2. 关键事实

- 仓库 WSL UNC:`\\wsl.localhost\Ubuntu\home\leon\projects\zhang-gui-zhi-ku-Matt\`
- GitHub:`chaosorder010/zhang-gui-zhi-ku-Matt`(remote=origin,feat/* push 仓;master 空 — 起手)
- gh auth 已配(repo/gist/read:org),`gh auth setup-git` 已跑 → push 不等交互
- 本机 python3 是 Windows 商店 stub — **永远用 WSL venv `.venv/bin/python`**(WSL = `Ubuntu` distro,调用路径: `wsl.exe -d Ubuntu -- bash -lc <cmd>`)
- venv Python 3.12.13 在 `/home/leon/projects/zhang-gui-zhi-ku-Matt/.venv/`,包 `pip install -e '.[test]'` 装通,pytest 5/5 绿

### ⚠️ WSL DNS 陷阱

`/etc/wsl.conf` 起手 `generateResolvConf=false` + boot 脚本 `/usr/local/bin/wsl-static-ip.sh` 写 `/etc/resolv.conf` 失败 → 只剩 stub `127.0.0.53`,域名解不出。

**已修**(每次 WSL 重启后重做):
```bash
sudo rm -f /etc/resolv.conf
sudo tee /etc/resolv.conf <<'EOF'
nameserver 114.114.114.114
nameserver 8.8.8.8
EOF
```
下次 pip / push 挂 → 先测 `ping pypi.org`,不通重做。

## 3. 已工件

| 工件 | 路径/URL |
|---|---|
| SPEC | `docs/spec/SPEC-全链路-MVP.md` |
| Compose | `docker-compose.yml`(Milvus standalone+Mongo+MinIO+backend+frontend) |
| Backend | `app/{api,core,ingestion,retrieval,graph}/` 模块骨架 + `/api/health` 真 probes + 文档/query 占位 501 |
| Frontend | `frontend/` HTML/CSS/JS + nginx 空 chat 壳 |
| 本地 ticket | `.scratch/mvp-全链路/issues/{01..04}-*.md` |

### GitHub 映射

| # | title | state |
|---|---|---|
| #1 | SPEC 父票 | OPEN |
| #2 | 01 — 骨架·容器·配置 | OPEN(待 /code-review 关,PR #6) |
| #3 | 02 — 核心 RAG 环路 | OPEN frontier |
| #4 | 03 — 多路检索 | blocked by #3 |
| #5 | 04 — MCP + trace | blocked by #4 |

## 4. 设计锚点(已签仓 — 不改)

检索三路(路1 原 query Milvus hybrid、路2 HyDE Milvus、路3 MCP stub 空);Milvus 内 `score = α·dense + (1-α)·sparse`(α 网格搜索);RRF `score(d) = Σ_routes 1/(k + rank_i(d))`(k 默认 60);rerank 二阶 BGE-Reranker-Large 或 Qwen3;item_name LLM 抽 + 拼 chunk 头 + Milvus 常量字段存;三层分块(标题切→超长二次切→过短合并);LLM 云端 + LangChain,embed + rerank 本机 GPU;`docker compose up` 一键;不做 MCP路3(MVP stub)/鉴权/MinerU容器化/前端构建。

## 5. 即做(下一会话)

1. **`/code-review PR #6`** → 关 #2(`gh issue close 2`)
2. **`/implement #3`**(详见 `.scratch/mvp-全链路/issues/02-rag-loop.md` + SPEC 决策 D2/D3/D4):
   - Milvus SDK 真连(pymilvus 3.x MilvusClient — 现 ORM `connections.connect` 已 deprecation warn,迁 MilvusClient)
   - bge-m3 dense/sparse embed + rerank 真接
   - `ingestion/{parsers,chunker,item_name,embed,index}` 三段分块真 impl + item_name LLM 抽
   - `retrieval/{milvus_search,hyde,rrf,rerank}` 真 RRF 算术 + HyDE + rerank
   - `graph/` LangGraph 状态机 `receive → (hyde‖retrieve) → rrf → rerank → generate → respond`;rerank 无高质量命中 → `route="reject"` 输出"文档未覆盖"
   - `/api/documents`(POST/GET/DELETE)+ `/api/query` 实质化
   - 测试三件套,embed/rerank mock 小矩阵,RRF/α 算术断言,fixtures `tests/fixtures/`
   - 开 `feat/02-rag-loop`,push
3. `/clear` → `/implement #4` → `/implement #5`

## 6. 建议 skills

| Skill | 何时 |
|---|---|
| `/tdd` | RRF 算术 + 检索融合 + rerank 单调 + chunk 切点 — 单测级 |
| `/implement` | 每 ticket 主循环(必用) |
| `/code-review` | PR #6 关票;#3/#4/#5 提 PR 前 |
| `/verify` | compose 冒烟 + 上传→查询闭环 |
| `/caveman-commit` | Conventional Commit |
| `/security-review` | #5 MCP 外网接入前后 |

## 7. Conventions(延续)

- Caveman mode active(中文压缩体,技术词/API 名精确不动)
- 中文主语言;module 顶 `#:`;public 给 Google-style docstring;单 public 单文件,private 前缀 `_`
- 新模块三件套 `test_unit + test_integration + test_contract`,复用 `tests/_kit`
- embed/rerank 测 mock 小矩阵;compose 实库连 Milvus/Mongo
- Conventional Commits

## 8. 已知陷阱

- WSL DNS 重启易复发 — §2
- pydantic-settings module-load 即实例化 `Settings()`;`tests/conftest.py` 在 module load 写 os.environ(不能等 autouse fixture,已踩修)
- pymilvus 3.x ORM `connections.connect` deprecation,迁 MilvusClient
- gh create PR `--label ready-for-agent`,单引号 heredoc 写 body
- 缺 `.env` `LLM_API_KEY` → Settings raise `ValidationError` 明报缺啥 — 满足 #2 AC
