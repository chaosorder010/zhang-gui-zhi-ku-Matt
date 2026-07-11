# 掌柜智库 (zhang-gui-zhi-ku)

垂直领域企业级智能知识库,RAG 驱动,中文优先。

## 快速起

```bash
cp .env.example .env
# 编辑 .env: 填 LLM_API_KEY 等
docker compose up --build
```

后端 `http://localhost:8000/api/health`,前端 `http://localhost:8080`。

## 目录

- `app/` — FastAPI 后端 + RAG 管线 + LangGraph 编排
- `frontend/` — 原生 HTML/CSS/JS 聊天壳
- `docker/` — Dockerfile + nginx 配置
- `tests/` — pytest 套件
- `docs/` — SPEC / ADR / handoff
