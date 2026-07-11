# 01 — 仓库骨架 · 容器编排 · 配置

## What to build
仓库长出可 `docker compose up` 的 FastAPI + Milvus + MongoDB + MinIO + nginx 拓扑;`.env.example` 全量启动期校验;`/api/health` 后端 + 三依赖健康共答。浏览器打开仅空 chat 壳。

## Blocked by
None — 可立即开始

## Status
ready-for-agent

## Acceptance criteria
- [ ] `docker compose up` 五容器启,`/api/health` 共 4 项健康
- [ ] 缺 `.env` 必填键,后端起不来并明报缺啥
- [ ] `pyproject.toml` 列出全部依赖,pytest 具装,`tests/` 冒烟绿
- [ ] `/api/documents` `/api/query` 404 误(路由注册)
