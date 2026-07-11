# 02 — 核心 RAG 环路(单路 · 流式 · 回证)

## What to build
上传小 PDF/MD → 自动解析+三层分块+LLM 抽 item_name 拼头+embed+Milvus 入库 → 中文问 → 单路 dense top-N 召回 → LangGraph 状态机(串行,无多路)→ LLM 生成带 citation 答,流式推前端。三轮对话带历史。

## Blocked by
#01 — 仓库骨架 · 容器编排 · 配置

## Status
ready-for-agent

## Acceptance criteria
- [ ] 上传 fixtures 样本 PDF → `GET /api/documents` 返回含 item_name 的元数据,chunk 数 > 0
- [ ] 中文 query → `POST /api/query` 流式 SSE答,`citations[]` 至少 1 条含来源预览
- [ ] 多轮同 session_id 问,后续答引用前文前提不重复
- [ ] 未覆盖问题 → 答"文档未覆盖"+ `citations` 空
- [ ] 坏 PDF → 4xx 明报;不阻塞其余端点
- [ ] 串行覆盖率 pytest ≥ 单路召回 + LangGraph 分支 + citation 解析
