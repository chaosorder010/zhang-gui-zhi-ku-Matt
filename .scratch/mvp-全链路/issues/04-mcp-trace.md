# 04 — MCP 网络搜索路(路3)· 调试 trace 面板

## What to build
路3 从 stub 真接(MCP client 封装,URL 由 `.env` 配;若不配仍保持 stub 不报错)。前端加"调试"开关:展示本轮各路候选数、RRF 分数、rerank 分数、命中 span。大文件异步入库不扣留问答。

## Blocked by
#03 — 多路检索

## Status
ready-for-agent

## Acceptance criteria
- [ ] 配 MCP URL → 路 3 召回真混进 RRF;不配 → 静默 stub,query 依然成功
- [ ] 前端 debug toggle 开,`/api/query` 返 `retrieval_trace`(含 hyde_text、各路 id/RRF/rerank 分数、命中 span)
- [ ] 上传 500+ 页 PDF,入库期间同时 `POST /api/query` 可正常返回(异步不扣)
- [ ] MCP 路异常(超时/网络) → 自动降级为 stub,query 不受影响,log 里 warn
- [ ] 端到端 pytest:配 MCP URL vs 不配,两分支 query 都成功,trace 路3候选数不一样
