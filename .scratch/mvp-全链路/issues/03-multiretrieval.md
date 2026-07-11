# 03 — 多路检索(Milvus hybrid · HyDE · RRF · rerank 二阶)

## What to build
三路并行(原始 dense+sparse 混合 α + HyDE + 路3 stub 空)+ RRF 融合 + rerank 精排。网格调参 α/k 直接环境变量生效;召回/precision 可量化对比 T2 基线。

## Blocked by
#02 — 核心 RAG 环路

## Status
ready-for-agent

## Acceptance criteria
- [ ] ≥20 手写 QA 对 seed → Milvus → 召回命中率基线比 T2 dense-only 升 ≥ 相对 15%(示 hybrid/RRF/rerank 真起作用)
- [ ] 改 `.env` 的 `ALPHA` 从 0→1,分数边界切换对(全 dense / 全 sparse)
- [ ] RRF 公式独立 pytest,手算 k=60 与代码同
- [ ] 路3 当前 stub 空,但 RRF 求和不报错、结果仍可用
- [ ] 手写同一 query 换 rerank 模型,精排相对序保序
