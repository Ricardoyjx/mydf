# my-df RAG 功能整理：调用链与路线图

> 整理时间：2026-08-22，基于当前代码快照。

## 一、RAG 如何被调用

### 1. 两个入口

**入口 A：对话内自动检索（主路径）**

```
用户发消息
  -> POST /api/threads/{id}/runs（runs worker 启动）
  -> supervisor 图 model_call 节点
  -> MemoryMiddleware.abefore_model（记忆，content_type=conversation）
  -> RagMiddleware.abefore_model（知识库，content_type=knowledge）
      |- get_latest_human_text 取用户最新问题
      |- KnowledgeService.search()（混合检索管线）
      `- _format_rag_context() -> 注入 <rag_context> 到首条 HumanMessage
  -> LLM 基于注入的上下文回答
```

**入口 B：API 直查（知识库管理/调试）** —— `backend/app/gateway/routers/rag.py`

| 接口 | 作用 |
|---|---|
| `POST /api/rag/documents` | 纯文本入库 |
| `POST /api/rag/documents/upload` | JSON 文本文件入库 |
| `POST /api/rag/files` | base64 二进制（docx/txt/md）入库 |
| `GET /api/rag/search` | 语义检索（q + user_id + top_k + min_score） |
| `POST /api/rag/search` | POST 版语义检索 |
| `GET /api/rag/documents` | 文档列表（按 document_id 聚合） |
| `DELETE /api/rag/documents/{id}` | 删除文档（含父块） |

### 2. 检索核心管线

`backend/packages/harness/my_df/rag/service.py` -> `KnowledgeService.search(query, top_k, min_score)`

```
KnowledgeService.search(query, top_k, min_score)
|
|- 分支1：small-to-big 开启（MYDF_RAG_SMALL_TO_BIG=true）
|   子块向量召回(top_k*4) -> 按 parent_id 去重 -> PG/内存 docstore 取父块
|   -> rerank -> min_score -> top_k（不参与 RRF）
|
`- 分支2：flat 模式（当前 .env）
    |
    |- 向量路：embedding(query) -> Milvus embedding_search(top_k*4)  IP 相似度
    |- BM25 路：Milvus 原生全文搜索 bm25_search(top_k*4)（sparse 索引，中文分词）
    |
    |- 两路都有命中 -> rrf_fuse(k=60) 按排名融合去重（分数约 1/(60+rank)）
    |- 仅一路命中   -> 保留该路原始分（不融合，避免稀释）
    |
    `- rerank（bge-reranker 懒加载，覆盖分数）
       -> min_score（只在“分数可比”时生效：重排成功或纯原始分）
       -> top_k 截断
```

### 3. 写入链路

```
上传文件 -> extract_text_from_file（docx 解 XML/w:t；txt/md 按 UTF-8）
  -> split_text 分块（段落优先 + 长段滑动窗口 + overlap）
  -> embedding.encode_batch
  -> Milvus insert（稠密 vector + text 字段，BM25 函数自动生成 sparse）
  -> metadata: document_id / title / source / chunk_index
```

### 4. 关键开关（`.env`）

| 配置 | 作用 |
|---|---|
| `MYDF_EMBEDDING_MODEL` | 向量模型（paraphrase-multilingual-MiniLM-L12-v2，384 维） |
| `MYDF_RERANK_ENABLED` | 重排开关（bge-reranker-base） |
| `MYDF_RAG_RRF_ENABLED` | 混合检索 + RRF 融合开关 |
| `MYDF_RAG_SMALL_TO_BIG` | 父子块模式（需 PG；与 RRF 互斥） |
| `MYDF_MILVUS_*` | Milvus 连接/集合/索引 |

### 5. 调用链上的注意点

1. **子代理不自动携带 RAG 上下文**：只有 supervisor 自己的模型调用会被注入 `<rag_context>`；委派给 general-purpose 时，任务描述里是否包含知识库内容取决于模型自己组织，系统不保证传递。
2. **small-to-big 与 RRF 互斥**：开启父子块就走不到混合检索，这是当前设计预期。
3. **旧集合无 BM25 Schema**：`bm25_search` 降级为空，检索自动退化为纯向量（日志有 WARNING 和“RRF 跳过融合”提示）。

## 二、RAG Roadmap

### 已完成

| 阶段 | 内容 |
|---|---|
| V1 基础链路 | 文件解析（docx/txt）-> 分块 -> 向量化 -> Milvus 入库/检索/删除；知识库管理 API + 前端页面 |
| V2 检索质量 | bge-reranker 精排（懒加载）、min_score、small-to-big 父子块（PG 父块 + Milvus 子块）、检索降级保护 |
| V3 混合检索 | Milvus 原生 BM25（中文 analyzer）+ 向量双路召回、RRF 融合、融合/降级日志、重排失败与分数不可比时的 min_score 保护 |

### 下一步建议（按优先级）

**P0（体验/正确性）**

1. **旧集合迁移**：提供一键重建集合脚本或自动检测升级，解决当前 `BM25召回=0`；或者把内存 jieba BM25 接成旧集合降级第二路。
2. **子代理 RAG 上下文传递**：委派时把 `<rag_context>` 摘要随 task 传给子代理，保证知识问答委派后不丢上下文。

**P1（检索质量）**

3. **small-to-big + RRF 融合**：以 `parent_id` 为去重键，让父子块模式也能双路召回融合。
4. **评估体系**：建 eval（Recall@K / MRR / NDCG），对比 向量-only vs 向量+BM25+RRF vs +rerank 三档效果，用数据决定参数（k、top_k*4 粗召回倍数）。
5. **min_score 语义统一**：放开 API 上限或按重排分数归一化，让前端阈值可调可控。

**P2（工程化）**

6. list_documents 按文档分页（当前按 chunk 分页会截断大文档）。
7. delete_document 去掉 1 万条上限，改用过滤表达式批量删。
8. embedding/rerank 模型热切换、集合增量重建。
9. 多租户：集合级/文档级权限与 user_id 隔离强化。

## 三、审查时发现的已知问题（对照开发）

| 级别 | 问题 | 位置 |
|---|---|---|
| P1（已修） | 重排失败时 min_score 把结果清空 | service.py `_finalize_search` |
| P1（待修） | RagMiddleware 双重检索（一次 encode+search 浪费） | rag_middleware.py |
| P1（已修） | 向量空 + BM25 有命中时不对称处理 | service.py search |
| P2 | `bm25.py`（内存 jieba BM25）已成死代码 | rag/bm25.py |
| P2 | list_documents 按 chunk 分页，大文档截断 | service.py |
| P2 | delete_document 上限 10_000 chunks | service.py |
| P2 | API min_score 上限 1.0 与 rerank logits 不匹配 | routers/rag.py |
| P3 | `hybrid_search` 空壳方法残留 | milvus/client.py |
