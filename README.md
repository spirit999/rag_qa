# 企业制度智能问答与检索系统

项目包含制度文档解析、Query Rewrite、BM25/向量混合检索、Rerank、上下文构建、答案生成、引用校验、聊天 API 和增量索引更新。

## 运行环境

- Python 3.10+
- 本地 Qwen3-8B 与 Qwen3-Embedding-0.6B 模型，或兼容 OpenAI Chat Completions 的模型服务
- 处理 DOC/PDF 时需要 LibreOffice 和 `pdftotext`

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

模型路径配置见 `query-rewrite/qr/config.py` 和 `retrieval/hr/config.py`。如使用 OpenAI 兼容模型服务：

```bash
export QR_BACKEND=openai
export QR_BASE_URL=http://127.0.0.1:8000/v1
export QR_MODEL=/path/to/model
```

## 启动问答服务

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

请确保 `python` 指向项目 Conda 环境；可用 `conda run -n wzc_event python -m uvicorn ...` 显式指定环境。避免直接调用可能来自其他 Python 环境的全局 `uvicorn` 脚本。

浏览器打开 `http://127.0.0.1:8000/`。接口文档在 `/docs`。

- `POST /api/chat`：问答，返回答案、条款引用和各阶段耗时
- `POST /api/chat/stream`：流式问答；生成中的答案会标记为待校验，完成后以引用校验结果为准
- `POST /api/rewrite`：调试 Query Rewrite
- `POST /api/retrieve`：查看检索上下文（管理员接口）
- `POST /api/feedback`：提交 `1`（有帮助）或 `-1`（不准确）
- `GET /api/health`：服务和索引健康状态
- `GET /api/stats`：最近请求统计（管理员接口）
- `POST /api/index/update`：增量更新文档与检索索引（管理员接口）

`/api/chat` 对命中术语且表达明确的问题直接使用原问题检索；包含指代、追问或未识别术语的问题仍调用 LLM 改写。`/api/rewrite` 保留强制 LLM 改写，便于调试。

管理接口默认关闭。启用前设置长随机令牌，并通过 `X-Admin-Token` 请求头传入：

```bash
export RAG_ADMIN_TOKEN='replace-with-a-long-random-secret'
```

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/index/update \
  -H "X-Admin-Token: $RAG_ADMIN_TOKEN"
```

## 增量更新索引

命令行执行：

```bash
python retrieval/scripts/build_index.py
```

更新流程仅重新解析哈希变化的文档，复用内容未变化的 Chunk 向量，重建 BM25，并将完整索引写入新 generation。校验成功后通过原子更新 `retrieval/data/current_index.json` 切换索引；失败时保留上一个可用索引。更新报告会列出变更文档、向量复用数量和解析错误。

`--skip-embed` 仅适用于所有 Chunk 都可复用现有向量的情况；若有新内容，该命令会显式失败且不切换索引。

旧索引 generation 会保留，便于排障和回滚；确认无回滚需求后再手动清理旧 generation。

## 可观测性与隐私

请求统计保存在 `data/observability.sqlite3`，包括分阶段延迟、回答状态、引用数量与反馈评分。默认不保存原始问题、对话历史或答案正文。聊天接口只返回答案引用，不返回完整检索上下文；检索调试接口需要管理员令牌。

## 离线测试

```bash
python query-rewrite/scripts/test_offline.py
pip install pytest
python -m pytest -q tests
```
