# 企业制度智能问答与检索系统

项目包含制度文档解析、Query Rewrite、BM25/向量混合检索、Rerank、上下文构建、答案生成、引用校验、聊天 API 和增量索引更新。

## 运行环境

- Python 3.10+
- 本地 Qwen3-8B 与 Qwen3-Embedding-0.6B 模型，或兼容 OpenAI Chat Completions 的模型服务
- 处理 DOC/PDF 时需要 LibreOffice 和 `pdftotext`

HF 改写后端在 CUDA 可用时默认将 Qwen3-8B 固定加载到逻辑 GPU 0。若要指定其他物理 GPU，可在启动服务前设置 `CUDA_VISIBLE_DEVICES`；例如 `CUDA_VISIBLE_DEVICES=4` 会将物理 GPU 4 映射为逻辑 GPU 0。

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

模型路径配置见 `query-rewrite/qr/config.py` 和 `retrieval/hr/config.py`。如使用 OpenAI 兼容模型服务：

```bash
export QR_BACKEND=openai
export QR_BASE_URL=http://127.0.0.1:8001/v1
export QR_MODEL=Qwen3-8B
```

如在本机用 vLLM 提供模型服务，可单独创建其运行环境并启动服务（vLLM 不需要安装到本项目环境）：

```bash
conda activate wzc_rag
"$CONDA_PREFIX/bin/vllm" serve /home/ps/Model/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --enable-prefix-caching
```

随后在项目环境终端配置并启动 API：

```bash
conda activate wzc_event
export QR_BACKEND=openai
export QR_BASE_URL=http://127.0.0.1:8001/v1
export QR_MODEL=Qwen3-8B
export QR_DISABLE_THINKING=1
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

`QR_DISABLE_THINKING=1` 会在 OpenAI 兼容请求中向 vLLM 传递 `enable_thinking=false`，避免 Qwen3 生成额外的思考文本；其他 OpenAI 兼容后端默认不传此参数。推理服务与问答 API 使用不同端口。项目默认答案最大生成长度为 512 token，可用 `QR_MAX_NEW_TOKENS` 调整。部署后可通过 `/api/chat/stream` 观察生成中的答案；流式文本在最终引用校验完成前会标记为待校验。

## 并发与小模型级联

问答请求不再由一把全局锁串行化：检索、校验和 CPU 阶段可以并行；本地 HF 模型的 GPU 前向由共享锁保护，避免同一张卡上的模型互相争抢显存。OpenAI 兼容后端不在应用内加锁，由 vLLM/SGLang 做连续批处理。索引更新使用独立锁，进行更新时请求继续读旧索引，发布后原子切换到新索引。

Qwen2.5-0.5B-Instruct 已配置为可选的小模型答案路由，默认关闭，避免未完成人工准确率评估就改变生产答案。启用前设置：

```bash
export QR_SMALL_MODEL_MODE=cascade
export QR_SMALL_MODEL_PATH=/home/ps/wuzhichao/base_model/qwen2.5-0.5B-Instruct
```

级联只尝试处理未指定版本、要求一个简单数字、检索上下文未截断且候选条款只有一个明确数字的问题。小模型答案必须通过现有数字与引用校验；无法解析、拒答或校验失败时回退到主模型。响应会带 `generation_route`（`small`、`primary` 或 `primary_fallback`）。现有校验不能证明所有自然语言结论都语义正确，因此应先用实际问题集对照大模型和用户反馈，再启用级联。

## 推理优化与量化

本地 Transformers 推理默认使用 PyTorch SDPA（`QR_ATTENTION_IMPLEMENTATION=sdpa`）、BF16 和 KV Cache。可选择 `eager` 或 `flash_attention_2`；后者需要额外安装匹配当前 PyTorch/CUDA 的 FlashAttention。

BitsAndBytes 量化是可选项，默认为不量化：

```bash
export QR_QUANTIZATION=4bit  # 或 8bit；恢复默认可设为 none
```

量化可能影响答案质量，需先在相同问题集上验证引用通过率、拒答率和负反馈率。量化模式要求 CUDA 与 bitsandbytes 原生库兼容；依赖加载失败会明确报错，不会静默退回未量化模型。

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

请求统计保存在 `data/observability.sqlite3`，包括分阶段延迟、锁等待、pipeline 加载时间、改写/答案缓存命中、回答状态、引用数量与反馈评分。改写缓存和答案缓存仅保存在进程内，不写入数据库；答案缓存最多保留 128 条，仅缓存无历史、当前索引版本下通过引用校验且非低置信度的答案。索引 generation 切换后缓存失效；收到负反馈时清空当前进程的答案缓存。默认不持久化原始问题、对话历史或答案正文。聊天接口只返回答案引用，不返回完整检索上下文；检索调试接口需要管理员令牌。

## 离线测试

```bash
python query-rewrite/scripts/test_offline.py
pip install pytest
python -m pytest -q tests
```
