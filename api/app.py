from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'retrieval'))

from api.telemetry import EventStore

logger = logging.getLogger('agent_rag.api')
STATIC_DIR = PROJECT / 'api' / 'static'
_pipeline = None
_event_store = None


@asynccontextmanager
async def lifespan(_app):
    global _event_store
    _event_store = EventStore()
    yield


app = FastAPI(
    title='Enterprise Regulation RAG',
    version='1.1',
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: Optional[List[Annotated[str, Field(max_length=4000)]]] = Field(
        default=None, max_length=20,
    )
    topk_chunks: int = Field(default=50, ge=1, le=200)
    topk_docs: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default='hybrid', pattern='^(bm25|dense|hybrid)$')
    rerank: bool = True


class RewriteRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: Optional[List[Annotated[str, Field(max_length=4000)]]] = Field(
        default=None, max_length=20,
    )


class RetrieveRequest(ChatRequest):
    rewrite: Optional[dict] = None


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    rating: int = Field(ge=-1, le=1)


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        from answer.pipeline import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline


def _load_pipeline_timed():
    started = time.perf_counter()
    pipeline = _load_pipeline()
    return pipeline, (time.perf_counter() - started) * 1000


def _admin_required(token):
    expected = os.environ.get('RAG_ADMIN_TOKEN')
    if not expected:
        raise HTTPException(
            status_code=503,
            detail='管理接口未启用；请配置 RAG_ADMIN_TOKEN。',
        )
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail='管理令牌无效。')


def _record_result(request_id, route, started, result):
    answer = result.get('answer') or {}
    rewrite = result.get('rewrite') or {}
    context = result.get('context') or {}
    timings = result.get('timings_ms') or {}
    cache = result.get('cache') or {}
    _event_store.record_request({
        'request_id': request_id,
        'route': route,
        'status': answer.get('status', 'ok'),
        'duration_ms': (time.perf_counter() - started) * 1000,
        'rewrite_ms': timings.get('rewrite', 0),
        'lock_wait_ms': timings.get('lock_wait', 0),
        'pipeline_load_ms': timings.get('pipeline_load', 0),
        'retrieval_ms': timings.get('retrieval', 0),
        'rerank_ms': timings.get('rerank', 0),
        'generation_ms': timings.get('generation', 0),
        'verification_ms': timings.get('verification', 0),
        'rewrite_cache_hit': cache.get('rewrite_cache_hit', False),
        'answer_cache_hit': cache.get('answer_cache_hit', False),
        'document_count': len(context.get('documents', [])),
        'citation_count': len(answer.get('citations', [])),
    })


def _record_error(request_id, route, started, exc):
    if _event_store:
        _event_store.record_request({
            'request_id': request_id,
            'route': route,
            'status': 'error',
            'duration_ms': (time.perf_counter() - started) * 1000,
            'error_type': type(exc).__name__,
        })


def _sanitize_rewrite(result):
    rewrite = result.get('rewrite')
    if isinstance(rewrite, dict):
        rewrite.pop('raw', None)
    return result


@app.get('/')
def home():
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/api/health')
def health():
    if _pipeline is None:
        return {'status': 'starting', 'pipeline_loaded': False}
    return {
        'status': 'ok',
        'pipeline_loaded': True,
        **_pipeline.health(),
    }


@app.post('/api/chat')
def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        pipeline, pipeline_load_ms = _load_pipeline_timed()
        result = pipeline.run(
            question=request.question,
            history=request.history or [],
            topk_chunks=request.topk_chunks,
            topk_docs=request.topk_docs,
            mode=request.mode,
            rerank=request.rerank,
        )
        result.setdefault('timings_ms', {})['pipeline_load'] = round(
            pipeline_load_ms, 2,
        )
        result['timings_ms']['total'] = round(
            (time.perf_counter() - started) * 1000, 2,
        )
        result['request_id'] = request_id
        _sanitize_rewrite(result)
        _record_result(request_id, '/api/chat', started, result)
        result.pop('context', None)
        return result
    except Exception as exc:
        logger.exception('chat request failed request_id=%s', request_id)
        _record_error(request_id, '/api/chat', started, exc)
        raise HTTPException(status_code=503, detail='问答服务暂时不可用。') from exc


@app.post('/api/chat/stream')
def chat_stream(request: ChatRequest):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    def events():
        try:
            yield 'data: %s\n\n' % json.dumps(
                {'type': 'stage', 'stage': 'starting'}, ensure_ascii=False,
            )
            pipeline, pipeline_load_ms = _load_pipeline_timed()
            stream = pipeline.run_stream(
                question=request.question,
                history=request.history or [],
                topk_chunks=request.topk_chunks,
                topk_docs=request.topk_docs,
                mode=request.mode,
                rerank=request.rerank,
            )
            for event in stream:
                if event['type'] == 'complete':
                    result = event['result']
                    result.setdefault('timings_ms', {})['pipeline_load'] = round(
                        pipeline_load_ms, 2,
                    )
                    result['timings_ms']['total'] = round(
                        (time.perf_counter() - started) * 1000, 2,
                    )
                    result['request_id'] = request_id
                    _sanitize_rewrite(result)
                    _record_result(request_id, '/api/chat/stream', started, result)
                    result.pop('context', None)
                    event = {'type': 'done', 'result': result}
                yield 'data: %s\n\n' % json.dumps(event, ensure_ascii=False)
                if event['type'] == 'done':
                    return
            raise RuntimeError('流式问答没有返回最终结果。')
        except Exception as exc:
            logger.exception('streaming chat request failed request_id=%s', request_id)
            _record_error(request_id, '/api/chat/stream', started, exc)
            yield 'data: %s\n\n' % json.dumps({
                'type': 'error',
                'message': '问答服务暂时不可用，请稍后重试。',
            }, ensure_ascii=False)

    return StreamingResponse(
        events(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.post('/api/rewrite')
def rewrite(request: RewriteRequest):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        pipeline, pipeline_load_ms = _load_pipeline_timed()
        result = pipeline.rewrite_only(
            request.question, history=request.history or [],
        )
        timings = result.setdefault('timings_ms', {})
        timings['pipeline_load'] = round(pipeline_load_ms, 2)
        timings['total'] = round((time.perf_counter() - started) * 1000, 2)
        result['request_id'] = request_id
        _sanitize_rewrite(result)
        rewrite = result.get('rewrite', {})
        _event_store.record_request({
            'request_id': request_id,
            'route': '/api/rewrite',
            'status': 'ok',
            'duration_ms': timings.get('total', 0),
            'rewrite_ms': timings.get('rewrite', 0),
            'lock_wait_ms': timings.get('lock_wait', 0),
            'pipeline_load_ms': timings.get('pipeline_load', 0),
            'rewrite_cache_hit': rewrite.get('from_cache', False),
        })
        return result
    except Exception as exc:
        logger.exception('rewrite request failed request_id=%s', request_id)
        _record_error(request_id, '/api/rewrite', started, exc)
        raise HTTPException(status_code=503, detail='查询改写服务暂时不可用。') from exc


@app.post('/api/retrieve')
def retrieve(request: RetrieveRequest, x_admin_token: Optional[str] = Header(default=None)):
    _admin_required(x_admin_token)
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        pipeline, pipeline_load_ms = _load_pipeline_timed()
        rewritten = request.rewrite
        rewrite_ms = 0
        rewrite_lock_wait_ms = 0
        if rewritten is None:
            rewrite_result = pipeline.rewrite_only(
                request.question, history=request.history or [],
            )
            rewritten = rewrite_result['rewrite']
            rewrite_ms = rewrite_result.get('timings_ms', {}).get('rewrite', 0)
            rewrite_lock_wait_ms = rewrite_result.get(
                'timings_ms', {},
            ).get('lock_wait', 0)
        result = pipeline.retrieve_only(
            request.question,
            rewritten,
            topk_chunks=request.topk_chunks,
            topk_docs=request.topk_docs,
            mode=request.mode,
            rerank=request.rerank,
        )
        result['request_id'] = request_id
        timings = result.get('timings_ms', {})
        timings['pipeline_load'] = round(pipeline_load_ms, 2)
        timings['lock_wait'] = round(
            timings.get('lock_wait', 0) + rewrite_lock_wait_ms, 2,
        )
        _event_store.record_request({
            'request_id': request_id,
            'route': '/api/retrieve',
            'status': 'ok',
            'duration_ms': (time.perf_counter() - started) * 1000,
            'rewrite_ms': rewrite_ms,
            'lock_wait_ms': timings.get('lock_wait', 0),
            'pipeline_load_ms': pipeline_load_ms,
            'retrieval_ms': timings.get('retrieval', 0),
            'rerank_ms': timings.get('rerank', 0),
            'document_count': len(result.get('context', {}).get('documents', [])),
        })
        return result
    except Exception as exc:
        logger.exception('retrieve request failed request_id=%s', request_id)
        _record_error(request_id, '/api/retrieve', started, exc)
        raise HTTPException(status_code=503, detail='检索服务暂时不可用。') from exc


@app.post('/api/feedback')
def feedback(request: FeedbackRequest):
    if request.rating == 0:
        raise HTTPException(status_code=422, detail='rating 必须为 1 或 -1。')
    if not _event_store.record_feedback(request.request_id, request.rating):
        raise HTTPException(status_code=404, detail='请求记录不存在。')
    if request.rating < 0 and _pipeline is not None:
        _pipeline.clear_answer_cache()
    return {'status': 'recorded'}


@app.get('/api/stats')
def stats(x_admin_token: Optional[str] = Header(default=None)):
    _admin_required(x_admin_token)
    return _event_store.summary()


@app.post('/api/index/update')
def update_index(x_admin_token: Optional[str] = Header(default=None)):
    _admin_required(x_admin_token)
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        pipeline = _load_pipeline()
        from hr.index_manager import IncrementalIndexManager
        with pipeline.index_update_lock:
            result = IncrementalIndexManager().update(pipeline.embedder)
            result['active_index'] = pipeline.reload_index()
        result['request_id'] = request_id
        result['duration_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return result
    except Exception as exc:
        logger.exception('index update failed request_id=%s', request_id)
        if _event_store:
            _event_store.record_request({
                'request_id': request_id,
                'route': '/api/index/update',
                'status': 'error',
                'duration_ms': (time.perf_counter() - started) * 1000,
                'error_type': type(exc).__name__,
            })
        raise HTTPException(
            status_code=500,
            detail='索引更新失败；当前服务继续使用上一个已发布索引。',
        ) from exc


@app.get('/api/stats/health')
def admin_health(x_admin_token: Optional[str] = Header(default=None)):
    _admin_required(x_admin_token)
    return {
        'pipeline_loaded': _pipeline is not None,
        'pipeline': _pipeline.health() if _pipeline else None,
        'telemetry': _event_store.summary() if _event_store else None,
    }
