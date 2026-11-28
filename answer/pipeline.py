from collections import OrderedDict
import copy
import json
import threading
import time
import unicodedata
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'retrieval'))
sys.path.insert(0, str(PROJECT / 'query-rewrite'))

from answer.generator import AnswerGenerator
from answer.verifier import AnswerVerifier
from hr.context_builder import build_context
from qr.config import SMALL_MODEL_MODE
from qr.llm import make_llm, make_small_llm
from qr.rewrite import QueryRewriter
from hr.embedder import QwenEmbedder
from hr.reranker import QwenReranker
from hr.searcher import HybridSearcher
from hr.config import INDEX_CURRENT_PATH, RERANK_MODEL_PATH


class RAGPipeline:
    def __init__(self, backend=None, load_reranker=True, load_dense=True,
                 llm=None, embedder=None, reranker=None, searcher=None,
                 small_llm=None, small_model_mode=None):
        self.gpu_inference_lock = threading.RLock()
        self.index_update_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self.llm = llm or make_llm(
            backend, inference_lock=self.gpu_inference_lock,
        )
        self.small_model_mode = small_model_mode or SMALL_MODEL_MODE
        if self.small_model_mode not in ('off', 'cascade'):
            raise ValueError('small_model_mode must be "off" or "cascade"')
        if small_llm is None and self.small_model_mode == 'cascade':
            small_llm = make_small_llm(
                inference_lock=self.gpu_inference_lock,
            )
        self.rewriter = QueryRewriter(llm=self.llm)
        self.answer_generator = AnswerGenerator(
            self.llm, small_llm=small_llm,
            small_model_mode=self.small_model_mode,
        )
        self.verifier = AnswerVerifier()
        self.embedder = embedder or (
            QwenEmbedder(inference_lock=self.gpu_inference_lock)
            if load_dense else None
        )
        self.reranker = reranker
        if self.reranker is None and load_reranker:
            reranker_path = Path(RERANK_MODEL_PATH)
            complete_weight = (
                (reranker_path / 'model.safetensors').exists()
                or bool(list(reranker_path.glob('model-*.safetensors')))
                or (reranker_path / 'pytorch_model.bin').exists()
            )
            if complete_weight:
                try:
                    self.reranker = QwenReranker(
                        inference_lock=self.gpu_inference_lock,
                    )
                except Exception as exc:
                    print('Reranker 加载失败，使用未重排 Hybrid 结果：%r' % exc)
        self.searcher = searcher or HybridSearcher.load(embedder=self.embedder)
        self._active_generation = self._index_generation()
        self._answer_cache = OrderedDict()
        self._answer_cache_limit = 128
        self._answer_cache_lock = threading.RLock()

    def run(self, question, history=None, topk_chunks=50, topk_docs=5,
            mode='hybrid', rerank=True, max_context_chars=6000):
        history = history or []
        request_start = time.perf_counter()
        searcher, generation = self._searcher_snapshot()
        result = self._run_request(
            question, history, topk_chunks, topk_docs, mode,
            rerank, max_context_chars, searcher, generation,
        )
        timings = result.setdefault('timings_ms', {})
        timings['lock_wait'] = 0
        timings['total'] = round((time.perf_counter() - request_start) * 1000, 2)
        return result

    def _searcher_snapshot(self):
        with self._index_lock:
            return self.searcher, self._active_generation

    def _index_generation(self):
        pointer = Path(INDEX_CURRENT_PATH)
        if not pointer.exists():
            return 'legacy'
        with pointer.open(encoding='utf-8') as f:
            return json.load(f).get('generation', 'unknown')

    def _answer_cache_key(self, question, history, topk_chunks, topk_docs,
                          mode, rerank, max_context_chars, generation):
        if history:
            return None
        return json.dumps([
            'answer-v1',
            generation,
            unicodedata.normalize('NFC', question).strip(),
            topk_chunks,
            topk_docs,
            mode,
            bool(rerank),
            max_context_chars,
        ], ensure_ascii=False, separators=(',', ':'))

    def _answer_cache_get(self, key):
        if key is None:
            return None
        with self._answer_cache_lock:
            cached = self._answer_cache.get(key)
            if cached is None:
                return None
            self._answer_cache.move_to_end(key)
            result = copy.deepcopy(cached)
        timings = result.setdefault('timings_ms', {})
        for stage in ('rewrite', 'retrieval', 'rerank', 'generation', 'verification'):
            timings[stage] = 0
        result['cache'] = {
            'answer_cache_hit': True,
            'rewrite_cache_hit': False,
        }
        return result

    def _answer_cache_put(self, key, result):
        if key is None:
            return
        answer = result.get('answer') or {}
        verification = answer.get('verification') or {}
        if (
            answer.get('status') != 'answered'
            or answer.get('confidence') == 'low'
            or not verification.get('passed')
            or not answer.get('citations')
        ):
            return
        with self._answer_cache_lock:
            self._answer_cache[key] = copy.deepcopy(result)
            self._answer_cache.move_to_end(key)
            if len(self._answer_cache) > self._answer_cache_limit:
                self._answer_cache.popitem(last=False)

    def clear_answer_cache(self):
        with self._answer_cache_lock:
            self._answer_cache.clear()

    @staticmethod
    def _small_answer_passed(answer):
        return (
            answer.status == 'answered'
            and answer.confidence != 'low'
            and answer.verification.passed
            and bool(answer.citations)
        )

    def _small_context(self, question, rewrite, context):
        build_small_context = getattr(
            self.answer_generator, 'small_context', None,
        )
        if build_small_context is None:
            return None
        return build_small_context(question, rewrite, context)

    def _generate_verified(self, question, rewrite, context):
        small_context = self._small_context(question, rewrite, context)
        generation_ms = 0.0
        verification_ms = 0.0
        if small_context is not None:
            started = time.perf_counter()
            raw = self.answer_generator.generate_small(
                question, rewrite, small_context,
            )
            generation_ms += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            answer = self.verifier.verify(raw, small_context)
            verification_ms += (time.perf_counter() - started) * 1000
            if self._small_answer_passed(answer):
                return raw, answer, generation_ms, verification_ms, 'small'

        started = time.perf_counter()
        raw = self.answer_generator.generate(question, rewrite, context)
        generation_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        answer = self.verifier.verify(raw, context)
        verification_ms += (time.perf_counter() - started) * 1000
        route = 'primary_fallback' if small_context is not None else 'primary'
        return raw, answer, generation_ms, verification_ms, route

    def _run_request(self, question, history, topk_chunks, topk_docs,
                     mode, rerank, max_context_chars, searcher, generation):
        total_start = time.perf_counter()
        cache_key = self._answer_cache_key(
            question, history, topk_chunks, topk_docs,
            mode, rerank, max_context_chars, generation,
        )
        cached = self._answer_cache_get(cache_key)
        if cached is not None:
            cached['timings_ms']['total'] = round(
                (time.perf_counter() - total_start) * 1000, 2,
            )
            return cached

        rewrite_start = time.perf_counter()
        rewrite = self.rewriter.rewrite(question, history=history)
        rewrite_ms = (time.perf_counter() - rewrite_start) * 1000
        if not rewrite.need_retrieval:
            from answer.schema import AnswerResult, VerificationResult
            result = AnswerResult(
                answer='该问题不需要检索公司制度，或不属于制度知识库可回答范围。',
                status='insufficient_context',
                confidence='high',
                verification=VerificationResult(
                    passed=True,
                    should_refuse=True,
                ),
            )
            result = {
                'rewrite': rewrite.to_dict(),
                'context': None,
                'answer': result.to_dict(),
                'timings_ms': {
                    'rewrite': round(rewrite_ms, 2), 'retrieval': 0,
                    'rerank': 0, 'generation': 0, 'verification': 0,
                    'total': round((time.perf_counter() - total_start) * 1000, 2),
                },
                'cache': {
                    'answer_cache_hit': False,
                    'rewrite_cache_hit': rewrite.from_cache,
                },
            }
            return result

        retrieve_start = time.perf_counter()
        retrieval_timings = {}
        context = searcher.retrieve_context(
            build_context,
            self.reranker if rerank else None,
            question,
            rewrite.keywords,
            rewrite.standalone_query,
            rewrite.sub_queries,
            topk_chunks=topk_chunks,
            topk_docs=topk_docs,
            mode=mode,
            rerank=rerank,
            max_context_chars=max_context_chars,
            timing_out=retrieval_timings,
        )
        retrieval_ms = retrieval_timings.get(
            'retrieval', (time.perf_counter() - retrieve_start) * 1000,
        )
        rerank_ms = retrieval_timings.get('rerank', 0)
        raw, answer, generation_ms, verification_ms, generation_route = (
            self._generate_verified(
                question, rewrite.to_dict(), context,
            )
        )
        result = {
            'rewrite': rewrite.to_dict(),
            'context': context.to_dict(),
            'answer': answer.to_dict(),
            'generation_route': generation_route,
            'timings_ms': {
                'rewrite': round(rewrite_ms, 2),
                'retrieval': round(retrieval_ms, 2),
                'rerank': round(rerank_ms, 2),
                'generation': round(generation_ms, 2),
                'verification': round(verification_ms, 2),
                'total': round((time.perf_counter() - total_start) * 1000, 2),
            },
            'cache': {
                'answer_cache_hit': False,
                'rewrite_cache_hit': rewrite.from_cache,
            },
        }
        self._answer_cache_put(cache_key, result)
        return result

    def run_stream(self, question, history=None, topk_chunks=50, topk_docs=5,
                   mode='hybrid', rerank=True, max_context_chars=6000):
        history = history or []
        request_start = time.perf_counter()
        total_start = request_start
        searcher, generation = self._searcher_snapshot()
        yield {'type': 'stage', 'stage': 'rewrite'}
        cache_key = self._answer_cache_key(
            question, history, topk_chunks, topk_docs,
            mode, rerank, max_context_chars, generation,
        )
        cached = self._answer_cache_get(cache_key)
        if cached is not None:
            cached['timings_ms']['lock_wait'] = 0
            cached['timings_ms']['total'] = round(
                (time.perf_counter() - total_start) * 1000, 2,
            )
            yield {'type': 'complete', 'result': cached}
            return

        rewrite_start = time.perf_counter()
        rewrite = self.rewriter.rewrite(question, history=history)
        rewrite_ms = (time.perf_counter() - rewrite_start) * 1000
        if not rewrite.need_retrieval:
            from answer.schema import AnswerResult, VerificationResult
            result = {
                'rewrite': rewrite.to_dict(),
                'context': None,
                'answer': AnswerResult(
                    answer='该问题不需要检索公司制度，或不属于制度知识库可回答范围。',
                    status='insufficient_context',
                    confidence='high',
                    verification=VerificationResult(
                        passed=True, should_refuse=True,
                    ),
                ).to_dict(),
                'timings_ms': {
                    'rewrite': round(rewrite_ms, 2),
                    'retrieval': 0,
                    'rerank': 0,
                    'generation': 0,
                    'verification': 0,
                    'lock_wait': 0,
                    'total': round(
                        (time.perf_counter() - total_start) * 1000, 2,
                    ),
                },
                'cache': {
                    'rewrite_cache_hit': rewrite.from_cache,
                    'answer_cache_hit': False,
                },
            }
            yield {'type': 'complete', 'result': result}
            return

        yield {'type': 'stage', 'stage': 'retrieval'}
        retrieval_timings = {}
        context = searcher.retrieve_context(
            build_context,
            self.reranker if rerank else None,
            question,
            rewrite.keywords,
            rewrite.standalone_query,
            rewrite.sub_queries,
            topk_chunks=topk_chunks,
            topk_docs=topk_docs,
            mode=mode,
            rerank=rerank,
            max_context_chars=max_context_chars,
            timing_out=retrieval_timings,
        )
        retrieval_ms = retrieval_timings.get('retrieval', 0)
        rerank_ms = retrieval_timings.get('rerank', 0)

        yield {'type': 'stage', 'stage': 'generation'}
        rewrite_data = rewrite.to_dict()
        small_context = self._small_context(
            question, rewrite_data, context,
        )
        generation_ms = 0.0
        verification_ms = 0.0
        generation_route = 'primary'
        raw = None
        answer = None
        if small_context is not None:
            generation_start = time.perf_counter()
            small_raw = self.answer_generator.generate_small(
                question, rewrite_data, small_context,
            )
            generation_ms += (time.perf_counter() - generation_start) * 1000
            verification_start = time.perf_counter()
            small_answer = self.verifier.verify(small_raw, small_context)
            verification_ms += (
                time.perf_counter() - verification_start
            ) * 1000
            if self._small_answer_passed(small_answer):
                raw = small_raw
                answer = small_answer
                generation_route = 'small'
                yield {'type': 'delta', 'text': answer.answer}
            else:
                generation_route = 'primary_fallback'

        if answer is None:
            generation_start = time.perf_counter()
            for event in self.answer_generator.generate_stream(
                    question, rewrite_data, context):
                if event['type'] == 'delta':
                    yield event
                else:
                    raw = event['raw']
            generation_ms += (time.perf_counter() - generation_start) * 1000
            if raw is None:
                raise RuntimeError('流式答案生成没有返回最终结果。')

            verification_start = time.perf_counter()
            answer = self.verifier.verify(raw, context)
            verification_ms += (
                time.perf_counter() - verification_start
            ) * 1000

        result = {
            'rewrite': rewrite_data,
            'context': context.to_dict(),
            'answer': answer.to_dict(),
            'generation_route': generation_route,
            'timings_ms': {
                'rewrite': round(rewrite_ms, 2),
                'retrieval': round(retrieval_ms, 2),
                'rerank': round(rerank_ms, 2),
                'generation': round(generation_ms, 2),
                'verification': round(verification_ms, 2),
                'lock_wait': 0,
                'total': round(
                    (time.perf_counter() - total_start) * 1000, 2,
                ),
            },
            'cache': {
                'rewrite_cache_hit': rewrite.from_cache,
                'answer_cache_hit': False,
            },
        }
        self._answer_cache_put(cache_key, result)
        yield {'type': 'complete', 'result': result}

    def rewrite_only(self, question, history=None):
        request_start = time.perf_counter()
        rewrite = self.rewriter.rewrite(
            question, history=history or [], force_llm=True,
        )
        return {
            'rewrite': rewrite.to_dict(),
            'timings_ms': {
                'rewrite': round(
                    (time.perf_counter() - request_start) * 1000, 2,
                ),
                'lock_wait': 0,
                'total': round(
                    (time.perf_counter() - request_start) * 1000, 2,
                ),
            },
        }

    def retrieve_only(self, question, rewrite, topk_chunks=50, topk_docs=5,
                      mode='hybrid', rerank=True, max_context_chars=6000):
        request_start = time.perf_counter()
        searcher, _ = self._searcher_snapshot()
        timings = {}
        context = searcher.retrieve_context(
            build_context,
            self.reranker if rerank else None,
            question,
            rewrite.get('keywords', []),
            rewrite.get('standalone_query', question),
            rewrite.get('sub_queries', []),
            topk_chunks=topk_chunks,
            topk_docs=topk_docs,
            mode=mode,
            rerank=rerank,
            max_context_chars=max_context_chars,
            timing_out=timings,
        )
        result = {
            'context': context.to_dict(),
            'timings_ms': dict(timings),
        }
        if not timings:
            result['timings_ms'].update({
                'retrieval': round(
                    (time.perf_counter() - request_start) * 1000, 2,
                ),
                'rerank': 0,
            })
        result['timings_ms']['lock_wait'] = 0
        result['timings_ms']['total'] = round(
            (time.perf_counter() - request_start) * 1000, 2,
        )
        return result

    def reload_index(self):
        with self.index_update_lock:
            new_searcher = HybridSearcher.load(embedder=self.embedder)
            generation = self._index_generation()
            with self._index_lock:
                self.searcher = new_searcher
                self._active_generation = generation
                self.clear_answer_cache()
            return {
                'generation': generation,
                'n_chunks': len(new_searcher.chunks),
            }

    def health(self):
        searcher, generation = self._searcher_snapshot()
        return {
            'index_generation': generation,
            'n_chunks': len(searcher.chunks),
            'dense_available': searcher.vectors is not None,
            'reranker_available': self.reranker is not None,
            'small_model_mode': self.small_model_mode,
        }
