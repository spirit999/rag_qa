import threading
import time
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'retrieval'))
sys.path.insert(0, str(PROJECT / 'query-rewrite'))

from answer.generator import AnswerGenerator
from answer.verifier import AnswerVerifier
from hr.context_builder import build_context
from qr.llm import make_llm
from qr.rewrite import QueryRewriter
from hr.embedder import QwenEmbedder
from hr.reranker import QwenReranker
from hr.searcher import HybridSearcher
from hr.config import INDEX_CURRENT_PATH, RERANK_MODEL_PATH


class RAGPipeline:
    def __init__(self, backend=None, load_reranker=True, load_dense=True,
                 llm=None, embedder=None, reranker=None, searcher=None):
        self.llm = llm or make_llm(backend)
        self.rewriter = QueryRewriter(llm=self.llm)
        self.answer_generator = AnswerGenerator(self.llm)
        self.verifier = AnswerVerifier()
        self.embedder = embedder or (QwenEmbedder() if load_dense else None)
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
                    self.reranker = QwenReranker()
                except Exception as exc:
                    print('Reranker 加载失败，使用未重排 Hybrid 结果：%r' % exc)
        self.searcher = searcher or HybridSearcher.load(embedder=self.embedder)
        self.inference_lock = threading.RLock()

    def run(self, question, history=None, topk_chunks=50, topk_docs=5,
            mode='hybrid', rerank=True, max_context_chars=12000):
        history = history or []
        with self.inference_lock:
            return self._run_locked(
                question, history, topk_chunks, topk_docs, mode,
                rerank, max_context_chars,
            )

    def _index_generation(self):
        pointer = Path(INDEX_CURRENT_PATH)
        if not pointer.exists():
            return 'legacy'
        import json
        with pointer.open(encoding='utf-8') as f:
            return json.load(f).get('generation', 'unknown')

    def _run_locked(self, question, history, topk_chunks, topk_docs,
                    mode, rerank, max_context_chars):
        total_start = time.perf_counter()
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
                    'rewrite_cache_hit': False,
                },
            }
            return result

        retrieve_start = time.perf_counter()
        retrieval_timings = {}
        context = self.searcher.retrieve_context(
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
        generation_start = time.perf_counter()
        raw = self.answer_generator.generate(
            question,
            rewrite.to_dict(),
            context,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000
        verification_start = time.perf_counter()
        answer = self.verifier.verify(raw, context)
        verification_ms = (time.perf_counter() - verification_start) * 1000
        result = {
            'rewrite': rewrite.to_dict(),
            'context': context.to_dict(),
            'answer': answer.to_dict(),
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
                'rewrite_cache_hit': False,
            },
        }
        return result

    def run_stream(self, question, history=None, topk_chunks=50, topk_docs=5,
                   mode='hybrid', rerank=True, max_context_chars=12000):
        history = history or []
        with self.inference_lock:
            total_start = time.perf_counter()
            yield {'type': 'stage', 'stage': 'rewrite'}
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
                        'total': round(
                            (time.perf_counter() - total_start) * 1000, 2,
                        ),
                    },
                    'cache': {
                        'rewrite_cache_hit': False,
                        'answer_cache_hit': False,
                    },
                }
                yield {'type': 'complete', 'result': result}
                return

            yield {'type': 'stage', 'stage': 'retrieval'}
            retrieval_timings = {}
            context = self.searcher.retrieve_context(
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
            generation_start = time.perf_counter()
            raw = None
            for event in self.answer_generator.generate_stream(
                    question, rewrite.to_dict(), context):
                if event['type'] == 'delta':
                    yield event
                else:
                    raw = event['raw']
            generation_ms = (time.perf_counter() - generation_start) * 1000
            if raw is None:
                raise RuntimeError('流式答案生成没有返回最终结果。')

            verification_start = time.perf_counter()
            answer = self.verifier.verify(raw, context)
            verification_ms = (time.perf_counter() - verification_start) * 1000
            result = {
                'rewrite': rewrite.to_dict(),
                'context': context.to_dict(),
                'answer': answer.to_dict(),
                'timings_ms': {
                    'rewrite': round(rewrite_ms, 2),
                    'retrieval': round(retrieval_ms, 2),
                    'rerank': round(rerank_ms, 2),
                    'generation': round(generation_ms, 2),
                    'verification': round(verification_ms, 2),
                    'total': round(
                        (time.perf_counter() - total_start) * 1000, 2,
                    ),
                },
                'cache': {
                    'rewrite_cache_hit': False,
                    'answer_cache_hit': False,
                },
            }
            yield {'type': 'complete', 'result': result}

    def rewrite_only(self, question, history=None):
        with self.inference_lock:
            start = time.perf_counter()
            rewrite = self.rewriter.rewrite(
                question, history=history or [], force_llm=True,
            )
            return {
                'rewrite': rewrite.to_dict(),
                'timings_ms': {
                    'rewrite': round((time.perf_counter() - start) * 1000, 2),
                },
            }

    def retrieve_only(self, question, rewrite, topk_chunks=50, topk_docs=5,
                      mode='hybrid', rerank=True, max_context_chars=12000):
        with self.inference_lock:
            start = time.perf_counter()
            timings = {}
            context = self.searcher.retrieve_context(
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
            return {
                'context': context.to_dict(),
                'timings_ms': timings or {
                    'retrieval': round((time.perf_counter() - start) * 1000, 2),
                    'rerank': 0,
                },
            }

    def reload_index(self):
        with self.inference_lock:
            new_searcher = HybridSearcher.load(embedder=self.embedder)
            self.searcher = new_searcher
            return {
                'generation': self._index_generation(),
                'n_chunks': len(self.searcher.chunks),
            }

    def health(self):
        return {
            'index_generation': self._index_generation(),
            'n_chunks': len(self.searcher.chunks),
            'dense_available': self.searcher.vectors is not None,
            'reranker_available': self.reranker is not None,
        }
