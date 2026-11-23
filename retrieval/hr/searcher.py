from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from .bm25 import BM25Index, add_terms_to_jieba, load_synonym_groups
from .config import (
    BM25_PATH, CHUNKS_PATH, INDEX_CURRENT_PATH, INDEX_GENERATIONS_DIR,
    RRF_K, VEC_PATH,
)
from .chunker import load_chunks

__all__ = ['HybridSearcher', 'BM25Index', 'add_terms_to_jieba', 'load_synonym_groups']


class HybridSearcher:
    def __init__(self, chunks, bm25, vectors, embedder=None):
        self.chunks = chunks
        self.bm25 = bm25
        self.vectors = vectors
        self.embedder = embedder
        self._vec_norm = None

    @classmethod
    def load(cls, embedder=None, chunks_path=CHUNKS_PATH, bm25_path=BM25_PATH, vec_path=VEC_PATH):
        pointer = Path(INDEX_CURRENT_PATH)
        if pointer.exists():
            import json
            with pointer.open(encoding='utf-8') as f:
                generation = json.load(f)['generation']
            base = Path(INDEX_GENERATIONS_DIR) / generation
            chunks_path = base / 'chunks.jsonl'
            bm25_path = base / 'bm25.pkl'
            vec_path = base / 'vectors.npy'
        chunks = load_chunks(chunks_path)
        surface_to_group, groups = load_synonym_groups()
        add_terms_to_jieba(groups)
        bm25 = BM25Index.load(bm25_path, surface_to_group, groups)
        vectors = np.load(vec_path) if Path(vec_path).exists() else None
        return cls(chunks, bm25, vectors, embedder)

    def _dense_scores(self, queries, topk):
        if self.embedder is None or self.vectors is None or not queries:
            return []
        q = self.embedder.encode(queries, is_query=True)
        sims = self.vectors @ q.T
        best = sims.max(axis=1)
        idx = np.argpartition(-best, min(topk, len(best) - 1))[:topk]
        idx = idx[np.argsort(-best[idx])]
        return [(int(i), float(best[i])) for i in idx]

    def search_chunks(self, keywords, standalone_query, sub_queries=None, topk_chunks=60,
                      mode='hybrid'):
        bm_res = self.bm25.search(keywords, standalone_query, topk=topk_chunks)

        dense_res = []
        if mode in ('hybrid', 'dense'):
            queries = [standalone_query] + list(sub_queries or [])
            dense_res = self._dense_scores(queries, topk_chunks)

        if mode == 'bm25':
            return bm_res
        if mode == 'dense':
            return dense_res
        return rrf_fuse(bm_res, dense_res)

    def search(self, keywords, standalone_query, sub_queries=None, topk_chunks=60, topk_docs=10,
               mode='hybrid'):
        fused = self.search_chunks(
            keywords, standalone_query, sub_queries, topk_chunks, mode,
        )
        return self.aggregate_docs(fused, topk_docs)

    def aggregate_docs(self, fused, topk_docs):
        by_doc = defaultdict(list)
        for cid, score in fused:
            c = self.chunks[cid]
            by_doc[c['doc_title']].append((score, cid))
        docs = []
        for title, items in by_doc.items():
            items.sort(key=lambda x: -x[0])
            c = self.chunks[items[0][1]]
            docs.append({
                'doc_title': title,
                'category': c['category'],
                'doc_number': c['doc_number'],
                'score': items[0][0],
                'n_chunks': len(items),
                'top_cid': items[0][1],
                'section': c['section'],
                'article': c['article'],
                'snippet': c['text'][:120],
                'sources': sorted({self.chunks[cid]['source'] for _, cid in items}),
            })
        docs.sort(key=lambda d: -d['score'])
        return docs[:topk_docs]

    def rerank_chunks(self, reranker, query, candidates, batch_size=16):
        """对 Hybrid 候选执行重排序，返回 (cid, rerank_score)。"""
        if not candidates:
            return []
        texts = [self.chunks[cid]['embed_text'] for cid, _ in candidates]
        scores = reranker.score(query, texts, batch_size=batch_size)
        return sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda item: -item[1],
        )

    def retrieve_context(self, context_builder, reranker, question,
                         keywords, standalone_query, sub_queries,
                         topk_chunks=50, topk_docs=5, mode='hybrid',
                         rerank=True, max_context_chars=12000, timing_out=None):
        search_started = time.perf_counter()
        candidates = self.search_chunks(
            keywords, standalone_query, sub_queries,
            topk_chunks=topk_chunks, mode=mode,
        )
        search_ms = (time.perf_counter() - search_started) * 1000
        rerank_ms = 0.0
        if rerank and reranker is not None:
            rerank_started = time.perf_counter()
            candidates = self.rerank_chunks(
                reranker, standalone_query, candidates,
                batch_size=getattr(reranker, 'batch_size', 16),
            )
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        context_started = time.perf_counter()
        context = context_builder(question, self.chunks, candidates,
                                  max_chars=max_context_chars,
                                  topk_docs=topk_docs)
        if timing_out is not None:
            timing_out.update({
                'retrieval': round(
                    search_ms + (time.perf_counter() - context_started) * 1000, 2,
                ),
                'rerank': round(rerank_ms, 2),
            })
        return context


def rrf_fuse(rank_a, rank_b, k=RRF_K):
    scores = defaultdict(float)
    for rank, (cid, _) in enumerate(rank_a):
        scores[cid] += 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(rank_b):
        scores[cid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])
