import csv
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import jieba

from .config import BM25_B, BM25_K1, DOC_LIST_CSV, GLOSSARY_CSV

STOP_WORDS = set('''
的 了 是 在 和 与 或 对 为 及 等 被 把 将 从 向 由 于 按 据 经 因 予 并 但 而 则 就 也 都 又
我 你 他 她 它 我们 你们 他们 什么 怎么 怎样 如何 多少 哪个 哪些 可以 能否 是否 有没有
公司 员工 本办法 本制度 规定 相关 有关 进行 应当 应该 需要 一个 这个 那个 情况 问题
'''.split())

RE_TOKEN_OK = re.compile(r'^[0-9a-zA-Z一-鿿]+$')


def normalize(surface):
    return surface.strip().strip('()（）“”" ').replace('（', '(').replace('）', ')')


# 返回所有同义词组列表和每个词所属的组号映射
def load_synonym_groups(glossary_csv=GLOSSARY_CSV, doc_list_csv=DOC_LIST_CSV):
    """返回 (surface→group_id 映射, groups 列表)。同组词可互换检索。"""
    groups = []
    surface_to_group = {}
    with open(glossary_csv, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            term = r['公文术语（规范说法）'].strip()
            if not term:
                continue
            variants = [term]
            for s in r['口语/同义说法（建议待人工复核）'].split('/'):
                s = s.strip()
                if s:
                    variants.append(s)
            variants = [v for v in dict.fromkeys(variants) if v]
            gid = len(groups)
            groups.append(variants)
            for v in variants:
                surface_to_group.setdefault(normalize(v), gid)
                surface_to_group.setdefault(v, gid)
    if Path(doc_list_csv).exists():
        seen = set()
        with open(doc_list_csv, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                t = r['制度名称'].strip()
                if t and t not in seen:
                    seen.add(t)
                    groups.append([t])
                    surface_to_group.setdefault(normalize(t), len(groups) - 1)
    return surface_to_group, groups


def add_terms_to_jieba(groups):
    for variants in groups:
        for v in variants:
            if len(v) >= 2:
                jieba.add_word(v, freq=2000)


class BM25Index:
    def __init__(self, chunks, surface_to_group=None, groups=None, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.surface_to_group = surface_to_group or {}
        self.groups = groups or []
        self.n = len(chunks)
        self.dl = []
        self.postings = defaultdict(list)
        self.df = defaultdict(int)
        self.cid_to_doc = []
        total = 0
        for c in chunks:
            tokens = self.tokenize(c['embed_text'])
            self.dl.append(len(tokens))
            total += len(tokens)
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, k in tf.items():
                self.postings[t].append([c['cid'], k])
                self.df[t] += 1
            self.cid_to_doc.append(c['doc_title'])
        self.avgdl = (total / self.n) if self.n else 1.0

    @staticmethod
    def tokenize(text):
        out = []
        for w in jieba.lcut(text):
            w = w.strip()
            if not w or w in STOP_WORDS or not RE_TOKEN_OK.match(w):
                continue
            out.append(w)
        return out

    def idf(self, token):
        df = self.df.get(token, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _tf_score(self, tf, dl):
        denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        return tf * (self.k1 + 1) / denom

    # 关键词权重高于普通词
    def search(self, keywords, query_text='', topk=50, text_weight=0.5):
        groups = []
        for kw in keywords or []:
            gid = self.surface_to_group.get(normalize(kw)) or self.surface_to_group.get(kw)
            if gid is not None:
                groups.append((self.groups[gid], 1.0))
            else:
                tokens = self.tokenize(kw)
                if tokens:
                    groups.append((tokens, 1.0))
        kw_surfaces = {normalize(v) for vs, _ in groups for v in vs}
        if query_text:
            tokens = [t for t in self.tokenize(query_text) if normalize(t) not in kw_surfaces]
            for t in tokens:
                gid = self.surface_to_group.get(t)
                variants = self.groups[gid] if gid is not None else [t]
                groups.append((variants, text_weight))
        if not groups:
            return []
        scores = defaultdict(float)
        for variants, weight in groups:
            best = {}
            for t in variants:
                postings = self.postings.get(t)
                if not postings:
                    continue
                idf = self.idf(t)
                for cid, tf in postings:
                    s = idf * self._tf_score(tf, self.dl[cid])
                    if s > best.get(cid, 0.0):
                        # 存在cid咋返回，没有咋返回0.0
                        best[cid] = s
            for cid, s in best.items():
                scores[cid] += weight * s
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:topk]
        return [(cid, s) for cid, s in ranked]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'k1': self.k1, 'b': self.b, 'n': self.n, 'avgdl': self.avgdl,
            'dl': self.dl, 'df': dict(self.df),
            'postings': {t: p for t, p in self.postings.items()},
            'cid_to_doc': self.cid_to_doc,
        }
        with path.open('wb') as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path, surface_to_group=None, groups=None):
        with Path(path).open('rb') as f:
            p = pickle.load(f)
        obj = cls.__new__(cls)
        obj.k1 = p['k1']
        obj.b = p['b']
        obj.n = p['n']
        obj.avgdl = p['avgdl']
        obj.dl = p['dl']
        obj.df = defaultdict(int, p['df'])
        obj.postings = defaultdict(list, p['postings'])
        obj.cid_to_doc = p['cid_to_doc']
        obj.surface_to_group = surface_to_group or {}
        obj.groups = groups or []
        return obj
