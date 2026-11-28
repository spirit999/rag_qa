from collections import OrderedDict
import copy
import json
import re
import threading
import time

from .config import CATEGORIES, MAX_REWRITE_NEW_TOKENS, PROMPT_VERSION
from .glossary import Glossary, title_digest
from .prompts import build_messages
from .schema import RewriteResult, parse_llm_output

DEPENDENCY_MARKERS = re.compile(r'(那|这|它|呢|也|还有|同样|一样|上面|前面|别的|其他)')
FOLLOWUP_MARKERS = re.compile(
    r'(那|这|它|呢|也|还有|同样|一样|上面|前面|刚才|别的|其他|'
    r'继续|剩下|前者|后者|举例|展开|补充说明|具体呢|为什么|怎么理解)'
)
QUESTION_MARKERS = re.compile(
    r'(如何|怎么|多少|几|是否|能否|能不能|什么|哪些|哪个|哪种|多久|'
    r'何时|什么时候|标准|条件|流程|手续|申请|办理|规定|计算|结算|报销|'
    r'审批|需要|适用|要求|天数|金额|额度|比例|期限|文号|版本|有哪些|是什么)'
)
SHORT_LEN = 14


def needs_history(query, history):
    if not history:
        return False
    if len(query) <= SHORT_LEN and DEPENDENCY_MARKERS.search(query):
        return True
    return len(query) <= 8


class QueryRewriter:
    def __init__(self, llm=None, glossary=None):
        self.llm = llm
        self.glossary = glossary or Glossary()
        self._cache = OrderedDict()
        self._cache_limit = 256
        self._cache_lock = threading.RLock()

    def _cache_key(self, query, history):
        return json.dumps(
            [PROMPT_VERSION, query, history or []],
            ensure_ascii=False,
            separators=(',', ':'),
        )

    def _cache_get(self, query, history):
        key = self._cache_key(query, history)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            result = copy.deepcopy(cached)
        result.from_cache = True
        result.latency = 0.0
        return result

    def _cache_put(self, query, history, result):
        key = self._cache_key(query, history)
        with self._cache_lock:
            self._cache[key] = copy.deepcopy(result)
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)

    def _fallback(self, query):
        r = RewriteResult(fallback=True)
        r.standalone_query = query
        r.keywords = self.glossary.match_terms(query)[:6]
        r.sub_queries = [query]
        r.intent = '事实数字'
        r.category = keyword_category_vote(query, self.glossary)
        return r

    def _can_use_fast_path(self, query, history):
        if FOLLOWUP_MARKERS.search(query):
            return False
        if not QUESTION_MARKERS.search(query):
            return False
        if history and needs_history(query, history) and not self.glossary.match_terms(query):
            return False
        return bool(self.glossary.match_terms(query))

    def _fast_rewrite(self, query):
        result = RewriteResult()
        result.standalone_query = query.strip()
        result.keywords = self.glossary.match_terms(query)[:6]
        result.category = keyword_category_vote(query, self.glossary)
        result.sub_queries = []
        if re.search(r'(流程|手续|申请|办理|审批|材料|报销)', query):
            result.intent = '流程手续'
        elif re.search(r'(条件|资格|哪些人|谁能|是否|能否|可以)', query):
            result.intent = '条件资格'
        elif re.search(r'(文号|版本|发布日期|生效日期|全称)', query):
            result.intent = '制度元信息'
        elif re.search(r'(计算|怎么算|核算|结算|金额|比例)', query):
            result.intent = '计算判断'
        else:
            result.intent = '事实数字'
        return result

    def rewrite(self, query, history=None, force_llm=False):
        return self.rewrite_batch(
            [{'query': query, 'history': history}], force_llm=force_llm,
        )[0]

    def rewrite_batch(self, items, batch_size=8, force_llm=False):
        from tqdm import tqdm
        n = len(items)
        results = [None] * n
        pending = []
        for i, item in enumerate(items):
            if not force_llm and self._can_use_fast_path(
                    item['query'], item.get('history')):
                results[i] = self._fast_rewrite(item['query'])
            else:
                cached = None if force_llm else self._cache_get(
                    item['query'], item.get('history'),
                )
                if cached is not None:
                    results[i] = cached
                else:
                    pending.append((i, item))
        if not pending:
            return results

        for s in tqdm(range(0, len(pending), batch_size), desc='LLM批量改写'):
            chunk = pending[s:s + batch_size]
            msgs = []
            for _, it in chunk:
                hist = it.get('history')
                use_hist = hist if needs_history(it['query'], hist) else None
                prompt_context = ' '.join((use_hist or []) + [it['query']])
                category = keyword_category_vote(prompt_context, self.glossary)
                digest = self.glossary.prompt_digest(
                    max_chars=900, query=prompt_context,
                )
                titles = title_digest(categories=[category]) if category else ''
                msgs.append(build_messages(
                    it['query'], use_hist, digest, titles=titles,
                ))
            t0 = time.perf_counter()
            token_limit = (
                {'max_new_tokens': MAX_REWRITE_NEW_TOKENS}
                if getattr(self.llm, 'supports_max_new_tokens', False) else {}
            )
            if hasattr(self.llm, 'chat_batch'):
                raws = self.llm.chat_batch(msgs, **token_limit)
            else:
                raws = [self.llm.chat(m, **token_limit) for m in msgs]
            latency = (time.perf_counter() - t0) / max(len(chunk), 1)

            retry_idx = []
            for j, ((result_index, it), raw) in enumerate(zip(chunk, raws)):
                r = parse_llm_output(raw, it['query'])
                if r is None or (r.need_retrieval and r.intent == '闲聊超域'):
                    retry_idx.append(j)
                    continue
                r.latency = round(latency, 3)
                results[result_index] = r
                if not force_llm:
                    self._cache_put(it['query'], it.get('history'), r)
            for j in retry_idx:
                result_index, it = chunk[j]
                try:
                    raw = self.llm.chat(msgs[j], **token_limit)
                except Exception as e:
                    print('LLM调用失败:', repr(e)[:120])
                    raw = ''
                r = parse_llm_output(raw, it['query'])
                if r is None:
                    r = self._fallback(it['query'])
                r.latency = round(latency, 3)
                results[result_index] = r
                if not force_llm and not r.fallback:
                    self._cache_put(it['query'], it.get('history'), r)

        return results


GLOSSARY_TO_CATEGORY = {
    '机构主体': '', '归口部门': '', '人群身份': '',
    '假期考勤': '人力资源', '差旅报销': '财务管理', '薪酬福利': '人力资源',
    '行政事务': '公共管理', '外部依据': '', '常用表单': '',
}


def keyword_category_vote(text, glossary):
    votes = {}
    for term_row in glossary.rows:
        surfaces = [term_row['term']] + term_row['colloquial']
        for s in surfaces:
            if s and s in text:
                cat = GLOSSARY_TO_CATEGORY.get(term_row['category'], '')
                if cat:
                    votes[cat] = votes.get(cat, 0) + 1
                break
    return max(votes, key=votes.get) if votes else ''
