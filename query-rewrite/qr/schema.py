import json
from dataclasses import dataclass, field, asdict

from .config import CATEGORIES, INTENTS


@dataclass
class RewriteResult:
    standalone_query: str = ''
    intent: str = ''
    keywords: list = field(default_factory=list)
    sub_queries: list = field(default_factory=list)
    category: str = ''
    subject: str = ''
    time_scope: str = 'unspecified'
    need_retrieval: bool = True
    from_cache: bool = False
    fallback: bool = False
    raw: str = ''
    latency: float = 0.0

    def to_dict(self):
        return asdict(self)


def _coerce_str_list(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    return []


def extract_json(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def parse_llm_output(text, original_query):
    data = extract_json(text)
    if data is None:
        return None
    r = RewriteResult(raw=text)
    r.standalone_query = str(data.get('standalone_query') or original_query).strip()
    intent = str(data.get('intent') or '').strip()
    r.intent = intent if intent in INTENTS else ''
    r.keywords = _coerce_str_list(data.get('keywords'))
    r.sub_queries = _coerce_str_list(data.get('sub_queries'))
    category = str(data.get('category') or '').strip()
    r.category = category if category in CATEGORIES else ''
    r.subject = str(data.get('subject') or '').strip()
    ts = str(data.get('time_scope') or '').strip()
    r.time_scope = ts if ts in ('latest', 'specific', 'unspecified') else 'unspecified'
    nr = data.get('need_retrieval')
    r.need_retrieval = bool(nr) if isinstance(nr, bool) else r.intent != '闲聊超域'
    if r.intent == '闲聊超域':
        r.need_retrieval = False
        r.category = ''
    if r.need_retrieval and not r.sub_queries:
        r.sub_queries = [r.standalone_query]
    return r
