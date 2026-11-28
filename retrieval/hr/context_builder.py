from dataclasses import asdict, dataclass, field
from typing import Iterable, List


@dataclass
class ContextArticle:
    article: str
    text: str
    chunk_id: str
    score: float
    source: str


@dataclass
class ContextSection:
    section: str
    articles: List[ContextArticle] = field(default_factory=list)


@dataclass
class ContextDocument:
    doc_title: str
    doc_number: str
    category: str
    sections: List[ContextSection] = field(default_factory=list)
    score: float = 0.0


@dataclass
class ContextPack:
    question: str
    documents: List[ContextDocument] = field(default_factory=list)
    context_text: str = ''
    truncated: bool = False

    def to_dict(self):
        return asdict(self)


def _deduplicate_text(text):
    text = text.strip()
    if not text:
        return ''
    return text


def build_context(question, chunks, ranked_candidates, max_chars=6000, topk_docs=5):
    """将 Rerank/检索候选整理为按制度、章、条去重的完整上下文。"""
    if not ranked_candidates:
        return ContextPack(question=question)

    by_doc = {}
    seen_articles = set()
    seen_chunks = set()

    for item in ranked_candidates:
        if isinstance(item, tuple):
            cid, score = item
        else:
            cid = item
            score = 1.0
        cid = int(cid)
        if cid < 0 or cid >= len(chunks) or cid in seen_chunks:
            continue
        seen_chunks.add(cid)
        chunk = chunks[cid]
        doc_title = chunk.get('doc_title', '')
        doc_key = doc_title or chunk.get('source', str(cid))
        doc = by_doc.setdefault(doc_key, {
            'doc_title': doc_title,
            'doc_number': chunk.get('doc_number', ''),
            'category': chunk.get('category', ''),
            'sections': {},
            'score': float(score),
        })
        doc['score'] = max(doc['score'], float(score))

        section_name = chunk.get('section') or '章标题未识别'
        article_name = chunk.get('article') or '条款未识别'
        article_key = (doc_key, section_name, article_name)
        text = _deduplicate_text(chunk.get('text', ''))
        if not text:
            continue

        section = doc['sections'].setdefault(section_name, {})
        articles = section
        existing = articles.get(article_name)
        if existing:
            if len(text) > len(existing.text):
                existing.text = text
                existing.chunk_id = chunk.get('chunk_id', str(cid))
                existing.score = max(existing.score, float(score))
            continue
        if article_key in seen_articles:
            continue
        seen_articles.add(article_key)
        articles[article_name] = ContextArticle(
            article=article_name,
            text=text,
            chunk_id=chunk.get('chunk_id', str(cid)),
            score=float(score),
            source=chunk.get('source', ''),
        )

    ranked_docs = sorted(by_doc.values(), key=lambda d: -d['score'])[:topk_docs]
    documents = []
    lines = []
    truncated = False
    for doc in ranked_docs:
        doc_header = '制度：%s' % doc['doc_title']
        if doc['doc_number']:
            doc_header += '（%s）' % doc['doc_number']
        doc_lines = [doc_header]
        selected_sections = []
        for section_name, articles in doc['sections'].items():
            section_lines = []
            selected_articles = []
            section_articles = sorted(
                articles.values(),
                key=lambda a: (a.article != '条款未识别', a.article),
            )
            for article in section_articles:
                article_lines = [
                    '[%s %s]' % (article.article, article.chunk_id),
                    article.text,
                ]
                candidate_section = (
                    section_lines
                    + ([] if selected_articles else [section_name])
                    + article_lines
                )
                candidate_doc = '\n'.join(doc_lines + candidate_section)
                if len('\n\n'.join(lines + [candidate_doc])) > max_chars:
                    truncated = True
                    break
                if not selected_articles:
                    section_lines.append(section_name)
                section_lines.extend(article_lines)
                selected_articles.append(article)
            if selected_articles:
                doc_lines.extend(section_lines)
                selected_sections.append(ContextSection(
                    section_name, selected_articles,
                ))
            if truncated:
                break

        if selected_sections:
            lines.append('\n'.join(doc_lines))
            documents.append(ContextDocument(
                doc_title=doc['doc_title'],
                doc_number=doc['doc_number'],
                category=doc['category'],
                sections=selected_sections,
                score=doc['score'],
            ))
        if truncated:
            break

    return ContextPack(
        question=question,
        documents=documents,
        context_text='\n\n'.join(lines),
        truncated=truncated,
    )
