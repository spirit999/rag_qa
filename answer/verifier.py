import re
from collections import defaultdict

from .schema import AnswerCitation, AnswerResult, VerificationResult


ARABIC_NUMBER = re.compile(r'\d+(?:\.\d+)?')
CHINESE_NUMBER = re.compile(r'[零〇一二三四五六七八九十百千万亿]+')


def extract_numbers(text):
    out = set(ARABIC_NUMBER.findall(text or ''))
    out.update(CHINESE_NUMBER.findall(text or ''))
    return out


def flatten_context_articles(context_pack):
    result = {}
    for doc in context_pack.documents:
        for section in doc.sections:
            for article in section.articles:
                result[article.chunk_id] = AnswerCitation(
                    chunk_id=article.chunk_id,
                    doc_title=doc.doc_title,
                    doc_number=doc.doc_number,
                    section=section.section,
                    article=article.article,
                    source=article.source,
                    text=article.text,
                )
    return result


class AnswerVerifier:
    def verify(self, raw_answer, context_pack):
        status = raw_answer.get('status', 'insufficient_context')
        answer = str(raw_answer.get('answer') or '').strip()
        citations_by_id = flatten_context_articles(context_pack)
        cited = set()
        unsupported_claims = []
        citation_errors = []

        claim_chunks = raw_answer.get('claim_chunks', [])
        if status == 'answered':
            if not claim_chunks:
                citation_errors.append('回答没有绑定任何制度条款。')
            for claim in claim_chunks:
                if not isinstance(claim, dict):
                    citation_errors.append('结论引用格式无效。')
                    continue
                ids = claim.get('chunk_ids', [])
                if not isinstance(ids, list) or not ids:
                    citation_errors.append('关键结论缺少 chunk_id 引用。')
                    continue
                for chunk_id in ids:
                    if chunk_id not in citations_by_id:
                        citation_errors.append('引用的 chunk_id 不存在于检索上下文：%s' % chunk_id)
                    else:
                        cited.add(chunk_id)

        context_numbers = extract_numbers(context_pack.context_text)
        answer_numbers = extract_numbers(answer)
        unsupported_numbers = sorted(answer_numbers - context_numbers)
        if unsupported_numbers:
            unsupported_claims.extend(
                ['答案中的数字/数值未在检索上下文中找到：%s' % n for n in unsupported_numbers]
            )

        if not answer:
            unsupported_claims.append('最终回答为空。')

        missing = raw_answer.get('missing_conditions', [])
        if not isinstance(missing, list):
            missing = []
        missing = [str(x) for x in missing if str(x).strip()]

        passed = status == 'answered' and not unsupported_claims and not citation_errors
        should_refuse = (
            status in ('insufficient_context', 'conflict')
            or bool(unsupported_claims)
            or bool(citation_errors)
            or not answer
        )
        if should_refuse and status == 'answered':
            status = 'insufficient_context'

        citations = [citations_by_id[k] for k in sorted(cited)]
        verification = VerificationResult(
            passed=passed,
            unsupported_claims=unsupported_claims,
            missing_conditions=missing,
            citation_errors=citation_errors,
            should_refuse=should_refuse,
        )
        return AnswerResult(
            answer=answer,
            status=status,
            confidence=raw_answer.get('confidence', 'low'),
            citations=citations,
            unsupported_claims=unsupported_claims,
            missing_conditions=missing,
            verification=verification,
        )
