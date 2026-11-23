import json
import re

from .prompts import build_answer_messages

ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"')


class _AnswerFieldStreamer:
    def __init__(self):
        self.pending = ''
        self.started = False
        self.finished = False
        self.escaped = False
        self.unicode_digits = ''
        self.pending_high_surrogate = None

    def _emit_codepoint(self, value, output):
        if 0xD800 <= value <= 0xDBFF:
            if self.pending_high_surrogate is not None:
                output.append('\ufffd')
            self.pending_high_surrogate = value
        elif 0xDC00 <= value <= 0xDFFF:
            if self.pending_high_surrogate is None:
                output.append('\ufffd')
            else:
                codepoint = (
                    0x10000
                    + ((self.pending_high_surrogate - 0xD800) << 10)
                    + (value - 0xDC00)
                )
                output.append(chr(codepoint))
                self.pending_high_surrogate = None
        else:
            if self.pending_high_surrogate is not None:
                output.append('\ufffd')
                self.pending_high_surrogate = None
            output.append(chr(value))

    def feed(self, text):
        if self.finished:
            return ''
        if not self.started:
            self.pending += text
            match = ANSWER_FIELD.search(self.pending)
            if not match:
                return ''
            text = self.pending[match.end():]
            self.pending = ''
            self.started = True

        output = []
        escapes = {
            '"': '"', '\\': '\\', '/': '/', 'b': '\b',
            'f': '\f', 'n': '\n', 'r': '\r', 't': '\t',
        }
        for char in text:
            if self.unicode_digits:
                self.unicode_digits += char
                if len(self.unicode_digits) == 5:
                    self._emit_codepoint(
                        int(self.unicode_digits[1:], 16), output,
                    )
                    self.unicode_digits = ''
                continue
            if self.escaped:
                self.escaped = False
                if char == 'u':
                    self.unicode_digits = 'u'
                elif char in escapes:
                    output.append(escapes[char])
                else:
                    output.append(char)
                continue
            if char == '\\':
                self.escaped = True
            elif char == '"':
                if self.pending_high_surrogate is not None:
                    output.append('\ufffd')
                    self.pending_high_surrogate = None
                self.finished = True
                break
            else:
                if self.pending_high_surrogate is not None:
                    output.append('\ufffd')
                    self.pending_high_surrogate = None
                output.append(char)
        return ''.join(output)


def extract_json(text):
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question, rewrite, context_pack):
        if not context_pack.context_text.strip():
            return self._empty_context_result()
        messages = build_answer_messages(question, rewrite, context_pack.context_text)
        try:
            raw = self.llm.chat(messages)
        except Exception as exc:
            return self._error_result(exc)
        return self._parse_response(raw)

    def generate_stream(self, question, rewrite, context_pack):
        if not context_pack.context_text.strip():
            yield {'type': 'complete', 'raw': self._empty_context_result()}
            return
        messages = build_answer_messages(question, rewrite, context_pack.context_text)
        parser = _AnswerFieldStreamer()
        parts = []
        try:
            for part in self.llm.chat_stream(messages):
                parts.append(part)
                delta = parser.feed(part)
                if delta:
                    yield {'type': 'delta', 'text': delta}
        except Exception as exc:
            yield {'type': 'complete', 'raw': self._error_result(exc)}
            return
        yield {
            'type': 'complete',
            'raw': self._parse_response(''.join(parts)),
        }

    @staticmethod
    def _empty_context_result():
        return {
            'answer': '当前没有检索到能够支持回答该问题的制度条款，因此无法给出确定结论。',
            'status': 'insufficient_context',
            'confidence': 'high',
            'claim_chunks': [],
            'missing_conditions': ['相关制度条款'],
        }

    @staticmethod
    def _error_result(exc):
        return {
            'answer': '答案生成服务暂时不可用，请稍后重试。',
            'status': 'insufficient_context',
            'confidence': 'low',
            'claim_chunks': [],
            'missing_conditions': ['答案生成服务恢复'],
            '_error': repr(exc)[:200],
        }

    @staticmethod
    def _parse_response(raw):
        data = extract_json(raw)
        if data is None:
            return {
                'answer': '当前回答无法稳定解析为有依据的制度答案，因此未提供结论。',
                'status': 'insufficient_context',
                'confidence': 'low',
                'claim_chunks': [],
                'missing_conditions': ['有效的答案生成结果'],
            }
        status = data.get('status')
        if status not in ('answered', 'insufficient_context', 'conflict'):
            status = 'insufficient_context'
        confidence = data.get('confidence')
        if confidence not in ('high', 'medium', 'low'):
            confidence = 'low'
        claim_chunks = data.get('claim_chunks')
        if not isinstance(claim_chunks, list):
            claim_chunks = []
        return {
            'answer': str(data.get('answer') or '').strip(),
            'status': status,
            'confidence': confidence,
            'claim_chunks': claim_chunks,
            'missing_conditions': data.get('missing_conditions')
            if isinstance(data.get('missing_conditions'), list) else [],
        }
