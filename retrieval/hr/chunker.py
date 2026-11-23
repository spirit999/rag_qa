import json
import re
from pathlib import Path

from .config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, OVERLAP_CHARS, PARSED_DIR

RE_CHAPTER = re.compile(r'^第[一二三四五六七八九十百零0-9]+章')
RE_ARTICLE = re.compile(r'^第[一二三四五六七八九十百零0-9]+条')
SENT_SPLIT = re.compile(r'(?<=[。；！？])')
NL = chr(10) # 换行符


def read_parsed(path):
    meta = {}
    body = []
    text = path.read_text(encoding='utf-8')
    if '# meta' not in text or '# body' not in text:
        return meta, body
    head, rest = text.split('# body', 1)
    for line in head.split(NL):
        if ':' in line and not line.startswith('#'):
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    body = [l.strip() for l in rest.split(NL) if l.strip()]
    return meta, body


# max_chars是软上限，只有大于它的两倍的时候才会强制切分
def split_long(text, max_chars=MAX_CHUNK_CHARS, overlap=OVERLAP_CHARS):
    out = []
    cur = ''
    for sent in SENT_SPLIT.split(text):
        if not sent:
            continue
        if cur and len(cur) + len(sent) > max_chars:
            out.append(cur)
            cur = cur[-overlap:] if overlap else ''
        cur += sent
        while len(cur) > max_chars * 2:
            out.append(cur[:max_chars])
            cur = cur[max_chars - overlap:]
    if cur.strip():
        out.append(cur)
    return out


def segment(meta, body):
    """按 章/条 切分，返回 [{section, article, text}]。"""
    sections = []
    section = ''
    article = ''
    buf = []
    for line in body:
        if RE_CHAPTER.match(line) and len(line) <= 26:
            if buf:
                sections.append((section, article, NL.join(buf)))
                buf = []
            section = line
            article = ''
            continue
        if line == '[表格]':
            buf.append(line)
            continue
        if RE_ARTICLE.match(line):
            if buf:
                sections.append((section, article, NL.join(buf)))
                buf = []
            article = line[:line.find('条') + 1]
            buf = [line]
            continue
        buf.append(line)
    if buf:
        sections.append((section, article, NL.join(buf)))
    return [{'section': s, 'article': a, 'text': t} for s, a, t in sections if t.strip()]


def merge_short(sections, min_chars=MIN_CHUNK_CHARS, max_chars=MAX_CHUNK_CHARS):
    out = []
    for s in sections:
        if out and len(s['text']) < min_chars:
            prev = out[-1]
            if len(prev['text']) + len(s['text']) <= max_chars:
                prev['text'] = prev['text'] + NL + s['text']
                continue
        out.append(dict(s))
    return out


def build_chunks(parsed_dir=PARSED_DIR):
    chunks = []
    for path in sorted(Path(parsed_dir).glob('*.txt')):
        meta, body = read_parsed(path)
        if not body:
            continue
        title = meta.get('doc_title', path.stem)
        category = meta.get('category', '')
        source = meta.get('source', '')
        doc_number = meta.get('doc_number', '')
        sections = merge_short(segment(meta, body))
        for s in sections:
            for piece_index, piece in enumerate(split_long(s['text'])):
                text = piece.strip()
                if len(text) < 20:
                    continue
                ctx = title
                if s['section']:
                    ctx += '｜' + s['section']
                if s['article']:
                    ctx += '｜' + s['article']
                article = s['article'] or '条款未识别'
                section = s['section'] or '章标题未识别'
                stable_source = source or path.name
                chunk_id = '%s::%s::%s::%d' % (
                    stable_source, section, article, piece_index,
                )
                chunks.append({
                    'cid': len(chunks),
                    'chunk_id': chunk_id,
                    'source': source,
                    'doc_title': title,
                    'category': category,
                    'doc_number': doc_number,
                    'section': s['section'],
                    'article': s['article'],
                    'text': text,
                    'embed_text': ctx + '｜' + text,
                })
    return chunks


def save_chunks(chunks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + NL)


def load_chunks(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
