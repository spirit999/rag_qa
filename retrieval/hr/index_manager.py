import hashlib
import json
import os
import re
import sys
import tempfile
import time
import fcntl
from pathlib import Path

import numpy as np

from .bm25 import BM25Index, add_terms_to_jieba, load_synonym_groups
from .chunker import build_chunks, load_chunks, save_chunks
from .config import (
    DATA_DIR, DOCS_DIR, INDEX_CURRENT_PATH, INDEX_GENERATIONS_DIR,
    MANIFEST_PATH,
)

PROJECT = Path(__file__).resolve().parents[2]
QUERY_REWRITE = PROJECT / 'query-rewrite'
if str(QUERY_REWRITE) not in sys.path:
    sys.path.insert(0, str(QUERY_REWRITE))


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_jsonl(path):
    if not Path(path).exists():
        return []
    records = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _safe_parsed_name(category, source_path):
    stem = Path(source_path).stem
    safe_stem = re.sub(r'[/\\:*?"<>|]', '_', stem)
    return category + '__' + safe_stem + '.txt'


class IncrementalIndexManager:
    def __init__(self, docs_dir=DOCS_DIR, parsed_dir=None, manifest_path=None,
                 data_dir=DATA_DIR, generations_dir=None, current_path=None):
        self.docs_dir = Path(docs_dir)
        self.parsed_dir = Path(parsed_dir or (PROJECT / 'data' / 'parsed'))
        self.manifest_path = Path(manifest_path or MANIFEST_PATH)
        self.data_dir = Path(data_dir)
        self.generations_dir = Path(generations_dir or INDEX_GENERATIONS_DIR)
        self.current_path = Path(current_path or INDEX_CURRENT_PATH)

    def _parse_changed(self, path, rel_path, file_hash):
        import parse_docs

        category = rel_path.parts[0]
        ext = path.suffix.lower()
        title, doc_number = parse_docs.parse_filename_meta(path)
        blocks = []
        status, note = 'ok', ''
        try:
            if ext == '.docx':
                blocks = parse_docs.parse_docx(path)
            elif ext == '.doc':
                with tempfile.TemporaryDirectory(prefix='rag-doc-convert-') as tmp:
                    ok, log = parse_docs.convert_doc_group([path], Path(tmp))
                    converted = Path(tmp) / (path.stem + '.docx')
                    if not ok or not converted.exists():
                        raise RuntimeError('DOC 转换失败：' + log[:300])
                    blocks = parse_docs.parse_docx(converted)
            elif ext == '.pdf':
                blocks = parse_docs.parse_pdf(path)
            elif ext in ('.xlsx', '.xls'):
                blocks = parse_docs.parse_xlsx(path)
            else:
                status, note = 'skip', '不支持的类型'
        except Exception as exc:
            return None, {
                'source': str(rel_path),
                'status': 'error',
                'note': repr(exc)[:300],
            }

        parsed_name = _safe_parsed_name(category, rel_path)
        lines = [
            '# meta',
            'category: ' + category,
            'source: ' + str(rel_path),
            'doc_title: ' + title,
            'doc_number: ' + doc_number,
            'n_tables: ' + str(sum(1 for kind, _ in blocks if kind == 'tbl')),
            '# body',
        ]
        for kind, text in blocks:
            if kind == 'tbl':
                lines.extend(['[表格]', text])
            else:
                lines.append(text)
        parsed_path = self.parsed_dir / parsed_name
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = parsed_path.with_suffix(parsed_path.suffix + '.tmp')
        tmp_path.write_text('\n'.join(lines), encoding='utf-8')
        os.replace(tmp_path, parsed_path)

        chars = sum(len(text) for _, text in blocks)
        if chars == 0 and status == 'ok':
            status = 'empty'
        return {
            'category': category,
            'source': str(rel_path),
            'ext': ext,
            'doc_title': title,
            'doc_number': doc_number,
            'parsed_file': parsed_name,
            'paragraphs': sum(1 for kind, _ in blocks if kind == 'p'),
            'tables': sum(1 for kind, _ in blocks if kind == 'tbl'),
            'chars': chars,
            'status': status,
            'note': note,
            'sha256': file_hash,
        }, None

    def _update_parsed_documents(self):
        if not self.docs_dir.is_dir():
            raise FileNotFoundError('制度文档目录不存在：%s' % self.docs_dir)
        old_records = {
            r['source']: r for r in _load_jsonl(self.manifest_path)
            if r.get('source')
        }
        source_paths = sorted(p for p in self.docs_dir.rglob('*') if p.is_file())
        if not source_paths:
            raise RuntimeError('制度文档目录为空，拒绝发布空索引。')
        found_sources = set()
        records = []
        errors = []
        changed_count = 0
        unchanged_count = 0

        for path in source_paths:
            rel_path = path.relative_to(self.docs_dir)
            source = str(rel_path)
            found_sources.add(source)
            old = old_records.get(source)
            current_hash = _sha256_file(path)
            if old and old.get('sha256') == current_hash:
                records.append(old)
                unchanged_count += 1
                continue
            record, error = self._parse_changed(path, rel_path, current_hash)
            if error:
                errors.append(error)
                if old:
                    records.append(old)
                continue
            if record and record.get('status') != 'skip':
                records.append(record)
            changed_count += 1

        deleted_sources = set(old_records) - found_sources
        for source in deleted_sources:
            parsed_file = old_records[source].get('parsed_file')
            if parsed_file:
                target = (self.parsed_dir / parsed_file).resolve()
                if target.parent == self.parsed_dir.resolve() and target.exists():
                    target.unlink()

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_manifest = self.manifest_path.with_suffix('.jsonl.tmp')
        with tmp_manifest.open('w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_manifest, self.manifest_path)
        return {
            'documents': len(records),
            'changed': changed_count,
            'unchanged': unchanged_count,
            'deleted': len(deleted_sources),
            'errors': errors,
        }

    def _old_vectors_by_text(self, embedder):
        pointer = self._current_generation()
        if pointer:
            base = self.generations_dir / pointer['generation']
            chunks_path = base / 'chunks.jsonl'
            vectors_path = base / 'vectors.npy'
            meta_path = base / 'vectors_meta.json'
        else:
            chunks_path = self.data_dir / 'chunks.jsonl'
            vectors_path = self.data_dir / 'vectors.npy'
            meta_path = self.data_dir / 'vectors_meta.json'
        if not chunks_path.exists() or not vectors_path.exists():
            return {}, None
        if embedder is not None and meta_path.exists():
            with meta_path.open(encoding='utf-8') as f:
                old_meta = json.load(f)
            old_model = Path(old_meta.get('model', '')).name
            new_model = Path(getattr(embedder, 'path', '')).name
            old_max_len = old_meta.get('max_len', 512)
            new_max_len = getattr(embedder, 'max_len', 512)
            if old_model and new_model and (
                old_model != new_model or old_max_len != new_max_len
            ):
                return {}, pointer
        old_chunks = load_chunks(chunks_path)
        old_vectors = np.load(vectors_path, mmap_mode='r')
        if len(old_chunks) != len(old_vectors):
            raise ValueError('旧索引 Chunk 与向量数量不一致，拒绝复用向量缓存。')
        mapping = {}
        for chunk, vector in zip(old_chunks, old_vectors):
            mapping.setdefault(chunk['embed_text'], vector)
        return mapping, pointer

    def _current_generation(self):
        if not self.current_path.exists():
            return None
        with self.current_path.open(encoding='utf-8') as f:
            return json.load(f)

    def _publish_index(self, embedder):
        chunks = build_chunks(self.parsed_dir)
        if not chunks:
            raise RuntimeError('没有可用于建索引的 Chunk，保留当前索引。')
        surface_to_group, groups = load_synonym_groups()
        add_terms_to_jieba(groups)
        bm25 = BM25Index(chunks, surface_to_group, groups)

        previous_vectors, previous = self._old_vectors_by_text(embedder)
        vectors = [None] * len(chunks)
        missing_indices = []
        missing_texts = []
        reused = 0
        for i, chunk in enumerate(chunks):
            vector = previous_vectors.get(chunk['embed_text'])
            if vector is None:
                missing_indices.append(i)
                missing_texts.append(chunk['embed_text'])
            else:
                vectors[i] = vector
                reused += 1
        if missing_texts:
            if embedder is None:
                raise RuntimeError('有新增或变更文档，需要 Embedding 模型生成新向量。')
            encoded = embedder.encode(missing_texts, show_progress=True)
            if len(encoded) != len(missing_indices):
                raise ValueError('Embedding 返回数量与新增 Chunk 数量不一致。')
            for i, vector in zip(missing_indices, encoded):
                vectors[i] = vector
        matrix = np.asarray(vectors, dtype=np.float32)
        if (
            matrix.ndim != 2
            or len(matrix) != len(chunks)
            or not matrix.shape[1]
            or not np.isfinite(matrix).all()
        ):
            raise ValueError('新索引向量校验失败。')

        manifest_records = _load_jsonl(self.manifest_path)
        manifest_digest = hashlib.sha256(
            ''.join(sorted(r.get('sha256', '') for r in manifest_records)).encode('utf-8')
        ).hexdigest()[:12]
        generation = '%d-%s' % (time.time_ns(), manifest_digest)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        staging = self.generations_dir / ('.building-' + generation)
        final = self.generations_dir / generation
        staging.mkdir()
        try:
            save_chunks(chunks, staging / 'chunks.jsonl')
            bm25.save(staging / 'bm25.pkl')
            np.save(staging / 'vectors.npy', matrix)
            meta = {
                'generation': generation,
                'created_at': time.time(),
                'n_chunks': len(chunks),
                'n_docs': len({chunk['source'] for chunk in chunks}),
                'n_vectors_reused': reused,
                'n_vectors_encoded': len(missing_texts),
                'dim': int(matrix.shape[1]) if matrix.ndim == 2 else 0,
                'model': getattr(embedder, 'path', ''),
                'max_len': getattr(embedder, 'max_len', 512),
            }
            _atomic_json(staging / 'vectors_meta.json', meta)
            os.replace(staging, final)
            _atomic_json(self.current_path, {
                'generation': generation,
                'created_at': meta['created_at'],
                'n_chunks': meta['n_chunks'],
                'n_docs': meta['n_docs'],
            })
        except Exception:
            if staging.exists():
                for child in staging.iterdir():
                    if child.is_file():
                        child.unlink()
                staging.rmdir()
            raise

        return {
            'generation': generation,
            'n_chunks': len(chunks),
            'n_docs': meta['n_docs'],
            'n_vectors_reused': reused,
            'n_vectors_encoded': len(missing_texts),
            'previous_generation': previous.get('generation') if previous else None,
        }

    def update(self, embedder):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.data_dir / '.index-update.lock'
        with lock_path.open('w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            document_stats = self._update_parsed_documents()
            errors = document_stats.pop('errors', [])
            index_stats = self._publish_index(embedder)
            return {
                'documents': document_stats,
                'index': index_stats,
                'errors': errors,
            }
