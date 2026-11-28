from contextlib import closing
import sqlite3
import time
from pathlib import Path

from retrieval.hr.config import PROJECT


class EventStore:
    def __init__(self, path=None):
        self.path = Path(path or PROJECT / 'data' / 'observability.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = self._connect()
        try:
            with db:
                db.execute('PRAGMA journal_mode=WAL')
                db.execute('''
                CREATE TABLE IF NOT EXISTS request_events (
                    request_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    route TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    rewrite_ms REAL NOT NULL DEFAULT 0,
                    retrieval_ms REAL NOT NULL DEFAULT 0,
                    rerank_ms REAL NOT NULL DEFAULT 0,
                    generation_ms REAL NOT NULL DEFAULT 0,
                    verification_ms REAL NOT NULL DEFAULT 0,
                    lock_wait_ms REAL NOT NULL DEFAULT 0,
                    pipeline_load_ms REAL NOT NULL DEFAULT 0,
                    rewrite_cache_hit INTEGER NOT NULL DEFAULT 0,
                    answer_cache_hit INTEGER NOT NULL DEFAULT 0,
                    document_count INTEGER NOT NULL DEFAULT 0,
                    citation_count INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT NOT NULL DEFAULT ''
                )
                ''')
                columns = {
                    row[1] for row in db.execute('PRAGMA table_info(request_events)')
                }
                if 'rerank_ms' not in columns:
                    db.execute(
                        'ALTER TABLE request_events ADD COLUMN rerank_ms REAL NOT NULL DEFAULT 0'
                    )
                if 'lock_wait_ms' not in columns:
                    db.execute(
                        'ALTER TABLE request_events ADD COLUMN lock_wait_ms REAL NOT NULL DEFAULT 0'
                    )
                if 'pipeline_load_ms' not in columns:
                    db.execute(
                        'ALTER TABLE request_events ADD COLUMN pipeline_load_ms REAL NOT NULL DEFAULT 0'
                    )
                db.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    request_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    rating INTEGER NOT NULL
                )
                ''')
        finally:
            db.close()

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        return db

    def record_request(self, event):
        now = time.time()
        with closing(self._connect()) as db:
            with db:
                db.execute('''
                INSERT OR REPLACE INTO request_events (
                    request_id, created_at, route, status, duration_ms,
                    rewrite_ms, retrieval_ms, generation_ms, verification_ms,
                    rerank_ms, lock_wait_ms, pipeline_load_ms, rewrite_cache_hit,
                    answer_cache_hit, document_count, citation_count, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event['request_id'], now, event['route'], event['status'],
                    event.get('duration_ms', 0), event.get('rewrite_ms', 0),
                    event.get('retrieval_ms', 0), event.get('generation_ms', 0),
                    event.get('verification_ms', 0), event.get('rerank_ms', 0),
                    event.get('lock_wait_ms', 0), event.get('pipeline_load_ms', 0),
                    int(event.get('rewrite_cache_hit', False)),
                    int(event.get('answer_cache_hit', False)),
                    event.get('document_count', 0), event.get('citation_count', 0),
                    event.get('error_type', ''),
                ))

    def record_feedback(self, request_id, rating):
        with closing(self._connect()) as db:
            with db:
                exists = db.execute(
                    'SELECT 1 FROM request_events WHERE request_id = ?',
                    (request_id,),
                ).fetchone()
                if not exists:
                    return False
                db.execute('''
                INSERT OR REPLACE INTO user_feedback (request_id, created_at, rating)
                VALUES (?, ?, ?)
                ''', (request_id, time.time(), rating))
        return True

    def summary(self, limit=1000):
        with closing(self._connect()) as db:
            rows = db.execute('''
                SELECT status, duration_ms, rewrite_ms, retrieval_ms,
                       generation_ms, verification_ms, rewrite_cache_hit,
                       answer_cache_hit, rerank_ms, lock_wait_ms, pipeline_load_ms
                FROM request_events
                ORDER BY created_at DESC LIMIT ?
            ''', (limit,)).fetchall()
            feedback = db.execute('''
                SELECT rating, COUNT(*) FROM user_feedback
                GROUP BY rating
            ''').fetchall()
        if not rows:
            return {
                'sample_size': 0,
                'status_counts': {},
                'latency_ms': {},
                'rewrite_cache_hits': 0,
                'answer_cache_hits': 0,
                'feedback_counts': {str(r): c for r, c in feedback},
            }
        fields = {
            'total': 1,
            'rewrite': 2,
            'retrieval': 3,
            'generation': 4,
            'verification': 5,
            'rerank': 8,
            'lock_wait': 9,
            'pipeline_load': 10,
        }
        timings = {}
        for name, idx in fields.items():
            values = sorted(float(row[idx]) for row in rows)
            timings[name] = {
                'avg': round(sum(values) / len(values), 2),
                'p95': round(values[max(0, int(len(values) * 0.95) - 1)], 2),
            }
        status_counts = {}
        for row in rows:
            status_counts[row[0]] = status_counts.get(row[0], 0) + 1
        return {
            'sample_size': len(rows),
            'status_counts': status_counts,
            'latency_ms': timings,
            'rewrite_cache_hits': sum(int(row[6]) for row in rows),
            'answer_cache_hits': sum(int(row[7]) for row in rows),
            'feedback_counts': {str(r): c for r, c in feedback},
        }
