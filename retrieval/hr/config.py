from pathlib import Path

PROJECT = Path('/home/ps/wuzhichao/agent_rag')
DOCS_DIR = PROJECT / 'docs'
PARSED_DIR = PROJECT / 'data' / 'parsed'
MANIFEST_PATH = PROJECT / 'data' / 'manifest.jsonl'
GLOSSARY_CSV = PROJECT / 'query-rewrite' / 'glossary' / '01_核心术语对照表.csv'
DOC_LIST_CSV = PROJECT / 'query-rewrite' / 'glossary' / '00_制度清单.csv'

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULT_DIR = Path(__file__).resolve().parent.parent / 'result'
CHUNKS_PATH = DATA_DIR / 'chunks.jsonl'
BM25_PATH = DATA_DIR / 'bm25.pkl'
VEC_PATH = DATA_DIR / 'vectors.npy'
META_PATH = DATA_DIR / 'vectors_meta.json'
INDEX_GENERATIONS_DIR = DATA_DIR / 'indexes'
INDEX_CURRENT_PATH = DATA_DIR / 'current_index.json'

EMBED_MODEL_PATH = '/home/ps/Model/Qwen3-Embedding-0.6B'
RERANK_MODEL_PATH = '/home/ps/.cache/models/Qwen3-Reranker-0.6B'
RERANK_INSTRUCTION = '给定企业制度相关问题，判断该条款是否能回答这个问题'
QUERY_INSTRUCTION = (
    'Instruct: Given a web search query, '
    'retrieve relevant passages that answer the query' + chr(10) + 'Query:'
)

EVAL_QUERIES = PROJECT / 'query-rewrite' / 'data' / 'eval' / 'queries.jsonl'
REWRITE_RESULTS = PROJECT / 'query-rewrite' / 'result' / 'rewrite_results.jsonl'

MAX_CHUNK_CHARS = 480
MIN_CHUNK_CHARS = 60
OVERLAP_CHARS = 80

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
