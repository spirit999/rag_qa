from pathlib import Path

PROJECT = Path('/home/ps/wuzhichao/agent_rag')
GLOSSARY_DIR = PROJECT / 'query-rewrite' / 'glossary'
PARSED_DIR = PROJECT / 'data' / 'parsed'
MANIFEST = PROJECT / 'data' / 'manifest.jsonl'
EVAL_DIR = PROJECT / 'query-rewrite' / 'data' / 'eval'
RESULT_DIR = PROJECT / 'query-rewrite' / 'result'
MODEL_PATH = '/home/ps/Model/Qwen3-8B'
EMBED_MODEL_PATH = '/home/ps/Model/Qwen3-Embedding-0.6B'

CATEGORIES = [
    '人力资源', '财务管理', '党的建设', '法人治理', '公共管理', '考核办法',
    '科技创新', '科研生产', '内部监督', '市场营销', '业务支撑', '质量技术', '资产经营',
]

INTENTS = ['事实数字', '流程手续', '条件资格', '计算判断', '制度元信息', '闲聊超域']

MAX_NEW_TOKENS = 700
MAX_REWRITE_NEW_TOKENS = 192
TEMPERATURE = 0.1

PROMPT_VERSION = 'v3'  # prompt 变更时递增，旧缓存自动失效
