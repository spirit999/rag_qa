import os
from pathlib import Path

PROJECT = Path('/home/ps/wuzhichao/agent_rag')
GLOSSARY_DIR = PROJECT / 'query-rewrite' / 'glossary'
PARSED_DIR = PROJECT / 'data' / 'parsed'
MANIFEST = PROJECT / 'data' / 'manifest.jsonl'
EVAL_DIR = PROJECT / 'query-rewrite' / 'data' / 'eval'
RESULT_DIR = PROJECT / 'query-rewrite' / 'result'
MODEL_PATH = '/home/ps/Model/Qwen3-8B'
SMALL_MODEL_PATH = Path(os.environ.get(
    'QR_SMALL_MODEL_PATH',
    '/home/ps/wuzhichao/base_model/qwen2.5-0.5B-Instruct',
))
SMALL_MODEL_MODE = os.environ.get('QR_SMALL_MODEL_MODE', 'off').lower()
if SMALL_MODEL_MODE not in ('off', 'cascade'):
    raise ValueError('QR_SMALL_MODEL_MODE must be "off" or "cascade"')
QUANTIZATION = os.environ.get('QR_QUANTIZATION', 'none').lower()
if QUANTIZATION not in ('none', '8bit', '4bit'):
    raise ValueError('QR_QUANTIZATION must be "none", "8bit", or "4bit"')
ATTENTION_IMPLEMENTATION = os.environ.get(
    'QR_ATTENTION_IMPLEMENTATION', 'sdpa',
).lower()
if ATTENTION_IMPLEMENTATION not in ('sdpa', 'eager', 'flash_attention_2'):
    raise ValueError(
        'QR_ATTENTION_IMPLEMENTATION must be "sdpa", "eager", or "flash_attention_2"'
    )
EMBED_MODEL_PATH = '/home/ps/Model/Qwen3-Embedding-0.6B'

CATEGORIES = [
    '人力资源', '财务管理', '党的建设', '法人治理', '公共管理', '考核办法',
    '科技创新', '科研生产', '内部监督', '市场营销', '业务支撑', '质量技术', '资产经营',
]

INTENTS = ['事实数字', '流程手续', '条件资格', '计算判断', '制度元信息', '闲聊超域']

MAX_NEW_TOKENS = int(os.environ.get('QR_MAX_NEW_TOKENS', '512'))
MAX_REWRITE_NEW_TOKENS = 128
TEMPERATURE = 0.1

PROMPT_VERSION = 'v4'  # prompt 变更时递增，旧缓存自动失效
