import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from .config import RERANK_INSTRUCTION, RERANK_MODEL_PATH

PREFIX = (
    '<|im_start|>system' + chr(10) +
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".' +
    '<|im_end|>' + chr(10) + '<|im_start|>user' + chr(10)
)
SUFFIX = (
    '<|im_end|>' + chr(10) + '<|im_start|>assistant' + chr(10) +
    '<think>' + chr(10) + chr(10) + '</think>' + chr(10) + chr(10)
)
NL = chr(10)


def format_pair(instruct, query, doc):
    return ('<Instruct>: ' + instruct + NL +
            '<Query>: ' + query + NL +
            '<Document>: ' + doc)


class QwenReranker:
    def __init__(self, path=RERANK_MODEL_PATH, max_len=640, batch_size=16):
        path = Path(path)
        if not (
            (path / 'model.safetensors').exists()
            or list(path.glob('model-*.safetensors'))
            or (path / 'pytorch_model.bin').exists()
        ):
            raise FileNotFoundError(
                'Reranker 权重不存在：%s；请安装完整的 Qwen3-Reranker 模型后重试。'
                % path
            )
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
        self.model.eval()
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.token_yes = self.tokenizer.convert_tokens_to_ids('yes')
        self.token_no = self.tokenizer.convert_tokens_to_ids('no')
        self.budget = max_len - len(self.prefix_ids) - len(self.suffix_ids)

    @torch.no_grad()
    def score(self, query, docs, instruct=RERANK_INSTRUCTION, batch_size=None, show_progress=False):
        bs = batch_size or self.batch_size
        pairs = [format_pair(instruct, query, d) for d in docs]
        scores = []
        rng = range(0, len(pairs), bs)
        if show_progress:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc='Rerank')
            except ImportError:
                pass
        for i in rng:
            batch = pairs[i:i + bs]
            enc = self.tokenizer(
                batch, padding=False, truncation=True,
                max_length=self.budget, return_attention_mask=False,
            )
            seqs = [self.prefix_ids + ids + self.suffix_ids for ids in enc['input_ids']]
            maxlen = max(len(s) for s in seqs)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            input_ids = torch.tensor(
                [[pad_id] * (maxlen - len(s)) + s for s in seqs], dtype=torch.long,
            )
            attention = torch.ones_like(input_ids)
            attention[input_ids == pad_id] = 0
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits[:, -1, :]
            pair_logits = torch.stack(
                [logits[:, self.token_no], logits[:, self.token_yes]], dim=1,
            )
            probs = F.log_softmax(pair_logits, dim=1)[:, 1].exp()
            scores.extend(probs.tolist())
        return scores
