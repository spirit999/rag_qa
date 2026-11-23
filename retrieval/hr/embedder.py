import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import EMBED_MODEL_PATH, QUERY_INSTRUCTION


class QwenEmbedder:
    def __init__(self, path=EMBED_MODEL_PATH, max_len=512, batch_size=32):
        self.max_len = max_len
        self.batch_size = batch_size
        self.path = str(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        self.model = AutoModel.from_pretrained(path, dtype=torch.float32)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, is_query=False, batch_size=None, max_len=None, show_progress=False):
        return self._encode_uncached(
            texts, is_query=is_query, batch_size=batch_size,
            max_len=max_len, show_progress=show_progress,
        )

    def _encode_uncached(self, texts, is_query=False, batch_size=None,
                         max_len=None, show_progress=False):
        bs = batch_size or self.batch_size
        ml = max_len or self.max_len
        if is_query:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        outs = []
        rng = range(0, len(texts), bs)
        if show_progress:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc='向量编码')
            except ImportError:
                pass
        for i in rng:
            batch = texts[i:i + bs]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=ml, return_tensors='pt',
            )
            hidden = self.model(**inputs).last_hidden_state
            emb = hidden[:, -1]
            emb = F.normalize(emb, p=2, dim=1)
            outs.append(emb.float().cpu().numpy())
        if not outs:
            return np.zeros((0, 1024), dtype=np.float32)
        return np.vstack(outs)
