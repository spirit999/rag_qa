import os
import threading

from .config import (
    MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE,
)


class OpenAIChat:
    supports_max_new_tokens = True

    def __init__(self, base_url=None, model=None, api_key='EMPTY'):
        from openai import OpenAI
        self.base_url = base_url or os.environ.get('QR_BASE_URL', 'http://127.0.0.1:8000/v1')
        self.model = model or os.environ.get('QR_MODEL', MODEL_PATH)
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=120)

    def chat(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content

    def chat_stream(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_new_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content


class HFChat:
    supports_max_new_tokens = True

    def __init__(self, model_path=MODEL_PATH, device_map='auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device_map,
        )
        self.model.eval()

    def chat(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        return self.chat_batch([messages], max_new_tokens=max_new_tokens)[0]

    def chat_batch(self, messages_list, max_new_tokens=MAX_NEW_TOKENS):
        import torch
        texts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for m in messages_list
        ]
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                top_p=0.9 if TEMPERATURE > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        results = []
        for seq in out:
            gen = seq[inputs['input_ids'].shape[1]:]
            results.append(self.tokenizer.decode(gen, skip_special_tokens=True))
        return results

    def chat_stream(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        import torch
        from transformers import TextIteratorStreamer

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        errors = []

        def generate():
            try:
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=TEMPERATURE > 0,
                        temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                        top_p=0.9 if TEMPERATURE > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        streamer=streamer,
                    )
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()
        if errors:
            raise errors[0]


def make_llm(backend=None):
    if backend is None:
        backend = os.environ.get('QR_BACKEND', 'openai' if os.environ.get('QR_BASE_URL') else 'hf')
    if backend == 'openai':
        return OpenAIChat()
    if backend == 'hf':
        return HFChat()
    raise ValueError('unknown backend: ' + backend)
