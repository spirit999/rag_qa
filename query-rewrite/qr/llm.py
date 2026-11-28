import os
import threading

from .config import (
    ATTENTION_IMPLEMENTATION, MODEL_PATH, MAX_NEW_TOKENS, QUANTIZATION,
    SMALL_MODEL_PATH, TEMPERATURE,
)


class OpenAIChat:
    supports_max_new_tokens = True

    def __init__(self, base_url=None, model=None, api_key='EMPTY'):
        from openai import OpenAI
        self.base_url = base_url or os.environ.get('QR_BASE_URL', 'http://127.0.0.1:8000/v1')
        self.model = model or os.environ.get('QR_MODEL', MODEL_PATH)
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=120)
        self.disable_thinking = os.environ.get(
            'QR_DISABLE_THINKING', '',
        ).lower() in ('1', 'true', 'yes', 'on')

    def _chat_template_options(self):
        if not self.disable_thinking:
            return {}
        return {
            'extra_body': {
                'chat_template_kwargs': {'enable_thinking': False},
            },
        }

    def chat(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_new_tokens,
            **self._chat_template_options(),
        )
        return resp.choices[0].message.content

    def chat_stream(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_new_tokens,
            stream=True,
            **self._chat_template_options(),
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content


class HFChat:
    supports_max_new_tokens = True

    def __init__(self, model_path=MODEL_PATH, device_map=None,
                 inference_lock=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device_map is None:
            device_map = {'': 0} if torch.cuda.is_available() else 'auto'
        self.inference_lock = inference_lock or threading.RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        load_options = {
            'dtype': torch.bfloat16,
            'device_map': device_map,
            'attn_implementation': ATTENTION_IMPLEMENTATION,
        }
        if QUANTIZATION != 'none':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    'QR_QUANTIZATION requires a CUDA device; got %s.'
                    % QUANTIZATION
                )
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    'QR_QUANTIZATION requires a Transformers build with '
                    'BitsAndBytesConfig and a compatible bitsandbytes install.'
                ) from exc
            if QUANTIZATION == '4bit':
                load_options['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_options['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_options,
            )
        except (ImportError, OSError, RuntimeError) as exc:
            if QUANTIZATION != 'none':
                raise RuntimeError(
                    'Failed to load %s-bit quantized model. Check that '
                    'bitsandbytes and its CUDA libraries match this PyTorch '
                    'environment.' % QUANTIZATION
                ) from exc
            raise
        self.model.eval()

    def chat(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        return self.chat_batch([messages], max_new_tokens=max_new_tokens)[0]

    def chat_batch(self, messages_list, max_new_tokens=MAX_NEW_TOKENS):
        import torch
        with self.inference_lock:
            texts = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                for m in messages_list
            ]
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                texts, return_tensors='pt', padding=True,
            ).to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=TEMPERATURE > 0,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                    top_p=0.9 if TEMPERATURE > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
        results = []
        for seq in out:
            gen = seq[inputs['input_ids'].shape[1]:]
            results.append(self.tokenizer.decode(gen, skip_special_tokens=True))
        return results

    def chat_stream(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        import torch
        from transformers import TextIteratorStreamer

        with self.inference_lock:
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
                    with torch.inference_mode():
                        self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=TEMPERATURE > 0,
                            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                            top_p=0.9 if TEMPERATURE > 0 else None,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=True,
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


def make_llm(backend=None, inference_lock=None):
    if backend is None:
        backend = os.environ.get('QR_BACKEND', 'openai' if os.environ.get('QR_BASE_URL') else 'hf')
    if backend == 'openai':
        return OpenAIChat()
    if backend == 'hf':
        return HFChat(inference_lock=inference_lock)
    raise ValueError('unknown backend: ' + backend)


def make_small_llm(inference_lock=None):
    if not SMALL_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            '小模型目录不存在：%s' % SMALL_MODEL_PATH
        )
    return HFChat(
        model_path=str(SMALL_MODEL_PATH),
        inference_lock=inference_lock,
    )
