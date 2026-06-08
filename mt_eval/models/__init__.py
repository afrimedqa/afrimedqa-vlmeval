import os

from .chat_model import ChatModel
from .gemini_model import GeminiModel
from .hf_model import HFModel
from .seq2seq_model import Seq2SeqModel
from .vllm_model import VLLMModel


def _resolve_api_key(value: str) -> str:
    """If value names an env var, return its value; otherwise return it as-is."""
    return os.environ.get(value, value)


def build_model(model_cfg: dict):
    """ dispatch to the correct model class based on config 'type'.

    Supported types:
      - "vllm"    : HuggingFace model loaded in-process via vLLM Python API
      - "openai"  : OpenAI API Endpoint
      - "gemini"  : Google GenAI SDK (Gemini 2.0+)
      - "seq2seq" : Encoder-decoder translation model (NLLB-200, MarianMT, etc.)
    """
    model_type = model_cfg.get('type', 'gemini')

    if model_type == 'vllm':
        return VLLMModel(
            model_name=model_cfg['model_name'],
            max_tokens=model_cfg.get('max_tokens', 512),
            temperature=model_cfg.get('temperature', 0.0),
            repetition_penalty=model_cfg.get('repetition_penalty', 1.15),
            **model_cfg.get('vllm_kwargs', {}),
        )

    if model_type == 'openai':
        return ChatModel(
            model_name=model_cfg['model_name'],
            api_key=_resolve_api_key(model_cfg['api_key']),
            base_url=model_cfg.get('base_url', None),
            max_tokens=model_cfg.get('max_tokens', 512),
            temperature=model_cfg.get('temperature', 0.0),
        )

    if model_type == 'gemini':
        return GeminiModel(
            model_name=model_cfg['model_name'],
            api_key=_resolve_api_key(model_cfg['api_key']),
            max_tokens=model_cfg.get('max_tokens', 512),
            temperature=model_cfg.get('temperature', 0.0),
        )

    if model_type == 'hf':
        return HFModel(
            model_name=model_cfg['model_name'],
            max_tokens=model_cfg.get('max_tokens', 512),
            temperature=model_cfg.get('temperature', 0.0),
            repetition_penalty=model_cfg.get('repetition_penalty', 1.15),
            **model_cfg.get('hf_kwargs', {}),
        )

    if model_type == 'seq2seq':
        return Seq2SeqModel(
            model_name=model_cfg['model_name'],
            src_lang=model_cfg['src_lang'],
            tgt_lang=model_cfg['tgt_lang'],
            max_tokens=model_cfg.get('max_tokens', 512),
            device=model_cfg.get('device', None),
        )

    raise ValueError(
        f"Unknown model type '{model_type}'. Choose from: vllm, hf, openai, gemini, seq2seq"
    )
