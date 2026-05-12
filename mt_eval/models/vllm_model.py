from __future__ import annotations

from .base import BaseMTModel


class VLLMModel(BaseMTModel):
    """Runs a HuggingFace model in-process via the vLLM Python API.

    Messages are formatted using the model's chat template and passed to LLM.generate().
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        **vllm_kwargs,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )
        trust_remote_code = vllm_kwargs.get("trust_remote_code", False)
        self.llm = LLM(model=model_name, **vllm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    def translate(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text
