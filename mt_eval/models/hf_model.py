from __future__ import annotations

from .base import BaseMTModel


class HFModel(BaseMTModel):
    """Runs a HuggingFace Language model using the transformers library.

    set device_map="auto" by default to spread across available GPUs.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        **model_kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        dtype = getattr(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=dtype,
                **model_kwargs,
            )
        except (ValueError, OSError):
            # VLMs (e.g. Qwen3-VL) register under AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=dtype,
                **model_kwargs,
            )
        self.model.eval()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

    def translate(self, messages: list[dict]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = self.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            repetition_penalty=self.repetition_penalty,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
