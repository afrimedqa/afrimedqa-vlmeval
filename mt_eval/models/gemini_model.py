from __future__ import annotations

import time

from google import genai
from google.genai import types

from .base import BaseMTModel

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds


class GeminiModel(BaseMTModel):
    """Google GenAI SDK (Gemini 2.0+).

    Expects messages in the  chat format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = genai.Client(api_key=api_key)

    def translate(self, messages: list[dict]) -> str:
        system_msg = next(
            (m['content'] for m in messages if m['role'] == 'system'), None
        )
        user_msg = next(
            (m['content'] for m in messages if m['role'] == 'user'), ''
        )

        config = types.GenerateContentConfig(
            system_instruction=system_msg,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        for attempt in range(1, _MAX_RETRIES + 1):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_msg,
                config=config,
            )
            if response.text is not None:
                return response.text
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

        return ""
