from __future__ import annotations

import json
import os

import requests

from .base import BaseMTModel


class AzureOpenAIModel(BaseMTModel):
    """GPT models via Azure AI Foundry (Azure OpenAI deployment).

    Reads from env:
        AZURE_FOUNDRY_GPT5_URL     – full deployment URL incl. api-version query param
        AZURE_FOUNDRY_GPT5_API_KEY – subscription key
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.url = os.environ.get('AZURE_FOUNDRY_GPT5_URL', '')
        assert self.url, 'Set AZURE_FOUNDRY_GPT5_URL in your environment.'

        self.headers = {
            'api-key': api_key,
            'content-type': 'application/json',
        }
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def translate(self, messages: list[dict]) -> str:
        payload = {
            'model': self.model_name,
            'messages': messages,
            'max_completion_tokens': self.max_tokens,
        }

        response = requests.post(
            self.url, headers=self.headers,
            data=json.dumps(payload), timeout=120
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
