from __future__ import annotations

import json
import os

import requests

from .base import BaseMTModel


class AnthropicFoundryModel(BaseMTModel):
    """Claude models via Azure AI Foundry (Anthropic-native Messages API).

    Reads from env:
        AZURE_FOUNDRY_ENDPOINT               – base URL
        AZURE_FOUNDRY_API_KEY                – Bearer token key
        AZURE_FOUNDRY_CLAUDE_OPUS47_DEPLOYMENT   – deployment name for claude-opus-4-7
        AZURE_FOUNDRY_CLAUDE_SONNET46_DEPLOYMENT – deployment name for claude-sonnet-4-6
    """

    _DEPLOYMENT_ENV = {
        'claude-opus-4-7': 'AZURE_FOUNDRY_CLAUDE_OPUS47_DEPLOYMENT',
        'claude-sonnet-4-6': 'AZURE_FOUNDRY_CLAUDE_SONNET46_DEPLOYMENT',
    }

    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        endpoint = os.environ.get('AZURE_FOUNDRY_ENDPOINT', '').rstrip('/')
        assert endpoint, 'Set AZURE_FOUNDRY_ENDPOINT in your environment.'

        env_key = self._DEPLOYMENT_ENV.get(model_name)
        self.deployment = os.environ.get(env_key, model_name) if env_key else model_name

        self.url = f'{endpoint}/anthropic/v1/messages'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        self.max_tokens = max_tokens
        self.temperature = temperature

    def translate(self, messages: list[dict]) -> str:
        system = None
        user_msgs = []
        for m in messages:
            if m['role'] == 'system':
                system = m['content']
            else:
                user_msgs.append(m)

        payload: dict = {
            'model': self.deployment,
            'max_tokens': self.max_tokens,
            'messages': user_msgs,
        }
        if system:
            payload['system'] = system

        response = requests.post(
            self.url, headers=self.headers,
            data=json.dumps(payload), timeout=120
        )
        response.raise_for_status()
        body = response.json()
        content = body.get('content', [])
        if not content:
            raise RuntimeError(
                f'AnthropicFoundry returned empty content. '
                f'stop_reason={body.get("stop_reason")!r} '
                f'full_response={body}'
            )
        return content[0]['text']
