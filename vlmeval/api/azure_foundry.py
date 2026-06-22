from vlmeval.smp import *
from vlmeval.api.claude import Claude3V, Claude_Wrapper
from vlmeval.api.gpt import GPT4V, OpenAIWrapper
from vlmeval.api.base import BaseAPI
import os
import requests
import json


class AzureFoundryWrapper(Claude_Wrapper):
    """Claude models via Azure AI Foundry (Serverless API).

    Required env vars:
        AZURE_FOUNDRY_ENDPOINT  
        AZURE_FOUNDRY_API_KEY   – Azure subscription key

    The wrapper calls {AZURE_FOUNDRY_ENDPOINT}/anthropic/v1/messages using the
    Anthropic-native Messages API format, authenticated with the 'api-key' header.
    """

    is_api: bool = True

    def __init__(self,
                 model: str = 'claude-opus-4-7',
                 key: str = None,
                 endpoint: str = None,
                 retry: int = 10,
                 timeout: int = 120,
                 wait: int = 5,
                 system_prompt: str = None,
                 verbose: bool = False,
                 temperature: float = 0,
                 max_tokens: int = 2048,
                 **kwargs):

        if endpoint is None:
            endpoint = os.environ.get('AZURE_FOUNDRY_ENDPOINT', '')
        assert endpoint, (
            'Azure Foundry endpoint not set. '
            'Set AZURE_FOUNDRY_ENDPOINT, e.g. '
        )

        if key is None:
            key = os.environ.get('AZURE_FOUNDRY_API_KEY', '')
        assert key, 'Set AZURE_FOUNDRY_API_KEY to your Azure AI Foundry subscription key.'

        # Resolve logical model name to deployment name via env var so deployment
        # names never need to appear in configs or code.
        _deployment_env = {
            'claude-opus-4-7': 'AZURE_FOUNDRY_CLAUDE_OPUS47_DEPLOYMENT',
            'claude-sonnet-4-6': 'AZURE_FOUNDRY_CLAUDE_SONNET46_DEPLOYMENT',
        }
        env_key = _deployment_env.get(model)
        deployment = os.environ.get(env_key, model) if env_key else model

        self.backend = 'azure_foundry'
        self.url = f'{endpoint.rstrip("/")}/anthropic/v1/messages'
        self.model = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = {
            'Authorization': f'Bearer {key}',
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        self.timeout = timeout
        self.key = key

        BaseAPI.__init__(
            self,
            retry=retry,
            wait=wait,
            verbose=verbose,
            system_prompt=system_prompt,
            **kwargs
        )

    def generate_inner(self, inputs, **kwargs) -> str:
        payload = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': self.prepare_inputs(inputs),
            **kwargs
        }
        if self.system_prompt is not None:
            payload['system'] = self.system_prompt

        response = requests.request(
            'POST', self.url, headers=self.headers,
            data=json.dumps(payload), timeout=self.timeout * 1.1
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg

        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['content'][0]['text'].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response


class AzureFoundryVLM(AzureFoundryWrapper):

    def generate(self, message, dataset=None):
        return super(Claude_Wrapper, self).generate(message)


class AzureFoundryOpenAIWrapper(GPT4V):
    """GPT-5 (and other OpenAI-compatible models) via Azure AI Foundry.

    Required env vars:
        AZURE_FOUNDRY_GPT5_URL      – full deployment URL including api-version query param

        AZURE_FOUNDRY_GPT5_API_KEY  – Azure subscription key

    Authentication uses the 'api-key' header (Azure OpenAI style).
    """

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-5',
                 key: str = None,
                 api_url: str = None,
                 retry: int = 10,
                 timeout: int = 120,
                 wait: int = 5,
                 system_prompt: str = None,
                 verbose: bool = False,
                 temperature: float = 0,
                 max_tokens: int = 2048,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 **kwargs):

        if api_url is None:
            api_url = os.environ.get('AZURE_FOUNDRY_GPT5_URL', '')
        assert api_url, (
            'Azure Foundry GPT-5 URL not set. '
            'Set AZURE_FOUNDRY_GPT5_URL to the full deployment URL including api-version.'
        )

        if key is None:
            key = os.environ.get('AZURE_FOUNDRY_GPT5_API_KEY', '')
        assert key, 'Set AZURE_FOUNDRY_GPT5_API_KEY to your Azure subscription key.'

        # Set all fields that generate_inner / prepare_inputs expect, then call
        # BaseAPI.__init__ directly to avoid OpenAIWrapper's endpoint-construction logic.
        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = True        # triggers 'api-key' header in generate_inner
        self.api_base = api_url
        self.key = key
        self.img_size = img_size
        self.img_detail = img_detail
        self.timeout = timeout
        # GPT-5 uses max_completion_tokens like the o-series
        self.o1_model = 'o1' in model or 'o3' in model or 'gpt-5' in model

        BaseAPI.__init__(
            self,
            retry=retry,
            wait=wait,
            verbose=verbose,
            system_prompt=system_prompt,
            **kwargs
        )
