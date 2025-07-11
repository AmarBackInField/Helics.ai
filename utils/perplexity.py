from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult
import requests
from typing import List

class PerplexityLLM(LLM):
    api_key: str 
    model: str = "sonar-pro"
    endpoint: str = "https://api.perplexity.ai/chat/completions"

    def _call(self, prompt: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "search_after_date_filter": "3/1/2025",
            "search_before_date_filter": "3/5/2025",
            "response_format": {
                    "type": "json_schema",
                "json_schema": {"schema": AnswerFormat.model_json_schema()},
            },
            "web_search_options": {
            "search_context_size": "medium"
        }
        }

        response = requests.post(self.endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise ValueError(f"API Error {response.status_code}: {response.text}")

    @property
    def _llm_type(self) -> str:
        return "perplexity"

