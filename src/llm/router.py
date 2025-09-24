from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LLMRouterClient:
    """Route LLM tasks to a selected provider.

    DOCJA_LLM_PROVIDER: 'gptoss' | 'gemma3' | 'ollama' | 'openai-compat'
    DOCJA_LLM_LANG: 'ja' | 'en'
    DOCJA_LLM_USE_IMAGE: '1' to send first page image to gemma3
    """

    def __init__(self):
        self.provider_name = (os.getenv('DOCJA_LLM_PROVIDER', 'gptoss') or 'gptoss').lower()
        self.target_lang = (os.getenv('DOCJA_LLM_LANG', 'ja') or 'ja').lower()
        self._provider = None
        try:
            if self.provider_name == 'gptoss':
                from .providers.gptoss_harmony import GptOssHarmonyProvider
                self._provider = GptOssHarmonyProvider()
            elif self.provider_name == 'gemma3':
                from .providers.gemma_openai import GemmaOpenAIProvider
                self._provider = GemmaOpenAIProvider()
            elif self.provider_name == 'ollama':
                from .providers.ollama import OllamaProvider
                self._provider = OllamaProvider()
            else:
                from ..llm_client import VLLMClient
                self._provider = VLLMClient()
        except Exception:
            # last resort fallback to text-only client (may still require endpoint)
            try:
                from ..llm_client import VLLMClient
                self._provider = VLLMClient()
            except Exception:
                self._provider = None

    def summarize(self, document: str, images: Optional[List[Any]] = None) -> Optional[str]:
        if hasattr(self._provider, 'summarize'):
            # gptoss ignores images; gemma may use one image
            try:
                return self._provider.summarize(document, images=images)  # type: ignore[arg-type]
            except TypeError:
                return self._provider.summarize(document)  # type: ignore[call-arg]
        return None

    def qa(self, document: str, question: str, images: Optional[List[Any]] = None) -> Optional[str]:
        if hasattr(self._provider, 'qa'):
            try:
                return self._provider.qa(document, question, images=images)  # type: ignore[arg-type]
            except TypeError:
                # gptoss variant
                return self._provider.qa(document, question)  # type: ignore[call-arg]
        return None

    def extract_json(self, document: str, schema: Dict[str, Any], images: Optional[List[Any]] = None) -> Optional[str]:
        if hasattr(self._provider, 'extract_json'):
            try:
                return self._provider.extract_json(document, schema, images=images)  # type: ignore[arg-type]
            except TypeError:
                return self._provider.extract_json(document, schema)  # type: ignore[call-arg]
        return None

    def chat(self, messages: List[Dict[str, Any]], image: Optional[Any] = None) -> Optional[str]:
        # gemma3 -> provider.chat; gptoss -> text-only chat
        try:
            if self._provider is None:
                return None
            # gemma3 with image
            if hasattr(self._provider, 'chat'):
                try:
                    return self._provider.chat(messages, image)  # type: ignore[call-arg]
                except TypeError:
                    return self._provider.chat(messages)  # type: ignore[call-arg]
            return None
        except Exception:
            return None
