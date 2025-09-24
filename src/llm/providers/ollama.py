from __future__ import annotations

import os
import json
import logging
from typing import Optional, Dict, Any, List

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Provider for Ollama local server.

    Env vars:
      - DOCJA_OLLAMA_ENDPOINT: base URL (default: http://localhost:11434)
      - DOCJA_OLLAMA_MODEL: model name (default: gptoss-20b)
      - DOCJA_LLM_TIMEOUT: request timeout seconds (default: 30)
      - DOCJA_LLM_LANG: ja|en (default: ja)
    """

    def __init__(self):
        if requests is None:  # pragma: no cover
            raise RuntimeError('requests is required for OllamaProvider')
        base = os.getenv('DOCJA_OLLAMA_ENDPOINT', 'http://localhost:11434').rstrip('/')
        self.base_generate = f"{base}/api/generate"
        self.base_chat = f"{base}/api/chat"
        self.model = os.getenv('DOCJA_OLLAMA_MODEL', 'gptoss-20b')
        try:
            self.timeout = float(os.getenv('DOCJA_LLM_TIMEOUT', '30'))
        except Exception:
            self.timeout = 30.0
        self.lang = (os.getenv('DOCJA_LLM_LANG', 'ja') or 'ja').lower()

    # --- Public tasks (prompt-oriented) ---
    def summarize(self, document: str, images: Optional[List[Any]] = None) -> Optional[str]:
        prompt = f"[文書]\n{document}\n\n[指示] 日本語で一行要約してください。"
        if self.lang == 'en':
            prompt = f"[Document]\n{document}\n\n[Instruction] Summarize in one line in English."
        return self._generate(prompt, max_tokens=256)

    def qa(self, document: str, question: str, images: Optional[List[Any]] = None) -> Optional[str]:
        prompt = f"[文書]\n{document}\n\n[質問]\n{question}\n\n日本語で答えのみを返してください。"
        if self.lang == 'en':
            prompt = f"[Document]\n{document}\n\n[Question]\n{question}\n\nAnswer concisely in English; output only the answer."
        ans = self._generate(prompt, max_tokens=128)
        if ans is None:
            return None
        trimmed = ans.strip()
        return trimmed.splitlines()[0] if trimmed else None

    def extract_json(self, document: str, schema: Dict[str, Any], images: Optional[List[Any]] = None) -> Optional[str]:
        schema_str = json.dumps(schema, ensure_ascii=False)
        if self.lang == 'en':
            prompt = f"[Document]\n{document}\n\n[Schema]\n{schema_str}\n\nReturn only valid JSON."
        else:
            prompt = f"[文書]\n{document}\n\n[スキーマ]\n{schema_str}\n\n有効なJSONのみを返してください。"
        ans = self._generate(prompt, max_tokens=256)
        if ans is None:
            return None
        # Try to coerce to pure JSON
        js = self._coerce_json(ans)
        return js if js is not None else ans

    # --- Public generic chat ---
    def chat(self, messages: List[Dict[str, Any]], image: Optional[Any] = None) -> Optional[str]:
        # images are ignored for now (use gemma3 provider if needed)
        try:
            payload = {
                'model': self.model,
                'messages': [{'role': str(m.get('role','user')), 'content': str(m.get('content',''))} for m in messages],
                'stream': False,
                'options': {
                    'temperature': 0.2,
                },
            }
            res = requests.post(self.base_chat, json=payload, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            # Ollama chat returns { message: { role, content }, ... }
            msg = (data.get('message') or {})
            text = msg.get('content')
            return text if (text and text.strip()) else None
        except Exception as e:  # pragma: no cover
            logger.warning(f"Ollama chat request failed: {e}")
            return None

    # --- Internal helpers ---
    def _generate(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        try:
            payload: Dict[str, Any] = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    # Some Ollama backends may not honor num_predict; keep small
                    'num_predict': max_tokens,
                }
            }
            res = requests.post(self.base_generate, json=payload, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            # Ollama generate returns { response: "...", done: true, ... }
            text = data.get('response')
            return text if (text and str(text).strip()) else None
        except Exception as e:  # pragma: no cover
            logger.warning(f"Ollama generate request failed: {e}")
            return None

    def _coerce_json(self, text: str) -> Optional[str]:
        try:
            import json as _json
            return _json.dumps(_json.loads(text), ensure_ascii=False)
        except Exception:
            pass
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                sub = text[start:end+1]
                import json as _json
                return _json.dumps(_json.loads(sub), ensure_ascii=False)
        except Exception:
            return None
        return None

