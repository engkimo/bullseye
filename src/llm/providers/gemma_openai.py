from __future__ import annotations

import os
import base64
import json
import logging
from io import BytesIO
from typing import Optional, Dict, Any, List

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)


def _img_to_data_url(img) -> str:
    try:
        from PIL import Image
        if not isinstance(img, Image.Image):
            return ''
        buf = BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ''


class GemmaOpenAIProvider:
    """
    Provider for Gemma 3 via OpenAI-compatible API (chat/completions).

    Env vars:
      - DOCJA_LLM_ENDPOINT (e.g., http://localhost:8000/v1)
      - DOCJA_LLM_MODEL    (e.g., google/gemma-3-12b-it)
      - DOCJA_LLM_TIMEOUT  (default 30)
      - DOCJA_LLM_LANG     (ja|en; default ja)
    """

    def __init__(self):
        if requests is None:  # pragma: no cover
            raise RuntimeError('requests is required for GemmaOpenAIProvider')
        self.base = os.getenv('DOCJA_LLM_ENDPOINT', 'http://localhost:8000/v1').rstrip('/')
        self.model = os.getenv('DOCJA_LLM_MODEL', 'google/gemma-3-12b-it')
        self.timeout = float(os.getenv('DOCJA_LLM_TIMEOUT', '30'))
        self.lang = (os.getenv('DOCJA_LLM_LANG', 'ja') or 'ja').lower()

    # --- Public tasks ---
    def summarize(self, document: str, images: Optional[List[Any]] = None) -> Optional[str]:
        sys_msg = "You are a helpful Japanese assistant. Prefer concise answers."
        user_text = f"以下の文書を日本語で一行要約してください。\n\n{document}"
        return self._chat(sys_msg, user_text, images=images, json_mode=False)

    def qa(self, document: str, question: str, images: Optional[List[Any]] = None) -> Optional[str]:
        sys_msg = "You are a helpful Japanese assistant. Respond with the answer only."
        user_text = f"[文書]\n{document}\n\n[質問]\n{question}\n\n日本語で回答のみ返してください。"
        return self._chat(sys_msg, user_text, images=images, json_mode=False)

    def extract_json(self, document: str, schema: Dict[str, Any], images: Optional[List[Any]] = None) -> Optional[str]:
        sys_msg = "You are a JSON-only assistant. Always return valid JSON matching the provided schema."
        user_text = f"[文書]\n{document}\n\n[スキーマ]\n{json.dumps(schema, ensure_ascii=False)}\n\nJSONのみを返してください。"
        return self._chat(sys_msg, user_text, images=images, json_mode=True)

    # --- Internal ---
    def _chat(self, system_text: str, user_text: str, images: Optional[List[Any]], json_mode: bool) -> Optional[str]:
        url = f"{self.base}/chat/completions"
        content = [{"type": "text", "text": user_text}]
        if images:
            # attach at most one image for now
            data_url = _img_to_data_url(images[0])
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            res = requests.post(url, json=payload, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            ch = (data.get('choices') or [{}])[0]
            msg = (ch.get('message') or {})
            text = msg.get('content')
            return text if (text and text.strip()) else None
        except Exception as e:  # pragma: no cover
            logger.warning(f"Gemma chat request failed: {e}")
            return None

    # --- Public generic chat ---
    def chat(self, messages: List[Dict[str, Any]], image: Optional[Any] = None) -> Optional[str]:
        """Generic chat with optional single image (OpenAI-compatible chat/completions).

        messages: [{'role':'system|user|assistant','content':str}, ...]
        image: PIL.Image or None (attached to the last user message)
        """
        if requests is None:
            return None
        url = f"{self.base}/chat/completions"
        # Build messages array; attach image to the last user message if provided
        openai_msgs: List[Dict[str, Any]] = []
        last_user_idx = -1
        for m in messages:
            role = str(m.get('role','user'))
            content_txt = str(m.get('content',''))
            openai_msgs.append({"role": role, "content": [{"type":"text","text": content_txt}]})
            if role == 'user':
                last_user_idx = len(openai_msgs) - 1
        if image is not None and last_user_idx >= 0:
            try:
                data_url = _img_to_data_url(image)
                if data_url:
                    openai_msgs[last_user_idx]['content'].append({"type":"image_url","image_url":{"url": data_url}})
            except Exception:
                pass
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_msgs,
            "temperature": 0.2,
            "max_tokens": 512,
        }
        try:
            res = requests.post(url, json=payload, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            ch = (data.get('choices') or [{}])[0]
            msg = (ch.get('message') or {})
            text = msg.get('content')
            return text if (text and text.strip()) else None
        except Exception as e:
            logger.warning(f"Gemma chat request failed: {e}")
            return None
