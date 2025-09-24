import os
import json
import logging
from typing import Optional, Dict, Any

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


logger = logging.getLogger(__name__)


class VLLMClient:
    """Minimal OpenAI-compatible client for vLLM.

    Environment variables:
      - DOCJA_LLM_ENDPOINT: base URL (default: http://localhost:8000/v1/completions)
      - DOCJA_LLM_MODEL: model name (default: gpt-oss-20B)
    """

    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("DOCJA_LLM_ENDPOINT", "http://localhost:8000/v1/completions")
        self.model = model or os.getenv("DOCJA_LLM_MODEL", "gpt-oss-20B")
        # 推論結果の言語指定（既定: 英語）。DOCJA_LLM_LANG=ja で日本語に変更可。
        self.target_lang = (os.getenv("DOCJA_LLM_LANG", "en") or "en").lower()
        # Harmony formatter
        try:
            from .harmony_formatter import HarmonyFormatter, ReasoningLevel
            self.formatter = HarmonyFormatter()
            self.ReasoningLevel = ReasoningLevel
        except Exception:
            self.formatter = None
            self.ReasoningLevel = None

    def _append_lang_instruction(self, prompt: str, task: str) -> str:
        """Append minimal language/use-style constraints to the prompt.

        task: 'summary' | 'qa' | 'json'
        """
        if self.target_lang == 'en':
            if task == 'summary':
                return f"{prompt}\n\nRespond in English only. Keep it to one line."
            if task == 'qa':
                return f"{prompt}\n\nAnswer concisely in English only. Output just the answer without explanation."
            if task == 'json':
                # JSONは言語非依存だが、不要な説明を防ぐ
                return f"{prompt}\n\nReturn only valid JSON. No extra text."
        elif self.target_lang == 'ja':
            if task == 'summary':
                return f"{prompt}\n\n日本語で一行に要約してください。"
            if task == 'qa':
                return f"{prompt}\n\n日本語で簡潔に、答えのみを出力してください。"
            if task == 'json':
                return f"{prompt}\n\n有効なJSONのみを返してください。説明は不要です。"
        return prompt

    def summarize(self, document: str, max_tokens: int = 256) -> Optional[str]:
        if self.formatter is not None:
            prompt = self.formatter.format_summary_prompt(document, max_length=None, focus_points=None)
        else:
            prompt = f"[文書] {document}\n[質問] 要点を3行で要約して"
        prompt = self._append_lang_instruction(prompt, 'summary')
        # 停止シーケンスは付けず、1行目を採用（改行で空になる事象を回避）
        ans = self._complete(prompt, max_tokens)
        if ans is None:
            return None
        trimmed = ans.strip()
        if not trimmed:
            return None
        return trimmed.splitlines()[0]

    def qa(self, document: str, question: str, max_tokens: int = 128) -> Optional[str]:
        if self.formatter is not None and self.ReasoningLevel is not None:
            prompt = self.formatter.format_prompt(document, question, self.ReasoningLevel.LOW)
        else:
            prompt = f"[文書] {document}\n[質問] {question}"
        # 言語/出力形式の指示を付与
        prompt = self._append_lang_instruction(prompt, 'qa')
        # 停止シーケンスは付けず、1行目のみ採用
        ans = self._complete(prompt, max_tokens)
        if ans is None:
            return None
        # Trim to first line unless explicitly disabled, guard empty
        trimmed = ans.strip()
        if not trimmed:
            return None
        if os.getenv("DOCJA_LLM_QA_NO_TRIM", "0") != "1":
            lines = trimmed.splitlines()
            return lines[0] if lines else trimmed
        return trimmed

    def extract_json(self, document: str, schema: Dict[str, Any], max_tokens: int = 256) -> Optional[str]:
        if self.formatter is not None:
            prompt = self.formatter.format_json_extraction_prompt(document, schema)
        else:
            prompt = f"[文書] {document}\n[質問] {json.dumps(schema, ensure_ascii=False)} でJSON抽出"
        # 説明を抑制
        prompt = self._append_lang_instruction(prompt, 'json')
        ans = self._complete(prompt, max_tokens)
        if ans is None:
            return None
        # JSON抽出の簡易フォールバック
        js = self._coerce_json(ans)
        return js if js is not None else ans

    def _complete(self, prompt: str, max_tokens: int, stop: Optional[list] = None) -> Optional[str]:
        if requests is None:
            logger.warning("requests not installed; skipping LLM call")
            return None
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }
            if stop:
                payload["stop"] = stop
            # Allow override of timeout via env; default 30s
            try:
                timeout = float(os.getenv("DOCJA_LLM_TIMEOUT", "30"))
            except Exception:
                timeout = 30.0
            res = requests.post(self.endpoint, json=payload, timeout=timeout)
            res.raise_for_status()
            data = res.json()
            # OpenAI-compatible: choices[0].text
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    text = choices[0].get("text")
                    if text is not None:
                        return str(text)
            return None
        except Exception as e:  # pragma: no cover
            logger.warning(f"LLM request failed: {e}")
            return None

    def _coerce_json(self, text: str) -> Optional[str]:
        # 純粋JSONでなければ先頭の{}ブロックを抽出して試行
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

    # --- Generic chat over text-only completions ---
    def chat(self, messages: list, max_tokens: int = 512) -> Optional[str]:
        # Simple role-tagged prompt
        sys = ''
        conv = []
        for m in messages:
            role = str(m.get('role','user'))
            content = str(m.get('content',''))
            if role == 'system':
                sys = content
            else:
                conv.append(f"{role.capitalize()}: {content}")
        prompt = ''
        if sys:
            prompt += f"[System]\n{sys}\n\n"
        prompt += "\n".join(conv) + "\nAssistant:"
        return self._complete(prompt, max_tokens)
