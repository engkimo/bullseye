from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class GptOssHarmonyProvider:
    """
    Provider for gpt-oss family using vLLM with OpenAI Harmony encoding/decoding.

    Env vars:
      - DOCJA_GPTOSS_MODEL: path/name to model (default: openai/gpt-oss-20b)
      - DOCJA_GPTOSS_TRUST_REMOTE_CODE: '1' to enable trust_remote_code (default: '1')
      - DOCJA_VLLM_TEMPERATURE: float (default: 0.1)
      - DOCJA_VLLM_MAX_TOKENS: int (default: 256)
    """

    def __init__(self):
        try:
            from openai_harmony import (
                HarmonyEncodingName,
                load_harmony_encoding,
                Conversation,
                Message,
                Role,
                SystemContent,
                DeveloperContent,
            )
            from vllm import LLM, SamplingParams
        except Exception as e:  # pragma: no cover
            raise RuntimeError("openai_harmony and vllm are required for GptOssHarmonyProvider") from e

        self._harmony = {
            'HarmonyEncodingName': HarmonyEncodingName,
            'load_harmony_encoding': load_harmony_encoding,
            'Conversation': Conversation,
            'Message': Message,
            'Role': Role,
            'SystemContent': SystemContent,
            'DeveloperContent': DeveloperContent,
        }
        self._LLM = LLM
        self._SamplingParams = SamplingParams

        model = os.getenv('DOCJA_GPTOSS_MODEL', 'openai/gpt-oss-20b')
        trc = os.getenv('DOCJA_GPTOSS_TRUST_REMOTE_CODE', '1') == '1'
        self._llm = self._LLM(model=model, trust_remote_code=trc)
        self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # --- Public tasks ---
    def summarize(self, document: str, target_lang: str = 'ja') -> Optional[str]:
        prompt = f"[文書]\n{document}\n[指示] 日本語で要点を一行で要約してください。"
        return self._generate_simple(prompt, target_lang=target_lang)

    def qa(self, document: str, question: str, target_lang: str = 'ja') -> Optional[str]:
        prompt = f"[文書]\n{document}\n[質問]\n{question}\n[指示] 日本語で簡潔に答えのみを返してください。"
        return self._generate_simple(prompt, target_lang=target_lang)

    def extract_json(self, document: str, schema: Dict[str, Any], target_lang: str = 'ja') -> Optional[str]:
        prompt = (
            f"[文書]\n{document}\n[質問] 次のスキーマに従ってJSONのみを返してください:\n"
            f"{schema}"
        )
        return self._generate_simple(prompt, target_lang=target_lang, json_only=True)

    # --- Internal helpers ---
    def _generate_simple(self, user_text: str, target_lang: str = 'ja', json_only: bool = False) -> Optional[str]:
        H = self._harmony
        Role = H['Role']
        # System/Developer instructions
        sys_msg = H['Message'].from_role_and_content(Role.SYSTEM, H['SystemContent'].new())
        dev_instr = "Always respond in Japanese."
        if json_only:
            dev_instr += " Return only valid JSON. No extra text."
        dev_msg = H['Message'].from_role_and_content(
            Role.DEVELOPER,
            H['DeveloperContent'].new().with_instructions(dev_instr),
        )
        user_msg = H['Message'].from_role_and_content(Role.USER, user_text)
        convo = H['Conversation'].from_messages([sys_msg, dev_msg, user_msg])

        # Prefill + stop tokens
        prefill_ids = self._encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        stop_token_ids = self._encoding.stop_tokens_for_assistant_actions()

        # Sampling params
        temperature = float(os.getenv('DOCJA_VLLM_TEMPERATURE', '0.1'))
        max_tokens = int(os.getenv('DOCJA_VLLM_MAX_TOKENS', '256'))
        sampling = self._SamplingParams(max_tokens=max_tokens, temperature=temperature, stop_token_ids=stop_token_ids)

        # Generate
        try:
            outputs = self._llm.generate(prompt_token_ids=[prefill_ids], sampling_params=sampling)
            gen = outputs[0].outputs[0]
            out_ids: List[int] = gen.token_ids or []
            # Prefer structured decode from tokens
            entries = self._encoding.parse_messages_from_completion_tokens(out_ids, Role.ASSISTANT)
            # Collect assistant text parts in order
            parts: List[str] = []
            for m in entries:
                try:
                    d = m.to_dict()
                    # Standard assistant text
                    if d.get('role') == 'assistant':
                        for c in d.get('content', []):
                            if c.get('type') == 'text' and c.get('text'):
                                parts.append(str(c['text']))
                except Exception:
                    pass
            text = ("\n".join(parts)).strip() if parts else (gen.text or '').strip()
            return text or None
        except Exception as e:  # pragma: no cover
            logger.warning(f"gpt-oss vLLM generation failed: {e}")
            return None
