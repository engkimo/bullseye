from __future__ import annotations

import os
import io
import json
import base64
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

import requests
import gradio as gr

# Workaround for Gradio /info schema bug (additionalProperties: true)
if os.getenv("GRADIO_DISABLE_API_INFO", "1") == "1":
    try:
        import gradio.routes as _gr_routes  # type: ignore

        def _safe_api_info(_: bool = False):
            return {}

        _gr_routes.api_info = _safe_api_info  # type: ignore
    except Exception:
        pass


API_BASE = os.getenv("DOCJA_API_BASE", "http://localhost:8001")


def _b64_file(file_obj: Tuple[str, bytes]) -> Tuple[str, str]:
    """Return (filename, base64) from Gradio file object (name, bytes)."""
    if not file_obj:
        return "", ""
    name, data = file_obj
    if isinstance(data, bytes):
        return os.path.basename(name or "upload"), base64.b64encode(data).decode("ascii")
    # If gradio returns str path
    if isinstance(data, str) and os.path.exists(data):
        with open(data, "rb") as f:
            raw = f.read()
        return os.path.basename(name or os.path.basename(data)), base64.b64encode(raw).decode("ascii")
    return os.path.basename(name or "upload"), ""


def _analysis_context(udj: Dict[str, Any], max_chars: int = 8000) -> str:
    pages = udj.get("pages", [])
    lines: List[str] = []
    lines.append(f"pages={len(pages)}")
    # first page text by reading order
    if pages:
        pg = pages[0]
        tbs = pg.get("text_blocks", [])
        order = pg.get("reading_order") or list(range(len(tbs)))
        texts = []
        for i in order:
            if 0 <= i < len(tbs):
                tx = (tbs[i].get("text") or "").strip()
                if tx:
                    texts.append(tx)
        txt = "\n".join(texts)
        if txt:
            lines.append("--- first_page_text ---")
            lines.append(txt)
    # any gantt/chart info
    try:
        charts = pages[0].get("charts", []) if pages else []
        if charts:
            lines.append(f"charts={len(charts)}")
    except Exception:
        pass
    ctx = "\n".join(lines)
    return ctx[:max_chars]


def analyze(file_obj: Tuple[str, bytes], provider: str, api_base: str) -> Tuple[str, str, str, str, List[Image.Image]]:
    if not file_obj:
        return "ファイルをアップロードしてください", "", "", ""
    filename, b64 = _b64_file(file_obj)
    options = {
        "output_format": "json",
        "detect_layout": True,
        "detect_tables": True,
        "extract_reading_order": True,
        "with_llm": False,
        "lite": True,
        "vis": True,
    }
    files = {
        "file": (filename, base64.b64decode(b64)),
        "options": (None, json.dumps(options, ensure_ascii=False)),
    }
    try:
        r = requests.post(f"{api_base}/v1/di/analyze", files=files, timeout=120)
        r.raise_for_status()
        udj = r.json()
    except Exception as e:
        return f"解析失敗: {e}", "", "", "", []
    ctx = _analysis_context(udj)
    # vis previews (Base64 PNGs) → PIL Images
    previews: List[Image.Image] = []
    try:
        for b64 in (udj.get('vis_previews') or [])[:4]:
            raw = base64.b64decode(b64)
            previews.append(Image.open(io.BytesIO(raw)).copy())
    except Exception:
        previews = []
    # 初期systemメッセージ（gptoss用）
    system = (
        "あなたはドキュメント解析アシスタントです。以降の会話では、以下の解析結果を事実として参照し、短く正確に日本語で答えてください。\n\n"
        + ctx
    )
    return f"解析完了: ページ数 {len(udj.get('pages', []))}", json.dumps(udj, ensure_ascii=False), filename, b64, previews


def chat(user_msg: str,
         history: List[Tuple[str, str]],
         udj_str: str,
         filename: str,
         file_b64: str,
         provider: str,
         api_base: str,
         include_image: bool) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    if not user_msg.strip():
        return history, []
    # Build messages from history
    messages: List[Dict[str, str]] = []
    # If provider is gptoss, prepend analysis context as system
    udj: Dict[str, Any] = {}
    try:
        if udj_str:
            udj = json.loads(udj_str)
    except Exception:
        udj = {}
    if provider in ('gptoss', 'ollama') and udj:
        ctx = _analysis_context(udj)
        messages.append({"role": "system", "content": (
            "You are a helpful assistant. Use the following analysis as context.\n\n" + ctx
        )})
    # replay conversation
    for h in history:
        u, a = h
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    # append new user message
    messages.append({"role": "user", "content": user_msg})

    payload: Dict[str, Any] = {
        "messages": messages,
        "provider": provider,
        "attach_image": (provider == 'gemma3' and include_image and bool(file_b64)),
    }
    if payload["attach_image"]:
        payload["file_base64"] = file_b64
    try:
        r = requests.post(f"{api_base}/v1/di/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        ans = data.get("message") or "(no response)"
    except Exception as e:
        ans = f"エラー: {e}"
    history = history + [(user_msg, ans)]
    # Also return a structured transcript for debugging
    transcript = [{"role": m["role"], "content": m["content"]} for m in messages] + [{"role":"assistant","content": ans}]
    return history, transcript


def build_ui():
    with gr.Blocks(title="DocJA + Gemma/GPTOSS/Ollama Chat") as demo:
        gr.Markdown("# DocJA Uploader + Multimodal Chat (Gemma 3 / GPT‑OSS / Ollama)")
        with gr.Row():
            provider = gr.Dropdown(choices=["gemma3", "gptoss", "ollama"], value=os.getenv("DOCJA_LLM_PROVIDER", "gemma3"), label="LLM Provider")
            api_base = gr.Textbox(value=API_BASE, label="API Base", scale=2)
            attach_image = gr.Checkbox(value=True, label="Gemma3に画像を添付")
        with gr.Row():
            file = gr.File(label="ドキュメントをアップロード (PDF/画像)")
            analyze_btn = gr.Button("解析する", variant="primary")
        status = gr.Markdown()
        vis_gallery = gr.Gallery(label="Visualization Overlays", columns=2, height=220)
        with gr.Row():
            chatbot = gr.Chatbot(height=420)
        with gr.Row():
            msg = gr.Textbox(placeholder="質問を入力…")
            send_btn = gr.Button("送信", variant="primary")

        # Hidden stores (avoid gr.State JSON schema issue)
        st_udj = gr.Textbox(visible=False)
        st_filename = gr.Textbox(visible=False)
        st_fileb64 = gr.Textbox(visible=False)

        analyze_btn.click(
            fn=analyze,
            inputs=[file, provider, api_base],
            outputs=[status, st_udj, st_filename, st_fileb64, vis_gallery],
        )

        def _chat_wrapper(user_msg, chat_hist, udj, fname, fb64, prov, base, inc_img):
            hist, _ = chat(user_msg, chat_hist or [], udj or {}, fname or "", fb64 or "", prov, base, inc_img)
            return hist

        send_btn.click(
            fn=_chat_wrapper,
            inputs=[msg, chatbot, st_udj, st_filename, st_fileb64, provider, api_base, attach_image],
            outputs=[chatbot],
        ).then(lambda: ("",), None, [msg])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name=os.getenv("HOST", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        share=bool(int(os.getenv("GRADIO_SHARE", "0"))),
        show_api=bool(int(os.getenv("GRADIO_SHOW_API", "0"))),  # /info を無効化してschema相性問題を回避
        inbrowser=bool(int(os.getenv("GRADIO_INBROWSER", "0"))),
        prevent_thread_lock=True,
    )
