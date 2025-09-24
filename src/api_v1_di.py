import os
import time
import uuid
import json
import base64
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, status
from fastapi.responses import JSONResponse

from .pipeline import DocumentProcessor
from .schemas import ProcessOptions
from .parsers.flow import parse_flow_from_page
from .parsers.gantt import parse_gantt_from_page
try:  # optional: LLM router for post-enrichment prompts
    from .llm.router import LLMRouterClient as _LLMRouterClient
except Exception:
    _LLMRouterClient = None  # type: ignore


logger = logging.getLogger("docja.api_v1_di")


def _auth_guard(x_api_key: Optional[str] = Header(None)):
    expected = os.getenv('DOCJA_API_KEY', '')
    if expected and (x_api_key or '') != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid API key')
    return True


def mount_api(app: FastAPI):
    @app.post('/v1/di/analyze')
    async def analyze(
        file: UploadFile = File(...),
        options: Optional[str] = Form(None),
        x_api_key: Optional[str] = Header(None),
    ):
        _auth_guard(x_api_key)
        req_id = str(uuid.uuid4())
        t0 = time.time()
        root_tmp = Path('/tmp/docja_v1')
        root_tmp.mkdir(parents=True, exist_ok=True)
        job_dir = root_tmp / req_id
        job_dir.mkdir(parents=True, exist_ok=True)
        in_path = job_dir / file.filename
        with in_path.open('wb') as f:
            f.write(await file.read())

        # Parse options
        po = ProcessOptions()
        if options:
            try:
                po = ProcessOptions(**json.loads(options))
            except Exception:
                pass

        proc = DocumentProcessor(
            det_provider=os.getenv('DOCJA_DET_PROVIDER', 'bullseye'),
            rec_provider=os.getenv('DOCJA_REC_PROVIDER', 'bullseye'),
            layout_provider=os.getenv('DOCJA_LAYOUT_PROVIDER', 'bullseye') if po.detect_layout else 'internal',
            layout_hf_model_id=os.getenv('DOCJA_LAYOUT_MODEL_ID', None),
            enable_table=po.detect_tables,
            table_provider=os.getenv('DOCJA_TABLE_PROVIDER', 'bullseye') if po.detect_tables else 'internal',
            table_hf_model_id=os.getenv('DOCJA_TABLE_MODEL_ID', None),
            enable_reading_order=po.extract_reading_order,
            enable_llm=po.enable_llm,
            device='cuda',
            lite_mode=po.lite,
        )

        # Visualization dir (if requested)
        vis_dir: Optional[Path] = (job_dir / 'vis') if (po and getattr(po, 'vis', False)) else None
        result = proc.process(
            str(in_path),
            max_pages=po.max_pages,
            vis_save_dir=str(vis_dir) if vis_dir else None,
            llm_task=(po.llm_task or 'summary') if po.enable_llm else None,
            llm_question=po.llm_question,
            llm_schema=po.llm_schema,
        )

        # Enrich pages with graphs/charts (v0 heuristic) unless disabled
        udj = result.to_dict()
        if (os.getenv('DOCJA_PARSE_DIAGRAMS', '1') or '1').strip() != '0':
            for page in udj.get('pages', []):
                try:
                    graphs = parse_flow_from_page(page)
                    charts = parse_gantt_from_page(page)
                    page['graphs'] = graphs
                    page['charts'] = charts
                except Exception as e:
                    logger.warning(f"Flow/Gantt parse failed: {e}")

        # Optional: LLM with Gantt schedule context (after enrichment)
        if po.enable_llm and (po.llm_task or 'summary') in ('summary', 'qa', 'json') and _LLMRouterClient is not None:
            try:
                router = _LLMRouterClient()
                # Compose concise schedule context from first page chart (if any)
                first_page = udj.get('pages', [{}])[0]
                charts = first_page.get('charts') or []
                first_chart = charts[0] if charts else None
                if first_chart:
                    # Prefer gemma3 with image; build a strict JSON schema for Gantt extraction
                    schema = {
                        "gantt": {
                            "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
                            "unit": "day|week|month",
                            "ticks": [{"label": "str", "date": "YYYY-MM-DD"}],
                            "tasks": [{
                                "name": "str",
                                "start_date": "YYYY-MM-DD",
                                "end_date": "YYYY-MM-DD"
                            }]
                        }
                    }
                    # Load first page image (PNG/JPG or render 1st page of PDF)
                    from PIL import Image as _PIL
                    img_for_llm = None
                    try:
                        suffix = in_path.suffix.lower()
                        if suffix in ('.png', '.jpg', '.jpeg', '.bmp'):
                            img_for_llm = _PIL.open(str(in_path)).convert('RGB')
                        elif suffix == '.pdf':
                            try:
                                import pypdfium2 as pdfium  # type: ignore
                                doc = pdfium.PdfDocument(str(in_path))
                                page = doc.get_page(0)
                                bmp = page.render(scale=300.0/72.0)
                                img_for_llm = bmp.to_pil().convert('RGB')
                                try:
                                    page.close()
                                except Exception:
                                    pass
                                try:
                                    doc.close()
                                except Exception:
                                    pass
                            except Exception:
                                img_for_llm = None
                    except Exception:
                        img_for_llm = None

                    # Build minimal text context (title or first few row labels)
                    context_lines = []
                    for el in (first_page.get('layout_elements') or [])[:10]:
                        t = (el.get('type') or '')
                        if t in ('title', 'section_heading'):
                            context_lines.append(str(t))
                    for tb in (first_page.get('text_blocks') or [])[:20]:
                        txt = (tb.get('text') or '').strip()
                        if len(txt) >= 2:
                            context_lines.append(txt)
                    context = "\n".join(context_lines[-10:])

                    res = router.extract_json(context or '(chart)', schema, images=[img_for_llm] if img_for_llm is not None else None)
                    udj.setdefault('metadata', {}).setdefault('llm', {})['gantt'] = {'extraction': res}
            except Exception as e:
                logger.warning(f"Post-enrichment LLM call failed: {e}")

        # Persist job metrics for /v1/di/jobs
        try:
            job_dir = Path('logs') / 'jobs'
            job_dir.mkdir(parents=True, exist_ok=True)
            metrics = udj.get('metadata', {}).get('metrics', {})
            job_blob = {
                'job_id': req_id,
                'filename': file.filename,
                'created_at': time.time(),
                'pages': [p.get('page_num') for p in udj.get('pages', [])],
                'metrics': metrics,
            }
            (job_dir / f'{req_id}.json').write_text(json.dumps(job_blob, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logger.warning(f"failed to persist job metrics: {e}")

        # Collect visualization previews (first 1-2)
        vis_previews: list[str] = []
        try:
            if vis_dir and vis_dir.exists():
                for i, p in enumerate(sorted(vis_dir.glob('*.png'))):
                    if i >= 2:
                        break
                    vis_previews.append(base64.b64encode(p.read_bytes()).decode('ascii'))
        except Exception:
            vis_previews = []

        if po.output_format == 'json':
            payload = {
                'document_id': req_id,
                'providers': udj.get('metadata', {}).get('providers', {}),
                'pages': udj.get('pages', []),
                'metadata': udj.get('metadata', {}),
                'vis_previews': vis_previews,
            }
            return JSONResponse(content=payload)

        # Non-JSON artifact generation: reuse CLI/exporters via API v0
        from .exporters import HTMLExporter, MarkdownExporter, JSONLinesExporter, PDFSearchableExporter
        exporters = {
            'md': MarkdownExporter(),
            'html': HTMLExporter(),
            'csv': JSONLinesExporter(),
            'pdf': PDFSearchableExporter(),
        }
        fmt = po.output_format
        exporter = exporters.get(fmt)
        if exporter is None:
            return JSONResponse(content={'error': f'unsupported format: {fmt}'}, status_code=400)

        # Serialize enriched result back to a lightweight object for exporters
        class _Shim:
            def __init__(self, d: Dict[str, Any]):
                self.filename = d.get('filename', file.filename)
                self.pages = []
                for p in d.get('pages', []):
                    # exporter expects certain keys; keep as-is
                    self.pages.append(type('P', (), p))
                self.metadata = d.get('metadata', {})

        shim = _Shim(udj | {'filename': file.filename})
        out_path = job_dir / f"artifact.{fmt}"
        exporter.export(shim, str(out_path))
        meta_path = job_dir / f"artifact.{fmt}.meta.json"
        meta_path.write_text(json.dumps({'filename': file.filename, 'metadata': shim.metadata}, ensure_ascii=False), encoding='utf-8')

        if fmt in ('md', 'html', 'csv'):
            content = out_path.read_text(encoding='utf-8')
            return JSONResponse(content={
                'content_type': f'text/{"markdown" if fmt=="md" else fmt}',
                'artifact_text': content,
                'meta_sidecar': meta_path.name,
            })
        else:
            b = out_path.read_bytes()
            return JSONResponse(content={
                'content_type': 'application/pdf',
                'artifact_base64': base64.b64encode(b).decode('ascii'),
                'meta_sidecar': meta_path.name,
            })

    @app.post('/v1/di/reprocess')
    async def reprocess(body: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        _auth_guard(x_api_key)
        # Placeholder: accept and echo
        return {'job_id': body.get('job_id', ''), 'accepted_pages': body.get('pages', []), 'status': 'queued'}

    @app.get('/v1/di/jobs/{job_id}')
    async def job_status(job_id: str, x_api_key: Optional[str] = Header(None)):
        _auth_guard(x_api_key)
        job_path = Path('logs') / 'jobs' / f'{job_id}.json'
        if not job_path.exists():
            raise HTTPException(status_code=404, detail='job not found')
        try:
            blob = json.loads(job_path.read_text(encoding='utf-8'))
        except Exception:
            raise HTTPException(status_code=500, detail='failed to load job')
        # Compute p50/p95 for total_ms and per-stage if present
        def _pct(values, q):
            if not values:
                return 0
            arr = sorted(values)
            idx = int(max(0, min(len(arr)-1, round((q/100.0)*(len(arr)-1)))))
            return arr[idx]
        pages = blob.get('metrics', {}).get('pages') or []
        total = [p.get('total_ms', 0) for p in pages]
        metrics_summary = {
            'p50_ms': _pct(total, 50),
            'p95_ms': _pct(total, 95),
        }
        # stage-wise percentiles
        for key in ('detect_ms','layout_ms','recog_ms','table_ms','read_ms','parse_ms','page_llm_ms'):
            vals = [p.get(key, 0) for p in pages if key in p]
            if vals:
                metrics_summary[key] = {'p50': _pct(vals,50), 'p95': _pct(vals,95)}
        return {
            'job_id': job_id,
            'status': 'completed',
            'pages': blob.get('pages', []),
            'metrics': metrics_summary,
        }

    @app.post('/v1/di/chat')
    async def di_chat(
        body: Dict[str, Any],
        x_api_key: Optional[str] = Header(None)
    ):
        """Multimodal chat about an image/PDF page with Gemma 3, GPT‑OSS, or Ollama.

        Request (application/json):
          {
            "messages": [{"role":"user","content":"..."}, ...],
            "provider": "gemma3|gptoss|ollama" (optional, env default),
            "file_base64": "..." OR "file_url": "..." (optional),
            "attach_image": true|false (default true for gemma3)
          }
        """
        _auth_guard(x_api_key)
        msgs = body.get('messages') or []
        if not isinstance(msgs, list) or not msgs:
            raise HTTPException(status_code=400, detail='messages[] required')
        provider = str(body.get('provider') or '').lower().strip()
        if provider:
            os.environ['DOCJA_LLM_PROVIDER'] = provider
        attach_image = bool(body.get('attach_image', True))
        # optional image
        img_obj = None
        if attach_image and (body.get('file_base64') or body.get('file_url')):
            try:
                from PIL import Image as _PIL
                import io, base64, requests as _req
                if body.get('file_base64'):
                    raw = base64.b64decode(body['file_base64'])
                else:
                    r = _req.get(str(body['file_url']), timeout=10)
                    r.raise_for_status()
                    raw = r.content
                img_obj = _PIL.open(io.BytesIO(raw)).convert('RGB')
            except Exception:
                img_obj = None
        # route to LLM
        if _LLMRouterClient is None:
            return {'provider': None, 'message': 'LLM未構成です。provider=gptoss|gemma3 を設定し、エンドポイントを用意してください。'}
        try:
            router = _LLMRouterClient()
            text = router.chat(msgs, image=img_obj)
        except Exception as e:
            text = None
        if text is None:
            prov = os.getenv('DOCJA_LLM_PROVIDER', 'gptoss')
            ep = os.getenv('DOCJA_LLM_ENDPOINT', '(未設定)')
            return {'provider': prov, 'message': f'LLMに接続できません。provider={prov}, endpoint={ep} を確認してください。'}
        return {'provider': os.getenv('DOCJA_LLM_PROVIDER', 'gptoss'), 'message': text}
