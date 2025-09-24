import os
import time
import uuid
import hashlib
import logging
from fastapi import FastAPI, UploadFile, File, Form, Depends, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional, Dict, Any, Union
from pathlib import Path
import shutil
import json
import base64
import tempfile
import requests

from .pipeline import DocumentProcessor
from .logging_filters import install_bullseye_log_filter, sanitize_bullseye_loggers
from .exporters import HTMLExporter, MarkdownExporter, JSONLinesExporter, PDFSearchableExporter
from .schemas import ProcessOptions, ProcessJSONRequest, DocumentResponse, ProcessArtifactResponse
from .api_v1_di import mount_api as mount_api_v1_di


tags_metadata = [
    {"name": "process", "description": "Document processing endpoints"},
]

def _setup_logging():
    try:
        os.makedirs('logs', exist_ok=True)
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(level=logging.INFO)
        # Avoid duplicate handlers
        names = [getattr(h, 'baseFilename', '') for h in root.handlers if hasattr(h, 'baseFilename')]
        if not any(name.endswith('logs/api.log') for name in names if isinstance(name, str)):
            fh = logging.FileHandler('logs/api.log', encoding='utf-8')
            fh.setLevel(logging.INFO)
            fmt = logging.Formatter('%(message)s')
            fh.setFormatter(fmt)
            root.addHandler(fh)
        # Install rename filter on existing handlers (pre-uvicorn)
        install_bullseye_log_filter()
        sanitize_bullseye_loggers()
    except Exception:
        pass


_setup_logging()

app = FastAPI(
    title="DocJA API",
    version="0.1.0",
    description="OCR/Layout/Table/RO + LLM 要約のAPI。/docs でOpenAPI UIを参照してください。",
    openapi_tags=tags_metadata,
)

logger = logging.getLogger("docja.api")

# CORS (dev-friendly; restrict in production)
try:
    origins_env = os.getenv('DOCJA_CORS_ORIGINS', '*')
    origins = [o.strip() for o in origins_env.split(',')] if origins_env else ['*']
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# Mount v1 DI API (Unified Doc JSON / Flow / Gantt / reprocess)
try:
    mount_api_v1_di(app)
except Exception as e:
    logger.warning(f"Failed to mount v1 DI API: {e}")


@app.on_event("startup")
async def _on_startup_install_filters():
    # Ensure filters are installed after uvicorn configures its handlers
    try:
        install_bullseye_log_filter()
        sanitize_bullseye_loggers()
    except Exception:
        pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _log_event(event: Dict[str, Any]):
    try:
        logger.info(json.dumps(event, ensure_ascii=False))
    except Exception:
        # Fallback
        logger.info(str(event))


def api_key_auth(x_api_key: Optional[str] = Header(None)):
    expected = os.getenv('DOCJA_API_KEY', '')
    if expected:
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid API key')
    return True


def _get_exporter(fmt: str):
    mapping = {
        'md': MarkdownExporter(),
        'html': HTMLExporter(),
        'json': JSONLinesExporter(),
        'csv': JSONLinesExporter(),
        'pdf': PDFSearchableExporter(),
    }
    return mapping.get(fmt)


@app.get("/healthz", tags=["process"], summary="Liveness/health check")
async def healthz():
    return {"status": "ok"}


@app.get("/ui", tags=["process"], summary="Simple Web UI")
async def simple_ui():
    html = """
<!doctype html>
<html lang=ja>
<meta charset=utf-8>
<title>DocJA UI</title>
<style>
body{font-family:sans-serif;max-width:960px;margin:20px auto;padding:0 12px}
textarea{width:100%;height:120px}
#chat{border:1px solid #ccc;padding:8px;height:300px;overflow:auto}
.msg{margin:6px 0}
.user{color:#1976d2}
.assistant{color:#2e7d32}
</style>
<h2>DocJA Uploader + Chat</h2>
<div>
  <label>Provider: <select id=provider>
    <option value=gptoss selected>gptoss</option>
    <option value=gemma3>gemma3</option>
    <option value=ollama>ollama</option>
  </select></label>
  <label style="margin-left:16px"><input type=checkbox id=attach checked> 画像をGemma3に添付</label>
</div>
<div style="margin-top:8px">
  <input type=file id=file>
  <button id=analyze>解析する</button>
  <span id=status></span>
  <input type=hidden id=udj>
  <input type=hidden id=fname>
  <input type=hidden id=fb64>
  <div id=chat></div>
  <div style="margin-top:8px">
    <input id=prompt style="width:80%" placeholder="質問を入力…">
    <button id=send>送信</button>
  </div>
</div>
<script>
const $ = sel => document.querySelector(sel);
const apiBase = '';
function analysisContext(){
  try{
    const udj = JSON.parse($('#udj').value||'{}');
    const pages = udj.pages||[]; let lines=[]; lines.push('pages='+(pages.length||0));
    if(pages.length>0){
      const pg = pages[0]; const tbs = pg.text_blocks||[]; const order = pg.reading_order||tbs.map((_,i)=>i);
      const texts=[]; order.forEach(i=>{ if(i>=0 && i<tbs.length){ const tx=(tbs[i].text||'').trim(); if(tx) texts.push(tx); }});
      const txt = texts.join('\n'); if(txt){ lines.push('--- first_page_text ---'); lines.push(txt); }
      const charts = pg.charts||[]; if(charts.length){ lines.push('charts='+charts.length); }
    }
    return lines.join('\n').slice(0,8000);
  }catch(e){ return ''; }
}
async function analyze(){
  const f = $('#file').files[0];
  if(!f){$('#status').textContent='ファイルを選択してください'; return}
  const arr = await f.arrayBuffer();
  const b64 = btoa(String.fromCharCode(...new Uint8Array(arr)));
  const fd = new FormData();
  fd.append('file', new Blob([new Uint8Array(arr)], {type: f.type||'application/octet-stream'}), f.name);
  fd.append('options', JSON.stringify({output_format:'json',detect_layout:true,detect_tables:true,extract_reading_order:true,with_llm:false,lite:true,vis:false}));
  const r = await fetch(apiBase + '/v1/di/analyze', {method:'POST', body:fd});
  if(!r.ok){$('#status').textContent='解析失敗'; return}
  const udj = await r.json();
  $('#udj').value = JSON.stringify(udj);
  $('#fname').value = f.name;
  $('#fb64').value = b64;
  $('#status').textContent = '解析完了: ページ数 ' + (udj.pages?udj.pages.length:0);
}
function append(role, text){
  const div = document.createElement('div');
  div.className='msg '+(role==='user'?'user':'assistant');
  div.textContent = (role==='user'?'You: ':'Assistant: ') + text;
  $('#chat').appendChild(div); $('#chat').scrollTop = 1e8;
}
async function send(){
  const q = $('#prompt').value.trim(); if(!q) return; $('#prompt').value=''; append('user', q);
  const msgs = []; const nodes = $('#chat').querySelectorAll('.msg');
  nodes.forEach(n=>{ const t=n.textContent; if(t.startsWith('You: ')) msgs.push({role:'user',content:t.slice(5)}); else msgs.push({role:'assistant',content:t.slice(10)}); });
  msgs.push({role:'user',content:q});
  const provider = $('#provider').value; const attach = $('#attach').checked;
  // Inject analysis context for text-only providers
  if(provider !== 'gemma3'){
    const ctx = analysisContext();
    if(ctx){ msgs.unshift({role:'system', content:'You are a helpful assistant. Use the following analysis as context.\n\n'+ctx}); }
  }
  const payload = {messages: msgs, provider: provider, attach_image: (provider==='gemma3' && attach)};
  if(payload.attach_image){ payload.file_base64 = $('#fb64').value; }
  const r = await fetch(apiBase + '/v1/di/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if(!r.ok){ append('assistant','(error)'); return }
  const data = await r.json(); append('assistant', data.message || '(no response)');
}
$('#analyze').onclick = analyze; $('#send').onclick = send;
</script>
"""
    return HTMLResponse(content=html)


@app.post(
    "/api/process",
    tags=["process"],
    summary="Process document (multipart)",
    response_model=Union[DocumentResponse, ProcessArtifactResponse],
)
async def process_document(
    file: UploadFile = File(...),
    layout: bool = Form(False),
    table: bool = Form(False),
    reading_order: bool = Form(False),
    llm: bool = Form(False),
    lite: bool = Form(False),
    options: Optional[str] = Form(None),
    auth: bool = Depends(api_key_auth),
):
    req_id = str(uuid.uuid4())
    t0 = time.time()
    tmp_dir = Path("/tmp/docja")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    file_hash = _sha256_file(tmp_path)

    fmt = 'json'
    vis = False
    max_pages = None
    if options:
        try:
            obj = json.loads(options)
            validated = ProcessOptions(**obj)
            layout = validated.detect_layout
            table = validated.detect_tables
            reading_order = validated.extract_reading_order
            llm = validated.enable_llm
            lite = validated.lite
            fmt = validated.output_format
            vis = validated.vis
            max_pages = validated.max_pages
        except Exception:
            pass

    processor = DocumentProcessor(
        det_model='dbnet',
        rec_model='abinet',
        layout_model='yolo' if layout else None,
        enable_table=table,
        enable_reading_order=reading_order,
        enable_llm=llm,
        device='cuda',
        weights_dir='weights',
        lite_mode=lite,
    )

    vis_dir = str(tmp_dir / 'vis') if vis else None
    try:
        # LLM task options
        llm_task = None
        llm_question = None
        llm_schema = None
        try:
            if options:
                obj = json.loads(options)
                po = ProcessOptions(**obj)
                llm_task = po.llm_task
                llm_question = po.llm_question
                llm_schema = po.llm_schema
        except Exception:
            pass
        result = processor.process(
            str(tmp_path),
            max_pages=max_pages,
            extract_figures=False,
            vis_save_dir=vis_dir,
            llm_task=llm_task or ('summary' if llm else None),
            llm_question=llm_question,
            llm_schema=llm_schema,
        )
    except Exception as e:
        _log_event({
            'ts': time.time(), 'req_id': req_id, 'route': '/api/process', 'method': 'POST',
            'status': 500, 'filename': file.filename, 'sha256': file_hash,
            'options': {'layout': layout, 'table': table, 'reading_order': reading_order, 'llm': llm, 'lite': lite},
            'error': str(e)
        })
        raise

    exporter = _get_exporter(fmt)
    if exporter is None or fmt == 'json':
        payload: Dict[str, Any] = result.to_dict()
    else:
        out_path = tmp_dir / f"artifact.{fmt}"
        exporter.export(result, str(out_path))
        if fmt in ('md', 'html', 'csv'):
            content = out_path.read_text(encoding='utf-8')
            payload = {
                'content_type': f'text/{"markdown" if fmt=="md" else fmt}',
                'artifact_text': content,
                'file_name': out_path.name,
                'size_bytes': len(content.encode('utf-8')),
            }
        else:
            b = out_path.read_bytes()
            payload = {
                'content_type': 'application/pdf',
                'artifact_base64': base64.b64encode(b).decode('ascii'),
                'file_name': out_path.name,
                'size_bytes': len(b),
            }

    if vis and vis_dir:
        previews = []
        for i, p in enumerate(sorted(Path(vis_dir).glob('*.png'))):
            if i >= 2:
                break
            previews.append(base64.b64encode(p.read_bytes()).decode('ascii'))
        payload['vis_previews'] = previews

    resp = JSONResponse(content=payload)
    _log_event({
        'ts': time.time(), 'req_id': req_id, 'route': '/api/process', 'method': 'POST',
        'status': 200, 'filename': file.filename, 'sha256': file_hash,
        'duration_ms': int((time.time() - t0) * 1000),
        'options': {
            'layout': layout, 'table': table, 'reading_order': reading_order, 'llm': llm, 'lite': lite,
            'format': fmt, 'vis': vis, 'max_pages': max_pages
        }
    })
    return resp


@app.post(
    "/api/process-json",
    tags=["process"],
    summary="Process document (JSON)",
    response_model=Union[DocumentResponse, ProcessArtifactResponse],
)
async def process_document_json(payload: ProcessJSONRequest, auth: bool = Depends(api_key_auth)):
    req_id = str(uuid.uuid4())
    t0 = time.time()
    tmp_dir = Path(tempfile.mkdtemp(prefix='docja_'))
    filename = payload.filename or (payload.file_url.split('/')[-1] if payload.file_url else 'upload')
    tmp_path = tmp_dir / filename

    if payload.file_base64:
        raw = base64.b64decode(payload.file_base64)
        tmp_path.write_bytes(raw)
    else:
        resp = requests.get(str(payload.file_url), timeout=10)
        resp.raise_for_status()
        tmp_path.write_bytes(resp.content)
    file_hash = _sha256_file(tmp_path)

    opts = payload.options or ProcessOptions()
    processor = DocumentProcessor(
        det_model='dbnet',
        rec_model='abinet',
        layout_model='yolo' if opts.detect_layout else None,
        enable_table=opts.detect_tables,
        enable_reading_order=opts.extract_reading_order,
        enable_llm=opts.enable_llm,
        device='cuda',
        weights_dir='weights',
        lite_mode=opts.lite,
    )

    vis_dir = str(tmp_dir / 'vis') if opts.vis else None
    try:
        result = processor.process(
            str(tmp_path),
            max_pages=opts.max_pages,
            extract_figures=False,
            vis_save_dir=vis_dir,
            llm_task=getattr(opts, 'llm_task', 'summary') if opts.enable_llm else None,
            llm_question=getattr(opts, 'llm_question', None),
            llm_schema=getattr(opts, 'llm_schema', None),
        )
    except Exception as e:
        _log_event({
            'ts': time.time(), 'req_id': req_id, 'route': '/api/process-json', 'method': 'POST',
            'status': 500, 'filename': filename, 'sha256': file_hash,
            'options': opts.dict() if hasattr(opts, 'dict') else {},
            'error': str(e)
        })
        raise

    exporter = _get_exporter(opts.output_format)
    if exporter is None or opts.output_format == 'json':
        payload_out: Dict[str, Any] = result.to_dict()
    else:
        out_path = tmp_dir / f"artifact.{opts.output_format}"
        exporter.export(result, str(out_path))
        if opts.output_format in ('md', 'html', 'csv'):
            content = out_path.read_text(encoding='utf-8')
            payload_out = {
                'content_type': f'text/{"markdown" if opts.output_format=="md" else opts.output_format}',
                'artifact_text': content,
                'file_name': out_path.name,
                'size_bytes': len(content.encode('utf-8')),
            }
        else:
            b = out_path.read_bytes()
            payload_out = {
                'content_type': 'application/pdf',
                'artifact_base64': base64.b64encode(b).decode('ascii'),
                'file_name': out_path.name,
                'size_bytes': len(b),
            }

    if opts.vis and vis_dir:
        previews = []
        for i, p in enumerate(sorted(Path(vis_dir).glob('*.png'))):
            if i >= 2:
                break
            previews.append(base64.b64encode(p.read_bytes()).decode('ascii'))
        payload_out['vis_previews'] = previews

    resp = JSONResponse(content=payload_out)
    _log_event({
        'ts': time.time(), 'req_id': req_id, 'route': '/api/process-json', 'method': 'POST',
        'status': 200, 'filename': filename, 'sha256': file_hash,
        'duration_ms': int((time.time() - t0) * 1000),
        'options': opts.dict() if hasattr(opts, 'dict') else {}
    })
    return resp
