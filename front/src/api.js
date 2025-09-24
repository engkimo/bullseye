function docjaHeaders() {
  const h = {}
  const k = (import.meta?.env?.VITE_DOCJA_API_KEY) || (typeof localStorage !== 'undefined' ? localStorage.getItem('DOCJA_API_KEY') : null)
  if (k) h['x-api-key'] = k
  return h
}

export async function analyzeDocument(file, options = {}) {
  const form = new FormData()
  form.append('file', file)
  const defaultOptions = {
    output_format: 'json',
    detect_layout: true,
    detect_tables: true,
    extract_reading_order: true,
    with_llm: false,
    lite: true,
    vis: true,
  }
  const merged = { ...defaultOptions, ...options }
  form.append('options', JSON.stringify(merged))
  const r = await fetch('/docja/v1/di/analyze', {
    method: 'POST',
    body: form,
    headers: docjaHeaders(),
  })
  if (!r.ok) {
    throw new Error(`analyze failed: ${r.status}`)
  }
  return r.json()
}

export async function chatOllama(model, messages) {
  const payload = { model, messages, stream: false }
  const r = await fetch('/ollama/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!r.ok) throw new Error(`ollama chat failed: ${r.status}`)
  const data = await r.json()
  const msg = (data.message || {})
  return msg.content || ''
}

export async function chatDocja(messages, provider = 'ollama', attachImage = false, fileBase64 = null) {
  const payload = { messages, provider, attach_image: !!attachImage }
  if (attachImage && fileBase64) payload.file_base64 = fileBase64
  const r = await fetch('/docja/v1/di/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...docjaHeaders() },
    body: JSON.stringify(payload),
  })
  if (!r.ok) throw new Error(`docja chat failed: ${r.status}`)
  const data = await r.json()
  return data.message || ''
}

export function composeSystemFromUDJ(udj) {
  try {
    const pages = udj?.pages || []
    let lines = [`pages=${pages.length}`]
    if (pages.length > 0) {
      const pg = pages[0]
      const tbs = pg.text_blocks || []
      const order = pg.reading_order || tbs.map((_, i) => i)
      const texts = []
      order.forEach((i) => {
        if (i >= 0 && i < tbs.length) {
          const tx = (tbs[i].text || '').trim()
          if (tx) texts.push(tx)
        }
      })
      const txt = texts.join('\n')
      if (txt) {
        lines.push('--- first_page_text ---')
        lines.push(txt)
      }
      // omit charts count in summary to reduce noise
    }
    return (
      'You are a helpful assistant. Use the following analysis as factual context.\n\n' +
      lines.join('\n').slice(0, 8000)
    )
  } catch {
    return 'You are a helpful assistant.'
  }
}
