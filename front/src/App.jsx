import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { analyzeDocument, chatOllama, chatDocja, composeSystemFromUDJ } from './api.js'

function Spinner({ size=16 }) {
  return <span className="spinner" style={{ width: size, height: size }} aria-label="loading" />
}

function Gallery({ images, zoom=1 }) {
  if (!images?.length) return null
  return (
    <div className="gallery-canvas" style={{ transform: `scale(${zoom})`, transformOrigin: '0 0' }}>
      {images.map((b64, i) => (
        <img className="doc-page" key={i} src={`data:image/png;base64,${b64}`} alt={`overlay-${i}`} />
      ))}
    </div>
  )
}

function ChatMessage({ role, content }) {
  const bubble = (
    <div className="bubble">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || ''}</ReactMarkdown>
    </div>
  )
  const avatar = <div className="avatar">{role === 'user' ? 'You' : 'AI'}</div>
  return (
    <div className={`msg ${role}`}>
      {role === 'assistant' ? (<>{avatar}{bubble}</>) : (<>{bubble}{avatar}</>)}
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('')
  const [udj, setUdj] = useState(null)
  const [visPreviews, setVisPreviews] = useState([])
  const [provider] = useState('ollama')
  const [ollamaModel, setOllamaModel] = useState('gpt-oss:20b')
  const [messages, setMessages] = useState([]) // [{role, content}]
  const [sysPreview, setSysPreview] = useState('')
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const chatRef = useRef(null)
  const [zoom, setZoom] = useState(1)
  const [isComposing, setIsComposing] = useState(false)
  const zoomIn = () => setZoom((z) => Math.min(3, Number((z + 0.1).toFixed(2))))
  const zoomOut = () => setZoom((z) => Math.max(0.5, Number((z - 0.1).toFixed(2))))
  const zoomReset = () => setZoom(1)

  const onAnalyze = async () => {
    if (!file) return
    try {
      setStatus('解析中...')
      setAnalyzing(true)
      const res = await analyzeDocument(file)
      setUdj(res)
      setVisPreviews(res.vis_previews || [])
      const sys = composeSystemFromUDJ(res)
      setMessages([{ role: 'system', content: sys }])
      setSysPreview(sys)
      setStatus(`解析完了: ページ数 ${res.pages?.length || 0}`)
    } catch (e) {
      setStatus(`解析失敗: ${e}`)
    } finally {
      setAnalyzing(false)
    }
  }

  const onSend = async () => {
    const text = input.trim()
    if (!text) return
    const next = [...messages, { role: 'user', content: text }]
    setMessages(next)
    setInput('')
    try {
      setSending(true)
      let reply = ''
      if (provider === 'ollama') {
        reply = await chatOllama(ollamaModel, next)
      } else {
        reply = await chatDocja(next, provider)
      }
      setMessages([...next, { role: 'assistant', content: reply || '(no response)' }])
    } catch (e) {
      setMessages([...next, { role: 'assistant', content: `エラー: ${e}` }])
    } finally {
      setSending(false)
    }
  }

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages])

  const transcript = useMemo(
    () => messages.filter((m) => m.role !== 'system').map((m, i) => <ChatMessage key={i} role={m.role} content={m.content} />),
    [messages]
  )

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">BullsEye</div>
        <div className="header-actions"></div>
      </header>
      {/* Responsive two-column layout */}
      <div className="two-col">
        <div>
          <div className="control-bar">
            <label className="file-picker" title="ファイルを選択">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z" stroke="#e5e7eb" strokeWidth="1.5"/>
                <path d="M14 2v6h6" stroke="#e5e7eb" strokeWidth="1.5"/>
              </svg>
              <span>Choose file</span>
              <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </label>
            <span className="file-name" title={file?.name || ''}>{file?.name || 'No file selected'}</span>
            <button onClick={onAnalyze} className="btn primary" disabled={!file}>
              {analyzing ? <Spinner /> : '解析する'}
            </button>
          </div>
          <div className="status-line" aria-live="polite">
            {analyzing && <Spinner />}
            <span className={`status-text ${analyzing ? 'shine-text' : ''}`}>{status || ''}</span>
          </div>
          <div className="viewer">
            <div className="viewer-toolbar">
              <span>Zoom:</span>
              <button className="btn ghost" onClick={zoomOut} title="-">−</button>
              <button className="btn ghost" onClick={zoomReset} title="Reset">100%</button>
              <button className="btn ghost" onClick={zoomIn} title="+">＋</button>
            </div>
            <div className="viewer-stage">
              {analyzing && (
                <div className="viewer-overlay"><Spinner size={28} /></div>
              )}
              <Gallery images={visPreviews} zoom={zoom} />
            </div>
          </div>
          <details style={{ marginTop: 8 }}>
            <summary>解析サマリ（最初のページのテキスト抜粋）</summary>
            <pre className="panel-pre">{sysPreview || '(empty)'}</pre>
          </details>
          <details>
            <summary>UDJ (Unified Doc JSON)</summary>
            <pre className="panel-pre">{udj ? JSON.stringify(udj, null, 2) : '(empty)'}</pre>
          </details>
        </div>
        <div>
          <div className="controls-row" style={{ marginBottom: 11 }}>
            {/* Provider selector hidden; default is Ollama. Keep model input visible for convenience. */}
            <div className="model-select" role="group" aria-label="Model selector">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <path d="M4 7a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7Z" stroke="#94a3b8" strokeWidth="1.5"/>
                <path d="M8 12h8M8 16h6M8 8h4" stroke="#94a3b8" strokeWidth="1.5"/>
              </svg>
              <input className="input-text" value={ollamaModel} onChange={(e) => setOllamaModel(e.target.value)} placeholder="model tag" />
            </div>
            <button className="btn ghost" onClick={() => setMessages((m) => m.filter((x) => x.role === 'system'))}>Clear</button>
          </div>
          {/* Keep chat aligned with viewer by reserving the same status-line height */}
          <div className="status-line status-spacer" aria-hidden="true"></div>
          <div className="chat-wrap" ref={chatRef}>{transcript}</div>
          <div className="input-row">
            <div className="textarea-wrap">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing && !isComposing) {
                    e.preventDefault()
                    onSend()
                  }
                }}
                placeholder={sending ? '' : '質問を入力（Shift+Enterで改行）'}
                disabled={sending}
              />
              {sending && !input.trim() && (
                <div className="textarea-overlay shine-text" aria-hidden="true">送信中...</div>
              )}
            </div>
            <button className={`btn primary`} onClick={onSend} disabled={sending} title="送信">
              {sending ? (
                <Spinner />
              ) : (
                <svg className="send-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M2 21L23 12L2 3L2 10L17 12L2 14L2 21Z" fill="#0b0f14"/>
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
      <footer className="app-footer" />
    </div>
  )
}
