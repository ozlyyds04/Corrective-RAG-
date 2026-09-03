const BASE = ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || res.statusText
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  health: () => request('/api/health'),
  config: () => request('/api/config'),
  listKbs: () => request('/api/kbs'),
  createKb: (name) =>
    request('/api/kbs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  deleteKb: (name) => request(`/api/kbs/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  upload: (files, kbName, maxFiles) => {
    const form = new FormData()
    files.forEach((f) => form.append('file', f))
    form.append('kb_name', kbName)
    form.append('max_files', String(maxFiles))
    return request('/api/ingest/upload', { method: 'POST', body: form })
  },
  watchStatus: () => request('/api/watch/status'),
  startWatch: (dir, kbName) =>
    request('/api/watch/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dir, kb_name: kbName }),
    }),
  stopWatch: () => request('/api/watch/stop', { method: 'POST' }),
}

// 读取 fetch 流式响应的 SSE 帧，按 event 名分发到 handlers
export async function readSSE(res, handlers = {}) {
  if (!res.ok || !res.body) {
    handlers.error?.({ message: `请求失败：${res.statusText}` })
    return
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let event = 'message'
  let dataLines = []

  const dispatch = () => {
    const payload = dataLines.join('\n')
    dataLines = []
    if (!payload) {
      event = 'message'
      return
    }
    let parsed = payload
    try {
      parsed = JSON.parse(payload)
    } catch {
      /* keep raw */
    }
    handlers[event]?.(parsed)
    event = 'message'
  }

  const consumeLine = (line) => {
    if (line === '') {
      dispatch()
    } else if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 1)
      consumeLine(raw.replace(/\r$/, ''))
    }
  }
  if (buffer.trim()) consumeLine(buffer.trim())
  if (dataLines.length) dispatch()
}

export async function queryStream(payload, handlers = {}) {
  const res = await fetch(`${BASE}/api/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  await readSSE(res, handlers)
}

export async function ingestStream(payload, handlers = {}) {
  const res = await fetch(`${BASE}/api/ingest/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  await readSSE(res, handlers)
}
