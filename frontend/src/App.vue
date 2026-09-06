<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { api, queryStream, ingestStream, readSSE } from './api' 

const STEP_LABELS = {
  retrieve: '多路召回',
  grade_documents: '相关性过滤',
  transform_query: '查询改写',
  web_search: '网络搜索',
  generate: '生成',
}

// ---------- 配置与知识库 ----------
const config = ref(null)
const kbs = ref([])
const currentKb = ref('默认')
const newKbName = ref('')

// ---------- 会话 ----------
const sessions = ref([])
const activeSessionId = ref('')

// ---------- 文档入库 ----------
const urlsText = ref('')
const localPath = ref('')
const maxFiles = ref(50)
const uploadFiles = ref([])
const ingest = ref({ running: false, processed: 0, total: 0, chunks: 0 })

// ---------- 问答 ----------
const messages = ref([])
const question = ref('')
const answering = ref(false)
const activeCite = ref(null)

// ---------- 增量更新 ----------
const watchDir = ref('')
const watchStatus = ref({ running: false, directory: '', kb_name: '默认', queue_size: 0 })
const watchLog = ref([])
let watchStop = false
const apiToken = ref(
  (() => {
    try {
      return localStorage.getItem('rag_api_token') || ''
    } catch {
      return ''
    }
  })(),
)

const toast = ref('')
let toastTimer = null
function notify(text) {
  toast.value = text
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => (toast.value = ''), 3200)
}

function saveToken() {
  try {
    localStorage.setItem('rag_api_token', apiToken.value.trim())
  } catch {
    /* ignore */
  }
  notify('已保存 API Token')
}

function clearToken() {
  apiToken.value = ''
  try {
    localStorage.removeItem('rag_api_token')
  } catch {
    /* ignore */
  }
  notify('已清除 API Token')
}

function genSessionId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `s-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function fmtTime(ts) {
  if (!ts) return ''
  const d = new Date(Number(ts) * 1000)
  return d.toLocaleString('zh-CN', { hour12: false })
}

// ---------- 生命周期 ----------
onMounted(async () => {
  try {
    config.value = await api.config()
    const res = await api.listKbs()
    kbs.value = res.kbs
  } catch (e) {
    notify(`加载失败：${e.message}`)
  }
  await refreshSessions()
  if (sessions.value.length) {
    await switchSession(sessions.value[0].id)
  } else {
    await newSession()
  }
  refreshWatch()
  connectWatch()
})
onUnmounted(() => {
  watchStop = true
})

// ---------- 会话 ----------
async function refreshSessions() {
  try {
    const res = await api.listSessions()
    sessions.value = res.sessions || []
  } catch {
    /* ignore */
  }
}

async function newSession() {
  activeSessionId.value = genSessionId()
  messages.value = []
  activeCite.value = null
  notify('已新建会话')
}

async function switchSession(id) {
  if (id === activeSessionId.value && messages.value.length) return
  try {
    const res = await api.sessionMessages(id)
    activeSessionId.value = id
    messages.value = (res.messages || []).map((m) => ({
      role: m.role,
      content: m.content,
      sources: m.sources || [],
      steps: m.steps || [],
      streaming: false,
    }))
    activeCite.value = null
  } catch (e) {
    notify(e.message)
  }
}

async function removeSession(id) {
  try {
    await api.deleteSession(id)
    if (id === activeSessionId.value) await newSession()
    await refreshSessions()
  } catch (e) {
    notify(e.message)
  }
}

// ---------- 知识库操作 ----------
async function createKb() {
  if (!newKbName.value.trim()) return
  try {
    const res = await api.createKb(newKbName.value.trim())
    kbs.value = res.kbs
    currentKb.value = newKbName.value.trim()
    newKbName.value = ''
    notify('知识库已创建')
  } catch (e) {
    notify(e.message)
  }
}

async function deleteKb() {
  if (currentKb.value === '默认') return
  try {
    const res = await api.deleteKb(currentKb.value)
    kbs.value = res.kbs
    currentKb.value = '默认'
    notify('知识库已删除')
  } catch (e) {
    notify(e.message)
  }
}

// ---------- 增量更新 ----------
function refreshWatch() {
  api.watchStatus().then((s) => (watchStatus.value = s)).catch(() => {})
}

function localHeaders() {
  const h = {}
  const t = apiToken.value.trim()
  if (t) h['X-API-Token'] = t
  return h
}

function connectWatch() {
  const push = (text) => {
    watchLog.value.push(text)
    if (watchLog.value.length > 30) watchLog.value.shift()
  }
  let retryDelay = 3000
  const run = async () => {
    if (watchStop) return
    let everConnected = false
    try {
      const res = await fetch('/api/watch/stream', { headers: localHeaders() })
      await readSSE(res, {
        connected: (d) => {
          everConnected = true
          retryDelay = 3000
          push(`监听已连接：${d.watch_dir || '（未设置）'}`)
        },
        file_enqueued: (d) => push(`发现新文件，已入队：${d.path}`),
        file_started: (d) => push(`开始入库：${d.path}`),
        file_done: (d) => {
          push(`入库完成：${d.path}（${d.chunks} 片段）`)
          refreshWatch()
        },
        file_error: (d) => push(`入库失败：${d.path} - ${d.error}`),
        error: (d) => push(`增量更新：${d.message || '连接中断'}`),
      })
    } catch {
      push('增量更新连接中断')
    }
    if (watchStop) return
    if (everConnected) push(`连接已断开，${retryDelay / 1000} 秒后重连...`)
    setTimeout(run, retryDelay)
    // 一直连不上时指数退避（上限 30 秒），避免持续打爆服务端
    retryDelay = Math.min(retryDelay * 2, 30000)
  }
  run()
}

async function startWatch() {
  if (!watchDir.value.trim()) {
    notify('请填写监听目录')
    return
  }
  try {
    await api.startWatch(watchDir.value.trim(), currentKb.value)
    refreshWatch()
    notify('文件监听已启动')
  } catch (e) {
    notify(e.message)
  }
}

async function stopWatch() {
  try {
    await api.stopWatch()
    refreshWatch()
    notify('文件监听已停止')
  } catch (e) {
    notify(e.message)
  }
}

// ---------- 文档入库 ----------
function runIngest() {
  const urls = urlsText.value.split('\n').map((s) => s.trim()).filter(Boolean)
  const localPaths = localPath.value.trim() ? [localPath.value.trim()] : []
  if (!urls.length && !localPaths.length) {
    notify('请填写 URL 或本地路径')
    return
  }
  ingest.value = { running: true, processed: 0, total: 0, chunks: 0 }
  ingestStream(
    { urls, local_paths: localPaths, kb_name: currentKb.value, max_files: maxFiles.value },
    {
      start: () => notify('开始入库...'),
      progress: (d) => {
        ingest.value = { running: true, processed: d.processed, total: d.total, chunks: d.chunks }
      },
      done: (d) => {
        ingest.value = { running: false, processed: d.total, total: d.total, chunks: d.chunk_count }
        notify(`入库完成，共 ${d.chunk_count} 个片段`)
        if (d.errors?.length) notify(`有 ${d.errors.length} 个来源失败：${d.errors[0][1]}`)
      },
      error: (d) => {
        ingest.value = { ...ingest.value, running: false }
        notify(`入库失败：${d.message || d.error}`)
      },
    },
  )
}

async function runUpload() {
  if (!uploadFiles.value.length) {
    notify('请先选择文件')
    return
  }
  ingest.value = { running: true, processed: 0, total: uploadFiles.value.length, chunks: 0 }
  try {
    const res = await api.upload(uploadFiles.value, currentKb.value, maxFiles.value)
    ingest.value = {
      running: false,
      processed: uploadFiles.value.length,
      total: uploadFiles.value.length,
      chunks: res.chunk_count,
    }
    notify(`上传入库完成，共 ${res.chunk_count} 个片段`)
    if (res.skipped_unsupported) notify(`${res.skipped_unsupported} 个文件类型不受支持，已跳过`)
    if (res.skipped_over_limit) notify(`超过数量上限（${maxFiles.value}），${res.skipped_over_limit} 个文件未处理`)
    if (res.errors?.length) notify(`有 ${res.errors.length} 个文件失败`)
  } catch (e) {
    ingest.value = { ...ingest.value, running: false }
    notify(`上传失败：${e.message}`)
  }
}

function onFilePicked(event) {
  uploadFiles.value = Array.from(event.target.files || [])
}

const ingestPercent = computed(() => {
  if (!ingest.value.total) return 0
  return Math.round((ingest.value.processed / ingest.value.total) * 100)
})

// ---------- 回答与溯源 ----------
function parseContent(content, maxCite = 0) {
  return content
    .split(/(\[\d+\])/g)
    .filter(Boolean)
    .map((part) => {
      const m = part.match(/^\[(\d+)\]$/)
      if (m) {
        const n = Number(m[1])
        if (n >= 1 && n <= maxCite) return { kind: 'cite', n, text: part }
      }
      return { kind: 'text', text: part }
    })
}

function focusSource(n) {
  activeCite.value = n
  nextTick(() => {
    const el = document.querySelector(`[data-source="${n}"]`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  })
}

function ask() {
  const q = question.value.trim()
  if (!q || answering.value) return
  messages.value.push({ role: 'user', content: q })
  const assistant = { role: 'assistant', content: '', sources: [], steps: [], streaming: true }
  messages.value.push(assistant)
  question.value = ''
  answering.value = true
  queryStream(
    { question: q, kb_name: currentKb.value, session_id: activeSessionId.value },
    {
      sources: (d) => {
        assistant.sources = d.sources || []
        assistant.steps = d.steps || []
      },
      token: (d) => {
        assistant.content += d.text || ''
      },
      done: async (d) => {
        assistant.content = d.answer || assistant.content
        assistant.streaming = false
        answering.value = false
        refreshSessions()
      },
      error: (d) => {
        assistant.content = d.message || d.error || '出错了'
        assistant.streaming = false
        answering.value = false
      },
    },
  ).catch((e) => {
    assistant.content = `请求失败：${e.message}`
    assistant.streaming = false
    answering.value = false
  })
}

function onQuestionKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    ask()
  }
}
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="topbar" style="border: none; padding: 0 0 16px">
        <h1>纠正式 RAG</h1>
        <span class="badge" :class="config && config.has_llm_key ? 'ok' : 'warn'">
          {{ config ? config.llm_model : '...' }}
        </span>
      </div>

      <!-- API Token -->
      <div class="field">
        <label>API Token（服务端设置 API_TOKEN 时需要）</label>
        <div class="row">
          <input v-model="apiToken" type="password" placeholder="Bearer Token" />
          <el-button size="small" @click="saveToken">保存</el-button>
          <el-button size="small" @click="clearToken">清除</el-button>
        </div>
      </div>

      <!-- 会话列表 -->
      <div class="section">
        <h3>会话</h3>
        <el-button size="small" type="primary" style="width: 100%; margin-bottom: 8px" @click="newSession">
          新建会话
        </el-button>
        <div class="session-list">
          <div
            v-for="s in sessions"
            :key="s.id"
            class="session-item"
            :class="{ active: s.id === activeSessionId }"
            @click="switchSession(s.id)"
          >
            <span class="s-title">{{ s.title }}</span>
            <span class="s-time">{{ fmtTime(s.updated_at) }}</span>
            <span class="s-del" @click.stop="removeSession(s.id)">×</span>
          </div>
          <div v-if="!sessions.length" class="hint">还没有历史会话</div>
        </div>
      </div>

      <!-- 知识库管理 -->
      <div class="section">
        <h3>知识库</h3>
        <el-select v-model="currentKb" placeholder="选择知识库" style="width: 100%">
          <el-option v-for="k in kbs" :key="k" :label="k" :value="k" />
        </el-select>
        <div class="row" style="margin-top: 8px">
          <input v-model="newKbName" type="text" placeholder="新建库名称" @keyup.enter="createKb" />
          <el-button size="small" type="primary" @click="createKb">新建</el-button>
        </div>
        <div class="row" style="margin-top: 8px">
          <el-button size="small" type="danger" @click="deleteKb" :disabled="currentKb === '默认'">
            删除当前库
          </el-button>
        </div>
      </div>

      <!-- 文档入库 -->
      <div class="section">
        <h3>文档入库</h3>
        <div class="field">
          <label>URL（每行一个）</label>
          <textarea v-model="urlsText" rows="3" placeholder="https://..."></textarea>
        </div>
        <div class="field">
          <label>本地路径/文件夹</label>
          <input v-model="localPath" type="text" placeholder="D:\docs" />
          <div class="hint">Docker 模式下填宿主机路径会自动映射到容器（见 .env 的 HOST_MOUNT_PATH）</div>
        </div>
        <div class="row">
          <label class="hint" style="margin: 0">上限</label>
          <input
            v-model.number="maxFiles"
            type="number"
            min="1"
            max="200"
            style="width: 90px; padding: 7px"
          />
          <el-button size="small" type="primary" @click="runIngest" :disabled="ingest.running">入库</el-button>
        </div>
        <div class="field" style="margin-top: 10px">
          <label>文件上传</label>
          <input type="file" multiple accept=".pdf,.txt,.md,.docx,.html,.csv" @change="onFilePicked" />
        </div>
        <el-button size="small" @click="runUpload" :disabled="ingest.running">上传入库</el-button>
        <div v-if="ingest.running || ingest.total" class="progress" style="margin-top: 10px">
          <div :style="{ width: ingestPercent + '%' }"></div>
        </div>
        <div class="hint" v-if="ingest.total">
          {{ ingest.processed }}/{{ ingest.total }} 个来源 · {{ ingest.chunks }} 片段
        </div>
      </div>

      <!-- 增量更新 -->
      <div class="section">
        <h3>增量更新（文件监听）</h3>
        <div class="field">
          <label>监听目录</label>
          <input v-model="watchDir" type="text" placeholder="D:\docs" />
        </div>
        <div class="row">
          <el-button size="small" type="primary" @click="startWatch">启动监听</el-button>
          <el-button size="small" @click="stopWatch">停止</el-button>
        </div>
        <div class="hint" style="margin-top: 6px">
          状态：{{ watchStatus.running ? '运行中' : '未运行' }} · 队列 {{ watchStatus.queue_size ?? 0 }}
        </div>
        <div class="watch-log">
          <div v-for="(l, i) in watchLog" :key="i">{{ l }}</div>
        </div>
      </div>
    </aside>

    <main class="main">
      <div class="chat">
        <div v-if="!messages.length" class="msg hint">
          示例问题：总结这套系统的检索与纠错流程，并说明重排分数如何体现。
        </div>
        <div v-for="(m, i) in messages" :key="i" class="msg">
          <div class="bubble" :class="m.role">
            <template v-if="m.role === 'assistant'">
              <template v-for="(seg, k) in parseContent(m.content, m.sources ? m.sources.length : 0)" :key="k">
                <span v-if="seg.kind === 'text'">{{ seg.text }}</span>
                <span v-else class="cite" @click="focusSource(seg.n)">[{{ seg.n }}]</span>
              </template>
            </template>
            <template v-else>{{ m.content }}</template>
            <span v-if="m.streaming" class="cursor"></span>
          </div>

          <!-- 检索过程可视化 -->
          <div v-if="m.role === 'assistant' && m.steps && m.steps.length" class="steps">
            <template v-for="(st, idx) in m.steps" :key="idx">
              <span class="step">{{ STEP_LABELS[st] || st }}</span>
              <span v-if="idx < m.steps.length - 1" class="arrow">→</span>
            </template>
          </div>
          <el-alert
            v-if="m.role === 'assistant' && (m.steps || []).includes('web_search')"
            type="info"
            :closable="false"
            title="已触发 Tavily 网络搜索"
            style="margin-top: 8px"
          />

          <details v-if="m.role === 'assistant' && m.sources && m.sources.length" class="sources">
            <summary>用于生成的来源与重排分数（{{ m.sources.length }}），点 [n] 高亮</summary>
            <div
              v-for="(s, j) in m.sources"
              :key="j"
              class="source-card"
              :data-source="s.index"
              :class="{ active: activeCite === s.index }"
            >
              <div class="meta">
                <span>[{{ s.index }}] {{ s.source }}</span>
                <span v-if="s.score != null" class="score">重排 {{ Number(s.score).toFixed(1) }}</span>
              </div>
              <div class="snippet">{{ s.snippet }}</div>
            </div>
          </details>
        </div>
      </div>

      <div class="composer">
        <textarea
          v-model="question"
          rows="1"
          placeholder="请输入问题..."
          @keydown="onQuestionKeydown"
        ></textarea>
        <el-button type="primary" @click="ask" :disabled="answering">发送</el-button>
      </div>
    </main>

    <div v-if="toast" class="toast">{{ toast }}</div>
  </div>
</template>
