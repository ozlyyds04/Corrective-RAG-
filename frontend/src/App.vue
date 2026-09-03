<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api, queryStream, ingestStream } from './api'

// ---------- 配置与知识库 ----------
const config = ref(null)
const kbs = ref([])
const currentKb = ref('默认')
const newKbName = ref('')

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

// ---------- 增量更新 ----------
const watchDir = ref('')
const watchStatus = ref({ running: false, directory: '', kb_name: '默认', queue_size: 0 })
const watchLog = ref([])
let watchSource = null

const toast = ref('')
let toastTimer = null
function notify(text) {
  toast.value = text
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => (toast.value = ''), 3200)
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
  refreshWatch()
  connectWatch()
})
onUnmounted(() => {
  watchSource?.close()
})

function refreshWatch() {
  api.watchStatus().then((s) => (watchStatus.value = s)).catch(() => {})
}

function connectWatch() {
  watchSource?.close()
  watchSource = new EventSource('/api/watch/stream')
  const push = (text) => {
    watchLog.value.push(text)
    if (watchLog.value.length > 30) watchLog.value.shift()
  }
  watchSource.addEventListener('connected', (e) => {
    const d = JSON.parse(e.data)
    push(`监听已连接：${d.watch_dir || '（未设置）'}`)
  })
  watchSource.addEventListener('file_enqueued', (e) => {
    const d = JSON.parse(e.data)
    push(`发现新文件，已入队：${d.path}`)
  })
  watchSource.addEventListener('file_started', (e) => {
    const d = JSON.parse(e.data)
    push(`开始入库：${d.path}`)
  })
  watchSource.addEventListener('file_done', (e) => {
    const d = JSON.parse(e.data)
    push(`入库完成：${d.path}（${d.chunks} 片段）`)
    refreshWatch()
  })
  watchSource.addEventListener('file_error', (e) => {
    const d = JSON.parse(e.data)
    push(`入库失败：${d.path} - ${d.error}`)
  })
  watchSource.addEventListener('error', (e) => {
    push('增量更新连接中断，稍后自动重连...')
  })
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
        ingest.value = {
          running: true,
          processed: d.processed,
          total: d.total,
          chunks: d.chunks,
        }
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
    ingest.value = { running: false, processed: uploadFiles.value.length, total: uploadFiles.value.length, chunks: res.chunk_count }
    notify(`上传入库完成，共 ${res.chunk_count} 个片段`)
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

// ---------- 增量更新控制 ----------
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

// ---------- 问答 ----------

function ask() {
  const q = question.value.trim()
  if (!q || answering.value) return
  messages.value.push({ role: 'user', content: q })
  const assistant = { role: 'assistant', content: '', sources: [], steps: [], streaming: true }
  messages.value.push(assistant)
  question.value = ''
  answering.value = true
  queryStream(
    { question: q, kb_name: currentKb.value },
    {
      sources: (d) => {
        assistant.sources = d.sources || []
        assistant.steps = d.steps || []
      },
      token: (d) => {
        assistant.content += d.text || ''
      },
      done: (d) => {
        assistant.content = d.answer || assistant.content
        assistant.streaming = false
        answering.value = false
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

      <!-- 知识库管理 -->
      <div class="section">
        <h3>知识库</h3>
        <div class="kb-list">
          <span
            v-for="k in kbs"
            :key="k"
            class="chip"
            :class="{ active: k === currentKb }"
            @click="currentKb = k"
          >
            {{ k }}
          </span>
        </div>
        <div class="row">
          <input v-model="newKbName" type="text" placeholder="新建库名称" @keyup.enter="createKb" />
          <button class="primary" @click="createKb">新建</button>
        </div>
        <div class="row" style="margin-top: 8px">
          <button class="danger" @click="deleteKb" :disabled="currentKb === '默认'">删除当前库</button>
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
        </div>
        <div class="row">
          <label class="hint" style="margin: 0">上限</label>
          <input
            v-model.number="maxFiles"
            type="number"
            min="1"
            max="500"
            style="width: 90px; padding: 7px"
          />
          <button class="primary" @click="runIngest" :disabled="ingest.running">入库</button>
        </div>
        <div class="field" style="margin-top: 10px">
          <label>文件上传</label>
          <input type="file" multiple accept=".pdf,.txt,.md,.docx,.html,.csv" @change="onFilePicked" />
        </div>
        <button @click="runUpload" :disabled="ingest.running">上传入库</button>
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
          <button class="primary" @click="startWatch">启动监听</button>
          <button @click="stopWatch">停止</button>
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
            <span>{{ m.content }}</span>
            <span v-if="m.streaming" class="cursor"></span>
          </div>
          <details v-if="m.role === 'assistant' && m.sources && m.sources.length" class="sources">
            <summary>召回来源与重排分数（{{ m.sources.length }}）</summary>
            <div v-for="(s, j) in m.sources" :key="j" class="source-card">
              <div class="meta">
                <span>{{ s.source }}</span>
                <span v-if="s.score != null" class="score">重排 {{ s.score }}</span>
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
        <button class="primary" @click="ask" :disabled="answering">发送</button>
      </div>
    </main>

    <div v-if="toast" class="toast">{{ toast }}</div>
  </div>
</template>
