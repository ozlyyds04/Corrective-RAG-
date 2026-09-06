# 🔄 Corrective RAG Agent

## 中文

一个基于 LangGraph 构建的纠正式检索增强生成（Corrective RAG）系统，配以异步全栈界面：FastAPI 后端 + Vue 3 前端，通过 Docker 编排（Redis + Qdrant）。从本地文档检索、相关性评分、必要时改写查询并联网兜底，最后严格基于上下文生成，并通过 SSE 逐字流式输出。通过配套基准评测验证：混合检索 + LLM 重排 + 相关性过滤 + 严格生成，使 20 题子集的**幻觉率降到 0%**，**事实正确率 95%**。

### 功能特性

- **混合检索**：稠密向量（qwen3.7-text-embedding）+ 关键词稀疏向量（jieba 分词），RRF 融合
- **LLM 重排**：默认大语言模型（deepseek-v4-flash-vision-exp）对候选打 0-10 分，取 Top-5
- **相关性评分**：本地片段全部不相关时才触发网络搜索
- **网络搜索兜底**：Tavily 搜索，结果逐条过滤后才进入生成
- **严格生成**：只基于上下文回答，资料不足时拒绝猜测
- **异步后端**：FastAPI + SSE 流式接口，支持检索可视化和 JSON 接口
- **流式输出**：SSE 逐 token 推送，ChatGPT 式打字效果
- **检索可视化**：界面展示召回片段、来源与 LLM 重排分数
- **增量更新**：文件监听把新增/变更文档推入 Redis 队列，后台 worker 自动向量化入库并流式推送进度
- **三层记忆**：短期对话历史（Redis）、长期文档知识（Qdrant）、情景问答摘要（Qdrant），一并注入生成
- **会话列表**：左侧边栏多会话切换，类似 ChatGPT 体验
- **检索过程可视化**：步骤条展示"多路召回 → 重排 → 过滤"；触发联网兜底时明确提示
- **答案溯源**：回答用 [n] 标注所用原文片段，点击高亮对应来源
- **上下文压缩**：在 token 预算内先按重排分数择优保留，超长片段截断后再生成
- **安全与并发**：可选 `API_TOKEN` 鉴权（默认关）、收紧 CORS、流式请求断连自动取消，Redis leader 锁避免多 worker 重复入库
- **多知识库**：单集合 + `kb` 标签过滤（免费档友好）
- **多来源输入**：URL（每行一个）、本地文件/文件夹（递归）、多文件上传（上限可配置，默认 50）
- **Docker 部署**：`docker-compose` 编排 `api`、`web`、`redis`（可选本地 `qdrant`）
- **健壮性**：网络断连/超时自动重试（最多 4 次，指数退避）
- **.env 配置**：所有密钥从 `.env` 读取；中文界面

### 项目结构

```text
src/corrective_rag/
  __init__.py       # 包初始化
  core.py           # 核心逻辑：加载器、混合检索、重排、LangGraph 工作流
  api.py            # FastAPI 后端（REST + SSE 流式、入库、监听接口）
  jobs.py           # Redis 任务队列 + 文件监听 worker（增量更新）
  memory.py         # 三层记忆：短期（Redis）+ 情景（Qdrant）检索
  evaluate.py       # LLM-as-judge 评估（基线 RAG vs 纠错 RAG）
frontend/           # Vue 3 + Vite 单页应用（聊天流式、检索可视化、入库、监听）
  src/App.vue       # 主界面
  src/api.js        # API 客户端 + SSE 解析器
tests/              # 52 个 pytest 用例（无需 API 密钥）
eval/               # 评测语料 + 47 题评测集
docs/               # 架构图与界面截图
Dockerfile          # 后端镜像
frontend/Dockerfile # 前端构建 + nginx 镜像
docker-compose.yml  # api / web / redis / qdrant
.env.example        # 配置模板
```

### 架构图

![架构图](docs/corrective_rag.svg)

### 如何运行

#### 方式一 — Docker Compose（推荐）

1. **克隆仓库**：
   ```bash
   git clone https://github.com/ozlyyds04/Corrective-RAG-.git
   cd Corrective-RAG-
   ```

2. **配置 API 密钥**：
   ```bash
   cp .env.example .env
   ```
   填写 `.env`：
   - `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` —— 大语言模型（默认 DeepSeek 官方 API；兼容任意 OpenAI 风格接口）
   - `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` —— 向量模型（默认 qwen3.7-text-embedding，走 DashScope 兼容接口）
   - `TAVILY_API_KEY` —— 网络搜索
   - `QDRANT_URL` / `QDRANT_API_KEY` —— Qdrant 集群（云端，或 `http://qdrant:6333` 使用内置本地服务）
   - `HOST_MOUNT_PATH` / `CONTAINER_MOUNT_PATH` —— 把宿主机目录挂入容器，让本地文件夹入库可用（Docker）
   - `MAX_CONTEXT_TOKENS` —— 上下文压缩的 token 预算（默认 2600）
   - `REDIS_URL` / `WATCH_DIR` / `WATCH_KB` —— 增量文件监听（容器内由 compose 注入 `REDIS_URL`）
   - `API_TOKEN` —— 可选；设置后除 `/api/health`、`/api/config` 外都需 `Authorization: Bearer <token>`
   - `ALLOWED_ORIGINS` —— 允许跨域的来源（逗号分隔，默认 `http://localhost:5173,http://localhost:8081`）
   - `APP_DEBUG` —— 设为 `1` 时返回具体错误信息

3. **启动服务**：
   ```bash
   docker compose up --build
   ```
   - 网页界面：<http://localhost:8081>
   - API：<http://localhost:8001>
   - 本地 Qdrant：<http://localhost:6333>

#### 方式二 — 本地开发

1. **安装后端依赖**：
   ```bash
   uv sync
   ```

2. **启动后端 API**（监听 `http://localhost:8001`）：
   ```bash
   uv run uvicorn corrective_rag.api:app --reload --port 8001
   ```

3. **启动前端**（另开一个终端）：
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Vite 会把 `/api` 代理到后端，打开 <http://localhost:5173>。

### 使用说明

- **知识库**：新建 / 选择 / 删除知识库
- **文档入库**：粘贴 URL（每行一个）、本地文件或文件夹路径、或上传文件（上限可配置，最多 200 个）
- **增量更新**：对某个目录启动文件监听，新增/变更文档会自动向量化入库（按来源替换旧片段，不产生重复数据），进度实时推送
- **会话**：左侧边栏可新建 / 切换 / 删除会话，对话历史保存在 Redis（默认保留一天）
- **提问**：回答逐字流式展示；回答中的 [n] 标注可点击，高亮对应的原文来源片段；每个回答附检索步骤与来源重排分数

### 测试与评估

运行测试套件（不需要任何 API 密钥）：

```bash
uv run pytest
```

使用 LLM-as-judge 指标（检索精度 precision@k、命中率、忠实度/幻觉率、无支撑断言、事实正确率）对比基线 RAG 与纠错 RAG：

```bash
uv run python -m corrective_rag.evaluate --llm-api-key $LLM_API_KEY \
    --tavily-api-key $TAVILY_API_KEY --qdrant-url $QDRANT_URL
```

评测问题集在 `eval/questions.json`（47 题），语料在 `eval/data/`，报告输出到 `eval/reports/`。

### 技术栈

- **LangChain / LangGraph**：编排与工作流管理
- **FastAPI**：异步后端，REST + SSE 流式
- **Vue 3 + Vite**：单页前端
- **Element Plus**：UI 组件（知识库选择、提示等）
- **Qdrant**：向量数据库（单集合，稠密 + 稀疏向量，`kb` 标签过滤）
- **Redis**：增量更新任务队列 + 事件发布订阅
- **Docker Compose**：服务编排（api / web / redis / qdrant）
- **qwen3.7-text-embedding**：向量模型（DashScope 兼容接口）
- **deepseek-v4-flash-vision-exp**（默认大语言模型）：生成、相关性评分、查询改写、重排
- **Tavily**：网络搜索兜底
- **jieba**：关键词稀疏检索的分词
