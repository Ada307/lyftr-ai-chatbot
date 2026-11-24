# RelayChat

RelayChat is a resilient, full-stack LLM chat application built with React, FastAPI, LiteLLM, and SQLite. It provides automatic multi-key provider failover, in-memory response caching, multi-turn conversational context assembly, atomic transaction persistence, and rich per-message metadata tracking (resolution source, token usage, latency, and key provenance).

---

## Live URLs

- **Frontend Application (Vercel)**: [https://lyftr-ai-chatbot-nine.vercel.app](https://lyftr-ai-chatbot-nine.vercel.app)
- **Backend API (Railway)**: [https://lyftr-ai-chatbot-production-0b264854.up.railway.app](https://lyftr-ai-chatbot-production-0b264854.up.railway.app)
- **Backend Health Check**: [https://lyftr-ai-chatbot-production-0b264854.up.railway.app/health](https://lyftr-ai-chatbot-production-0b264854.up.railway.app/health)
- **GitHub Repository**: [https://github.com/Ada307/lyftr-ai-chatbot](https://github.com/Ada307/lyftr-ai-chatbot)

---

## Run locally: with Docker and without

### Prerequisites

- **Docker & Docker Compose** (Docker Desktop on Windows/macOS or Docker Engine on Linux)
- **Node.js 20+** and **npm**
- **Python 3.11+**
- **Git** and **GNU Make** (optional, recommended)
- At least two Gemini API keys (or compatible LiteLLM provider keys)

### Environment Configuration

Create a root `.env` file by copying the provided `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and supply at least two distinct API keys:

```env
LLM_API_KEYS=your_first_gemini_api_key,your_second_gemini_api_key
LLM_MODEL=gemini/gemini-3.5-flash-lite
DATABASE_URL=sqlite:////data/app.db
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173
ENABLE_FAILURE_SIMULATION=true
VITE_API_BASE_URL=http://localhost:8000
```

---

### Option A: Run with Docker Compose (Recommended)

1. **Build and launch both services**:
   ```bash
   make up
   ```
   *(Or alternatively: `docker compose up --build -d`)*

2. **Access the application**:
   - **Frontend UI**: [http://localhost:3000](http://localhost:3000)
   - **Backend API**: [http://localhost:8000](http://localhost:8000)
   - **Health Readiness Check**: [http://localhost:8000/health](http://localhost:8000/health)

3. **Stop the services** (preserves the persistent SQLite volume):
   ```bash
   make down
   ```
   *(Or alternatively: `docker compose down`)*

---

### Option B: Run locally without Docker

1. **Backend Setup**:
   ```powershell
   # Create and activate Python virtual environment
   cd backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Start FastAPI backend
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Frontend Setup** (in a separate terminal):
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
   Access the frontend at [http://localhost:5173](http://localhost:5173).

3. **Run Backend Tests**:
   ```powershell
   cd backend
   python -m pytest tests -v
   ```

---

## Design decisions: key fallback, caching, SQLite session storage

### 1. Explicit Sequential Key Fallback Loop
- **Ordered Key Progression**: The backend parses `LLM_API_KEYS` into stable identifiers (`key_1`, `key_2`, ...). On every completion request, inference begins strictly at `key_1`.
- **Deterministic Failover**: If an inference attempt fails due to rate limits (`429`), temporary upstream errors (`500`, `502`, `503`, `504`), or provider connection issues, the gateway captures the exception, increments `fallback_attempts`, and transparently fails over to `key_2`.
- **Request-Scoped Failure Simulation**: The `X-Simulate-Failure` header (e.g. `key_1=429` or `*=429`) simulates failures for targeted keys on that specific request without mutating shared global state or affecting concurrent queries.
- **Fail-Fast Exhaustion**: If all configured keys fail, the service raises an `AllProvidersUnavailableException`, resulting in an HTTP `503 Service Unavailable` response with `{"detail": "all providers unavailable"}`.

### 2. LiteLLM In-Memory Response Caching
- **Official Built-in Cache**: Initialized once during application startup lifespan via `litellm.enable_cache(type="local")` using LiteLLM's process-local in-memory cache. No external Redis dependencies are required.
- **Cache Hit Provenance**: Cache hits are identified directly from LiteLLM's metadata (`response._hidden_params["cache_hit"] is True`).
- **Metadata Transparency**: Cache hits report `resolution_source="LLM_CACHE"`, `key_id=null`, `fallback_attempts=0`, exact retained token counts, and reduced processing latency.
- **Simulation Bypass**: When `X-Simulate-Failure` is active, caching is bypassed (`no-cache: True, no-store: True`) to ensure failover mechanics are genuinely exercised and prevent cache poisoning.
- **Process-Local Scope**: The in-memory cache resides in memory and resets whenever the backend server restarts.

### 3. Atomic SQLite Session Storage
- **Built-in `sqlite3`**: Uses Python's native `sqlite3` module with zero heavyweight ORM overhead.
- **Data Integrity**: Uses `PRAGMA foreign_keys = ON;`, `PRAGMA journal_mode = WAL;`, and an explicit index `idx_messages_session_id_id ON messages(session_id, id)`.
- **Atomic Chat Turns**: User and assistant messages, along with complete JSON metadata, are inserted in a single atomic database transaction (`save_chat_turn`). If inference fails (such as HTTP 503), no session or message records are written to the database.
- **Docker Persistence**: The SQLite file is stored at `/data/app.db` and mounted to the Docker named volume `relaychat-data`, ensuring chat histories survive container restarts.

---

## Verification: how to see failover, cache hits and per-message token usage

You can verify all core behaviors using the frontend UI or via PowerShell / cURL commands.

### 1. Verification via Frontend UI
1. Open [http://localhost:3000](http://localhost:3000).
2. Note the **Backend Status** badge (`Online • 2 Keys`).
3. Send a message (e.g. *"What is the capital of France?"*).
4. Observe the assistant bubble metadata:
   - `meta-source`: **`LIVE_INFERENCE`** (indigo badge)
   - `meta-model`: `gemini/gemini-3.5-flash-lite`
   - `meta-latency`: Wall clock backend processing time (e.g. `2400ms`)
   - `meta-input-tokens` & `meta-output-tokens`: Real token integers
   - `Key: key_1`, `Failovers: 0`
5. Click **New Chat** (creates an isolated session without `localStorage`).
6. Send the **exact same message**.
7. Observe the cache hit:
   - `meta-source`: **`LLM_CACHE`** (emerald badge)
   - `meta-latency`: Near-instantaneous response (e.g. `11ms`–`300ms`)
   - `Key: null`, `Failovers: 0`
   - Tokens match original inference exactly.

---

### 2. PowerShell API Verification Commands

> [!TIP]
> You can run these commands against local Docker (`http://localhost:8000`) or directly against the live Railway backend (`https://lyftr-ai-chatbot-production-0b264854.up.railway.app`).

#### Baseline Cache Miss (`LIVE_INFERENCE`)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"session_id": "test-session-1", "message": "Explain recursion in 10 words."}'
```
*Expected: `resolution_source`: "LIVE_INFERENCE", `key_id`: "key_1", `fallback_attempts`: 0.*

#### Cache Hit Verification across Sessions (`LLM_CACHE`)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"session_id": "test-session-2", "message": "Explain recursion in 10 words."}'
```
*Expected: `resolution_source`: "LLM_CACHE", `key_id`: null, `fallback_attempts`: 0, identical token counts and reply, significantly lower latency.*

#### Key Failover with Warm Cache (`key_1=429`)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method POST `
  -Headers @{"X-Simulate-Failure" = "key_1=429"} `
  -ContentType "application/json" `
  -Body '{"session_id": "test-session-3", "message": "Explain recursion in 10 words."}'
```
*Expected: Cache is bypassed, failover triggers, `resolution_source`: "LIVE_INFERENCE", `key_id`: "key_2", `fallback_attempts`: 1.*

#### Full Provider Outage Simulation (`*=429`)
```powershell
try {
  Invoke-RestMethod -Uri "http://localhost:8000/chat" `
    -Method POST `
    -Headers @{"X-Simulate-Failure" = "*=429"} `
    -ContentType "application/json" `
    -Body '{"session_id": "test-session-4", "message": "Explain recursion in 10 words."}'
} catch {
  $_.Exception.Response.StatusCode.value__
}
```
*Expected: HTTP 503 with `{"detail": "all providers unavailable"}`. Database confirms `test-session-4` was never created.*

---

## Setup Used: your editor and AI tools

- **Operating System**: Windows 11 (64-bit) with PowerShell 7 and GNU Make 4.4.1
- **IDE / Editor**: Visual Studio Code
- **Container Engine**: Docker Desktop 29.6.1 with Compose v2
- **Runtime Environments**: Python 3.11.9, Node.js v22.17.0, npm 10.9.2
- **AI Coding Assistants**:
  - **ChatGPT (GPT-4o)**: Used for architecture consultation, assignment requirements breakdown, and edge-case planning.
  - **Antigravity (Google DeepMind)**: Agentic coding assistant utilized for codebase scaffolding, unit/integration test authorship, iterative test verification, and automated Docker orchestration.

---

## Bonus features (if any): what you added beyond the requirements and why

1. **Live Backend Readiness Probe**: Enhanced `GET /health` to verify real SQLite connectivity, confirm $\ge 2$ configured API keys, and test response cache readiness, returning HTTP 503 if any component is degraded.
2. **Atomic Rollback on Failed Inference**: Integrated database writes exclusively after successful LLM completion inside `save_chat_turn()`. If an upstream failure occurs, no ghost session or orphan user message is created.
3. **Frontend Stale Reply Protection**: Implemented session token refs and `AbortController` cancellation in React so that clicking **New Chat** instantly discards in-flight API requests, preventing previous turn responses from bleeding into a new conversation.
4. **Interactive Prompt Suggestions**: Added one-click test prompts on the welcome screen that allow instant verification of live inference, key failover, and response caching.
5. **One-Click Response Copy**: Added clipboard copy action with temporary visual feedback on each assistant message bubble.