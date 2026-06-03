# Sovereign Gen AI LLM

> Sovereign Gen AI with local privacy.

A self-hosted chat app for people who want full control over their AI assistant: which model runs, what the root system prompt says, and exactly what gets remembered.

Your conversations, memories, and settings stay on **your machine** in a local SQLite database. Connect to a local LLM (Ollama, LM Studio, etc.) or an external API (OpenAI or any OpenAI-compatible endpoint). Nothing is sent to a vendor unless **you** configure it.

## Why this exists

Commercial chat products decide what to remember, what to forget, and how the assistant behaves. This project is for anyone with a decent computer who wants to:

- Run generative AI **locally** (or mix local and cloud models)
- Edit the **root system prompt** — the baseline instructions for every reply
- Choose **what is remembered** — save facts explicitly, merge or delete memories, move them between conversations
- See **exactly what goes to the model** before you send (context preview)
- Keep **all data local** by default

## Features

- **Multi-conversation chat** with markdown, math (KaTeX), and image input
- **Memory bank** per conversation — you decide what the model should retain long-term
- **Editable system prompt** (Settings → Default prompt)
- **Multiple LLM profiles** — Ollama, OpenAI, or OpenAI-compatible APIs; pick a model per conversation
- **Speech-to-text** via local Whisper (faster-whisper)
- **Browser text-to-speech** for reading replies aloud
- **Context preview** — inspect system prompt, memories, and history token estimates before sending
- **Docker** one-command deploy, or run backend + frontend in development mode

## Quick start (Docker)

**Requirements:** [Docker](https://docs.docker.com/get-docker/) and, for local models, [Ollama](https://ollama.com/) on your host.

1. **Pull a model in Ollama** (on your host, not inside Docker):

   ```bash
   ollama pull qwen3.5:9b
   ```

2. **Start the app:**

   ```bash
   docker compose up --build
   ```

3. Open **http://localhost:8087**

4. Go to **Settings** and confirm the Ollama address is reachable:
   - **Docker on Windows/macOS:** `http://host.docker.internal:11434` (default)
   - **Docker on Linux:** often `http://172.17.0.1:11434` or your host IP — see [docs/LLM_SETUP.md](docs/LLM_SETUP.md)

5. Send a message. Use **Remember** on replies you want kept in the memory bank.

Data is stored in a Docker volume (`chat-data`). To reset everything: `docker compose down -v`.

## Quick start (development)

**Requirements:** Python 3.11+, Node.js 22+, ffmpeg (for speech-to-text)

```bash
# Backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt

# Optional: point at your Ollama instance
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen3.5:9b

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8087
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the URL Vite prints (usually **http://localhost:5173**). The dev server proxies `/api` to port 8087.

## Connecting an LLM

Configure models under **Settings → LLM models**. Supported providers:

| Provider | Use case | API key |
|----------|----------|---------|
| **Ollama** | Local models on your PC | Not required |
| **OpenAI** | OpenAI cloud models | Required |
| **OpenAI-compatible** | LM Studio, LocalAI, vLLM, Groq, Together, etc. | Depends on service |

Each row needs a **base URL** and **model name**. You can add several profiles and set one as active, or assign a model per conversation.

Detailed setup for Ollama, LM Studio, OpenAI, and other endpoints: **[docs/LLM_SETUP.md](docs/LLM_SETUP.md)**

### Ollama (local) — minimal example

| Field | Value |
|-------|-------|
| Provider | Ollama |
| Address | `http://localhost:11434` (native) or `http://host.docker.internal:11434` (Docker) |
| Model | Whatever you pulled, e.g. `llama3.2`, `qwen3.5:9b`, `mistral` |

### OpenAI (cloud) — minimal example

| Field | Value |
|-------|-------|
| Provider | OpenAI API |
| Address | `https://api.openai.com/v1` |
| Model | e.g. `gpt-4o-mini` |
| API key | Your key from [platform.openai.com](https://platform.openai.com/) |

## Controlling memory and prompts

### Root system prompt

**Settings → Default prompt** sets the system message prepended to every LLM request. Change it to match how you want the assistant to behave. You can reset to the built-in baseline anytime.

### Memory bank

Memories are **opt-in**, not automatic:

- Click **Remember** on an assistant message, or type `remember: your fact here`
- View and manage all memories on the **Memories** page
- Merge, integrate (AI-combined summary), move between conversations, or delete
- Only **active** (non-archived) memories are injected into the model context

### What the model actually sees

Before sending, use **Preview context** to see the system prompt, memory bank, chat history, and rough token estimate. Toggle **Current message only** to skip history and send just the latest user message plus memories.

## Environment variables

Copy [`.env.example`](.env.example) for reference. Common variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHAT_DATA_DIR` | `./data` | SQLite DB and app data |
| `OLLAMA_HOST` | `http://localhost:11434` | Default Ollama URL (seeded on first run) |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Default model name (seeded on first run) |
| `WHISPER_MODEL` | `base.en` | Speech-to-text model |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Whisper compute type |

LLM settings can also be changed in the UI and are stored in the database.

## Project layout

```
backend/          FastAPI API, SQLite, LLM + Whisper integration
frontend/         React + Vite UI
data/             Local database (created at runtime; gitignored)
docker-compose.yml
Dockerfile
```

Legacy CLI scripts (`mychat.py`, `converse.py`, etc.) are older experiments; the supported app is the web UI + `backend/main.py`.

## Privacy

- Chat history and memories live in `CHAT_DATA_DIR/chat.db` on your machine
- API keys are stored in that same local database — not in git
- With Ollama or another local endpoint, prompts never leave your network
- Cloud providers only receive traffic if you configure them

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and pull requests are welcome.

## License

[MIT](LICENSE) — use, modify, and share freely.
