# LLM setup guide

Sovereign Gen AI LLM talks to language models over HTTP. You configure one or more **model profiles** in **Settings → LLM models**. Each profile has:

- **Provider** — how requests are formatted
- **Address (base URL)** — where the server lives
- **Model** — the model identifier that server expects
- **API key** — optional for local servers, required for OpenAI

You can run several profiles (e.g. a fast local model and a cloud fallback) and pick one per conversation.

---

## Ollama (recommended for local use)

[Ollama](https://ollama.com/) runs open-weight models on your computer with a simple API.

### 1. Install and pull a model

```bash
# Install from https://ollama.com/download
ollama pull qwen3.5:9b
# or: ollama pull llama3.2
# or: ollama pull mistral
```

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

### 2. Configure in the app

| Setting | Native install | Docker app + Ollama on host |
|---------|----------------|----------------------------|
| Provider | Ollama | Ollama |
| Address | `http://localhost:11434` | `http://host.docker.internal:11434` |
| Model | Name from `ollama list`, e.g. `qwen3.5:9b` | Same |

**Linux + Docker:** `host.docker.internal` may not work. Try:

- `http://172.17.0.1:11434` (default Docker bridge), or
- Your LAN IP, e.g. `http://192.168.1.10:11434`

Ensure Ollama listens on all interfaces if needed:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 3. Vision models

Image chat uses Ollama’s native image field. Use a vision-capable model, e.g.:

```bash
ollama pull llava
```

Set model to `llava` (or your vision model name) in Settings.

---

## LM Studio (local, OpenAI-compatible)

[LM Studio](https://lmstudio.ai/) serves models with an OpenAI-style API.

1. Load a model in LM Studio
2. Start the **Local Server** (default port **1234**)
3. In this app:

| Setting | Value |
|---------|-------|
| Provider | OpenAI-compatible API |
| Address | `http://localhost:1234/v1` |
| Model | The model id shown in LM Studio (often matches the loaded model name) |
| API key | Leave blank unless you enabled auth |

If the app runs in Docker but LM Studio runs on the host, replace `localhost` with `host.docker.internal` (or your host IP on Linux).

---

## LocalAI, vLLM, text-generation-webui

Any server that exposes **`POST /v1/chat/completions`** works with the **OpenAI-compatible** provider.

Examples:

| Software | Typical base URL |
|----------|------------------|
| LocalAI | `http://localhost:8080/v1` |
| vLLM | `http://localhost:8000/v1` |
| text-generation-webui (OpenAI extension) | `http://localhost:5000/v1` |

Use the model string your server documents. API keys depend on your server configuration.

---

## OpenAI (cloud)

1. Create an API key at [platform.openai.com](https://platform.openai.com/api-keys)
2. In this app:

| Setting | Value |
|---------|-------|
| Provider | OpenAI API |
| Address | `https://api.openai.com/v1` |
| Model | e.g. `gpt-4o-mini`, `gpt-4o` |
| API key | Your secret key |

**Privacy note:** Messages, memories, and your system prompt are sent to OpenAI when you use this provider.

---

## Other OpenAI-compatible cloud APIs

Use **OpenAI-compatible API** with the vendor’s base URL:

| Service | Base URL (example) |
|---------|-------------------|
| Groq | `https://api.groq.com/openai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Azure OpenAI | `https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT` |

Check each vendor’s docs for the exact model name and whether an API key is required.

---

## Multiple models

You can add several rows in Settings:

- Mark one as **Active** for new conversations
- Change model per conversation via the model picker in the chat header
- Add **Comments** to label profiles (e.g. “Fast local”, “Cloud backup”)

---

## Troubleshooting

### “Could not reach Ollama at …”

- Confirm Ollama is running: `ollama list`
- From inside Docker, `localhost` refers to the container — use `host.docker.internal` or the host IP
- Check firewall rules for port 11434

### HTTP 404 or “model not found”

- Run `ollama pull MODEL_NAME` or load the model in LM Studio
- Model name must match exactly (including tags like `:9b`)

### OpenAI: “API key required”

- OpenAI provider always needs a key
- Use **OpenAI-compatible** (not **OpenAI API**) for local servers without auth

### Slow first reply

- Local models load into RAM on first use — normal for Ollama/LM Studio
- Whisper downloads model weights on first transcription

### Connection works in browser but not from Docker

- On Linux, add to `docker-compose.yml`:

  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```

  (Already included in this repo’s compose file.)

---

## How requests are built

Each chat request sends (in order):

1. **Your default system prompt** (Settings)
2. **Memory bank** — bullet list of saved memories for that conversation
3. **Chat history** (up to 40 messages, unless “current message only” is on)
4. **Your new message** (and image, if attached)

Use **Preview context** in the chat UI to inspect this payload before sending.
