# Security

## Reporting a vulnerability

If you find a security issue, please **do not** open a public GitHub issue with exploit details.

Open a private [GitHub Security Advisory](https://github.com/YOUR_USERNAME/YOUR_REPO/security/advisories/new) or email the maintainer directly.

## Scope notes

- This app is designed for **local, single-user** use on your own machine
- There is **no built-in authentication** — do not expose port 8087 to the public internet without a reverse proxy and auth
- API keys are stored in the local SQLite database (`CHAT_DATA_DIR/chat.db`); protect that directory
- When using cloud LLM providers, your prompts and memories are sent to that provider

## Recommendations

- Run behind localhost or a trusted home network only
- Use local models (Ollama) when privacy is critical
- Back up `data/` if memories matter to you; treat `docker compose down -v` as destructive
