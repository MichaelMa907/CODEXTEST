# ChatGPT Wrapper Website (localhost-ready)

This project is now an actual deployable web app.

## What it does

- Serves a chat UI at `http://localhost:8000`.
- Uses a backend wrapper that remembers the whole transcript in memory.
- Sends full recent history until context limit is near.
- Rolls older turns into a summary when token budget gets tight.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY="your_key_here"
```

## Run locally

```bash
python app.py
```

Then open: `http://localhost:8000`

## Config via environment variables

- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `MAX_CONTEXT_TOKENS` (default: `128000`)
- `RESPONSE_TOKENS` (default: `512`)

## API

`POST /api/chat`

Request:

```json
{
  "message": "hello",
  "session_id": "optional-existing-session"
}
```

Response includes:
- `reply`
- `session_id`
- `full_transcript_count`
- `live_message_count`
- `has_summary`
