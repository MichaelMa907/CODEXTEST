from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

from chatgpt_wrapper import ContextMemoryChatWrapper


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "128000"))
RESPONSE_TOKENS = int(os.getenv("RESPONSE_TOKENS", "512"))

API_KEY = os.getenv("OPENAI_API_KEY")


def _build_client():
    if not API_KEY:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(api_key=API_KEY)


CLIENT = _build_client()
SESSIONS: dict[str, ContextMemoryChatWrapper] = {}

INDEX_HTML = """<!doctype html>
<html lang='en'>
  <head>
    <meta charset='UTF-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1.0' />
    <title>Memory Chat Wrapper</title>
    <style>
      body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; }
      .container { max-width: 860px; margin: 20px auto; padding: 16px; }
      #chat { border: 1px solid #334155; border-radius: 8px; min-height: 360px; padding: 12px; background: #111827; overflow-y: auto; }
      .msg { margin: 10px 0; padding: 10px; border-radius: 6px; white-space: pre-wrap; }
      .user { background: #1d4ed8; }
      .assistant { background: #065f46; }
      .row { display: flex; gap: 8px; margin-top: 12px; }
      input { flex: 1; padding: 12px; border-radius: 6px; border: 1px solid #334155; background: #0b1220; color: #e2e8f0; }
      button { padding: 12px 16px; border: 0; border-radius: 6px; background: #22c55e; color: #052e16; font-weight: 700; cursor: pointer; }
      #meta { margin-top: 10px; color: #94a3b8; font-size: 13px; }
    </style>
  </head>
  <body>
    <div class='container'>
      <h1>ChatGPT Memory Wrapper (localhost deployable)</h1>
      <div id='chat'></div>
      <div class='row'>
        <input id='prompt' placeholder='Type your message...' />
        <button id='send'>Send</button>
      </div>
      <div id='meta'>Session not started</div>
    </div>
    <script>
      const chat = document.getElementById('chat');
      const promptInput = document.getElementById('prompt');
      const sendBtn = document.getElementById('send');
      const meta = document.getElementById('meta');
      let sessionId = null;
      function addMessage(role, text) {
        const div = document.createElement('div');
        div.className = `msg ${role}`;
        div.textContent = `${role.toUpperCase()}: ${text}`;
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
      }
      async function sendMessage() {
        const message = promptInput.value.trim();
        if (!message) return;
        promptInput.value = '';
        addMessage('user', message);
        const res = await fetch('/api/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, session_id: sessionId })
        });
        const data = await res.json();
        if (!res.ok) { addMessage('assistant', `Error: ${data.error || 'Unknown error'}`); return; }
        sessionId = data.session_id;
        addMessage('assistant', data.reply);
        meta.textContent = `session_id=${sessionId} | full=${data.full_transcript_count} | live=${data.live_message_count} | summarized=${data.has_summary}`;
      }
      sendBtn.addEventListener('click', sendMessage);
      promptInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });
    </script>
  </body>
</html>
"""


def chat_logic(payload: dict, client=CLIENT):
    if client is None:
        return {"error": "Server is missing OPENAI_API_KEY."}, 500

    message = (payload.get("message") or "").strip()
    if not message:
        return {"error": "message is required"}, 400

    session_id = payload.get("session_id") or str(uuid4())
    wrapper = SESSIONS.get(session_id)
    if wrapper is None:
        wrapper = ContextMemoryChatWrapper(
            client=client,
            model=MODEL,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            response_tokens=RESPONSE_TOKENS,
            system_prompt="You are a helpful assistant.",
        )
        SESSIONS[session_id] = wrapper

    answer = wrapper.ask(message)
    return {
        "session_id": session_id,
        "reply": answer,
        "full_transcript_count": len(wrapper.state.full_transcript),
        "live_message_count": len(wrapper.state.live_messages),
        "has_summary": bool(wrapper.state.rolling_summary),
    }, 200


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}

        response, status = chat_logic(payload)
        body = json.dumps(response).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
