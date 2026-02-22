"""ChatGPT wrapper with context-aware memory management.

This wrapper keeps a full local transcript of every turn while sending as much
verbatim history as possible to the model. When the request would exceed the
model context limit, it compresses the oldest turns into a rolling summary and
keeps newer turns verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


Message = Dict[str, str]


@dataclass
class MemoryState:
    """In-memory conversation store.

    - full_transcript: every message ever added (never dropped)
    - live_messages: currently sent verbatim to the model
    - rolling_summary: compressed memory of older turns that no longer fit
    """

    full_transcript: List[Message] = field(default_factory=list)
    live_messages: List[Message] = field(default_factory=list)
    rolling_summary: str = ""


class ContextMemoryChatWrapper:
    """A chat wrapper that preserves history until context fills up.

    Behavior:
    1. Stores *all* turns in `full_transcript`.
    2. Sends full history verbatim while it fits.
    3. Once near context limit, compresses oldest turns into `rolling_summary`
       and keeps newest turns verbatim.
    """

    def __init__(
        self,
        client,
        model: str,
        max_context_tokens: int,
        response_tokens: int = 512,
        system_prompt: Optional[str] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.response_tokens = response_tokens
        self.state = MemoryState()
        self._token_counter = token_counter or self._default_token_counter

        if system_prompt:
            msg = {"role": "system", "content": system_prompt}
            self.state.full_transcript.append(msg)
            self.state.live_messages.append(msg)

    def ask(self, user_text: str) -> str:
        """Append user text, call model, append assistant response, and return it."""
        user_msg = {"role": "user", "content": user_text}
        self._append_message(user_msg)
        outbound_messages = self._build_outbound_messages()

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=outbound_messages,
            max_tokens=self.response_tokens,
        )
        answer = completion.choices[0].message.content or ""

        assistant_msg = {"role": "assistant", "content": answer}
        self._append_message(assistant_msg)
        return answer

    def _append_message(self, message: Message) -> None:
        self.state.full_transcript.append(message)
        self.state.live_messages.append(message)

    def _build_outbound_messages(self) -> List[Message]:
        """Fit messages into context window by rolling up oldest turns."""
        available_prompt_tokens = self.max_context_tokens - self.response_tokens
        if available_prompt_tokens <= 0:
            raise ValueError("response_tokens must be smaller than max_context_tokens")

        # Keep compressing oldest non-system live messages until prompt fits.
        while self._messages_tokens(self._render_with_summary()) > available_prompt_tokens:
            idx = self._oldest_compressible_index()
            if idx is not None:
                removed = self.state.live_messages.pop(idx)
                self._merge_into_summary(removed)
                continue

            # Nothing left to compress except system messages; trim summary itself.
            if self.state.rolling_summary:
                lines = self.state.rolling_summary.splitlines()
                self.state.rolling_summary = "\n".join(lines[1:])
                continue

            raise ValueError("Prompt cannot fit within context budget.")

        return self._render_with_summary()

    def _oldest_compressible_index(self) -> Optional[int]:
        for i, msg in enumerate(self.state.live_messages):
            if msg["role"] != "system":
                return i
        return None

    def _render_with_summary(self) -> List[Message]:
        rendered = list(self.state.live_messages)
        if self.state.rolling_summary:
            rendered.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Conversation memory summary of earlier turns:\n"
                        f"{self.state.rolling_summary}"
                    ),
                },
            )
        return rendered

    def _merge_into_summary(self, message: Message) -> None:
        """Simple deterministic compression of old turns into a rolling digest."""
        line = f"- {message['role']}: {message['content'].strip()}"
        if not self.state.rolling_summary:
            self.state.rolling_summary = line
        else:
            self.state.rolling_summary += "\n" + line

        # Cap summary size itself to avoid unbounded growth.
        summary_cap = max(256, self.max_context_tokens // 3)
        while self._token_counter(self.state.rolling_summary) > summary_cap:
            # Drop the oldest summary line first.
            parts = self.state.rolling_summary.splitlines()
            self.state.rolling_summary = "\n".join(parts[1:])
            if not self.state.rolling_summary:
                break

    def _messages_tokens(self, messages: List[Message]) -> int:
        # Lightweight estimate. Good enough for budget control.
        total = 0
        for msg in messages:
            total += 4  # role/message framing overhead
            total += self._token_counter(msg.get("content", ""))
        total += 2
        return total

    @staticmethod
    def _default_token_counter(text: str) -> int:
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Fallback approximation when tiktoken isn't available.
            return max(1, int(len(text.split()) * 1.3))


if __name__ == "__main__":
    import os

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install dependencies: pip install openai tiktoken") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=api_key)
    bot = ContextMemoryChatWrapper(
        client=client,
        model="gpt-4o-mini",
        max_context_tokens=128_000,
        response_tokens=512,
        system_prompt="You are a helpful assistant.",
    )

    print("Memory chat started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = bot.ask(user_input)
        print(f"Assistant: {reply}\n")
