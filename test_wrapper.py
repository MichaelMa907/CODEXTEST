from chatgpt_wrapper import ContextMemoryChatWrapper


class _FakeRespMsg:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeRespMsg(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, model, messages, max_tokens):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        return _FakeResponse("ok")


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_rollup_and_full_transcript_preserved():
    client = _FakeClient()
    # Tiny token budget to force rollups quickly
    bot = ContextMemoryChatWrapper(
        client=client,
        model="fake",
        max_context_tokens=40,
        response_tokens=10,
        token_counter=lambda s: len(s.split()),
    )

    for i in range(6):
        bot.ask(f"message number {i}")

    assert len(bot.state.full_transcript) == 12
    assert bot.state.rolling_summary
    # Ensure call happened and stayed within budget approximation.
    last_call = client.chat.completions.calls[-1]
    prompt_tokens = bot._messages_tokens(last_call["messages"])
    assert prompt_tokens <= (bot.max_context_tokens - bot.response_tokens)
