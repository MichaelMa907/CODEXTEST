import app as webapp


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
    def create(self, model, messages, max_tokens):
        return _FakeResponse("hello from fake model")


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_chat_logic_flow():
    webapp.SESSIONS.clear()
    fake = _FakeClient()

    payload, status = webapp.chat_logic({"message": "hi"}, client=fake)
    assert status == 200
    assert payload["reply"] == "hello from fake model"
    sid = payload["session_id"]

    payload2, status2 = webapp.chat_logic({"message": "remember", "session_id": sid}, client=fake)
    assert status2 == 200
    assert payload2["session_id"] == sid
    assert payload2["full_transcript_count"] > payload["full_transcript_count"]


def test_chat_logic_requires_message():
    fake = _FakeClient()
    payload, status = webapp.chat_logic({}, client=fake)
    assert status == 400
    assert "error" in payload
