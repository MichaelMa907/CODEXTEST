# Context-Memory ChatGPT Wrapper

A Python wrapper around Chat Completions that:

1. Keeps full verbatim history while it fits in the model context window.
2. When it no longer fits, compresses the oldest turns into a rolling summary.
3. Still retains the full local transcript in memory so your app can audit/export all turns.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
from openai import OpenAI
from chatgpt_wrapper import ContextMemoryChatWrapper

client = OpenAI(api_key="YOUR_KEY")

bot = ContextMemoryChatWrapper(
    client=client,
    model="gpt-4o-mini",
    max_context_tokens=128_000,
    response_tokens=512,
    system_prompt="You are a helpful assistant.",
)

print(bot.ask("Remember that my favorite color is green."))
print(bot.ask("What's my favorite color?"))
```

## Notes

- `full_transcript` always contains all turns ever seen in this process.
- `live_messages` contains only the newest verbatim turns sent to the model.
- `rolling_summary` stores compressed old turns once context fills up.
