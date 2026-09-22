"""Step 1: confirm your API key works."""
import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(msg.content[0].text)
print(f"tokens in/out: {msg.usage.input_tokens}/{msg.usage.output_tokens}")
