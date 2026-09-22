#!/usr/bin/env python
import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Try a simple request with the most stable model
try:
    print("Testing with claude-opus-4-1-20250805...")
    msg = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print("✓ Success!")
    print(msg.content[0].text)
except Exception as e:
    print(f"✗ Failed: {e}")

# Try with claude-4
try:
    print("\nTesting with claude-4...")
    msg = client.messages.create(
        model="claude-4",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print("✓ Success!")
    print(msg.content[0].text)
except Exception as e:
    print(f"✗ Failed: {e}")

# Try list models if available
try:
    print("\nTrying to list available models...")
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"  - {model.id}")
except Exception as e:
    print(f"✗ Models listing not available: {e}")
