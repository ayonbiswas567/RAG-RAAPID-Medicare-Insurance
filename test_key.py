import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the meaning of name ayon"}],
    )
    print("SUCCESS:", response.choices[0].message.content)
    print(f"Model used: {response.model}")
    print(f"Tokens used: {response.usage.total_tokens}")
except Exception as e:
    print(f"FAILED: {e}")
