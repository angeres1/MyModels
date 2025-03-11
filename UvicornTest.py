import ollama

try:
    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[{"role": "user", "content": "Test message"}]
    )
    print(response["message"]["content"])
except Exception as e:
    print(f"Direct call error: {e}")