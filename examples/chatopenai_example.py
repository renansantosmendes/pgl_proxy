"""
Example: using the PGL OpenAI Proxy with LangChain's ChatOpenAI.

The `api_key` is the student's matricula (registered and active in Neon),
not a real OpenAI key. `base_url` points at the proxy instead of OpenAI.

Usage
-----
    pip install langchain-openai
    python -m examples.chatopenai_example
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="1121387",  # the student's matricula
    model="gpt-4o-mini",
)

response = llm.invoke("What is the capital of Brazil?")
print(response.content)

# Streaming works the same way as talking to OpenAI directly.
for chunk in llm.stream("Count from 1 to 5."):
    print(chunk.content, end="", flush=True)
print()
