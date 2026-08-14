"""
Example: using the PGL OpenAI Proxy with LangChain's ChatOpenAI.

The `api_key` is a short-lived JWT obtained from `pgl_auth_server`, not the
student's raw registration_number/senha and not a real OpenAI key.
`base_url` points at the proxy instead of OpenAI.

Usage
-----
    pip install langchain-openai pgl-auth
    python -m examples.chatopenai_example
"""

from langchain_openai import ChatOpenAI
from pgl_auth import PGLAuthClient

# Exchanges registration_number/senha for a JWT via pgl_auth_server; never
# sent to, or stored by, this proxy.
token = PGLAuthClient().login(registration_number="1121387", senha="597582")

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key=token,
    model="gpt-4o-mini",
)

response = llm.invoke("What is the capital of Brazil?")
print(response.content)

# Streaming works the same way as talking to OpenAI directly.
for chunk in llm.stream("Count from 1 to 5."):
    print(chunk.content, end="", flush=True)
print()
