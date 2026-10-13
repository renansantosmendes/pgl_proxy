"""
Example: using the PGL proxy with LangChain's ChatAnthropic.

The `api_key` is a short-lived JWT obtained from `pgl_auth_server`, not the
student's raw registration_number/password and not a real Anthropic key.
`base_url` points at the proxy root (no `/v1` suffix: the Anthropic SDK
appends `/v1/messages` itself).

Usage
-----
    pip install langchain-anthropic pgl-auth
    python -m examples.chatanthropic_example
"""

import getpass

from langchain_anthropic import ChatAnthropic
from pgl_auth import PGLAuthClient

# Exchanges registration_number/password for a JWT via pgl_auth_server;
# never sent to, or stored by, this proxy.
registration_number = input("Registration number: ")
token = PGLAuthClient().login(registration_number, getpass.getpass("Password: "))

llm = ChatAnthropic(
    base_url="http://localhost:8000",
    api_key=token,
    model="claude-haiku-4-5",
    # The proxy caps max_tokens at 2048; anything higher is lowered.
    max_tokens=1024,
)

response = llm.invoke("What is the capital of Brazil?")
print(response.content)

# Streaming works the same way as talking to Anthropic directly.
for chunk in llm.stream("Count from 1 to 5."):
    print(chunk.text, end="", flush=True)
print()
