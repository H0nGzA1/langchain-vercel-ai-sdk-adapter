# LangChain AI SDK Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Python package that adapts LangChain streaming output to Vercel AI SDK SSE format, enabling seamless integration between Python FastAPI backends and AI SDK frontend clients.

**Architecture:** The adapter consists of three core functions: `to_base_messages()` converts AI SDK UIMessage format to LangChain messages; `to_ui_message_stream()` converts LangChain's `astream_events` output to AI SDK's SSE `data:` format; `create_ui_message_stream_response()` wraps the stream into a Starlette/FastAPI StreamingResponse. The package is framework-agnostic and has optional FastAPI integration.

**Tech Stack:** Python ≥3.10, langchain-core ≥0.3.0, starlette (optional, for StreamingResponse), pydantic ≥2.0

---

## 0. Project Setup

**Files:**
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\pyproject.toml`
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\README.md`
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\LICENSE`
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\.gitignore`

- [ ] **Step 1: Create project directory structure**

```bash
mkdir -p D:/open-source-project/langchain-ai-sdk-adapter
mkdir -p D:/open-source-project/langchain-ai-sdk-adapter/src/langchain_ai_sdk_adapter
mkdir -p D:/open-source-project/langchain-ai-sdk-adapter/tests
mkdir -p D:/open-source-project/langchain-ai-sdk-adapter/docs/superpowers/plans
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "langchain-ai-sdk-adapter"
version = "0.1.0"
description = "Adapter to integrate LangChain with Vercel AI SDK SSE streaming"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [
    { name = "Lewis Yan", email = "lewis@avc.com" }
]
keywords = ["langchain", "vercel-ai-sdk", "sse", "streaming", "adapter"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "langchain-core>=0.3.0",
]

[project.optional-dependencies]
fastapi = ["fastapi>=0.100.0", "starlette>=0.27.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
    "ruff>=0.4.0",
    "mypy>=1.9.0",
]

[project.urls]
Homepage = "https://github.com/your-org/langchain-ai-sdk-adapter"
Repository = "https://github.com/your-org/langchain-ai-sdk-adapter"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/langchain_ai_sdk_adapter"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
```

- [ ] **Step 3: Create README.md**

```markdown
# LangChain AI SDK Adapter

Adapter to integrate Python LangChain backends with Vercel AI SDK frontend clients.

## Features

- `to_base_messages()` - Convert AI SDK UIMessage format to LangChain messages
- `to_ui_message_stream()` - Convert LangChain stream to AI SDK SSE format
- `create_ui_message_stream_response()` - Create a Starlette/FastAPI StreamingResponse

## Installation

```bash
pip install langchain-ai-sdk-adapter
```

With FastAPI support:

```bash
pip install langchain-ai-sdk-adapter[fastapi]
```

## Usage

### Basic Integration with FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_ai_sdk_adapter import to_base_messages, to_ui_message_stream, create_ui_message_stream_response

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    langchain_messages = await to_base_messages(await request.json())
    stream = await model.astream(langchain_messages)
    return create_ui_message_stream_response(to_ui_message_stream(stream))
```

### Integration with LangGraph

```python
from langgraph.graph import StateGraph
from langchain_ai_sdk_adapter import to_base_messages, to_ui_message_stream, create_ui_message_stream_response

@app.post("/chat")
async def chat(request: Request):
    langchain_messages = await to_base_messages(await request.json())
    graph = create_graph()
    stream = graph.astream_events(
        {"messages": langchain_messages},
        version="v2"
    )
    return create_ui_message_stream_response(to_ui_message_stream(stream))
```

## SSE Format

The adapter produces SSE in AI SDK `useChat` format:

```
data: {"id":"msg_abc123","role":"assistant","content":"Hello"}\n\n
data: {"id":"msg_abc123","role":"assistant","content":" world"}\n\n
...
data: [DONE]\n\n
```

## License

MIT
```

- [ ] **Step 4: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp

# Env
.env
.env.local

# mypy
.mypy_cache/
```

- [ ] **Step 5: Create LICENSE (MIT)**

```text
MIT License

Copyright (c) 2026 Lewis Yan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 6: Initialize git and commit**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
git init
git add .
git commit -m "feat: initial project scaffold"
```

---

## 1. Message Conversion (to_base_messages)

Convert AI SDK UIMessage format to LangChain HumanMessage/AIMessage/SystemMessage.

**Files:**
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\src\langchain_ai_sdk_adapter\messages.py`
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\tests\test_messages.py`

**AI SDK UIMessage format:**
```json
{
  "id": "msg_abc123",
  "role": "user|assistant|system",
  "content": "Hello world"
}
```

**LangChain message conversion:**
| AI SDK role | LangChain class |
|-------------|----------------|
| user | HumanMessage |
| assistant | AIMessage |
| system | SystemMessage |

- [ ] **Step 1: Write failing test**

```python
# tests/test_messages.py
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ai_sdk_adapter.messages import to_base_messages

def test_to_base_messages_user():
    ui_messages = [{"id": "1", "role": "user", "content": "Hello"}]
    result = to_base_messages(ui_messages)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"

def test_to_base_messages_assistant():
    ui_messages = [{"id": "1", "role": "assistant", "content": "Hi there"}]
    result = to_base_messages(ui_messages)
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "Hi there"

def test_to_base_messages_system():
    ui_messages = [{"id": "1", "role": "system", "content": "You are helpful"}]
    result = to_base_messages(ui_messages)
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are helpful"

def test_to_base_messages_multiple():
    ui_messages = [
        {"id": "1", "role": "system", "content": "You are a helpful assistant"},
        {"id": "2", "role": "user", "content": "Hi"},
        {"id": "3", "role": "assistant", "content": "Hello!"},
    ]
    result = to_base_messages(ui_messages)
    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest tests/test_messages.py -v
```
Expected: FAIL - "ModuleNotFoundError: No module named 'langchain_ai_sdk_adapter'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/langchain_ai_sdk_adapter/messages.py
"""Convert AI SDK UIMessage format to LangChain messages."""
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


async def to_base_messages(ui_messages: list[dict[str, Any]]) -> list:
    """Convert AI SDK UIMessage list to LangChain messages.

    Args:
        ui_messages: List of AI SDK UIMessage dicts with keys:
            - id: str
            - role: "user" | "assistant" | "system"
            - content: str

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage, SystemMessage)
    """
    result = []
    for msg in ui_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            # Default to human for unknown roles
            result.append(HumanMessage(content=content))

    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest tests/test_messages.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/langchain_ai_sdk_adapter/messages.py tests/test_messages.py
git commit -m "feat: add to_base_messages() for UIMessage to LangChain conversion"
```

---

## 2. SSE Stream Conversion (to_ui_message_stream)

Convert LangChain `astream_events` output to AI SDK SSE format.

**Files:**
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\src\langchain_ai_sdk_adapter\converter.py`
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\tests\test_converter.py`

**AI SDK SSE format:**
```
data: {"id":"msg_xxx","role":"assistant","content":"Hello"}\n\n
data: {"id":"msg_xxx","role":"assistant","content":" world"}\n\n
...
data: [DONE]\n\n
```

- [ ] **Step 1: Write failing test**

```python
# tests/test_converter.py
import pytest
import uuid
from langchain_core.messages import AIMessageChunk
from langchain_ai_sdk_adapter.converter import to_ui_message_stream

def test_to_ui_message_stream_text_chunk():
    """Test that a single text chunk produces correct SSE line."""
    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}])
    ]

    # Consume the generator
    result = []
    async def consume():
        async for line in to_ui_message_stream(chunks):
            result.append(line)

    import asyncio
    asyncio.get_event_loop().run_until_complete(consume())

    assert len(result) == 1
    assert result[0].startswith("data: ")
    assert '"role":"assistant"' in result[0]
    assert '"content":"Hello"' in result[0]
    assert result[0].endswith("\n\n")

def test_to_ui_message_stream_multiple_chunks():
    """Test multiple text chunks accumulate correctly."""
    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}]),
        AIMessageChunk(content=[{"type": "text", "text": " world"}]),
    ]

    result = []
    async def consume():
        async for line in to_ui_message_stream(chunks):
            result.append(line)

    import asyncio
    asyncio.get_event_loop().run_until_complete(consume())

    # Each chunk emits its own SSE line
    assert len(result) == 2
    assert '"content":"Hello"' in result[0]
    assert '"content":" world"' in result[1]

def test_to_ui_message_stream_done():
    """Test that [DONE] is emitted at the end."""
    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Done"}])
    ]

    result = []
    async def consume():
        async for line in to_ui_message_stream(chunks):
            result.append(line)

    import asyncio
    asyncio.get_event_loop().run_until_complete(consume())

    assert result[-1] == "data: [DONE]\n\n"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest tests/test_converter.py -v
```
Expected: FAIL - module not found

- [ ] **Step 3: Write minimal implementation**

```python
# src/langchain_ai_sdk_adapter/converter.py
"""Convert LangChain streams to AI SDK SSE format."""
import json
import uuid
from typing import AsyncGenerator

from langchain_core.messages import AIMessageChunk


def _generate_message_id() -> str:
    """Generate a unique message ID matching AI SDK format."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def _chunk_to_content_text(chunk: AIMessageChunk) -> str:
    """Extract text content from an AIMessageChunk.

    Handles both list content blocks and string content.
    """
    content = getattr(chunk, "content", None)

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Extract text from all text-type blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    return ""


async def to_ui_message_stream(
    chunks,
) -> AsyncGenerator[str, None]:
    """Convert LangChain AIMessageChunk stream to AI SDK SSE format.

    Args:
        chunks: An async iterable of AIMessageChunk objects (from
            model.stream() or graph.astream_events() with on_chat_model_stream events).

    Yields:
        SSE-formatted strings in AI SDK useChat format:
        - data: {"id":"msg_xxx","role":"assistant","content":"..."}\n\n
        - data: [DONE]\n\n

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic
        from langchain_ai_sdk_adapter import to_ui_message_stream

        model = ChatAnthropic(model="claude-3-5-sonnet")
        stream = await model.astream([HumanMessage(content="Hi")])

        async for line in to_ui_message_stream(stream):
            print(line, end="")
        ```
    """
    msg_id = _generate_message_id()
    role = "assistant"

    async for chunk in chunks:
        if not isinstance(chunk, AIMessageChunk):
            continue

        text = _chunk_to_content_text(chunk)
        if not text:
            continue

        data = {
            "id": msg_id,
            "role": role,
            "content": text,
        }
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest tests/test_converter.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/langchain_ai_sdk_adapter/converter.py tests/test_converter.py
git commit -m "feat: add to_ui_message_stream() for LangChain to AI SDK SSE conversion"
```

---

## 3. StreamingResponse Creation (create_ui_message_stream_response)

Create a Starlette/FastAPI StreamingResponse from the converted stream.

**Files:**
- Modify: `D:\open-source-project\langchain-ai-sdk-adapter\src\langchain_ai_sdk_adapter\converter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_converter.py - add this test

def test_create_ui_message_stream_response_returns_streaming_response():
    """Test that create_ui_message_stream_response returns a StreamingResponse."""
    from fastapi.responses import StreamingResponse
    from langchain_ai_sdk_adapter.converter import create_ui_message_stream_response

    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}])
    ]

    response = create_ui_message_stream_response(
        to_ui_message_stream(chunks)
    )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest tests/test_converter.py::test_create_ui_message_stream_response_returns_streaming_response -v
```
Expected: FAIL - StreamingResponse not available

- [ ] **Step 3: Write implementation with optional FastAPI support**

```python
# src/langchain_ai_sdk_adapter/converter.py
# Add to the end of the file:

def create_ui_message_stream_response(stream) -> "StreamingResponse":
    """Create a Starlette StreamingResponse from a to_ui_message_stream generator.

    Requires starlette or fastapi to be installed.

    Args:
        stream: Async generator from to_ui_message_stream()

    Returns:
        StreamingResponse with text/event-stream media type

    Raises:
        ImportError: If starlette is not installed
    """
    try:
        from starlette.responses import StreamingResponse
    except ImportError:
        raise ImportError(
            "starlette is required for create_ui_message_stream_response. "
            "Install with: pip install langchain-ai-sdk-adapter[fastapi]"
        )

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest tests/test_converter.py::test_create_ui_message_stream_response_returns_streaming_response -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/langchain_ai_sdk_adapter/converter.py
git commit -m "feat: add create_ui_message_stream_response() helper"
```

---

## 4. Package Init and Exports

Expose public API in `__init__.py`.

**Files:**
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\src\langchain_ai_sdk_adapter\__init__.py`

- [ ] **Step 1: Create __init__.py**

```python
# src/langchain_ai_sdk_adapter/__init__.py
"""LangChain AI SDK Adapter.

Convert LangChain streaming output to Vercel AI SDK SSE format.
"""
from langchain_ai_sdk_adapter.messages import to_base_messages
from langchain_ai_sdk_adapter.converter import (
    to_ui_message_stream,
    create_ui_message_stream_response,
)

__all__ = [
    "to_base_messages",
    "to_ui_message_stream",
    "create_ui_message_stream_response",
]
```

- [ ] **Step 2: Verify exports**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
python -c "from langchain_ai_sdk_adapter import to_base_messages, to_ui_message_stream, create_ui_message_stream_response; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/langchain_ai_sdk_adapter/__init__.py
git commit -m "feat: expose public API in __init__.py"
```

---

## 5. Integration Test with LangChain

End-to-end test using actual LangChain models.

**Files:**
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\tests\test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration tests using actual LangChain components."""
import pytest
from langchain_core.messages import HumanMessage
from langchain_ai_sdk_adapter import to_base_messages, to_ui_message_stream

@pytest.mark.asyncio
async def test_to_ui_message_stream_with_fake_model():
    """Test SSE conversion with a simple fake model stream."""

    class FakeModelStream:
        async def stream(self, messages):
            yield HumanMessage(content="Hello")
            yield HumanMessage(content=" world")

    # The key is to test the converter with chunks that mimic AIMessageChunk
    from langchain_core.messages import AIMessageChunk

    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}]),
        AIMessageChunk(content=[{"type": "text", "text": " world"}]),
    ]

    result = []
    async for line in to_ui_message_stream(chunks):
        result.append(line)

    assert len(result) == 3  # 2 content + 1 [DONE]
    assert "Hello" in result[0]
    assert "world" in result[1]
    assert "[DONE]" in result[2]
```

- [ ] **Step 2: Run integration test**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest tests/test_integration.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests"
```

---

## 6. Code Quality

**Files:**
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\pyproject.toml` (update)

- [ ] **Step 1: Install dev dependencies and run linting**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
uv pip install -e ".[dev]"
ruff check src/
ruff format src/
```

- [ ] **Step 2: Run mypy**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
mypy src/
```

- [ ] **Step 3: Run full test suite**

```bash
cd D:/open-source-project/langchain-ai-sdk-adapter
pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "chore: add linting and type checking"
```

---

## 7. GitHub Actions CI

**Files:**
- Create: `D:\open-source-project\langchain-ai-sdk-adapter\.github\workflows\test.yml`

- [ ] **Step 1: Create CI workflow**

```yaml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest -v

      - name: Lint
        run: ruff check src/

      - name: Type check
        run: mypy src/
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add GitHub Actions workflow"
```

---

## 8. Publish to PyPI

**Files:**
- Modify: `D:\open-source-project\langchain-ai-sdk-adapter\pyproject.toml` (add publishing config)

- [ ] **Step 1: Add publishing config to pyproject.toml**

```toml
[tool.hatch.build.targets.sdist]
include = ["src"]

[tool.hatch.build.targets.wheel]
packages = ["src/langchain_ai_sdk_adapter"]

[[tool.hatch.metadata.urls]]
Repository = "https://github.com/your-org/langchain-ai-sdk-adapter"
```

- [ ] **Step 2: Tag and release**

```bash
git tag v0.1.0
git push origin v0.1.0
# Create release on GitHub, then:
pip install build
python -m build
twine upload dist/*
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: prepare for PyPI release"
git push
```

---

## Self-Review Checklist

- [ ] All 8 sections completed
- [ ] Each step has code blocks with actual content
- [ ] No "TBD", "TODO", or placeholder text
- [ ] Test commands verified
- [ ] All files use correct absolute paths
- [ ] Package structure follows standard Python layout
