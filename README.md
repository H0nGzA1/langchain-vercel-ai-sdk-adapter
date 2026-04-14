# LangChain AI SDK Adapter

Adapter to integrate Python LangChain backends with Vercel AI SDK frontend clients.

## Features

- `to_base_messages()` - Convert AI SDK UIMessage format to LangChain messages
- `to_ui_message_stream()` - Convert LangChain stream to AI SDK SSE format
- `create_ui_message_stream_response()` - Create a Starlette/FastAPI StreamingResponse

## Installation

```bash
pip install langchain-vercel-ai-sdk-adapter
```

With FastAPI support:

```bash
pip install langchain-vercel-ai-sdk-adapter[fastapi]
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

## SSE Format (AI SDK 5.0)

The adapter outputs SSE in AI SDK 5.0 protocol format, compatible with `useChat`:

```
data: {"type":"start","messageId":"msg_<uuid>"}
data: {"type":"text-start","id":"text_<uuid>"}
data: {"type":"text-delta","id":"text_<uuid>","delta":"Hello"}
data: {"type":"text-delta","id":"text_<uuid>","delta":" world"}
data: {"type":"text-end","id":"text_<uuid>"}
data: {"type":"finish","finishReason":"stop","usage":{"inputTokens":0,"outputTokens":0}}
```

Compatible with Vercel AI SDK 5.0 `useChat` hook.

## License

MIT
