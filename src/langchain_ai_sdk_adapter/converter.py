"""Convert LangChain streams to AI SDK SSE format."""
import json
from typing import AsyncGenerator, TYPE_CHECKING

from langchain_core.messages import AIMessageChunk

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse


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
    """Convert LangChain AIMessageChunk stream to AI SDK 5.0 SSE format.

    Yields SSE lines in AI SDK 5.0 protocol:
    - data: {"type":"start","messageId":"msg_<uuid>"}
    - data: {"type":"text-start","id":"text_<uuid>"}
    - data: {"type":"text-delta","id":"text_<uuid>","delta":"..."}  (per chunk)
    - data: {"type":"text-end","id":"text_<uuid>"}
    - data: {"type":"finish","finishReason":"stop","usage":{"inputTokens":0,"outputTokens":0}}

    No [DONE] sentinel — stream terminates after finish.
    Compatible with Vercel AI SDK 5.0 useChat hook.
    """
    import uuid

    message_id = f"msg_{uuid.uuid4().hex[:30]}"
    text_id = f"text_{uuid.uuid4().hex[:30]}"

    yield f"data: {json.dumps({'type': 'start', 'messageId': message_id})}\n\n"
    yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"

    async for chunk in chunks:
        if not isinstance(chunk, AIMessageChunk):
            continue

        text = _chunk_to_content_text(chunk)
        if not text:
            continue

        yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': text})}\n\n"

    yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"
    finish_data = {'type': 'finish', 'finishReason': 'stop', 'usage': {'inputTokens': 0, 'outputTokens': 0}}
    yield f"data: {json.dumps(finish_data)}\n\n"


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
            "x-vercel-ai-ui-message-stream": "v1",
        }
    )
