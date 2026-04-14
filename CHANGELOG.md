# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-04-14

### Changed

- `to_ui_message_stream()` now outputs AI SDK standard SSE format:
  - Changed from `{"id":"msg_xxx","role":"assistant","content":"..."}` to `{"type":"text","text":"..."}`
  - Added `x-vercel-ai-ui-message-stream: v1` header to `create_ui_message_stream_response()`

## [0.1.0] - 2026-04-14

### Added

- `to_base_messages()` - Convert AI SDK UIMessage format to LangChain messages
- `to_ui_message_stream()` - Convert LangChain AIMessageChunk stream to AI SDK SSE format
- `create_ui_message_stream_response()` - Create a Starlette/FastAPI StreamingResponse
