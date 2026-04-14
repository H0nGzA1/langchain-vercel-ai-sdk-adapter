"""Convert AI SDK UIMessage format to LangChain messages."""
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def to_base_messages(ui_messages: list[dict[str, Any]]) -> list:
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
