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
