import pytest
from harken.templates import AUDIO_TOKEN, build_chat, build_prompt, format_options


def test_format_options_letters():
    assert format_options(["cat", "dog"]) == "(a) cat\n(b) dog"


def test_format_options_too_many():
    with pytest.raises(ValueError):
        format_options([str(i) for i in range(27)])


def test_build_prompt_open_ended():
    prompt = build_prompt("What is making the sound?")
    assert prompt.startswith(AUDIO_TOKEN)
    assert "What is making the sound?" in prompt


def test_build_prompt_multiple_choice():
    prompt = build_prompt("Which animal?", options=["dog", "cat"])
    assert "(a) dog" in prompt
    assert "letter" in prompt.lower()


def test_build_chat_includes_system():
    messages = build_chat("Q?", system="be helpful")
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[-1]["role"] == "user"
    assert AUDIO_TOKEN in messages[-1]["content"]


def test_build_chat_without_system():
    messages = build_chat("Q?")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
