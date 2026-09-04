from relacats_v2.common import (
    build_reasoning_prompt,
    confidence_suffix,
    generation_defaults,
    remove_forced_think_from_chat_template,
)


class ForcedThinkTokenizer:
    chat_template = "{{'<｜Assistant｜><think>\\n'}}"

    def apply_chat_template(self, chat, **kwargs):
        del chat, kwargs
        return "<｜User｜>question<｜Assistant｜><think>\n"


def test_deepseek_prompt_does_not_force_think() -> None:
    prompt = build_reasoning_prompt(
        ForcedThinkTokenizer(), "What is 1+1?", "number"
    )
    assert prompt.endswith("<｜Assistant｜>")
    assert not prompt.endswith("<think>\n")


def test_deepseek_template_normalizer_is_narrow() -> None:
    source = "before {{'<｜Assistant｜><think>\\n'}} after <think>content</think>"
    normalized = remove_forced_think_from_chat_template(source)
    assert "{{'<｜Assistant｜>'}}" in normalized
    assert "<think>content</think>" in normalized


def test_deepseek_defaults_and_suffix() -> None:
    defaults = generation_defaults("DeepSeek-R1-Distill-Qwen-1.5B")
    assert defaults == {"max_new_tokens": 2048, "max_model_len": 8192}
    suffix = confidence_suffix("DeepSeek-R1-Distill-Qwen-1.5B")
    assert suffix.endswith("<｜Assistant｜>")
    assert "<think>" not in suffix
    assert "</think>" not in suffix
