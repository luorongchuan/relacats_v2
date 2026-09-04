"""Merge a RelaCaTS-v2 LoRA adapter while preserving its tokenizer protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from relacats_v2.common import (
    model_family,
    remove_forced_think_from_chat_template,
    remove_forced_think_from_prompt,
)


def _load_training_tokenizer(base_model: str, lora_path: str):
    """Prefer the tokenizer saved with the adapter, falling back to the base."""

    adapter = Path(lora_path)
    source = adapter if (adapter / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, use_fast=False
    )
    if not tokenizer.chat_template:
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model, local_files_only=True, use_fast=False
        )
        tokenizer.chat_template = base_tokenizer.chat_template
    if not tokenizer.chat_template:
        raise ValueError(
            "The training/base tokenizer has no chat_template; refusing to create "
            "a merged checkpoint with a different prompt protocol"
        )
    return tokenizer


def _normalise_deepseek_template(tokenizer, model_name: str) -> None:
    if model_family(model_name) != "deepseek":
        return
    tokenizer.chat_template = remove_forced_think_from_chat_template(
        tokenizer.chat_template
    )
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "protocol check"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if remove_forced_think_from_prompt(probe) != probe:
        raise ValueError("DeepSeek chat_template still forces <think> after assistant")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    output = Path(args.output_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty merged-model directory: {output}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = _load_training_tokenizer(args.base_model, args.lora_path)
    _normalise_deepseek_template(tokenizer, args.base_model)
    merged = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    reloaded = AutoTokenizer.from_pretrained(output, local_files_only=True, use_fast=False)
    if reloaded.chat_template != tokenizer.chat_template:
        raise RuntimeError("Merged tokenizer chat_template was not preserved exactly")
    print(f"Merged model saved to {output}")


if __name__ == "__main__":
    main()
