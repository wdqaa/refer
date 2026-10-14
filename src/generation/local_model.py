"""Minimal single-checkpoint Transformers backend for reproducible Stage 01 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    seed: int = 42
    enable_thinking: bool = False


class TransformersChatModel:
    """Load one local model lazily and reuse it for all prompt-based roles."""

    def __init__(self, model_path: Path, *, device: str = "cuda:0") -> None:
        if not model_path.is_dir():
            raise ValueError(f"Model path does not exist: {model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for local inference") from exc

        self._torch = torch
        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device_map: str | dict[str, str]
        if device == "auto":
            # Let Accelerate shard the checkpoint across the GPUs exposed by
            # CUDA_VISIBLE_DEVICES.  This is useful when no single card has enough
            # free memory; it does not alter generation settings or model weights.
            device_map = "auto"
        else:
            device_map = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    def render_messages(
        self, messages: list[dict[str, str]], *, enable_thinking: bool = False
    ) -> str:
        """Render messages exactly as they will be supplied to the checkpoint."""

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        """Generate only completion tokens for a system/user chat conversation."""

        torch = self._torch
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        rendered = self.render_messages(messages, enable_thinking=settings.enable_thinking)
        inputs = self.tokenizer([rendered], return_tensors="pt").to(self.input_device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if settings.temperature > 0:
            kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **kwargs)
        generated = output_ids[0, inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        conversations: list[list[dict[str, str]]],
        settings: GenerationSettings,
    ) -> list[str]:
        """Greedily generate independent completions in one padded model batch."""

        if not conversations:
            return []
        torch = self._torch
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        rendered = [
            self.render_messages(messages, enable_thinking=settings.enable_thinking)
            for messages in conversations
        ]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(rendered, return_tensors="pt", padding=True).to(
                self.input_device
            )
        finally:
            self.tokenizer.padding_side = original_padding_side
        kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if settings.temperature > 0:
            kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **kwargs)
        prompt_width = inputs.input_ids.shape[1]
        return [
            self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
            for row in output_ids
        ]

    def count_input_tokens(
        self, messages: list[dict[str, str]], *, enable_thinking: bool = False
    ) -> int:
        """Count the exact rendered prompt tokens used by ``generate``."""

        rendered = self.render_messages(messages, enable_thinking=enable_thinking)
        return len(self.tokenizer.encode(rendered, add_special_tokens=False))


class TransformersCompletionModel(TransformersChatModel):
    """Backend for checkpoints documented for plain completion prompts."""

    def render_messages(
        self, messages: list[dict[str, str]], *, enable_thinking: bool = False
    ) -> str:
        if enable_thinking:
            raise ValueError("plain-completion checkpoints do not support thinking mode")
        sections: list[str] = []
        labels = {"system": "Instructions", "user": "Question input", "assistant": "Response"}
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in labels or not isinstance(content, str):
                raise ValueError("messages must contain string system/user/assistant content")
            sections.append(f"{labels[role]}:\n{content.strip()}")
        sections.append("Response:\n")
        return "\n\n".join(sections)
