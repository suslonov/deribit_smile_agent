"""LLM proposer: send artifact bundle to an LLM and receive updated calculator.

Supports:
- Anthropic Claude (via anthropic SDK)
- OpenAI (via openai SDK)
- Local file injection (for testing without an API key)

The proposer extracts:
1. A revised calculator.py source block.
2. An updated calculator config dict (optional).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)
_CONFIG_BLOCK_RE = re.compile(
    r"```(?:yaml|json)?\s*\n(.*?)```",
    re.DOTALL,
)


class LLMProposer:
    """Send artifact → receive (new_source, new_config_overrides)."""

    def __init__(self, config: dict) -> None:
        llm_cfg = config.get("llm", {})
        self.provider: str = llm_cfg.get("provider", "anthropic")
        self.model: str = llm_cfg.get("model", "claude-opus-4-5")
        self.max_tokens: int = llm_cfg.get("max_tokens", 8192)
        self.temperature: float = llm_cfg.get("temperature", 0.2)
        self.api_key_env: str = llm_cfg.get("api_key_env", "ANTHROPIC_API_KEY")

    def propose(
        self,
        prompt: str,
        dry_run: bool = False,
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Send prompt to LLM. Returns (new_calculator_source, config_overrides).

        In dry_run mode returns (None, None) without hitting any API.
        """
        if dry_run:
            logger.info("Dry-run mode: skipping LLM call")
            return None, None

        if self.provider == "anthropic":
            response_text = self._call_anthropic(prompt)
        elif self.provider == "openai":
            response_text = self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")

        return _extract_from_response(response_text)

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        import os
        api_key = os.environ.get(self.api_key_env)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        import os
        api_key = os.environ.get(self.api_key_env)
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def _extract_from_response(
    text: str,
) -> tuple[Optional[str], Optional[dict]]:
    """Extract the first Python code block and optional YAML/JSON config block."""
    import yaml  # type: ignore[import]

    python_blocks = _CODE_BLOCK_RE.findall(text)
    new_source = python_blocks[0].strip() if python_blocks else None

    config_blocks = _CONFIG_BLOCK_RE.findall(text)
    new_config: Optional[dict] = None
    for block in config_blocks:
        try:
            parsed = yaml.safe_load(block)
            if isinstance(parsed, dict):
                new_config = parsed
                break
        except Exception:
            pass

    return new_source, new_config
