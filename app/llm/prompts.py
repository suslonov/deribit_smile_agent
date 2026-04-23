"""Prompt template rendering for the LLM proposer/critic loop."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def _get_env(prompts_dir: str | Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_rewrite_prompt(
    artifact: dict,
    prompts_dir: str | Path = "prompts",
) -> str:
    env = _get_env(prompts_dir)
    tmpl = env.get_template("rewrite_calculator.md")
    return tmpl.render(
        artifact=artifact,
        metrics_json=_safe_json(artifact.get("summary_metrics", {})),
        by_strategy_json=_safe_json(artifact.get("by_strategy", {})),
        by_horizon_json=_safe_json(artifact.get("by_horizon", {})),
        nan_json=_safe_json(artifact.get("nan_breakdown", {})),
        config_json=_safe_json(artifact.get("config", {}).get("calculator", {})),
    )


def render_critic_prompt(
    candidate_source: str,
    artifact: dict,
    prompts_dir: str | Path = "prompts",
) -> str:
    env = _get_env(prompts_dir)
    tmpl = env.get_template("critic_review.md")
    return tmpl.render(
        candidate_source=candidate_source,
        artifact=artifact,
    )


def _safe_json(obj: Any, indent: int = 2) -> str:
    def default(v: Any) -> Any:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        raise TypeError(type(v))
    return json.dumps(obj, indent=indent, default=default)
