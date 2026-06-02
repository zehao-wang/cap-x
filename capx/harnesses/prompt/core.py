"""Small prompt harness for cleaning and budgeting model inputs.

The harness keeps task-specific prompt rules out of trial runners and LLM
clients. It is intentionally deterministic for now; token counting and LLM
summarization can be added behind this boundary later.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

_CODE_AGENT_TEMPLATE_LINES = (
    re.compile(r"^\s*You are controlling .* API described below\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*You may write .* executable Python code .* code fences\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*After this code executes, you will receive .* stdout .* stderr.*$", re.IGNORECASE),
    re.compile(r"^\s*The functions \(APIs\) below are already imported .*$", re.IGNORECASE),
)


@dataclass(frozen=True)
class PromptBudgetPolicy:
    """Character budgets for verbose prompt sections.

    These are character budgets rather than token budgets by design. The
    current cap-x prompt paths do not have a model-specific tokenizer available
    at construction time, so this policy provides a conservative deterministic
    guardrail while preserving a clean upgrade point for token counting.
    """

    multiturn_stdout_chars: int = 12_000
    multiturn_stderr_chars: int = 8_000
    vdm_console_chars: int = 4_000


@dataclass(frozen=True)
class ConsolePromptText:
    """Budgeted console output ready to insert into a prompt."""

    stdout: str
    stderr: str


class PromptHarness:
    """Prepare task-scoped prompt content before it reaches an LLM."""

    def __init__(self, policy: PromptBudgetPolicy | None = None) -> None:
        self.policy = policy or PromptBudgetPolicy()

    def clean_vdm_task_description(
        self,
        prompt_text: str | None,
        task_only_prompt: str | None = None,
    ) -> str:
        """Return only the task goal text that should be sent to the VDM."""
        if task_only_prompt and task_only_prompt.strip():
            result = task_only_prompt.strip()
            if (prompt_text or "").strip() != result:
                logger.info(
                    "[PromptHarness] VDM task description sourced from task_only_prompt "
                    "(input_chars=%d, output_chars=%d)",
                    len((prompt_text or "").strip()),
                    len(result),
                )
            return result

        text = (prompt_text or "").strip()
        if not text:
            return ""

        text_before_apis = re.split(r"(?im)^\s*APIs:\s*$", text, maxsplit=1)[0].strip()
        goal_match = re.search(r"(?im)^\s*Goal:\s*(.+?)\s*$", text_before_apis)
        if goal_match:
            result = f"Goal: {goal_match.group(1).strip()}"
            if result != text:
                logger.info(
                    "[PromptHarness] VDM task description cleaned with goal extraction "
                    "(input_chars=%d, output_chars=%d)",
                    len(text),
                    len(result),
                )
            return result

        clean_lines: list[str] = []
        for line in text_before_apis.splitlines():
            if any(pattern.match(line) for pattern in _CODE_AGENT_TEMPLATE_LINES):
                continue
            clean_lines.append(line.rstrip())

        result = "\n".join(clean_lines).strip()
        if result != text:
            logger.info(
                "[PromptHarness] VDM task description cleaned "
                "(input_chars=%d, output_chars=%d)",
                len(text),
                len(result),
            )
        return result

    def get_vdm_task_description_from_env(
        self,
        env: Any,
        full_prompt_text: str | None,
        task_only_prompt: str | None = None,
    ) -> str:
        """Get a clean VDM task description from explicit config, env, or prompt text."""
        cfg = getattr(env, "cfg", None)
        cfg_task_only = getattr(cfg, "task_only_prompt", None) if cfg is not None else None
        explicit_task_only = task_only_prompt or cfg_task_only
        if explicit_task_only and str(explicit_task_only).strip():
            result = str(explicit_task_only).strip()
            logger.info(
                "[PromptHarness] VDM task description selected from explicit/env task prompt "
                "(output_chars=%d)",
                len(result),
            )
            return result

        env_task_prompt = getattr(env, "_task_prompt", None) or getattr(env, "prompt", None)
        source = "env_task_prompt" if env_task_prompt else "full_prompt"
        result = self.clean_vdm_task_description(env_task_prompt or full_prompt_text)
        if result != (full_prompt_text or "").strip():
            logger.info(
                "[PromptHarness] VDM task description prepared from %s "
                "(output_chars=%d)",
                source,
                len(result),
            )
        return result

    def truncate_text(self, text: str | None, *, max_chars: int, label: str = "text") -> str:
        """Bound verbose text before inserting it into an LLM prompt."""
        value = (text or "").strip()
        if len(value) <= max_chars:
            return value

        original_len = len(value)
        omitted = len(value) - max_chars
        marker = (
            f"\n\n[... {label} truncated: showing the first and last parts; "
            f"omitted {omitted} characters ...]\n\n"
        )
        if max_chars <= len(marker) + 20:
            marker = f"\n\n[... {label} truncated ...]\n\n"
        if max_chars <= len(marker) + 2:
            return value[:max_chars]

        remaining = max_chars - len(marker)
        head_len = max(1, remaining // 3)
        tail_len = remaining - head_len
        result = value[:head_len].rstrip() + marker + value[-tail_len:].lstrip()
        logger.info(
            "[PromptHarness] Truncated %s for prompt budget "
            "(input_chars=%d, output_chars=%d, max_chars=%d)",
            label,
            original_len,
            len(result),
            max_chars,
        )
        return result

    def prepare_multiturn_console_text(
        self,
        stdout: str | None,
        stderr: str | None,
    ) -> ConsolePromptText:
        """Prepare stdout/stderr for the multi-turn decision prompt."""
        return ConsolePromptText(
            stdout=self.truncate_text(
                stdout,
                max_chars=self.policy.multiturn_stdout_chars,
                label="console stdout",
            ),
            stderr=self.truncate_text(
                stderr,
                max_chars=self.policy.multiturn_stderr_chars,
                label="console stderr",
            ),
        )

    def prepare_vdm_console_text(self, console_output: str | None) -> str:
        """Prepare console output for visual differencing grounding."""
        return self.truncate_text(
            console_output,
            max_chars=self.policy.vdm_console_chars,
            label="console output",
        )


_DEFAULT_HARNESS = PromptHarness()


def clean_vdm_task_description(
    prompt_text: str | None,
    task_only_prompt: str | None = None,
) -> str:
    return _DEFAULT_HARNESS.clean_vdm_task_description(prompt_text, task_only_prompt)


def get_vdm_task_description_from_env(
    env: Any,
    full_prompt_text: str | None,
    task_only_prompt: str | None = None,
) -> str:
    return _DEFAULT_HARNESS.get_vdm_task_description_from_env(env, full_prompt_text, task_only_prompt)


def truncate_text_for_prompt(text: str | None, *, max_chars: int, label: str = "text") -> str:
    return _DEFAULT_HARNESS.truncate_text(text, max_chars=max_chars, label=label)


def prepare_multiturn_console_text(stdout: str | None, stderr: str | None) -> ConsolePromptText:
    return _DEFAULT_HARNESS.prepare_multiturn_console_text(stdout, stderr)


def prepare_vdm_console_text(console_output: str | None) -> str:
    return _DEFAULT_HARNESS.prepare_vdm_console_text(console_output)
