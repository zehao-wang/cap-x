"""Pure helpers for the interactive trial runner.

Self-contained, side-effect-free pieces split out of ``async_trial_runner`` to
keep that module focused on the trial control flow: the model-query args struct,
frame encoding, message collapsing, and the VDM (visual-differencing model)
prompt builders. Anything here is trivially unit-testable in isolation.
"""

from __future__ import annotations

import base64 as _b64
import io as _io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from capx.harnesses.prompt import clean_vdm_task_description, prepare_vdm_console_text


@dataclass
class LaunchArgsCompat:
    """Compatible args structure for ``query_model``."""

    model: str
    server_url: str
    api_key: str | None
    max_tokens: int
    temperature: float
    reasoning_effort: str
    debug: bool

    # Image differencing
    visual_differencing_model: str | None
    visual_differencing_model_server_url: str | None
    visual_differencing_model_api_key: str | None


def encode_frame_png(frame) -> tuple[Any, str]:
    """Encode an RGB frame to a (PIL image, ``data:image/png;base64,...`` URL)."""
    pil_img = Image.fromarray(frame)
    buf = _io.BytesIO()
    pil_img.save(buf, format="png")
    data_url = f"data:image/png;base64,{_b64.b64encode(buf.getvalue()).decode('utf-8')}"
    return pil_img, data_url


def merge_consecutive_messages(messages: list[dict]) -> list[dict]:
    """Merge adjacent messages with the same role into one.

    Some chat-template servers require strictly alternating roles. Appending
    human feedback / a reset note as its own user turn can produce consecutive
    ``user`` messages; this collapses them just before sending, concatenating
    their content parts. Content is normalized to the list-of-parts form so text
    and image parts merge cleanly. Input is not mutated.
    """
    def _as_parts(content: Any) -> list[dict]:
        if isinstance(content, list):
            return list(content)
        return [{"type": "text", "text": content if content is not None else ""}]

    merged: list[dict] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg.get("role"):
            merged[-1]["content"] = _as_parts(merged[-1]["content"]) + _as_parts(msg.get("content"))
        else:
            merged.append({**msg, "content": msg.get("content")})
    return merged


def build_initial_state_prompt(task_description: str, image_b64: str) -> list[dict]:
    """VDM prompt asking for a task-relevant description of the initial state."""
    task_description = clean_vdm_task_description(task_description)
    instruction = (
        "Describe the initial state of the environment with the goal of the task "
        "in mind. Do *NOT* write any code. Provide ONLY task-relevant information."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of "
                "the environment with the goal of the task in mind. Do *NOT* write "
                "any code. Provide ONLY task-relevant information."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": task_description},
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        },
    ]


def build_state_diff_prompt(
    task_description: str,
    prev_b64: str,
    cur_b64: str,
    console_output: str | None = None,
) -> list[dict]:
    """VDM prompt asking for the before/after difference and task completion.

    ``console_output`` is the stdout the executed code printed (e.g. measured
    positions, ``"Lifted the cube"``). A single fixed camera view often cannot
    show small height/lift changes, so this grounds the VDM with what the code
    reported. It is framed as the agent's own report (possibly imperfect), not
    ground truth.
    """
    task_description = clean_vdm_task_description(task_description)
    instruction = (
        "Describe the difference between the current state of the environment and "
        "the previous state of the environment with the goal of the task in mind "
        "and whether the task has been completed. Do *NOT* write any code.."
    )
    content: list[dict] = [
        {"type": "text", "text": task_description},
        {"type": "text", "text": instruction},
    ]
    console_output = prepare_vdm_console_text(console_output)
    if console_output and console_output.strip():
        content.append({
            "type": "text",
            "text": (
                "For grounding, the code that produced the current state printed "
                "the following console output (the agent's own reported actions / "
                "measurements — corroborate it against the images, do not take it as "
                f"ground truth):\n```\n{console_output.strip()}\n```"
            ),
        })
    content += [
        {"type": "text", "text": "Previous state:"},
        {"type": "image_url", "image_url": {"url": prev_b64}},
        {"type": "text", "text": "Current state:"},
        {"type": "image_url", "image_url": {"url": cur_b64}},
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between "
                "the current state of the environment and the previous state of the "
                "environment with the goal of the task in mind and whether the task "
                "has been completed. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": content},
    ]


def build_feedback_distill_prompt(
    task_description: str,
    prior_guidance: str,
    human_feedback: str,
    failure_stdout: str | None,
    failure_stderr: str | None,
    *,
    max_bullets: int = 10,
) -> list[dict]:
    """Dedicated text-only call that distils cumulative operator guidance.

    Run once per human-feedback round. It folds the new (free-form) human
    feedback — and the key failure evidence from the attempt just run — into a
    SHORT, deduplicated, persistent bullet list that is carried into every
    subsequent regeneration. This keeps the regeneration context bounded instead
    of accumulating raw feedback + full code each round: superseded items are
    replaced, not appended. The distiller never writes code.
    """
    task_description = clean_vdm_task_description(task_description)
    prior = (prior_guidance or "").strip() or "(none yet)"
    fb = (human_feedback or "").strip()
    parts = [
        f"Task:\n{task_description}",
        f"\nCurrent operator guidance (carried from earlier feedback rounds):\n{prior}",
    ]
    so = (failure_stdout or "").strip()
    se = (failure_stderr or "").strip()
    if so or se:
        parts.append(
            "\nKey console evidence from the attempt the human just judged as "
            "unsuccessful. The agent's own prints may falsely claim success "
            "('Task completed', 'Pot lifted', ...) — treat such claims with "
            f"suspicion:\nstdout:\n{so or '(empty)'}\nstderr:\n{se or '(empty)'}"
        )
    parts.append(f"\nNew human feedback (authoritative):\n{fb}")
    parts.append(
        "\nProduce the UPDATED operator guidance: fold the new feedback into the "
        "existing list, merge duplicates, and DROP or REPLACE any item the new "
        "feedback supersedes. Keep every item concrete, imperative, and durable "
        "(a rule that should hold for future attempts, not a one-off remark). "
        f"Output AT MOST {max_bullets} short bullet lines and NOTHING else — no "
        "preamble, no code, no explanation."
    )
    return [
        {
            "role": "system",
            "content": (
                "You maintain a concise, durable checklist of operator "
                "instructions for a robot-control coding agent. You translate a "
                "human operator's free-form feedback into a short, deduplicated "
                "bullet list that the agent must follow on its next attempt. You "
                "keep the list SHORT: superseded or redundant items are removed, "
                "not appended. You never write code."
            ),
        },
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_feedback_regeneration_block(
    operator_guidance: str,
    human_feedback: str,
    failure_stdout: str | None,
    failure_stderr: str | None,
) -> str:
    """Assemble the single labelled user turn appended on a feedback retry.

    Carries three clearly-attributed sections into the regeneration: the
    cumulative (distilled) operator guidance, the previous attempt's key failure
    as console evidence, and the verbatim latest human feedback. The previous
    attempt's full code is deliberately NOT carried.
    """
    guidance = (operator_guidance or "").strip() or "(none)"
    fb = (human_feedback or "").strip() or "(none)"
    so = (failure_stdout or "").strip() or "(empty)"
    se = (failure_stderr or "").strip() or "(empty)"
    return (
        "The simulator has been reset to the start of this episode; none of your "
        "earlier steps persist. Write a COMPLETE solution from the initial state, "
        "incorporating the operator guidance and feedback below.\n\n"
        "=== OPERATOR GUIDANCE (cumulative, authoritative — you MUST follow every "
        "item) ===\n"
        f"{guidance}\n\n"
        "=== WHY THE PREVIOUS ATTEMPT DID NOT SUCCEED ===\n"
        "A human operator judged the previous attempt as unsuccessful, regardless "
        "of any 'success'/'completed' text the code printed. Console evidence "
        "from that attempt:\n"
        f"stdout:\n{so}\n\nstderr:\n{se}\n\n"
        "=== LATEST HUMAN FEEDBACK (verbatim, authoritative) ===\n"
        f"{fb}"
    )
