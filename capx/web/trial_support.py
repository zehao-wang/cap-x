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
