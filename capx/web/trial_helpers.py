"""Pure helpers for the interactive trial runner.

Stateless utilities split out of :mod:`capx.web.async_trial_runner` to keep that
module focused on the trial control flow. No env / session / websocket state here.
"""

from __future__ import annotations

import os
import re


def append_task_to_prompt(full_prompt: list, task_text: str) -> None:
    """Fold the operator's runtime task into the last prompt message in place.

    Handles both content shapes the harness uses: a list of content parts whose
    first part is the ``{"type": "text", "text": ...}`` block, or a plain string.
    """
    addition = f"\n\nThe task for this trial is:\n{task_text}"
    if not full_prompt:
        return
    msg = full_prompt[-1]
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and "text" in part:
                part["text"] = (part.get("text") or "") + addition
                return
        content.insert(0, {"type": "text", "text": addition.strip()})
    elif isinstance(content, str):
        msg["content"] = content + addition
    else:
        msg["content"] = addition.strip()


def annotate_code(code_blocks: list, code_block_metadata: list) -> str:
    """Join code blocks into one annotated program (the form saved as code.py).

    Shared by the trial's top-level code.py and the per-attempt
    trial_NN/attempt_NN/code.py so they have identical formatting.
    """
    parts = []
    for i, (block, meta) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        header = f"# Code block {i}"
        if isinstance(meta, dict) and meta.get("regenerated"):
            header += f" (regenerated at step {meta.get('regenerated_at_idx', '?')})"
        parts.append(f"{header}\n{block}")
    return "\n\n".join(parts)


def next_trial_number(output_dir: str | None) -> int:
    """Next free ``trial_NN`` index in ``output_dir`` (1 if none / no dir).

    Interactive runs one trial per ``run_trial_async`` call but reuses the same
    per-session ``output_dir``; numbering by the next free index keeps each new
    trial's trace / handoff / artifacts in its own ``trial_NN`` instead of
    overwriting ``trial_01``.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return 1
    nums = [
        int(m.group(1))
        for name in os.listdir(output_dir)
        if (m := re.match(r"trial_(\d+)", name))
    ]
    return (max(nums) + 1) if nums else 1
