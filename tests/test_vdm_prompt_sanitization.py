import logging
from types import SimpleNamespace

from capx.harnesses.prompt import (
    clean_vdm_task_description,
    get_vdm_task_description_from_env,
    prepare_multiturn_console_text,
    truncate_text_for_prompt,
)
from capx.web.trial_support import (
    build_feedback_distill_prompt,
    build_feedback_regeneration_block,
    build_state_diff_prompt,
)


def test_prompt_harness_logs_when_it_cleans_or_truncates(caplog) -> None:
    caplog.set_level(logging.INFO, logger="capx.harnesses.prompt.core")

    clean_vdm_task_description("Goal: pick up the red cube.\n\nAPIs:\nmove_to_joints()")
    truncate_text_for_prompt("start-" + ("x" * 200) + "-end", max_chars=80, label="console stdout")

    messages = [record.getMessage() for record in caplog.records]
    assert any("VDM task description cleaned" in message for message in messages)
    assert any("Truncated console stdout" in message for message in messages)


def test_clean_vdm_task_description_extracts_goal_without_api_docs() -> None:
    full_prompt = """You are controlling a Franka Emika robot with API described below.
Goal: pick up the red cube and lift it.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
After this code executes, you will receive the current console stdout and stderr.
The functions (APIs) below are already imported to the environment.

APIs:

get_observation() -> dict[str, typing.Any]
  Doc:
    Get the observation of the environment.
"""

    assert clean_vdm_task_description(full_prompt) == "Goal: pick up the red cube and lift it."


def test_clean_vdm_task_description_prefers_task_only_prompt() -> None:
    assert (
        clean_vdm_task_description("Goal: noisy\n\nAPIs:\nfoo()", "pick up the red cube")
        == "pick up the red cube"
    )


def test_get_vdm_task_description_uses_env_task_only_prompt() -> None:
    env = SimpleNamespace(
        cfg=SimpleNamespace(task_only_prompt="place the red cube on the green cube"),
        _task_prompt="Goal: noisy",
    )

    assert get_vdm_task_description_from_env(env, "Goal: noisy\n\nAPIs:\nfoo()") == (
        "place the red cube on the green cube"
    )


def test_clean_vdm_task_description_strips_api_docs_without_goal() -> None:
    full_prompt = """You are controlling a Franka Emika robot with API described below.
Move the object into the target bin.
The functions (APIs) below are already imported to the environment.

APIs:

move_to_joints(joints: numpy.ndarray) -> None
"""

    assert clean_vdm_task_description(full_prompt) == "Move the object into the target bin."


def test_state_diff_builder_cleans_task_description_defensively() -> None:
    prompt = build_state_diff_prompt(
        "Goal: pick up the red cube.\n\nAPIs:\nmove_to_joints()",
        "data:image/png;base64,prev",
        "data:image/png;base64,cur",
    )

    task_text = prompt[1]["content"][0]["text"]
    assert task_text == "Goal: pick up the red cube."
    assert "APIs:" not in task_text


def test_state_diff_builder_budgets_console_output_defensively() -> None:
    prompt = build_state_diff_prompt(
        "Goal: pick up the red cube.",
        "data:image/png;base64,prev",
        "data:image/png;base64,cur",
        console_output="stdout-start-" + ("x" * 20_000) + "-stdout-end",
    )

    console_text = prompt[1]["content"][2]["text"]
    assert "console output truncated" in console_text
    assert "stdout-start-" in console_text
    assert "-stdout-end" in console_text


def test_truncate_text_for_prompt_preserves_head_and_tail() -> None:
    text = "start-" + ("x" * 200) + "-end"
    truncated = truncate_text_for_prompt(text, max_chars=80, label="console stdout")

    assert truncated.startswith("start-")
    assert truncated.endswith("-end")
    assert "console stdout truncated" in truncated
    assert len(truncated) <= 80


def test_prepare_multiturn_console_text_applies_stdout_stderr_policies() -> None:
    console_text = prepare_multiturn_console_text(
        "stdout-start-" + ("x" * 20_000) + "-stdout-end",
        "stderr-start-" + ("y" * 20_000) + "-stderr-end",
    )

    assert console_text.stdout.startswith("stdout-start-")
    assert console_text.stdout.endswith("-stdout-end")
    assert "console stdout truncated" in console_text.stdout
    assert len(console_text.stdout) <= 12_000

    assert console_text.stderr.startswith("stderr-start-")
    assert console_text.stderr.endswith("-stderr-end")
    assert "console stderr truncated" in console_text.stderr
    assert len(console_text.stderr) <= 8_000


def test_feedback_distill_prompt_strips_api_docs_and_carries_feedback() -> None:
    prompt = build_feedback_distill_prompt(
        "Goal: lift the pot.\n\nAPIs:\nmove_to_joints()",
        prior_guidance="- avoid top-down grasps",
        human_feedback="blue handle was not grasped; add a pre-grasp check",
        failure_stdout="Task completed successfully.",
        failure_stderr="",
        max_bullets=8,
    )

    assert prompt[0]["role"] == "system"
    user_text = prompt[1]["content"]
    # Task description is cleaned of API docs defensively.
    assert "move_to_joints()" not in user_text
    assert "APIs:" not in user_text
    # Prior guidance, the new feedback, and the failure evidence all appear.
    assert "avoid top-down grasps" in user_text
    assert "add a pre-grasp check" in user_text
    assert "Task completed successfully." in user_text
    assert "AT MOST 8" in user_text


def test_feedback_regeneration_block_labels_sections_and_omits_code() -> None:
    block = build_feedback_regeneration_block(
        operator_guidance="- bring both grippers above their handle first",
        human_feedback="the task is simple, just grasp",
        failure_stdout="Manual grasp for blue handle at [...]",
        failure_stderr="IK failed for arm 1",
    )

    # All three attributed sections are present and clearly labelled.
    assert "OPERATOR GUIDANCE (cumulative, authoritative" in block
    assert "WHY THE PREVIOUS ATTEMPT DID NOT SUCCEED" in block
    assert "LATEST HUMAN FEEDBACK (verbatim, authoritative)" in block
    # Carries the verbatim feedback + key failure evidence.
    assert "the task is simple, just grasp" in block
    assert "IK failed for arm 1" in block
    # Tells the model to disregard self-asserted success.
    assert "regardless of any" in block
