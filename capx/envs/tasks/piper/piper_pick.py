from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling an Agilex PIPER 6-DOF robot arm with the API described below.
The arm is mounted next to a table, and a ZED 2i camera is rigidly fixed looking
at the workspace. All poses returned by the API are expressed in the robot base frame.
If enabled, the wrist-mounted RealSense camera is available at
obs["robot0_eye_in_hand"], while the fixed ZED scene camera is obs["robot0_robotview"].

Goal: pick up the object described in the task instruction and lift it.
For grasp selection, call sample_grasp_pose(object_name) to automatically choose
the highest-scoring grasp. Do not ask the user to choose a grasp. Before moving
the arm, preview the planned poses and wait for the user's execution approval.
After approval, goto_pose with z_approach, close the gripper, and retract upward.

You may write python comments for reasoning but ONLY write executable Python code
and do not wrap it in code fences. After the code executes you will be able to get
a new observation and write additional code if the task is not yet complete.

The functions (APIs) below are already imported into the execution environment.
If you want to use numpy, you need to import it explicitly.
"""


class PiperRealPickCodeEnv(CodeExecutionEnvBase):
    """High-level code-execution env for a simple Piper real pick-and-lift task."""

    prompt = PROMPT
    oracle_code = ""


__all__ = ["PiperRealPickCodeEnv"]
