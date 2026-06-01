import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any, SupportsFloat, TypeVar, abstractmethod

from gymnasium import Env

logger = logging.getLogger(__name__)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnv(Env):
    """
    Base environment class for low level control environments.
    This is a generic environment class for low level mujoco / simulator control environments.
    It is a subclass of the Gymnasium Env class.
    """

    privileged: bool = False
    max_steps: int = 999999

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            seed: The seed to reset the environment with.
            options: The options to reset the environment with.
        Returns:
            tuple: A tuple containing the observation and info.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment with the given action.
        Here we assume the action is the low level control actions (joint position, gripper position, etc.)
        Args:
            action: The action to take in the environment.
        Returns:
            tuple: A tuple containing the observation, reward, terminated, truncated, and info.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> ObsType:
        """
        Gets the observation of the environment.
        Returns:
            ObsType: The observation of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self) -> SupportsFloat:
        """
        Computes the reward of the environment.
        Returns:
            SupportsFloat: The reward of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def task_completed(self) -> bool:
        """
        Checks if the task is completed.
        Returns:
            bool: True if the task is completed, False otherwise.
        """
        raise NotImplementedError

    # ---- Episode state snapshot / restore -------------------------------
    # Used by the interactive web UI to send the simulator back to the exact
    # state at the start of the current episode on a human-requested reset
    # (§9.1), without re-sampling a new episode. Restores both the MuJoCo
    # physics state and the small Python-side bookkeeping reset() sets.

    # Python attributes worth restoring alongside the MuJoCo state. Snapshotted
    # only when present on the concrete env (LIBERO has the extra three).
    _SNAPSHOT_ATTRS: tuple[str, ...] = (
        "_step_count",
        "_sim_step_count",
        "_gripper_fraction",
        "_current_joints",
        "gripper_link_wxyz_xyz",
        "home_joint_position",
        "_current_obs",
        "_current_info",
    )

    def _get_mj_sim(self) -> Any | None:
        """Locate the underlying MuJoCo sim across simulator families."""
        robosuite_env = getattr(self, "robosuite_env", None)
        if robosuite_env is not None and getattr(robosuite_env, "sim", None) is not None:
            return robosuite_env.sim
        handle = getattr(self, "handle", None)
        if handle is not None:
            inner = getattr(handle, "env", None)
            if inner is not None and getattr(inner, "sim", None) is not None:
                return inner.sim
        return None

    @staticmethod
    def _capture_mj_state(sim: Any) -> tuple[str, Any] | None:
        """Capture the *full* MuJoCo integration state for an exact restore.

        ``mjSTATE_INTEGRATION`` covers everything that affects forward dynamics
        — time, qpos, qvel, act, warmstart, ctrl, applied forces, and mocap —
        i.e. the complete state as if the episode were freshly loaded. Falls
        back to the (qpos/qvel/time only) wrapper state if the raw handles or
        the mujoco API are unavailable.
        """
        try:
            import mujoco  # DeepMind bindings (robosuite MjSim wraps these)
            import numpy as np

            model = sim.model._model
            data = sim.data._data
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            size = mujoco.mj_stateSize(model, spec)
            buf = np.empty(size, dtype=np.float64)
            mujoco.mj_getState(model, data, buf, spec)
            return ("integration", buf)
        except Exception:
            try:
                return ("simstate", sim.get_state())
            except Exception:
                return None

    @staticmethod
    def _apply_mj_state(sim: Any, captured: tuple[str, Any]) -> None:
        """Apply a state captured by :meth:`_capture_mj_state`."""
        kind, payload = captured
        if kind == "integration":
            import mujoco

            model = sim.model._model
            data = sim.data._data
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            mujoco.mj_setState(model, data, payload, spec)
            mujoco.mj_forward(model, data)
        else:
            sim.set_state(payload)
            sim.forward()  # synchronize derived quantities (see MjSim.set_state)

    def snapshot_state(self) -> dict[str, Any] | None:
        """Capture the full episode-start state. Returns None if no sim found."""
        import copy as _copy

        sim = self._get_mj_sim()
        if sim is None:
            return None
        mj_state = self._capture_mj_state(sim)
        if mj_state is None:
            return None
        snapshot: dict[str, Any] = {"mj_state": mj_state}
        for name in self._SNAPSHOT_ATTRS:
            if hasattr(self, name):
                try:
                    snapshot[name] = _copy.deepcopy(getattr(self, name))
                except Exception:
                    snapshot[name] = getattr(self, name)
        kind, payload = mj_state
        try:
            import numpy as _np
            _qsum = float(_np.sum(sim.data.qpos))
        except Exception:
            _qsum = None
        logger.info(
            "snapshot_state: captured mj_state kind=%s qpos_sum=%s", kind, _qsum,
        )
        return snapshot

    def restore_state(self, snapshot: dict[str, Any] | None) -> None:
        """Restore a snapshot from :meth:`snapshot_state` (physics + bookkeeping)."""
        import copy as _copy

        if not snapshot:
            logger.warning("restore_state: empty snapshot -> NO reset performed")
            return
        sim = self._get_mj_sim()
        if sim is None:
            logger.warning("restore_state: no MuJoCo sim found -> NO reset performed")
            return
        try:
            import numpy as _np
            _before = float(_np.sum(sim.data.qpos))
        except Exception:
            _before = None
        self._apply_mj_state(sim, snapshot["mj_state"])
        try:
            import numpy as _np
            _after = float(_np.sum(sim.data.qpos))
        except Exception:
            _after = None
        logger.info(
            "restore_state: applied mj_state qpos_sum %s -> %s (changed=%s)",
            _before, _after, (_before is not None and _after is not None and abs(_before - _after) > 1e-6),
        )
        for name, value in snapshot.items():
            if name == "mj_state":
                continue
            try:
                setattr(self, name, _copy.deepcopy(value))
            except Exception:
                setattr(self, name, value)
        # An in-trial reset goes back to the episode start, but we keep the
        # prior attempt browsable: start a NEW viser segment instead of wiping
        # the timeline. Each segment is one attempt, selectable from the
        # Attempt dropdown and saved separately under outputs/ (§9.4). A full
        # new trial uses reset() -> frame_history.clear() to drop everything.
        frame_history = getattr(self, "frame_history", None)
        if frame_history is not None:
            try:
                frame_history.new_segment()
            except Exception:
                pass
        if getattr(self, "viser_debug", False) and hasattr(self, "_update_viser_server"):
            try:
                self._update_viser_server()
            except Exception:
                pass


# Use user's BaseEnv for low-level envs

_ENV_FACTORIES: dict[str, Callable[[], BaseEnv]] = {}


def register_env(name: str, factory: Callable[[], BaseEnv]) -> None:
    _ENV_FACTORIES[name] = factory


@lru_cache(maxsize=256)
def get_env(
    name: str,
    privileged: bool = False,
    enable_render: bool = False,
    viser_debug: bool = False,
    **kwargs: Any,
) -> BaseEnv:
    if name not in _ENV_FACTORIES:
        raise KeyError(f"Environment '{name}' not registered")
    return _ENV_FACTORIES[name](
        privileged=privileged,
        enable_render=enable_render,
        viser_debug=viser_debug,
        **kwargs,
    )


def list_envs() -> list[str]:
    return list(_ENV_FACTORIES.keys())


__all__ = [
    "BaseEnv",
    "register_env",
    "get_env",
    "list_envs",
]
