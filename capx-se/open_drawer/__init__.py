"""Hand-authored cap-x skill for the LIBERO task
``open_the_middle_drawer_of_the_cabinet`` (suite ``libero_goal_task``, task_id 0).

This package is a *dogfooding* exercise: we write — by hand — the kind of code the
coding agent would have to produce, using ONLY the legitimate cap-x vocabulary
exposed by ``FrankaLiberoApiReducedSkillLibrary`` (cameras + proprioception in,
joint actions out; no privileged ground-truth object poses).  The point is to
surface what cap-x is missing.  Findings live in ``GAPS.md``.
"""

from .skill import solve  # noqa: F401
