"""Project-level RSL-RL entrypoints and CLI helpers.

The namespace is extended so the installed ``rsl_rl.runners`` package remains
available alongside these project-owned entrypoints.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
