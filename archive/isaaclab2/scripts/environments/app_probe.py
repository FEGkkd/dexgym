"""Minimal Isaac Sim launcher probe for local runtime diagnostics."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
print(f"PROBE before_launcher name={__name__} executable={sys.executable}", flush=True)
simulation_app = AppLauncher(args).app
print(
    f"PROBE after_launcher name={__name__} running={simulation_app.is_running()}",
    flush=True,
)


def main() -> None:
    print(
        f"PROBE main running={simulation_app.is_running()} alive={simulation_app.is_running()}",
        flush=True,
    )
    simulation_app.update()
    print(f"PROBE after_update running={simulation_app.is_running()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
