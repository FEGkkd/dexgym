"""Explicit task registration entry point for DexterousHands."""


def register_tasks():
    """Import task modules so their Gymnasium IDs are registered.

    Keeping registration explicit lets pure-Python tools import contracts and
    manifests without starting the Isaac Lab task stack.
    """

    from . import direct, manager_based

    return direct.register_tasks(), manager_based.register_tasks()


__all__ = ["register_tasks"]
