"""Robot asset adapters used by DexterousHands tasks.

The package is intentionally lazy so pure-Python contracts and manifests do
not import Isaac Sim. The simulator-backed adapter is loaded on attribute
access by task construction.
"""

from importlib import import_module

__all__ = [
    "WUJI_ISAAC_JOINT_NAMES",
    "WUJI_ISAAC_BODY_NAMES",
    "WUJI_ISAAC_TIP_BODY_NAMES",
    "WUJI_ISAAC_NON_TIP_BODY_NAMES",
    "make_wuji_hand_cfg",
    "wuji_description_root",
    "wuji_hand_usd_path",
    "WUJI_HAND_USD_RELATIVE_PATH",
    "WUJI_HAND_CONTACT_FRICTION",
    "WUJI_OBJECT_SIZE_M",
    "WUJI_OBJECT_MASS_KG",
    "WUJI_OBJECT_FRICTION",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".wuji_hand", __name__)
        return getattr(module, name)
    raise AttributeError(name)
