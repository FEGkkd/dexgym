"""Semantic contact groups for native WUJI body names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WujiContactGroups:
    """Map native body names to the groups used by official reward terms."""

    palm_proximal: tuple[int, ...]
    distal: tuple[int, ...]
    fingertips: tuple[int, ...]
    finger_collision_pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_body_names(cls, body_names: tuple[str, ...] | list[str]) -> "WujiContactGroups":
        names = tuple(body_names)
        palm = tuple(
            i for i, name in enumerate(names)
            if name.endswith("palm_link") or name.endswith(("link1", "link2"))
        )
        distal = tuple(
            i for i, name in enumerate(names)
            if name.endswith(("link3", "link4"))
        )
        fingertips = tuple(i for i, name in enumerate(names) if name.endswith("tip_link"))
        if len(fingertips) != 5 or not palm or not distal:
            raise ValueError(
                "WUJI contact mapping requires palm/proximal, distal, and five fingertip bodies"
            )
        finger_bodies: dict[int, list[int]] = {}
        for i, name in enumerate(names):
            if "finger" not in name:
                continue
            try:
                finger = int(name.split("finger", 1)[1].split("_", 1)[0])
            except (IndexError, ValueError):
                continue
            finger_bodies.setdefault(finger, []).append(i)
        pairs: list[tuple[int, int]] = []
        fingers = sorted(finger_bodies)
        for left, finger in enumerate(fingers):
            for other in fingers[left + 1 :]:
                pairs.extend(
                    (first, second)
                    for first in finger_bodies[finger]
                    for second in finger_bodies[other]
                )
        return cls(palm, distal, fingertips, tuple(pairs))

    @property
    def all_hand_bodies(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.palm_proximal + self.distal + self.fingertips)))
