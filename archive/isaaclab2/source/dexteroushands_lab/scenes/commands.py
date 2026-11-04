"""Future-compatible debug control layout: 26 floats per free hand."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class HandCommand:
    translation_world: tuple[float, float, float]
    rotation_local: tuple[float, float, float]
    finger_positions: tuple[float, ...]


@dataclass(frozen=True)
class BimanualCommand:
    left: HandCommand
    right: HandCommand

    @classmethod
    def from_flat(cls, values) -> "BimanualCommand":
        sequence = tuple(values)
        if len(sequence) != 52 or not all(isfinite(x) for x in sequence):
            raise ValueError("Expected 52 finite values: left 26, right 26")
        return cls(
            left=HandCommand(tuple(sequence[0:3]), tuple(sequence[3:6]), tuple(sequence[6:26])),
            right=HandCommand(tuple(sequence[26:29]), tuple(sequence[29:32]), tuple(sequence[32:52])),
        )
