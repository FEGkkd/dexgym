"""Read packaged asset provenance without importing a simulator backend."""

import json
from importlib.resources import files


def load_manifest() -> dict:
    return json.loads(files("dexgym.assets").joinpath("manifest.json").read_text())
