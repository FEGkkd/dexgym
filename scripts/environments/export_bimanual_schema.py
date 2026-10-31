"""Write the checked Phase 1 action and observation schema as JSON."""

import json
from pathlib import Path

from dexteroushands_lab.tasks.direct.bimanual_bottle.task_contract import schema_dict

path = Path(__file__).resolve().parents[2] / "docs/assets/phase1_bimanual_rl_schema.json"
path.write_text(json.dumps(schema_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
