"""Use Isaac Sim's working binaries, appending pure Python venv dependencies."""

import runpy
import sys
from pathlib import Path

# torch must resolve before the legacy shared venv path is appended.
import torch

sys.path.append("/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if sys.argv[1:2] == ["--pytest"]:
    import pytest

    raise SystemExit(pytest.main(sys.argv[2:]))
if sys.argv[1:2] == ["--witness"]:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    value = torch.randn(8, 8, device="cuda")
    print(
        "S0_KERNEL_WITNESS",
        tuple((value @ value).shape),
        torch.cuda.get_device_name(),
        float((value @ value).sum()),
        flush=True,
    )
elif sys.argv[1:2] == ["-m"]:
    sys.argv = sys.argv[2:]
    if not sys.argv:
        raise SystemExit("-m requires a module name")
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
elif len(sys.argv) > 1:
    sys.argv.pop(0)
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    raise SystemExit("Pass --witness, --pytest paths, -m module, or a repository script path")
