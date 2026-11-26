"""Read the real signed inputs and run a small seeded CUDA/Warp kernel witness."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from check_execution import REPO, read

HMB_ROOT = Path("/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-E-E1-20261126-3f90652")
TELEOP = Path("/home/ubuntu/dexgym_worktrees/teleop-sim0-study")
HOST = Path("/home/ubuntu/dexgym_data/s0-5")


def stdout(*args):
    return subprocess.check_output(args, text=True).strip()


def digest(path):
    with path.open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    from hashlib import file_digest

    return file_digest(stream, "sha256").hexdigest()


def resources():
    gpu = stdout("nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,driver_version",
                 "--format=csv,noheader,nounits").split(", ")
    processes = stdout("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits")
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            mem[name] = int(value.split()[0]) * 1024
    return {"gpu_name": gpu[0], "gpu_memory_used_mib": int(gpu[1]), "gpu_memory_total_mib": int(gpu[2]),
            "gpu_utilization_pct": int(gpu[3]), "gpu_temperature_c": int(gpu[4]), "driver": gpu[5],
            "gpu_processes": processes.splitlines(), "cpu_count": __import__("os").cpu_count(), **mem,
            "disk_available_bytes": shutil.disk_usage(HOST).free}


def signed_inputs():
    from dexgym.tasks.direct.wholebody_s05.hmb_reader import GEOMETRY, MANIFEST_SHA, HMBReader

    manifest = HMB_ROOT / "review-1/hmb_manifest.json"
    plan = HMB_ROOT / "production_plan.json"
    config = REPO / "docs/s0-5/evidence/P01_E_E0/candidate_config.json"
    data = {"manifest_sha256": digest(manifest), "plan_sha256": digest(plan), "config_sha256": digest(config)}
    if data["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("signed HMB identity differs")
    frozen_plan = json.loads(plan.read_bytes())
    roots = {side: {row["root"] for row in frozen_plan["sources"] if row["split"] == side}
             for side in ("Train", "Validation")}
    if roots["Train"] & roots["Validation"]:
        raise ValueError("HMB split overlap")
    samples = {}
    for split in roots:
        reader = HMBReader(manifest, plan, config, HMB_ROOT / "products", split, GEOMETRY)
        arrays, meta, sources, _ = reader.read(reader.keys()[0])
        counts = Counter(str(row["split"]) for row in frozen_plan["partitions"])
        samples[split] = {"roots": len(roots[split]), "partitions": len(reader.keys()),
                          "planned_partitions": counts[split], "audited_sample_ticks": len(arrays["tick_ns"]),
                          "audited_sample_joint_valid": int(arrays["joint_valid"].sum()),
                          "sample_root": meta["identity"]["root_episode_id"], "sample_source_entries": len(sources)}
    data["split"] = samples
    data["planned_ticks"] = sum(row["planned_ticks"] for row in frozen_plan["partitions"])
    data["role"] = "SIGNED_INPUT_BYTES_AND_ONE_PARTITION_PER_SPLIT; NOT_RMB"
    return data


def cuda_kernel():
    import mujoco
    import mujoco_warp
    import torch
    import warp as wp

    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    torch.manual_seed(49)
    values = torch.rand((64, 64), device="cuda", dtype=torch.float32)
    result = float((values @ values.T).sum().item())
    wp.init()
    source = wp.array([float(i) for i in range(32)], dtype=float, device="cuda:0")
    output = wp.empty_like(source)

    @wp.kernel
    def scale(x: wp.array(dtype=float), y: wp.array(dtype=float)):
        location = wp.tid()
        y[location] = x[location] * 2.0

    wp.launch(scale, dim=32, inputs=[source, output], device="cuda:0")
    wp.synchronize()
    last = float(output.numpy()[-1])
    if last != 62.0 or not math.isfinite(result) or result <= 0:
        raise ValueError("CUDA kernel output differs")
    return {"torch": torch.__version__, "cuda_build": torch.version.cuda, "warp": wp.__version__,
            "mujoco": mujoco.__version__, "mujoco_warp_path": str(mujoco_warp.__file__),
            "seed": 49, "torch_sum": result, "warp_last": last, "device": torch.cuda.get_device_name(0)}


def main():
    config, cfg_sha = read()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-witness", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    machine = resources()
    archived = signed_inputs()
    stan = REPO / "docs/s0-5/evidence/P00_V2_1/design_policy.json"
    expected_teleop = "6d0b0b5b2601618f1b0052d93ef2403eea9b064a"
    teleop_head = stdout("git", "-C", str(TELEOP), "rev-parse", "HEAD")
    if teleop_head != expected_teleop:
        raise ValueError("Teleop reference checkout differs")
    bags = []
    for row in json.loads(stan.read_bytes())["regression_recordings"]:
        path = Path(row["path"])
        checksum = digest(path)
        if checksum != row["sha256"]:
            raise ValueError("deployment replay recording identity differs")
        bags.append({"id": row["id"], "sha256": checksum, "bytes": path.stat().st_size, "training_admission": False})
    data = {"at": datetime.now(timezone.utc).isoformat(), "execution_config_sha256": cfg_sha,
            "host": stdout("hostname"), "resources": machine, "signed_hmb": archived, "teleop_base": teleop_head,
            "recordings": bags, "policy_stage": "PREFLIGHT", "train_transitions": 0, "eval_transitions": 0}
    if args.cuda_witness:
        if machine["gpu_processes"]:
            raise ValueError("another compute process is active; CUDA witness not started")
        data["cuda_witness"] = cuda_kernel()
    data["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
