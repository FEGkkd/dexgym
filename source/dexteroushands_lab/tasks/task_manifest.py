from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    workflow: str
    status: str
    source_task: str | None = None


TASK_MANIFEST = (
    TaskRecord(
        task_id="DexterousHands-Shadow-Hand-Pose-Direct-v0",
        workflow="direct",
        status="architecture-baseline",
    ),
    TaskRecord(
        task_id="DexterousHands-Shadow-Hand-Reorientation-Direct-v0",
        workflow="direct",
        status="native-port-in-progress",
        source_task="ShadowHandReOrientation",
    ),
    TaskRecord(
        task_id="DexterousHands-Wuji-Hand-Reorientation-Direct-v0",
        workflow="direct",
        status="isaac-runtime-and-export-chain-validated-task-effect-pending",
        source_task="wuji-mjlab.reorient",
    ),
    TaskRecord(
        task_id="DexterousHands-Allegro-Repose-ManagerBased-v0",
        workflow="manager_based",
        status="architecture-baseline",
        source_task="Isaac-Repose-Cube-Allegro-v0",
    ),
)


def get_task_record(task_id: str) -> TaskRecord:
    for record in TASK_MANIFEST:
        if record.task_id == task_id:
            return record
    raise KeyError(f"Unknown DexterousHands task: {task_id}")
