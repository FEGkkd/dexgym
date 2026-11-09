# Ported from CumulusAlpha/Dex-Supermimic @ 2becb738.
"""Metric logging backends for Dex SuperMimic."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

LoggerBackend = Literal["wandb", "tensorboard", "none"]


class MetricLogger:
    """Small facade over W&B and TensorBoard.

    TensorBoard files remain in the run directory. W&B uses the same directory as
    its ``dir`` so offline runs and local metadata stay colocated with checkpoints.
    """

    def __init__(
        self,
        backend: LoggerBackend,
        *,
        log_dir: str | Path,
        config: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        project: str,
        entity: str,
    ) -> None:
        self.backend = backend
        self.log_dir = Path(log_dir)
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._tensorboard = SummaryWriter(str(self.log_dir))
            self._wandb = None
        elif backend == "wandb":
            import wandb

            if os.getenv("WANDB_MODE", "").lower() == "disabled":
                self.backend = "none"
                self._wandb = self._tensorboard = None
                return
            if not project or not entity:
                raise ValueError("project and entity are required for wandb logging")
            self._wandb = wandb.init(
                project=project,
                entity=entity,
                name=run_name or self.log_dir.name,
                dir=str(self.log_dir),
                config=dict(config or {}),
            )
            self._tensorboard = None
        elif backend == "none":
            self._wandb = self._tensorboard = None
        else:
            raise ValueError(f"Unknown logger backend: {backend!r}")

    @property
    def wandb_run(self):
        """The active ``wandb.Run`` when W&B is selected."""
        return self._wandb

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        """Log one dictionary of scalar metrics at a common explicit step."""
        values = {name: float(value) for name, value in scalars.items()}
        if self.backend == "wandb":
            self._wandb.log(values, step=step)
        elif self.backend == "tensorboard":
            for name, value in values.items():
                self._tensorboard.add_scalar(name, value, step)

    def log_images(self, name: str, images, step: int) -> None:
        """Log a ``(N, C, H, W)`` image batch."""
        if self.backend == "wandb":
            self._wandb.log({name: wandb_image_batch(images)}, step=step)
        elif self.backend == "tensorboard":
            self._tensorboard.add_images(name, images, step)

    def flush(self) -> None:
        if self.backend == "tensorboard":
            self._tensorboard.flush()

    def close(self) -> None:
        if self._tensorboard is not None:
            self._tensorboard.close()
        if self._wandb is not None:
            self._wandb.finish()
        self._wandb = self._tensorboard = None


def wandb_image_batch(images) -> list:
    """Convert a tensor/ndarray image batch to W&B image objects."""
    import wandb

    if hasattr(images, "detach"):
        return [wandb.Image(image) for image in images.detach().cpu()]
    if hasattr(images, "transpose"):
        images = images.transpose(0, 2, 3, 1)
    return [wandb.Image(image) for image in images]
