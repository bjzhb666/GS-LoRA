"""Experiment tracking behind a tiny interface.

Trackers are injected rather than imported at module scope, so nothing in the
package depends on a tracking backend being installed or logged in. The
default writes JSONL to the output directory; ``--tracker wandb`` adds
Weights & Biases on top.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

__all__ = ["Tracker", "NullTracker", "JsonlTracker", "WandbTracker", "build_tracker"]


class Tracker(Protocol):
    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...
    def set_name(self, name: str) -> None: ...
    def finish(self) -> None: ...


class NullTracker:
    """Does nothing. Used in tests and when no backend is configured."""

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        return None

    def set_name(self, name: str) -> None:
        return None

    def finish(self) -> None:
        return None


class JsonlTracker:
    """Append metrics to ``<outdir>/metrics.jsonl`` and echo scalars to stdout.

    The default backend: no dependencies, and a durable record of the run that
    survives a tracking service being unreachable.
    """

    def __init__(self, outdir: str | Path, echo: bool = False) -> None:
        self.path = Path(outdir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._echo = echo
        self._t0 = time.time()

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        record: dict[str, Any] = {"_t": round(time.time() - self._t0, 3)}
        if step is not None:
            record["_step"] = step
        for key, value in metrics.items():
            record[key] = _to_jsonable(value)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self._echo:
            print("  ".join(f"{k}={v}" for k, v in record.items() if not k.startswith("_")))

    def set_name(self, name: str) -> None:
        self.log({"_run_name": name})

    def finish(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class WandbTracker:
    """Thin adapter over wandb, imported lazily."""

    def __init__(self, project: str, group: str | None, name: str | None, offline: bool, config: Mapping[str, Any]) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            group=group,
            name=name,
            mode="offline" if offline else "online",
        )
        wandb.config.update(dict(config), allow_val_change=True)

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self._wandb.log({k: _to_jsonable(v) for k, v in metrics.items()}, step=step)

    def set_name(self, name: str) -> None:
        if self._run is not None:
            self._run.name = name

    def finish(self) -> None:
        self._wandb.finish()


class MultiTracker:
    def __init__(self, *trackers: Tracker) -> None:
        self._trackers = [t for t in trackers if t is not None]

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        for t in self._trackers:
            t.log(metrics, step)

    def set_name(self, name: str) -> None:
        for t in self._trackers:
            t.set_name(name)

    def finish(self) -> None:
        for t in self._trackers:
            t.finish()


def build_tracker(cfg) -> Tracker:
    """Build the tracker described by a :class:`gslora.config.Config`."""
    trackers: list[Tracker] = [JsonlTracker(cfg.outdir, echo=False)]
    if cfg.tracker == "wandb":
        try:
            trackers.append(
                WandbTracker(
                    project=cfg.wandb_project,
                    group=cfg.wandb_group,
                    name=cfg.run_name or Path(cfg.outdir).name,
                    offline=cfg.wandb_offline,
                    config=cfg.to_dict(),
                )
            )
        except Exception as exc:  # pragma: no cover - depends on local wandb setup
            print(f"[tracking] wandb unavailable ({exc!r}); falling back to JSONL only.")
    return MultiTracker(*trackers)


def _to_jsonable(value: Any) -> Any:
    # torch tensors / numpy scalars show up constantly in the metric dicts.
    if hasattr(value, "item") and getattr(value, "numel", lambda: 1)() == 1:
        try:
            return value.item()
        except Exception:  # pragma: no cover
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover
            pass
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
