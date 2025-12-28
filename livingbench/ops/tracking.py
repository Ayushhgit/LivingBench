"""
Experiment tracking module - supports local JSON, MLflow, and wandb
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod


class TrackerBackend(ABC):
    """Base class for tracking backends"""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        ...

    @abstractmethod
    def log_artifact(self, path: str, name: str | None = None) -> None:
        ...

    @abstractmethod
    def finish(self) -> None:
        ...


class LocalJSONTracker(TrackerBackend):
    """Simple JSON file tracker - no external deps needed"""

    def __init__(self, out_dir: str, run_name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.run_dir = self.out_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.data = {
            "run_name": run_name,
            "started_at": datetime.now().isoformat(),
            "params": {},
            "metrics": [],
            "artifacts": [],
        }

    def log_params(self, params: dict[str, Any]) -> None:
        self.data["params"].update(params)
        self._save()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        entry = {"ts": datetime.now().isoformat(), "step": step, **metrics}
        self.data["metrics"].append(entry)
        self._save()

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self.data["artifacts"].append({
            "path": path,
            "name": name or Path(path).name,
            "logged_at": datetime.now().isoformat(),
        })
        self._save()

    def finish(self) -> None:
        self.data["finished_at"] = datetime.now().isoformat()
        self._save()

    def _save(self) -> None:
        with open(self.run_dir / "run.json", "w") as f:
            json.dump(self.data, f, indent=2, default=str)


class MLflowTracker(TrackerBackend):
    """MLflow backend wrapper"""

    def __init__(self, exp_name: str, run_name: str, tracking_uri: str | None = None):
        try:
            import mlflow
        except ImportError:
            raise ImportError("mlflow not installed. Run: pip install mlflow")

        self.mlflow = mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(exp_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        # mlflow doesnt like nested dicts so we flatten
        flat = self._flatten(params)
        self.mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self.mlflow.log_artifact(path)

    def finish(self) -> None:
        self.mlflow.end_run()

    def _flatten(self, d: dict, parent: str = "", sep: str = ".") -> dict:
        items = []
        for k, v in d.items():
            key = f"{parent}{sep}{k}" if parent else k
            if isinstance(v, dict):
                items.extend(self._flatten(v, key, sep=sep).items())
            else:
                items.append((key, str(v)[:250]))  # mlflow has 250 char limit
        return dict(items)


class WandbTracker(TrackerBackend):
    """Weights & Biases tracker"""

    def __init__(self, project: str, run_name: str, config: dict | None = None):
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb not installed. Run: pip install wandb")

        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
        )

    def log_params(self, params: dict[str, Any]) -> None:
        self.wandb.config.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        artifact = self.wandb.Artifact(name or Path(path).stem, type="result")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.wandb.finish()


class ExperimentTracker:
    """
    Main tracker class - picks backend based on what you want

    Example:
        tracker = ExperimentTracker(backend="mlflow")
        tracker.log_params({"model": "llama-70b"})
        tracker.log_metrics({"acc": 0.85})
        tracker.finish()
    """

    def __init__(
        self,
        experiment_name: str = "livingbench",
        run_name: str | None = None,
        backend: str = "local",
        output_dir: str = "outputs/experiments",
        **kwargs: Any,
    ):
        self.exp_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backend_type = backend

        # setup the right backend
        if backend == "local":
            self._backend = LocalJSONTracker(output_dir, self.run_name)
        elif backend == "mlflow":
            self._backend = MLflowTracker(
                experiment_name,
                self.run_name,
                tracking_uri=kwargs.get("tracking_uri"),
            )
        elif backend == "wandb":
            self._backend = WandbTracker(
                project=experiment_name,
                run_name=self.run_name,
                config=kwargs.get("config"),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"[Tracker] Started: {backend} backend, run={self.run_name}")

    def log_params(self, params: dict[str, Any]) -> None:
        self._backend.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._backend.log_metrics(metrics, step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self._backend.log_artifact(path, name)

    def log_model_result(
        self,
        model_id: str,
        acc: float,
        weighted_acc: float,
        num_tasks: int,
        latency_ms: float | None = None,
        cost: float | None = None,
        **extra: float,
    ) -> None:
        """Helper to log common model eval metrics"""
        m = {
            f"{model_id}/accuracy": acc,
            f"{model_id}/weighted_accuracy": weighted_acc,
            f"{model_id}/n_tasks": float(num_tasks),
        }
        if latency_ms:
            m[f"{model_id}/latency_ms"] = latency_ms
        if cost:
            m[f"{model_id}/cost_usd"] = cost

        for k, v in extra.items():
            m[f"{model_id}/{k}"] = v

        self._backend.log_metrics(m)

    def finish(self) -> None:
        self._backend.finish()
        print(f"[Tracker] Run '{self.run_name}' done")

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *args) -> None:
        self.finish()
