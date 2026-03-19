from __future__ import annotations

import atexit
import contextlib
import io
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from ruamel.yaml import YAML
from tqdm import tqdm

from eb_jepa.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "diagnostics"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _flatten_payload(prefix: str, value: Any, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    key = prefix.strip("/")
    if isinstance(value, torch.Tensor):
        arrays[key] = value.detach().cpu().numpy()
        return
    if isinstance(value, np.ndarray):
        arrays[key] = value
        return
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            next_key = f"{key}/{sub_key}" if key else str(sub_key)
            _flatten_payload(next_key, sub_value, arrays, meta)
        return
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
            if arr.dtype != object:
                arrays[key] = arr
                return
        except Exception:
            pass
        meta[key] = _to_jsonable(value)
        return
    meta[key] = _to_jsonable(value)


METRIC_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "train_loss",
        "pattern": r"^train/loss$",
        "display_name": "Training Loss",
        "description": "Total training objective logged from the JEPA optimization step.",
        "why_it_matters": "Tracks overall optimization progress and major training instability.",
        "preferred_direction": "lower_better",
        "heuristic_interpretation": "Persistent increase is usually bad. Early oscillation can be normal.",
    },
    {
        "id": "train_component_loss",
        "pattern": r"^train/(vc_loss|pred_loss|recon_loss|det_loss)$",
        "display_name": "Training Component Loss",
        "description": "One component of the total training objective.",
        "why_it_matters": "Separates which term is driving optimization or failure.",
        "preferred_direction": "lower_better",
        "heuristic_interpretation": "Interpret relative to the total loss and coefficient settings.",
    },
    {
        "id": "train_regularizer_family",
        "pattern": r"^train/.+$",
        "display_name": "Training Scalar",
        "description": "Additional training scalar or regularizer-derived metric.",
        "why_it_matters": "Can expose collapse, variance floor issues, or regularizer imbalance.",
        "preferred_direction": "context_dependent",
        "heuristic_interpretation": "Use together with loss decomposition and covariance diagnostics.",
    },
    {
        "id": "val_loss",
        "pattern": r"^(fast/)?val/(recon_loss|det_loss)$",
        "display_name": "Validation Loss",
        "description": "Validation-side probe loss measuring reconstruction or detection quality.",
        "why_it_matters": "Primary lightweight signal of quality outside training batches.",
        "preferred_direction": "lower_better",
        "heuristic_interpretation": "Fast-prefixed values come from subset probes and are lower-confidence.",
    },
    {
        "id": "ap_family",
        "pattern": r"^(fast/)?AP_\d+$",
        "display_name": "Average Precision by Horizon",
        "description": "Average precision at a rollout horizon index.",
        "why_it_matters": "Measures prediction usefulness over time horizons.",
        "preferred_direction": "higher_better",
        "heuristic_interpretation": "Compare horizons and trends rather than absolute values alone.",
    },
    {
        "id": "progress_family",
        "pattern": r"^progress/(step|epoch_float|epoch_int|epoch_idx)$",
        "display_name": "Progress Coordinate",
        "description": "Helper coordinate for aligning curves across train and eval.",
        "why_it_matters": "Makes logs interpretable on a common step or epoch axis.",
        "preferred_direction": "not_ordinal",
        "heuristic_interpretation": "Not a quality metric; use as x-axis metadata.",
    },
    {
        "id": "eval_checkpoint_family",
        "pattern": r"^eval/checkpoint_.+$",
        "display_name": "Evaluation Checkpoint Metadata",
        "description": "Metadata identifying which saved checkpoint was evaluated.",
        "why_it_matters": "Links curves and artifacts back to exact model states.",
        "preferred_direction": "not_ordinal",
        "heuristic_interpretation": "Metadata only.",
    },
    {
        "id": "covariance_family",
        "pattern": r"^(fast/)?val/diag/cov/.+/(effective_rank|participation_ratio|top1_frac|top5_frac|trace|sample_count|feature_dim)$",
        "display_name": "Covariance Diagnostic",
        "description": "Scalar summary derived from covariance eigenvalues for one representation source.",
        "why_it_matters": "Provides compact collapse and anisotropy monitoring without inspecting spectra manually.",
        "preferred_direction": "context_dependent",
        "heuristic_interpretation": "Higher effective rank/participation ratio usually means less concentration. Very high top1 fraction can indicate collapse risk.",
    },
    {
        "id": "temporal_diag_family",
        "pattern": r"^(fast/)?val/diag/temporal/.+$",
        "display_name": "Temporal Diagnostic",
        "description": "Summary statistic derived from temporal self-similarity matrices.",
        "why_it_matters": "Captures temporal smoothness, separation, and drift in latent dynamics.",
        "preferred_direction": "context_dependent",
        "heuristic_interpretation": "Interpret with the distance metric and whether the event is fast or canonical.",
    },
    {
        "id": "trajectory_diag_family",
        "pattern": r"^(fast/)?val/diag/trajectory/.+$",
        "display_name": "Trajectory Diagnostic",
        "description": "Summary statistic derived from PCA trajectory comparisons between ground truth and prediction.",
        "why_it_matters": "Helps quantify rollout drift without relying only on trajectory plots.",
        "preferred_direction": "lower_better",
        "heuristic_interpretation": "Lower divergence is usually better, but compare relative to horizon length and training stage.",
    },
    {
        "id": "embedding_diag_family",
        "pattern": r"^(fast/)?val/diag/embedding/.+$",
        "display_name": "Embedding Diagnostic",
        "description": "Summary statistic derived from occupancy embedding geometry.",
        "why_it_matters": "Measures separation or mixing between occupancy classes.",
        "preferred_direction": "context_dependent",
        "heuristic_interpretation": "Useful comparatively; absolute values depend on embedding method and subsampling.",
    },
    {
        "id": "activation_diag_family",
        "pattern": r"^(fast/)?val/diag/activation/.+$",
        "display_name": "Activation Diagnostic",
        "description": "Summary statistic from activation saliency maps.",
        "why_it_matters": "Flags concentration, sparsity, and unstable attention patterns.",
        "preferred_direction": "context_dependent",
        "heuristic_interpretation": "Interpret with layer choice and sample count.",
    },
    {
        "id": "health_family",
        "pattern": r"^(train|val)/(mem|shape)/.+$",
        "display_name": "Health Diagnostic",
        "description": "Operational health metric such as memory usage or tensor shape capture.",
        "why_it_matters": "Useful for debugging runtime regressions and environment issues.",
        "preferred_direction": "context_dependent",
        "heuristic_interpretation": "Hard failures matter more than absolute values unless tracking drift.",
    },
]


def _catalog_entry_for_metric(key: str) -> dict[str, Any]:
    for family in METRIC_FAMILIES:
        if re.match(family["pattern"], key):
            return {
                "family_id": family["id"],
                "key": key,
                "display_name": family["display_name"],
                "description": family["description"],
                "why_it_matters": family["why_it_matters"],
                "preferred_direction": family["preferred_direction"],
                "heuristic_interpretation": family["heuristic_interpretation"],
            }
    return {
        "family_id": "uncategorized",
        "key": key,
        "display_name": key,
        "description": "Scalar metric without an explicit catalog family yet.",
        "why_it_matters": "Review source payload or code path for interpretation.",
        "preferred_direction": "context_dependent",
        "heuristic_interpretation": "No catalog guidance registered yet.",
    }


def _build_metrics_readme_text() -> str:
    lines = [
        "# Diagnostics Metric Catalog",
        "",
        "This catalog explains exported scalar families and how to interpret them.",
        "",
        "Guidance policy:",
        "- Hard failure conditions are only used for truly invalid states such as NaNs, missing counts, or broken payload generation.",
        "- Most model-quality metrics are context dependent; treat the notes below as heuristics, not universal thresholds.",
        "",
    ]
    for family in METRIC_FAMILIES:
        lines.extend(
            [
                f"## {family['display_name']}",
                "",
                f"- Pattern: `{family['pattern']}`",
                f"- Meaning: {family['description']}",
                f"- Why it matters: {family['why_it_matters']}",
                f"- Direction: {family['preferred_direction']}",
                f"- Interpretation: {family['heuristic_interpretation']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


class DiagnosticsManager:
    def __init__(
        self,
        run_dir: Path | str,
        wandb_run: Any | None = None,
        enabled: bool = True,
        upload_artifacts: bool = True,
        flush_interval_sec: float = 3.0,
        run_kind: str = "train",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled)
        self.wandb_run = wandb_run
        self.upload_artifacts = bool(upload_artifacts and wandb_run is not None)
        self.flush_interval_sec = max(0.5, float(flush_interval_sec))
        self.run_kind = str(run_kind)
        self.diagnostics_root = self.run_dir / "diagnostics"
        self.scalars_dir = self.diagnostics_root / "scalars"
        self.checkpoints_dir = self.diagnostics_root / "checkpoints"
        self.events_dir = self.diagnostics_root / "events"
        self.media_dir = self.diagnostics_root / "media"
        self.metrics_dir = self.diagnostics_root / "metrics"
        self.queue_dir = self.diagnostics_root / "upload_queue"
        self.manifest_path = self.diagnostics_root / "manifest.json"
        self.index_path = self.diagnostics_root / "index.json"
        self.summary_path = self.diagnostics_root / "summary.md"
        self.catalog_path = self.metrics_dir / "catalog.yaml"
        self.metrics_readme_path = self.metrics_dir / "README.md"
        self.scalars_jsonl_path = self.scalars_dir / "scalars.jsonl"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._artifact_name = f"{_safe_name(self.run_dir.name)}-diagnostics"

        if not self.enabled:
            return

        for path in [
            self.diagnostics_root,
            self.scalars_dir,
            self.checkpoints_dir,
            self.events_dir,
            self.media_dir,
            self.metrics_dir,
            self.queue_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self._write_metric_catalog()
        self._ensure_manifest_and_index()
        self._update_summary()
        self._update_wandb_metadata()

        if self.upload_artifacts:
            self._start_worker()
            atexit.register(self.close)

    def close(self) -> None:
        if not self.enabled or self._worker is None:
            return
        self._stop_event.set()
        self._worker.join(timeout=max(2.0, self.flush_interval_sec + 1.0))
        self._worker = None

    def event_dir_for(self, event_type: str, step: int, epoch: int | None) -> Path:
        safe_event = _safe_name(event_type)
        step_name = f"step_{int(step):08d}"
        if event_type in {"canonical_diagnostics", "eval_checkpoint"}:
            return self.checkpoints_dir / step_name
        epoch_suffix = f"_epoch_{int(epoch):04d}" if epoch is not None else ""
        return self.events_dir / f"{safe_event}_{step_name}{epoch_suffix}"

    def media_root_for(self, category: str) -> Path:
        path = self.media_dir / _safe_name(category)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_event(
        self,
        *,
        event_type: str,
        phase: str,
        step: int,
        epoch: int | None,
        metrics: dict[str, Any],
        raw_payloads: dict[str, Any] | None = None,
        media_refs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        event_dir = self.event_dir_for(event_type, step, epoch)
        event_dir.mkdir(parents=True, exist_ok=True)
        event_id = event_dir.relative_to(self.diagnostics_root).as_posix()
        raw_payloads = raw_payloads or {}
        media_refs = media_refs or {}
        metadata = metadata or {}

        metric_docs: dict[str, Any] = {}
        scalar_rows = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalar_rows.append(
                    {
                        "timestamp": _utc_now_iso(),
                        "event_id": event_id,
                        "event_type": event_type,
                        "phase": phase,
                        "step": int(step),
                        "epoch": int(epoch) if epoch is not None else None,
                        "metric": key,
                        "value": float(value),
                    }
                )
            metric_docs[key] = _catalog_entry_for_metric(key)

        payload_files = self._write_payloads(event_dir, raw_payloads)
        self._write_json(event_dir / "metadata.json", {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type,
            "phase": phase,
            "step": int(step),
            "epoch": int(epoch) if epoch is not None else None,
            "created_at": _utc_now_iso(),
            "payload_files": payload_files,
            **_to_jsonable(metadata),
        })
        self._write_json(event_dir / "scalars.json", _to_jsonable(metrics))
        self._write_json(event_dir / "metric_metadata.json", metric_docs)
        self._write_json(event_dir / "media_refs.json", _to_jsonable(media_refs))
        self._write_event_summary(event_dir, event_type, phase, step, epoch, metrics, payload_files, media_refs)

        with self._lock:
            for row in scalar_rows:
                with self.scalars_jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            manifest = self._read_json(self.manifest_path, default={})
            index = self._read_json(self.index_path, default={})
            index.setdefault("events", {})[event_id] = {
                "event_type": event_type,
                "phase": phase,
                "step": int(step),
                "epoch": int(epoch) if epoch is not None else None,
                "relative_path": event_id,
                "metrics": sorted(metrics.keys()),
                "payload_files": payload_files,
                "media_keys": sorted(media_refs.keys()),
                "created_at": _utc_now_iso(),
            }
            manifest.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_kind": self.run_kind,
                    "run_dir": str(self.run_dir),
                    "diagnostics_root": str(self.diagnostics_root),
                    "last_event_id": event_id,
                    "last_step": int(step),
                    "last_epoch": int(epoch) if epoch is not None else None,
                    "updated_at": _utc_now_iso(),
                }
            )
            self._write_json(self.manifest_path, manifest)
            self._write_json(self.index_path, index)
            self._update_summary(manifest=manifest, index=index)

        if self.upload_artifacts:
            self._enqueue_artifact_job(event_id=event_id, event_dir=event_dir, event_type=event_type, step=step, epoch=epoch)

        self._update_wandb_metadata(last_event_id=event_id)
        return event_dir

    def drain_upload_queue(self, max_jobs: int | None = None) -> None:
        if not self.upload_artifacts:
            return
        job_paths = sorted(self.queue_dir.glob("*.json"))
        processed = 0
        for job_path in job_paths:
            if max_jobs is not None and processed >= max_jobs:
                break
            processed += int(self._process_artifact_job(job_path))

    def _ensure_manifest_and_index(self) -> None:
        if not self.manifest_path.exists():
            self._write_json(
                self.manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_kind": self.run_kind,
                    "run_dir": str(self.run_dir),
                    "diagnostics_root": str(self.diagnostics_root),
                    "created_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                },
            )
        if not self.index_path.exists():
            self._write_json(self.index_path, {"events": {}, "created_at": _utc_now_iso()})
        if not self.scalars_jsonl_path.exists():
            self.scalars_jsonl_path.touch()

    def _write_metric_catalog(self) -> None:
        yaml = YAML()
        yaml.default_flow_style = False
        catalog = {
            "schema_version": SCHEMA_VERSION,
            "families": METRIC_FAMILIES,
        }
        with self.catalog_path.open("w", encoding="utf-8") as handle:
            yaml.dump(catalog, handle)
        self.metrics_readme_path.write_text(_build_metrics_readme_text(), encoding="utf-8")

    def _write_payloads(self, event_dir: Path, raw_payloads: dict[str, Any]) -> list[str]:
        payload_files: list[str] = []
        for payload_name, payload_value in raw_payloads.items():
            arrays: dict[str, np.ndarray] = {}
            meta: dict[str, Any] = {}
            _flatten_payload("", payload_value, arrays, meta)
            base_name = _safe_name(payload_name)
            if arrays:
                npz_path = event_dir / f"{base_name}.npz"
                np.savez_compressed(npz_path, **cast(dict[str, Any], arrays))
                payload_files.append(npz_path.name)
            if meta:
                json_path = event_dir / f"{base_name}.json"
                self._write_json(json_path, meta)
                payload_files.append(json_path.name)
        return payload_files

    def _write_event_summary(
        self,
        event_dir: Path,
        event_type: str,
        phase: str,
        step: int,
        epoch: int | None,
        metrics: dict[str, Any],
        payload_files: list[str],
        media_refs: dict[str, Any],
    ) -> None:
        top_metrics = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float, np.integer, np.floating)):
                top_metrics.append(f"- `{key}`: {float(value):.6g}")
            if len(top_metrics) >= 12:
                break
        media_lines = [f"- `{key}` -> {value}" for key, value in sorted(_to_jsonable(media_refs).items())]
        lines = [
            f"# Diagnostics Event: {event_type}",
            "",
            f"- Phase: `{phase}`",
            f"- Step: `{int(step)}`",
            f"- Epoch: `{int(epoch) if epoch is not None else 'n/a'}`",
            "",
            "## Metrics",
            "",
            *(top_metrics or ["- No scalar metrics recorded."]),
            "",
            "## Payload Files",
            "",
            *[f"- `{name}`" for name in payload_files],
            "",
            "## Media References",
            "",
            *(media_lines or ["- No media references recorded."]),
            "",
        ]
        (event_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    def _update_summary(self, manifest: dict[str, Any] | None = None, index: dict[str, Any] | None = None) -> None:
        manifest = dict(manifest or self._read_json(self.manifest_path, default={}) or {})
        index = dict(index or self._read_json(self.index_path, default={}) or {})
        events = index.get("events", {})
        latest = events.get(manifest.get("last_event_id"), {}) if events else {}
        lines = [
            "# Diagnostics Summary",
            "",
            f"- Schema version: `{manifest.get('schema_version', SCHEMA_VERSION)}`",
            f"- Run directory: `{manifest.get('run_dir', self.run_dir)}`",
            f"- Diagnostics root: `{manifest.get('diagnostics_root', self.diagnostics_root)}`",
            f"- Total events: `{len(events)}`",
            f"- Last event: `{manifest.get('last_event_id', 'n/a')}`",
            "",
        ]
        if latest:
            lines.extend(
                [
                    "## Latest Event",
                    "",
                    f"- Event type: `{latest.get('event_type', 'n/a')}`",
                    f"- Phase: `{latest.get('phase', 'n/a')}`",
                    f"- Step: `{latest.get('step', 'n/a')}`",
                    f"- Epoch: `{latest.get('epoch', 'n/a')}`",
                    f"- Metrics recorded: `{len(latest.get('metrics', []))}`",
                    f"- Payload files: `{len(latest.get('payload_files', []))}`",
                    "",
                ]
            )
        self.summary_path.write_text("\n".join(lines), encoding="utf-8")

    def _update_wandb_metadata(self, last_event_id: str | None = None) -> None:
        if self.wandb_run is None or not self.enabled:
            return
        try:
            self.wandb_run.config.update(
                {
                    "diagnostics.run_dir": str(self.run_dir),
                    "diagnostics.root": str(self.diagnostics_root),
                    "diagnostics.schema_version": SCHEMA_VERSION,
                    "diagnostics.catalog_path": str(self.catalog_path),
                },
                allow_val_change=True,
            )
            summary = {
                "diagnostics/run_dir": str(self.run_dir),
                "diagnostics/root": str(self.diagnostics_root),
                "diagnostics/schema_version": SCHEMA_VERSION,
                "diagnostics/index_path": str(self.index_path),
                "diagnostics/summary_path": str(self.summary_path),
            }
            if last_event_id is not None:
                summary["diagnostics/last_event_id"] = last_event_id
            self.wandb_run.summary.update(summary)
        except Exception:
            logger.exception("Failed updating W&B diagnostics metadata")

    def _enqueue_artifact_job(self, *, event_id: str, event_dir: Path, event_type: str, step: int, epoch: int | None) -> None:
        job_path = self.queue_dir / f"{_safe_name(event_id)}.json"
        if job_path.exists():
            return
        self._write_json(
            job_path,
            {
                "event_id": event_id,
                "event_dir": str(event_dir),
                "event_type": event_type,
                "step": int(step),
                "epoch": int(epoch) if epoch is not None else None,
                "attempts": 0,
                "status": "pending",
                "created_at": _utc_now_iso(),
            },
        )

    def _start_worker(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._worker_loop, name="diagnostics-uploader", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.drain_upload_queue(max_jobs=1)
            except Exception:
                logger.exception("Diagnostics uploader loop failed")
            self._stop_event.wait(self.flush_interval_sec)

    @staticmethod
    def _emit_console_lines_safely(text: str) -> None:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                tqdm.write(line)
            except Exception:
                logger.info(line)

    def _run_with_tqdm_console(self, fn):
        # WandB artifact APIs print to stdout/stderr, which can break active tqdm bars.
        # Capture and replay with tqdm.write to preserve progress bar rendering.
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            result = fn()
        out_text = out_buf.getvalue()
        err_text = err_buf.getvalue()
        if out_text:
            self._emit_console_lines_safely(out_text)
        if err_text:
            self._emit_console_lines_safely(err_text)
        return result

    def _process_artifact_job(self, job_path: Path) -> bool:
        job = self._read_json(job_path, default={})
        if not job or job.get("status") == "done":
            return False
        if self.wandb_run is None:
            return False

        try:
            import wandb

            event_dir = Path(job["event_dir"])
            if not event_dir.exists():
                job["status"] = "missing"
                self._write_json(job_path, job)
                return False

            artifact = wandb.Artifact(
                name=self._artifact_name,
                type="diagnostics",
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "event_id": job["event_id"],
                    "event_type": job.get("event_type"),
                    "step": job.get("step"),
                    "epoch": job.get("epoch"),
                    "run_dir": str(self.run_dir),
                },
            )
            def _upload():
                artifact.add_dir(str(event_dir), name=event_dir.name)
                artifact.add_file(str(self.manifest_path), name="manifest.json")
                artifact.add_file(str(self.index_path), name="index.json")
                artifact.add_file(str(self.catalog_path), name="metrics/catalog.yaml")
                artifact.add_file(str(self.metrics_readme_path), name="metrics/README.md")
                aliases = ["latest", _safe_name(str(job["event_id"]))]
                self.wandb_run.log_artifact(artifact, aliases=aliases)

            self._run_with_tqdm_console(_upload)
            job["status"] = "done"
            job["uploaded_at"] = _utc_now_iso()
            job["attempts"] = int(job.get("attempts", 0)) + 1
            self._write_json(job_path, job)
            return True
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["attempts"] = int(job.get("attempts", 0)) + 1
            job["last_error_at"] = _utc_now_iso()
            self._write_json(job_path, job)
            logger.exception("Failed uploading diagnostics artifact for %s", job.get("event_id"))
            return False

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed reading JSON file %s", path)
            return default

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)


def build_media_reference(key: str, path: Path | str, wandb_key: str | None = None) -> dict[str, Any]:
    return {
        "key": key,
        "local_path": str(path),
        "wandb_key": wandb_key or key,
    }
