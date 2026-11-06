"""Pipeline orchestration shared by the command-line scripts.

The pipeline is intentionally explicit: cache -> gating -> window index ->
training -> evaluation.  Each stage writes a machine-readable report, so a
failed stage can be diagnosed without rerunning everything.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    get_profile,
    resolve_data_root,
    resolve_external_root,
)
from .data_inspection import (
    DatasetIndex,
    discover_dataset,
    save_index,
    validate_dataset,
)
from .data_loader import (
    SignalCache,
    build_signal_cache,
    iter_cached_pair,
    load_signal_cache,
    read_pair,
)
from .dataset import (
    WindowDataset,
    build_window_index,
    subject_table,
    subset_by_subjects,
)
from .evaluation import (
    aggregate_windows_to_recordings,
    aggregate_windows_to_subjects,
    bootstrap_metric_ci,
    compute_binary_metrics,
    summarise_predictions,
)
from .gating import gate_recording, save_gating_result
from .preprocessing import NormalizationStats, expected_window_length
from .training import build_model_from_checkpoint, load_checkpoint, predict_windows
from .utils import ensure_dir, project_root, resolve_path, save_json

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
def profile_name(cfg: Mapping[str, Any]) -> str:
    return str(get_profile(cfg)["name"])


def cache_path(cfg: Mapping[str, Any], profile: str | None = None, dataset_name: str = "main") -> Path:
    profile = profile or profile_name(cfg)
    return resolve_path(cfg.get("data", {}).get("cache_dir", "data/cache")) / profile / dataset_name


def gating_path(cfg: Mapping[str, Any], profile: str | None = None, dataset_name: str = "main") -> Path:
    profile = profile or profile_name(cfg)
    return project_root() / "data" / "gating" / profile / dataset_name


def windows_path(
    cfg: Mapping[str, Any],
    experiment: str,
    profile: str | None = None,
    dataset_name: str = "main",
) -> Path:
    profile = profile or profile_name(cfg)
    return project_root() / "data" / "windows" / profile / dataset_name / experiment.lower()


def results_path(
    cfg: Mapping[str, Any],
    experiment: str,
    profile: str | None = None,
    run_name: str = "run",
) -> Path:
    profile = profile or profile_name(cfg)
    root = resolve_path(cfg.get("project", {}).get("output_dir", "results"))
    return root / profile / experiment.lower() / run_name


# ---------------------------------------------------------------------------
# Dataset / cache
# ---------------------------------------------------------------------------
def prepare_dataset(
    cfg: Mapping[str, Any],
    root: str | Path | None = None,
    dataset_name: str = "main",
    external: bool = False,
    overwrite: bool = False,
    max_recordings: int | None = None,
    validate: bool = True,
) -> tuple[DatasetIndex, SignalCache]:
    """Discover, validate and cache one dataset split."""
    if root is None:
        if external:
            root = resolve_external_root(cfg, required=True)
        else:
            root = resolve_data_root(cfg)
    root = Path(root)
    threshold = float(cfg.get("data", {}).get("lvef_threshold_percent", 40.0))
    index = discover_dataset(root, lvef_threshold_percent=threshold, external=external)

    index_dir = cache_path(cfg, dataset_name=dataset_name).parent / f"{dataset_name}_index"
    save_index(index, index_dir)
    LOGGER.info("Dataset summary: %s", json.dumps(index.summary(), indent=2, default=str))

    if validate:
        data_cfg = dict(cfg.get("data", {}))
        frame, summary = validate_dataset(
            index,
            expected_ecg_fs=int(data_cfg.get("expected_ecg_fs", 500)),
            expected_pcg_fs=int(data_cfg.get("expected_pcg_fs", 4000)),
            min_duration_seconds=float(data_cfg.get("min_duration_seconds", 10.0)),
            max_duration_seconds=float(data_cfg.get("max_duration_seconds", 90.0)),
            duration_tolerance_seconds=float(data_cfg.get("duration_tolerance_seconds", 0.25)),
            max_recordings=max_recordings,
        )
        frame.to_csv(index_dir / "recording_validation.csv", index=False)
        save_json(summary, index_dir / "recording_validation_summary.json")
        if summary.get("n_errors"):
            LOGGER.warning("%d recordings had read errors; see %s", summary["n_errors"], index_dir)

    cache_base = resolve_path(cfg.get("data", {}).get("cache_dir", "data/cache")) / profile_name(cfg)
    cache = build_signal_cache(
        index,
        dict(cfg),
        cache_root=cache_base,
        name=dataset_name,
        overwrite=overwrite,
        max_recordings=max_recordings,
    )
    LOGGER.info("Signal cache ready: %s", cache.cache_dir)
    return index, cache


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------
def _summary_from_saved(data: Mapping[str, Any]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for item in data.get("rejected", []):
        for reason in item.get("reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    rr = data.get("rr_stats") or {}
    delay = data.get("delay") or {}
    return {
        "record_id": data.get("record_id"),
        "subject": data.get("subject"),
        "site": data.get("site"),
        "fold": data.get("fold"),
        "method": data.get("method"),
        "n_rpeaks": int(data.get("n_rpeaks", 0)),
        "n_windows": int(data.get("n_windows", 0)),
        "n_rejected": int(data.get("n_rejected", 0)),
        "success_rate": float(data.get("success_rate", 0.0)),
        "recording_ok": bool(data.get("recording_ok", False)),
        "reasons": "; ".join(data.get("reasons", [])),
        "rejection_reasons": "; ".join(
            f"{k}:{v}" for k, v in sorted(reason_counts.items())
        ),
        "median_rr": rr.get("median_rr"),
        "heart_rate_bpm": rr.get("heart_rate_bpm"),
        "best_lag_seconds": delay.get("best_lag_seconds"),
        "lag_contrast": delay.get("contrast"),
    }


def run_gating(
    cfg: Mapping[str, Any],
    cache: SignalCache,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    max_recordings: int | None = None,
    make_plots: bool = True,
    plot_count: int = 8,
    example_count: int = 3,
    plot_seconds: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run ECG-guided gating for every cached recording.

    Rejected windows and rejected recordings are recorded explicitly.  There is
    no fallback to ungated PCG.
    """
    output_dir = ensure_dir(output_dir or gating_path(cfg, dataset_name="main"))
    qc_dir = ensure_dir(output_dir / "qc")
    examples_dir = ensure_dir(output_dir / "examples")
    profile = get_profile(cfg)
    metadata = cache.metadata
    if max_recordings:
        metadata = metadata.iloc[: int(max_recordings)]
    rows: list[dict[str, Any]] = []
    plotted = 0
    examples = 0

    for _, record in tqdm(list(metadata.iterrows()), desc="ECG gating", unit="rec"):
        record_id = str(record["record_id"])
        json_path = output_dir / f"{record_id}.json"
        if json_path.exists() and not overwrite:
            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                rows.append(_summary_from_saved(saved))
                continue
            except Exception as exc:
                LOGGER.warning("Could not reuse %s (%s); recomputing.", json_path, exc)

        if str(record.get("status", "ok")) == "error":
            rows.append(
                {
                    "record_id": record_id,
                    "subject": record.get("subject"),
                    "site": record.get("site"),
                    "fold": record.get("fold"),
                    "method": cfg.get("gating", {}).get("method"),
                    "n_rpeaks": 0,
                    "n_windows": 0,
                    "n_rejected": 0,
                    "success_rate": 0.0,
                    "recording_ok": False,
                    "reasons": f"cache error: {record.get('reason', '')}",
                    "rejection_reasons": "",
                }
            )
            continue
        try:
            ecg, pcg, fs_ecg, fs_pcg = iter_cached_pair(cache, record)
            result = gate_recording(
                ecg,
                pcg,
                fs_ecg,
                fs_pcg,
                cfg,
                subject=str(record["subject"]),
                site=str(record["site"]),
                fold=int(record["fold"]) if pd.notna(record.get("fold")) else None,
                record_id=record_id,
                window_seconds=float(profile["window_seconds"]),
            )
            save_gating_result(result, output_dir)
            rows.append(result.summary_row())
            if make_plots and plotted < int(plot_count):
                try:
                    from .visualization import plot_signal_with_gating

                    plot_signal_with_gating(
                        ecg,
                        pcg,
                        fs_ecg,
                        fs_pcg,
                        result,
                        qc_dir / f"{record_id}.png",
                        max_seconds=plot_seconds,
                        title=f"{record_id} | method={result.method} | ok={result.recording_ok}",
                    )
                    plotted += 1
                except Exception as exc:
                    LOGGER.warning("QC plot failed for %s: %s", record_id, exc)
            if make_plots and examples < int(example_count) and result.n_windows:
                try:
                    from .visualization import plot_gated_window_examples

                    plot_gated_window_examples(
                        pcg,
                        fs_pcg,
                        result,
                        examples_dir / f"{record_id}_gated_windows.png",
                    )
                    examples += 1
                except Exception as exc:
                    LOGGER.warning("Gated-window plot failed for %s: %s", record_id, exc)
        except Exception as exc:
            LOGGER.exception("Gating failed for %s", record_id)
            rows.append(
                {
                    "record_id": record_id,
                    "subject": record.get("subject"),
                    "site": record.get("site"),
                    "fold": record.get("fold"),
                    "method": cfg.get("gating", {}).get("method"),
                    "n_rpeaks": 0,
                    "n_windows": 0,
                    "n_rejected": 0,
                    "success_rate": 0.0,
                    "recording_ok": False,
                    "reasons": f"gating pipeline error: {exc}",
                    "rejection_reasons": "",
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "gating_summary.csv", index=False)

    reason_counts: dict[str, int] = {}
    for value in summary_df.get("rejection_reasons", pd.Series(dtype=str)).dropna():
        for part in str(value).split(";"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            reason, number = part.rsplit(":", 1)
            try:
                reason_counts[reason] = reason_counts.get(reason, 0) + int(float(number))
            except ValueError:
                continue

    stats = {
        "profile": profile["name"],
        "window_seconds": float(profile["window_seconds"]),
        "method": cfg.get("gating", {}).get("method"),
        "n_recordings": int(len(summary_df)),
        "n_recording_ok": int(summary_df["recording_ok"].sum()) if len(summary_df) else 0,
        "fraction_recording_ok": (
            float(summary_df["recording_ok"].mean()) if len(summary_df) else 0.0
        ),
        "n_rpeaks_total": int(summary_df["n_rpeaks"].sum()) if len(summary_df) else 0,
        "n_windows_total": int(summary_df["n_windows"].sum()) if len(summary_df) else 0,
        "n_rejected_windows_total": int(summary_df["n_rejected"].sum()) if len(summary_df) else 0,
        "mean_success_rate": (
            float(summary_df["success_rate"].mean()) if len(summary_df) else 0.0
        ),
        "rejection_reason_counts": reason_counts,
        "note": (
            "Windows are ECG-anchored. Recordings that fail gating are excluded; "
            "there is no fallback to ungated PCG."
        ),
    }
    save_json(stats, output_dir / "gating_summary.json")
    try:
        from .visualization import plot_gating_rejections, plot_gating_summary

        plot_gating_summary(summary_df, output_dir / "gating_overview.png")
        plot_gating_rejections(summary_df, output_dir / "gating_rejections.png")
    except Exception as exc:
        LOGGER.warning("Could not create gating summary figures: %s", exc)
    LOGGER.info(
        "Gating: %d/%d recordings accepted, %d windows, %d rejected.",
        stats["n_recording_ok"],
        stats["n_recordings"],
        stats["n_windows_total"],
        stats["n_rejected_windows_total"],
    )
    return summary_df, stats


# ---------------------------------------------------------------------------
# Window index
# ---------------------------------------------------------------------------
def prepare_windows(
    cfg: Mapping[str, Any],
    cache: SignalCache,
    experiment: str,
    gating_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_windows_per_recording: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and save the E1/E2/E3 window table."""
    output_dir = ensure_dir(
        output_dir or windows_path(cfg, experiment, dataset_name="main")
    )
    frame, report = build_window_index(
        cache.metadata,
        cfg,
        experiment=experiment,
        gating_dir=gating_dir,
        max_windows_per_recording=max_windows_per_recording,
    )
    frame.to_csv(output_dir / "windows.csv", index=False)
    save_json(report, output_dir / "windows_summary.json")
    LOGGER.info(
        "Window index %s: %d windows from %d recordings (%d skipped).",
        experiment,
        report["n_windows"],
        report["n_recordings_used"],
        report["n_recordings_skipped"],
    )
    return frame, report


def load_windows(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "windows.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Window table not found: {path}\nRun scripts/prepare_windows.py first."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Window table is empty: {path}")
    return frame


# ---------------------------------------------------------------------------
# Smoke subset
# ---------------------------------------------------------------------------
def smoke_subset(
    windows: pd.DataFrame,
    max_subjects: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a small, class-balanced subject subset for a pipeline smoke test."""
    table = subject_table(windows)
    if max_subjects is None or len(table) <= max_subjects:
        return windows
    positives = table[table["label"] == 1]["subject"].astype(str).tolist()
    negatives = table[table["label"] == 0]["subject"].astype(str).tolist()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    n_pos = min(len(positives), max(1, max_subjects // 3))
    n_neg = min(len(negatives), max_subjects - n_pos)
    chosen = positives[:n_pos] + negatives[:n_neg]
    # Top up from whichever class still has subjects.
    remaining = [s for s in positives + negatives if s not in chosen]
    chosen = chosen + remaining[: max(0, max_subjects - len(chosen))]
    return subset_by_subjects(windows, chosen)


# ---------------------------------------------------------------------------
# Inference / external evaluation
# ---------------------------------------------------------------------------
def infer_with_checkpoint(
    checkpoint_path: str | Path,
    windows: pd.DataFrame,
    cache: SignalCache,
    cfg: Mapping[str, Any],
    device: torch.device | str = "cpu",
    batch_size: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one trained checkpoint over a window table."""
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model = build_model_from_checkpoint(checkpoint, cfg, device=device)
    model.eval()
    mode = "fusion" if str(checkpoint.get("model_type", "pcg")) == "fusion" else "pcg"
    dataset = WindowDataset(windows, cache, cfg, mode=mode, normalize=False)
    dataset.stats = NormalizationStats.from_dict(checkpoint["normalization"])
    if checkpoint.get("ecg_normalization"):
        dataset.ecg_stats = NormalizationStats.from_dict(checkpoint["ecg_normalization"])
    dataset.normalize = True
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    _, predictions = predict_windows(
        model,
        loader,
        criterion=None,
        device=torch.device(device),
        model_type=mode,
        use_amp=False,
    )
    return predictions, checkpoint


def evaluate_external(
    cfg: Mapping[str, Any],
    cache: SignalCache,
    windows: pd.DataFrame,
    checkpoint_paths: Sequence[str | Path],
    output_dir: str | Path,
    experiment: str,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
    note: str | None = None,
) -> dict[str, Any]:
    """Evaluate an ensemble of fold checkpoints on the external set.

    No threshold tuning is performed.  The decision threshold is the mean of
    the thresholds selected on the internal validation splits.  Predictions are
    averaged over the fold models.
    """
    output_dir = ensure_dir(output_dir)
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required for external evaluation.")
    if windows.empty:
        raise ValueError("External window table is empty; cannot evaluate.")

    subject_frames: list[pd.DataFrame] = []
    recording_frames: list[pd.DataFrame] = []
    thresholds: list[float] = []
    for path in checkpoint_paths:
        predictions, checkpoint = infer_with_checkpoint(
            path, windows, cache, cfg, device=device, batch_size=batch_size
        )
        subjects = aggregate_windows_to_subjects(predictions)
        recordings = aggregate_windows_to_recordings(predictions)
        fold = int(checkpoint.get("fold", -1))
        subjects["checkpoint_fold"] = fold
        recordings["checkpoint_fold"] = fold
        subject_frames.append(subjects)
        recording_frames.append(recordings)
        thresholds.append(float(checkpoint.get("threshold", 0.5)))

    def _average(frame_list: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
        combined = pd.concat(frame_list, ignore_index=True)
        grouped = (
            combined.groupby(keys, as_index=False)
            .agg(
                label=("label", "first"),
                prob=("prob", "mean"),
                n_windows=("n_windows", "mean") if "n_windows" in combined.columns else ("prob", "size"),
                n_models=("prob", "size"),
            )
            .reset_index(drop=True)
        )
        return grouped

    subjects = _average(subject_frames, ["subject"])
    recordings = _average(recording_frames, ["record_id", "subject"])
    threshold = float(np.mean(thresholds))
    subjects["threshold"] = threshold
    subjects["pred"] = (subjects["prob"] >= threshold).astype(int)
    recordings["threshold"] = threshold
    recordings["pred"] = (recordings["prob"] >= threshold).astype(int)
    subjects.sort_values("subject").to_csv(
        output_dir / "external_subject_predictions.csv", index=False
    )
    recordings.sort_values("record_id").to_csv(
        output_dir / "external_recording_predictions.csv", index=False
    )

    metrics = compute_binary_metrics(
        subjects["label"].to_numpy(), subjects["prob"].to_numpy(), threshold
    )
    recording_metrics = compute_binary_metrics(
        recordings["label"].to_numpy(), recordings["prob"].to_numpy(), threshold
    )
    metrics_at_05 = compute_binary_metrics(
        subjects["label"].to_numpy(), subjects["prob"].to_numpy(), 0.5
    )
    ev = dict(cfg.get("evaluation", {}))
    cis = {
        metric: bootstrap_metric_ci(
            subjects["label"].to_numpy(),
            subjects["prob"].to_numpy(),
            metric,
            threshold=threshold,
            n_boot=int(ev.get("bootstrap_iterations", 1000)),
            confidence=float(ev.get("bootstrap_confidence", 0.95)),
            seed=int(ev.get("bootstrap_seed", 42)),
        )
        for metric in ("roc_auc", "pr_auc", "f1", "recall_sensitivity", "specificity")
    }
    summary = {
        "experiment": experiment,
        "dataset": "external_community_test_set",
        "n_subjects": int(len(subjects)),
        "n_recordings": int(len(recordings)),
        "n_positive_subjects": int((subjects["label"] == 1).sum()),
        "n_negative_subjects": int((subjects["label"] == 0).sum()),
        "threshold": threshold,
        "threshold_source": "mean of internal validation thresholds; NOT tuned on external data",
        "checkpoint_folds": [int(p) for p in thresholds],
        "subject_metrics": metrics,
        "recording_metrics": recording_metrics,
        "subject_metrics_at_0.5": metrics_at_05,
        "bootstrap": cis,
        "limitation": (
            "The external set contains only 5 positive subjects. Estimates of "
            "sensitivity, AUPRC and F1 therefore have very wide uncertainty and "
            "must not be presented as precise performance. The external set was "
            "not used for model selection or threshold tuning."
        ),
        "note": note
        or "External community scenario test set; kept separate from cross-validation.",
    }
    save_json(summary, output_dir / "external_summary.json")
    pd.DataFrame(
        [
            {"level": "subject", **metrics},
            {"level": "recording", **recording_metrics},
            {"level": "subject_at_0.5", **metrics_at_05},
        ]
    ).to_csv(output_dir / "external_metrics.csv", index=False)

    try:
        from .visualization import (
            plot_confusion_matrix,
            plot_probability_distribution,
            plot_roc_pr,
        )

        plot_confusion_matrix(
            np.array(
                [
                    [metrics["tn"], metrics["fp"]],
                    [metrics["fn"], metrics["tp"]],
                ]
            ),
            output_dir / "external_confusion_matrix.png",
            title="External set - subject level",
        )
        plot_roc_pr(
            subjects["label"].to_numpy(),
            subjects["prob"].to_numpy(),
            output_dir / "external_roc_pr.png",
            title="External community test set (5 positives)",
            threshold=threshold,
            cfg=cfg,
        )
        plot_probability_distribution(
            subjects,
            output_dir / "external_probability_distribution.png",
            threshold=threshold,
            title="External subject-level probabilities",
        )
    except Exception as exc:
        LOGGER.warning("External figures could not be created: %s", exc)

    return summary
