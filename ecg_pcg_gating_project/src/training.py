"""Subject-level cross-validation, training and threshold selection.

Design decisions that matter for a valid comparison:

* The official five subject-level folds are used when available.  A
  subject-stratified fallback is only used if fold information is genuinely
  missing, and the fallback is recorded in the output.
* A validation subset is carved out of the *training* folds at the subject
  level.  The test fold is never touched during training or threshold choice.
* E1 and E2 use the same model class, loss, optimiser, schedule, seed policy
  and evaluation code.  Only the window definition differs.
* Normalisation statistics are estimated from the training subset only.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import get_profile
from .dataset import (
    WindowDataset,
    check_no_subject_leakage,
    compute_ecg_stats,
    compute_pcg_stats,
    subject_table,
    subset_by_subjects,
)
from .data_loader import SignalCache
from .evaluation import (
    aggregate_windows_to_recordings,
    aggregate_windows_to_subjects,
    compute_binary_metrics,
    metrics_to_text,
    select_threshold,
    summarise_predictions,
)
from .model import build_model, count_parameters, forward_dummy, model_summary
from .preprocessing import expected_window_length
from .utils import (
    choose_device,
    describe_device,
    ensure_dir,
    save_json,
    set_global_seed,
    timer,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def ensure_folds(
    windows: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Return windows with a complete ``fold`` column.

    Official folds are used when every subject has one.  If any fold is
    missing, subject-level stratified folds are generated and the reason is
    returned so it can be written to the run metadata.  This fallback is never
    silent.
    """
    df = windows.copy()
    if "fold" not in df.columns:
        df["fold"] = np.nan
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    missing = df["fold"].isna()
    if not missing.any():
        return df, "official_folds"

    subjects = (
        df.groupby("subject", as_index=False)
        .agg(label=("label", "first"), fold=("fold", "first"))
        .reset_index(drop=True)
    )
    labels = subjects["label"].fillna(0).astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    fold_map: dict[str, int] = {}
    try:
        for fold, (_, val_idx) in enumerate(splitter.split(subjects, labels)):
            for idx in val_idx:
                fold_map[str(subjects.iloc[int(idx)]["subject"])] = int(fold)
    except ValueError as exc:
        # Extremely small or single-class synthetic data: fall back to a
        # deterministic modulo assignment and record why.
        LOGGER.warning("Stratified fold generation failed (%s); using modulo assignment.", exc)
        fold_map = {
            str(subject): i % int(n_splits)
            for i, subject in enumerate(sorted(subjects["subject"].astype(str)))
        }
        df["fold"] = df["subject"].astype(str).map(fold_map)
        return df, "fallback_modulo_folds"
    df["fold"] = df["subject"].astype(str).map(fold_map)
    return df, "fallback_stratified_subject_folds"


def make_subject_splits(
    windows: pd.DataFrame,
    test_fold: int,
    val_fraction: float = 0.15,
    seed: int = 42,
    min_val_positives: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split windows into train / validation / test at the subject level."""
    df = windows[windows["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["subject"] = df["subject"].astype(str)
    df["fold"] = df["fold"].astype(int)

    test_subjects = sorted(df.loc[df["fold"] == int(test_fold), "subject"].unique())
    if not test_subjects:
        raise ValueError(f"Test fold {test_fold} contains no subjects with windows.")
    trainval = df[df["fold"] != int(test_fold)].copy()
    if trainval.empty:
        raise ValueError(f"No training windows outside fold {test_fold}.")

    info = subject_table(trainval)
    positives = info.loc[info["label"] == 1, "subject"].astype(str).tolist()
    negatives = info.loc[info["label"] == 0, "subject"].astype(str).tolist()
    n = len(info)
    rng = np.random.default_rng(int(seed))

    if n < 4 or val_fraction <= 0:
        val_subjects: list[str] = []
        reason = "validation split disabled (too few training subjects)"
    else:
        n_val = max(2, int(round(float(val_fraction) * n)))
        n_val = min(n_val, n - 2)
        pos_fraction = len(positives) / float(n) if n else 0.0
        n_val_pos = int(round(n_val * pos_fraction))
        if len(positives) >= 2:
            n_val_pos = max(int(min_val_positives), n_val_pos)
            n_val_pos = min(n_val_pos, len(positives) - 1)
        else:
            # Keep the only positive subject in training.
            n_val_pos = 0
        n_val_neg = n_val - n_val_pos
        n_val_neg = min(n_val_neg, max(len(negatives) - 1, 0))
        n_val_pos = min(n_val_pos, len(positives))
        # If one class is exhausted, shrink the validation set rather than
        # pushing a class out of training.
        n_val_total = n_val_pos + n_val_neg
        if n_val_total == 0:
            val_subjects = []
            reason = "no feasible subject-level validation split"
        else:
            chosen_pos = (
                rng.choice(positives, size=n_val_pos, replace=False).tolist()
                if n_val_pos
                else []
            )
            chosen_neg = (
                rng.choice(negatives, size=n_val_neg, replace=False).tolist()
                if n_val_neg
                else []
            )
            val_subjects = sorted(chosen_pos + chosen_neg)
            reason = "subject-level holdout from training folds"

    train_subjects = sorted(set(info["subject"].astype(str)) - set(val_subjects))
    train = subset_by_subjects(trainval, train_subjects)
    val = subset_by_subjects(trainval, val_subjects) if val_subjects else trainval.iloc[0:0].copy()
    test = subset_by_subjects(df, test_subjects)
    check_no_subject_leakage(train, val, test)

    split_info = {
        "test_fold": int(test_fold),
        "n_train_subjects": int(train["subject"].nunique()),
        "n_val_subjects": int(val["subject"].nunique()),
        "n_test_subjects": int(test["subject"].nunique()),
        "n_train_windows": int(len(train)),
        "n_val_windows": int(len(val)),
        "n_test_windows": int(len(test)),
        "val_positive_subjects": int(
            (subject_table(val)["label"] == 1).sum() if len(val) else 0
        ),
        "val_negative_subjects": int(
            (subject_table(val)["label"] == 0).sum() if len(val) else 0
        ),
        "reason": reason,
    }
    return train, val, test, split_info


# ---------------------------------------------------------------------------
# Loss / loader helpers
# ---------------------------------------------------------------------------
def compute_class_weights(labels: Iterable[float], mode: str = "balanced") -> torch.Tensor:
    """Balanced or square-root class weights for CrossEntropyLoss."""
    y = np.asarray(list(labels), dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    n = float(y.size)
    if n == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    present = counts > 0
    weights = np.ones(2, dtype=float)
    if str(mode).lower() == "sqrt":
        weights[present] = np.sqrt(n / (float(present.sum()) * counts[present]))
    else:
        weights[present] = n / (float(present.sum()) * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def _make_loader(
    dataset: WindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    use_cuda: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(1234)
    workers = int(num_workers)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(use_cuda),
        drop_last=False,
        worker_init_fn=_worker_init,
        generator=generator,
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=workers > 0,
    )


def _worker_init(worker_id: int) -> None:
    import random

    seed = int(torch.initial_seed() % (2**31 - 1))
    np.random.seed(seed)
    random.seed(seed)
    # Keep each worker single-threaded so that N loader processes do not
    # oversubscribe the CPU with intra-op thread pools.
    torch.set_num_threads(1)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _forward(model: nn.Module, batch: Mapping[str, Any], model_type: str) -> torch.Tensor:
    if model_type == "fusion":
        return model(batch["x"], batch["ecg"])
    return model(batch["x"])


def _amp_components(use_amp: bool):
    """Return ``(autocast_factory, scaler)`` compatible with PyTorch 2.x."""
    if not use_amp:
        return (lambda: torch.autocast(device_type="cpu", enabled=False)), None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    return (lambda: torch.amp.autocast(device_type="cuda", enabled=True)), scaler


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_windows(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    model_type: str = "pcg",
    use_amp: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Return mean loss and a window-level prediction table."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch["label"].long()
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            logits = _forward(model, batch, model_type)
            loss = criterion(logits, labels) if criterion is not None else None
        if loss is not None:
            total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_samples += int(labels.numel())
        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        label_np = labels.detach().cpu().numpy()
        subjects = batch["subject"]
        records = batch["record_id"]
        sites = batch["site"]
        folds = batch["fold"].detach().cpu().numpy()
        window_ids = batch["window_id"].detach().cpu().numpy()
        starts = batch["start_sec"].detach().cpu().numpy()
        for i in range(probs.size):
            rows.append(
                {
                    "subject": str(subjects[i]),
                    "record_id": str(records[i]),
                    "site": str(sites[i]),
                    "fold": int(folds[i]),
                    "label": int(label_np[i]),
                    "window_id": int(window_ids[i]),
                    "start_sec": float(starts[i]),
                    "prob": float(probs[i]),
                }
            )
    mean_loss = total_loss / max(total_samples, 1)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["experiment"] = ""
    return mean_loss, frame


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str,
    use_amp: bool,
    grad_clip: float,
    scaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch["label"].long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            logits = _forward(model, batch, model_type)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_samples += int(labels.numel())
    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def train_one_fold(
    windows: pd.DataFrame,
    cache: SignalCache,
    cfg: Mapping[str, Any],
    fold: int,
    experiment: str,
    output_dir: str | Path,
    model_type: str | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    save_window_predictions: bool = True,
) -> dict[str, Any]:
    """Train and evaluate one subject-independent fold."""
    output_dir = ensure_dir(output_dir)
    train_cfg = dict(cfg.get("training", {}))
    ev_cfg = dict(cfg.get("evaluation", {}))
    project_cfg = dict(cfg.get("project", {}))
    experiment = experiment.lower()
    model_type = model_type or ("fusion" if experiment == "e3" else "pcg")
    seed = int(seed if seed is not None else train_cfg.get("seed", project_cfg.get("seed", 42)))
    seed = int(seed) + int(fold)

    set_global_seed(seed, deterministic=bool(project_cfg.get("deterministic", True)))
    device = device or choose_device(prefer_cuda=True)
    use_cuda = device.type == "cuda"
    use_amp = bool(train_cfg.get("amp", True)) and use_cuda

    if windows.empty:
        raise ValueError(
            f"No windows available for fold {fold} / experiment {experiment}. "
            "Check the gating and window-index reports."
        )

    train_df, val_df, test_df, split_info = make_subject_splits(
        windows,
        test_fold=int(fold),
        val_fraction=float(train_cfg.get("val_fraction", 0.15)),
        seed=int(train_cfg.get("seed", seed)),
        min_val_positives=int(train_cfg.get("min_val_positives", 1)),
    )
    if len(val_df) == 0:
        LOGGER.warning("Fold %d has no validation windows; threshold falls back to 0.5.", fold)

    max_train = train_cfg.get("max_train_windows")
    if max_train:
        max_train = int(max_train)
        if len(train_df) > max_train:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train_df), size=max_train, replace=False)
            train_df = train_df.iloc[np.sort(idx)].reset_index(drop=True)
            split_info["max_train_windows_applied"] = max_train

    profile = get_profile(cfg)
    input_length = expected_window_length(profile)
    ecg_cfg = dict(cfg.get("preprocessing", {}).get("ecg", {}))
    ecg_input_length = int(round(float(profile["window_seconds"]) * int(ecg_cfg.get("target_fs", 500))))
    mode = "fusion" if model_type == "fusion" else "pcg"

    train_ds = WindowDataset(train_df, cache, cfg, mode=mode, normalize=False)
    val_ds = WindowDataset(val_df, cache, cfg, mode=mode, normalize=False)
    test_ds = WindowDataset(test_df, cache, cfg, mode=mode, normalize=False)

    stats_max = train_cfg.get("stats_max_windows", 5000)
    with timer(f"fold {fold}: normalisation statistics", LOGGER):
        pcg_stats = compute_pcg_stats(
            train_ds,
            max_windows=int(stats_max) if stats_max else None,
            seed=seed,
            source="train",
        )
        ecg_stats = (
            compute_ecg_stats(
                train_ds,
                max_windows=int(stats_max) if stats_max else None,
                seed=seed,
                source="train",
            )
            if mode == "fusion"
            else None
        )
    for ds in (train_ds, val_ds, test_ds):
        ds.stats = pcg_stats
        ds.ecg_stats = ecg_stats
        ds.normalize = True

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    sampler = None
    if bool(train_cfg.get("weighted_sampler", False)):
        labels = train_ds.labels().astype(int)
        counts = np.bincount(labels, minlength=2).astype(float)
        weights = np.asarray([1.0 / max(counts[y], 1.0) for y in labels])
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
    train_loader = _make_loader(
        train_ds, batch_size, True, num_workers, use_cuda, sampler=sampler
    )
    val_loader = _make_loader(val_ds, batch_size, False, num_workers, use_cuda)
    test_loader = _make_loader(test_ds, batch_size, False, num_workers, use_cuda)

    model = build_model(
        cfg,
        input_length=input_length,
        model_type=model_type,
        ecg_input_length=ecg_input_length if mode == "fusion" else None,
    ).to(device)
    info = model.info() if hasattr(model, "info") else None
    n_params = count_parameters(model, trainable_only=False)

    site_weights = train_cfg.get("site_weights") or ev_cfg.get("site_weights")

    class_weights = None
    if bool(train_cfg.get("class_weight", True)):
        class_weights = compute_class_weights(
            train_ds.labels(),
            mode=str(train_cfg.get("class_weight_mode", "balanced")),
        ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    lr = float(train_cfg.get("lr", 1e-3))
    if str(train_cfg.get("optimizer", "adamw")).lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    monitor = str(train_cfg.get("monitor", "auprc")).lower()
    scheduler = None
    if str(train_cfg.get("scheduler", "reduce_on_plateau")).lower() != "none":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if monitor == "auprc" else "min",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 3)),
            min_lr=1e-6,
        )

    autocast_factory, scaler = _amp_components(use_amp)
    epochs = int(train_cfg.get("epochs", 25))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 7))

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_monitor = -np.inf if monitor == "auprc" else np.inf
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    min_windows = int(ev_cfg.get("min_windows_per_subject", 1))

    LOGGER.info(
        "Fold %d | train %d subjects / %d windows | val %d / %d | test %d / %d",
        fold,
        split_info["n_train_subjects"],
        split_info["n_train_windows"],
        split_info["n_val_subjects"],
        split_info["n_val_windows"],
        split_info["n_test_subjects"],
        split_info["n_test_windows"],
    )
    LOGGER.info(
        "Model %s | %d parameters | pool=%s | profile=%s | device=%s",
        type(model).__name__,
        n_params,
        getattr(info, "pool_mode", "n/a"),
        profile["name"],
        describe_device(device),
    )

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss = _run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            model_type,
            use_amp,
            grad_clip,
            scaler,
        )
        val_loss, val_windows = predict_windows(
            model, val_loader, criterion, device, model_type, use_amp
        )
        val_subjects = aggregate_windows_to_subjects(val_windows, min_windows=min_windows, site_weights=site_weights)
        val_window_metrics = compute_binary_metrics(
            val_windows["label"].to_numpy(),
            val_windows["prob"].to_numpy(),
            0.5,
        ) if len(val_windows) else {"roc_auc": float("nan"), "pr_auc": float("nan")}
        val_subject_metrics = compute_binary_metrics(
            val_subjects["label"].to_numpy(),
            val_subjects["prob"].to_numpy(),
            0.5,
        ) if len(val_subjects) else {"roc_auc": float("nan"), "pr_auc": float("nan"), "f1": float("nan")}

        if monitor == "auprc":
            monitor_value = float(val_subject_metrics.get("pr_auc", float("nan")))
            if not np.isfinite(monitor_value):
                monitor_value = float(val_window_metrics.get("pr_auc", -np.inf))
            improved = monitor_value > best_monitor + 1e-6
        else:
            monitor_value = float(val_loss)
            improved = monitor_value < best_monitor - 1e-6

        if scheduler is not None:
            scheduler.step(monitor_value)

        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_window_auroc": float(val_window_metrics.get("roc_auc", float("nan"))),
            "val_window_auprc": float(val_window_metrics.get("pr_auc", float("nan"))),
            "val_subject_auroc": float(val_subject_metrics.get("roc_auc", float("nan"))),
            "val_subject_auprc": float(val_subject_metrics.get("pr_auc", float("nan"))),
            "val_subject_f1_at_0.5": float(val_subject_metrics.get("f1", float("nan"))),
            "monitor_value": float(monitor_value),
            "seconds": float(elapsed),
        }
        history.append(row)
        LOGGER.info(
            "fold %d epoch %2d/%d | train %.4f | val %.4f | subj AUROC %s | subj AUPRC %s | %.1fs",
            fold,
            epoch,
            epochs,
            train_loss,
            val_loss,
            f"{row['val_subject_auroc']:.3f}" if np.isfinite(row["val_subject_auroc"]) else "NA",
            f"{row['val_subject_auprc']:.3f}" if np.isfinite(row["val_subject_auprc"]) else "NA",
            elapsed,
        )

        if improved:
            best_monitor = monitor_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOGGER.info("Early stopping at epoch %d (best epoch %d).", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    val_loss, val_windows = predict_windows(
        model, val_loader, criterion, device, model_type, use_amp
    )
    test_loss, test_windows = predict_windows(
        model, test_loader, criterion, device, model_type, use_amp
    )
    val_subjects = aggregate_windows_to_subjects(val_windows, min_windows=min_windows, site_weights=site_weights)
    test_subjects = aggregate_windows_to_subjects(test_windows, min_windows=min_windows, site_weights=site_weights)
    val_recordings = aggregate_windows_to_recordings(val_windows)
    test_recordings = aggregate_windows_to_recordings(test_windows)

    threshold_metric = str(train_cfg.get("threshold_metric", "f1"))
    if len(val_subjects):
        threshold_info = select_threshold(
            val_subjects["label"].to_numpy(),
            val_subjects["prob"].to_numpy(),
            metric=threshold_metric,
            cfg=cfg,
            fallback=0.5,
        )
    else:
        threshold_info = {
            "threshold": 0.5,
            "metric": threshold_metric,
            "score": float("nan"),
            "reason": "empty validation set",
        }
    threshold = float(threshold_info["threshold"])

    test_metrics = compute_binary_metrics(
        test_subjects["label"].to_numpy(), test_subjects["prob"].to_numpy(), threshold
    ) if len(test_subjects) else {}
    test_metrics_at_05 = compute_binary_metrics(
        test_subjects["label"].to_numpy(), test_subjects["prob"].to_numpy(), 0.5
    ) if len(test_subjects) else {}
    val_metrics = compute_binary_metrics(
        val_subjects["label"].to_numpy(), val_subjects["prob"].to_numpy(), threshold
    ) if len(val_subjects) else {}
    test_recording_metrics = compute_binary_metrics(
        test_recordings["label"].to_numpy(), test_recordings["prob"].to_numpy(), threshold
    ) if len(test_recordings) else {}

    for frame in (val_subjects, test_subjects, val_recordings, test_recordings):
        if len(frame):
            frame["experiment"] = experiment
            frame["profile"] = profile["name"]
            frame["threshold"] = threshold
            frame["pred"] = (frame["prob"] >= threshold).astype(int)
    for frame in (val_windows, test_windows):
        if len(frame):
            frame["experiment"] = experiment
            frame["profile"] = profile["name"]
            frame["threshold"] = threshold
            frame["pred"] = (frame["prob"] >= threshold).astype(int)
            frame["fold"] = int(fold)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    val_subjects.to_csv(output_dir / "validation_subject_predictions.csv", index=False)
    test_subjects.to_csv(output_dir / "test_subject_predictions.csv", index=False)
    test_recordings.to_csv(output_dir / "test_recording_predictions.csv", index=False)
    if save_window_predictions:
        test_windows.to_csv(output_dir / "test_window_predictions.csv", index=False)
        val_windows.to_csv(output_dir / "validation_window_predictions.csv", index=False)

    checkpoint = {
        "model_state_dict": best_state,
        "model_type": model_type,
        "model_info": info.to_dict() if info is not None else {},
        "input_length": int(input_length),
        "ecg_input_length": int(ecg_input_length),
        "profile_name": profile["name"],
        "profile": profile,
        "normalization": pcg_stats.to_dict(),
        "ecg_normalization": ecg_stats.to_dict() if ecg_stats is not None else None,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "fold": int(fold),
        "experiment": experiment,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_monitor": float(best_monitor),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "model_num_classes": int(cfg.get("model", {}).get("num_classes", 2)),
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    fold_metrics = {
        "fold": int(fold),
        "experiment": experiment,
        "profile": profile["name"],
        "model_type": model_type,
        "n_parameters": int(n_params),
        "best_epoch": int(best_epoch),
        "threshold": threshold,
        "threshold_info": threshold_info,
        "split": split_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_0.5": test_metrics_at_05,
        "test_recording_metrics": test_recording_metrics,
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
    }
    save_json(fold_metrics, output_dir / "fold_metrics.json")
    save_json(split_info, output_dir / "split_summary.json")

    try:
        from .visualization import plot_training_curves

        plot_training_curves(history_df, output_dir / "training_curves.png", title=f"Fold {fold}")
    except Exception as exc:  # plotting must never abort training
        LOGGER.warning("Could not create training-curve figure: %s", exc)

    return {
        "fold": int(fold),
        "checkpoint": str(checkpoint_path),
        "history": history_df,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "val_recordings": val_recordings,
        "test_recordings": test_recordings,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_0.5": test_metrics_at_05,
        "test_recording_metrics": test_recording_metrics,
        "split": split_info,
        "best_epoch": int(best_epoch),
        "n_parameters": int(n_params),
        "profile": profile["name"],
        "experiment": experiment,
        "model_type": model_type,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def run_cross_validation(
    windows: pd.DataFrame,
    cache: SignalCache,
    cfg: Mapping[str, Any],
    experiment: str,
    output_dir: str | Path,
    folds: Sequence[int] | None = None,
    model_type: str | None = None,
    seed: int | None = None,
    save_window_predictions: bool = True,
) -> dict[str, Any]:
    """Run the official five-fold cross-validation for one experiment."""
    output_dir = ensure_dir(output_dir)
    data_cfg = dict(cfg.get("data", {}))
    folds = list(folds) if folds is not None else list(data_cfg.get("folds", [0, 1, 2, 3, 4]))
    windows = windows.copy()
    windows, fold_source = ensure_folds(
        windows,
        n_splits=len(folds),
        seed=int(seed if seed is not None else cfg.get("training", {}).get("seed", 42)),
    )
    if fold_source != "official_folds":
        LOGGER.warning(
            "Official folds were incomplete; using %s. This is recorded in cv_summary.json.",
            fold_source,
        )

    results: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = ensure_dir(output_dir / f"fold_{fold}")
        LOGGER.info("=" * 78)
        LOGGER.info("Experiment %s | fold %d", experiment, fold)
        LOGGER.info("=" * 78)
        result = train_one_fold(
            windows,
            cache,
            cfg,
            fold=fold,
            experiment=experiment,
            output_dir=fold_dir,
            model_type=model_type,
            seed=seed,
            save_window_predictions=save_window_predictions,
        )
        results.append(result)

    # ------------------------------------------------------------------
    # Aggregate artefacts
    # ------------------------------------------------------------------
    fold_rows: list[dict[str, Any]] = []
    val_subject_frames = []
    test_subject_frames = []
    test_recording_frames = []
    test_window_frames = []
    history_frames = []
    thresholds = []
    for result in results:
        fold = result["fold"]
        base = {
            "fold": fold,
            "experiment": result["experiment"],
            "profile": result["profile"],
            "model_type": result["model_type"],
            "n_parameters": result["n_parameters"],
            "best_epoch": result["best_epoch"],
            "threshold": result["threshold"],
            "threshold_metric": result["threshold_info"].get("metric"),
            "threshold_score": result["threshold_info"].get("score"),
        }
        test_metrics = result["test_metrics"]
        rec_metrics = result["test_recording_metrics"]
        val_metrics = result["val_metrics"]
        fold_rows.append(
            {
                **base,
                "split": "test",
                **{f"subject_{k}": v for k, v in test_metrics.items()},
                **{f"recording_{k}": v for k, v in rec_metrics.items()},
            }
        )
        fold_rows.append(
            {
                **base,
                "split": "validation",
                **{f"subject_{k}": v for k, v in val_metrics.items()},
            }
        )
        thresholds.append({"fold": fold, **base, **result["threshold_info"]})
        if len(result["val_subjects"]):
            frame = result["val_subjects"].copy()
            frame["fold"] = fold
            frame["split"] = "validation"
            val_subject_frames.append(frame)
        if len(result["test_subjects"]):
            frame = result["test_subjects"].copy()
            frame["fold"] = fold
            frame["split"] = "test"
            test_subject_frames.append(frame)
        if len(result["test_recordings"]):
            frame = result["test_recordings"].copy()
            frame["fold"] = fold
            frame["split"] = "test"
            test_recording_frames.append(frame)
        if len(result["test_windows"]):
            test_window_frames.append(result["test_windows"])
        if len(result["history"]):
            history_frames.append(result["history"])

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output_dir / "thresholds.csv", index=False)

    all_subject_predictions = (
        pd.concat(val_subject_frames + test_subject_frames, ignore_index=True)
        if (val_subject_frames or test_subject_frames)
        else pd.DataFrame()
    )
    all_subject_predictions.to_csv(
        output_dir / "subject_level_predictions.csv", index=False
    )
    oof = pd.concat(test_subject_frames, ignore_index=True) if test_subject_frames else pd.DataFrame()
    oof.to_csv(output_dir / "oof_subject_predictions.csv", index=False)
    recordings = (
        pd.concat(test_recording_frames, ignore_index=True)
        if test_recording_frames
        else pd.DataFrame()
    )
    recordings.to_csv(output_dir / "recording_level_predictions.csv", index=False)
    if test_window_frames:
        pd.concat(test_window_frames, ignore_index=True).to_csv(
            output_dir / "window_predictions_test.csv", index=False
        )
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(
            output_dir / "training_history.csv", index=False
        )

    # Pooled out-of-fold subject-level metrics.  Each subject is a test subject
    # exactly once, so this is a valid (though fold-mixed) summary.
    pooled_rows: list[dict[str, Any]] = []
    if len(oof):
        pooled_rows = summarise_predictions(
            experiment,
            oof,
            cfg=cfg,
            threshold=float(np.mean([r["threshold"] for r in results])),
            cohort="pooled_oof",
            recording_predictions=recordings if len(recordings) else None,
        )
        pd.DataFrame(pooled_rows).to_csv(output_dir / "oof_metrics.csv", index=False)

    # Mean and standard deviation across folds (test only).
    test_only = fold_metrics[fold_metrics["split"] == "test"]
    summary: dict[str, Any] = {
        "experiment": experiment,
        "profile": results[0]["profile"],
        "model_type": results[0]["model_type"],
        "n_parameters": results[0]["n_parameters"],
        "folds": [int(r["fold"]) for r in results],
        "fold_source": fold_source,
        "n_subjects_total": int(len(oof)) if len(oof) else 0,
        "pooled_oof_metrics": pooled_rows[0] if pooled_rows else {},
        "per_fold_test_metrics": test_only.to_dict(orient="records"),
        "thresholds": [float(r["threshold"]) for r in results],
        "mean_threshold": float(np.mean([r["threshold"] for r in results])),
        "device": results[0]["device"],
        "config": {
            "profile": results[0]["profile"],
            "experiment": experiment,
            "model_type": results[0]["model_type"],
        },
    }
    for metric in (
        "subject_roc_auc",
        "subject_pr_auc",
        "subject_f1",
        "subject_recall_sensitivity",
        "subject_specificity",
    ):
        if metric in test_only.columns:
            values = pd.to_numeric(test_only[metric], errors="coerce").dropna()
            summary[f"mean_{metric}"] = float(values.mean()) if len(values) else float("nan")
            summary[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) else float("nan")
    save_json(summary, output_dir / "cv_summary.json")

    try:
        from .visualization import plot_fold_metrics

        plot_fold_metrics(fold_metrics, output_dir / "fold_metrics.png")
    except Exception as exc:
        LOGGER.warning("Could not create fold metric figure: %s", exc)

    return {
        "results": results,
        "fold_metrics": fold_metrics,
        "subject_predictions": all_subject_predictions,
        "oof_subject_predictions": oof,
        "recording_predictions": recordings,
        "summary": summary,
        "output_dir": str(output_dir),
        "fold_source": fold_source,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a checkpoint created by :func:`train_one_fold`."""
    device = torch.device(device)
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(str(path), map_location=device)


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    cfg: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Rebuild a model and load its weights from a checkpoint."""
    model = build_model(
        cfg,
        input_length=int(checkpoint["input_length"]),
        model_type=str(checkpoint.get("model_type", "pcg")),
        ecg_input_length=int(checkpoint.get("ecg_input_length", 0)) or None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(torch.device(device))
