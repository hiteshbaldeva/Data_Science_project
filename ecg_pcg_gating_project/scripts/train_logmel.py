"""Train 2D Log-Mel Spectrogram CNN for ECG-gated PCG HFrEF Classification."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _common import add_common_arguments, configuration_from_args, initialise
from src.data_loader import load_signal_cache
from src.evaluation import (
    aggregate_windows_to_recordings,
    aggregate_windows_to_subjects,
    compute_binary_metrics,
    format_metric,
    select_threshold,
    summarise_predictions,
)
from src.logmel import LogMelCNN, LogMelDataset, LogMelExtractor
from src.model import count_parameters
from src.pipeline import cache_path, results_path, windows_path
from src.training import compute_class_weights, make_subject_splits
from src.utils import ensure_dir, save_json, set_global_seed

LOGGER = logging.getLogger("scripts.train_logmel")


def train_one_fold(
    fold: int,
    windows: pd.DataFrame,
    cache,
    cfg: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    train_cfg = cfg.get("training", {})
    ev_cfg = cfg.get("evaluation", {})
    seed = 42 + fold
    set_global_seed(seed)

    train_df, val_df, test_df, split_info = make_subject_splits(
        windows,
        test_fold=fold,
        val_fraction=float(train_cfg.get("val_fraction", 0.15)),
        seed=seed,
        min_val_positives=1,
    )

    extractor = LogMelExtractor(fs=1440, n_mels=32, nperseg=128, noverlap=64)
    train_ds = LogMelDataset(train_df, cache, extractor)
    val_ds = LogMelDataset(val_df, cache, extractor)
    test_ds = LogMelDataset(test_df, cache, extractor)

    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = LogMelCNN(in_channels=1, num_classes=2, dropout=0.2).to(device)
    n_params = count_parameters(model)
    LOGGER.info("Fold %d | Model LogMelCNN | %d parameters", fold, n_params)

    # Square-root class weighting
    class_weights = compute_class_weights(train_df["label"].to_numpy(), mode="sqrt").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    epochs = 20
    patience = 6
    best_auprc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    site_weights = train_cfg.get("site_weights") or ev_cfg.get("site_weights")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * len(y)
            n_train += len(y)
        train_loss /= max(n_train, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        val_rows = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += float(loss.detach()) * len(y)
                n_val += len(y)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for i in range(len(probs)):
                    val_rows.append({
                        "subject": batch["subject"][i],
                        "record_id": batch["record_id"][i],
                        "site": batch["site"][i],
                        "fold": int(batch["fold"][i]),
                        "label": int(batch["label"][i]),
                        "prob": float(probs[i]),
                    })
        val_loss /= max(n_val, 1)
        val_df_win = pd.DataFrame(val_rows)
        val_subj = aggregate_windows_to_subjects(val_df_win, site_weights=site_weights)
        val_metrics = compute_binary_metrics(val_subj["label"].to_numpy(), val_subj["prob"].to_numpy(), 0.5)
        auprc = val_metrics["pr_auc"]
        auroc = val_metrics["roc_auc"]
        elapsed = time.time() - t0
        LOGGER.info(
            "fold %d epoch %2d/%d | train %.4f | val %.4f | subj AUROC %.3f | subj AUPRC %.3f | %.1fs",
            fold, epoch, epochs, train_loss, val_loss, auroc, auprc, elapsed
        )

        if auprc > best_auprc:
            best_auprc = auprc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOGGER.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    # Final evaluation on test fold with best state
    model.load_state_dict(best_state)
    model.eval()
    test_rows = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for i in range(len(probs)):
                test_rows.append({
                    "subject": batch["subject"][i],
                    "record_id": batch["record_id"][i],
                    "site": batch["site"][i],
                    "fold": int(batch["fold"][i]),
                    "label": int(batch["label"][i]),
                    "prob": float(probs[i]),
                })
    test_df_win = pd.DataFrame(test_rows)
    val_thresh_info = select_threshold(val_subj["label"].to_numpy(), val_subj["prob"].to_numpy(), metric="f1")
    threshold = float(val_thresh_info["threshold"])

    test_subj = aggregate_windows_to_subjects(test_df_win, site_weights=site_weights)
    test_metrics = compute_binary_metrics(test_subj["label"].to_numpy(), test_subj["prob"].to_numpy(), threshold)

    fold_dir = ensure_dir(output_dir / f"fold_{fold}")
    torch.save(best_state, fold_dir / "checkpoint.pt")
    test_subj.to_csv(fold_dir / "test_subject_predictions.csv", index=False)
    save_json({"threshold": threshold, "metrics": test_metrics, "best_epoch": best_epoch}, fold_dir / "fold_summary.json")

    return {
        "fold": fold,
        "threshold": threshold,
        "test_subjects": test_subj,
        "metrics": test_metrics,
        "n_params": n_params,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--run-name", default="run")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load E2 gated windows
    p = windows_path(cfg, "e2", dataset_name="main")
    windows = pd.read_csv(p / "windows.csv")
    cache = load_signal_cache(cache_path(cfg, dataset_name="main"))

    output_dir = ensure_dir(Path(cfg["_project_root"]) / "results" / "logmel" / args.run_name)

    results = []
    all_test_subjects = []
    for fold in [0, 1, 2, 3, 4]:
        res = train_one_fold(fold, windows, cache, cfg, output_dir, device)
        results.append(res)
        all_test_subjects.append(res["test_subjects"])

    oof_subjects = pd.concat(all_test_subjects, ignore_index=True)
    oof_subjects.to_csv(output_dir / "oof_subject_predictions.csv", index=False)

    mean_thresh = float(np.mean([r["threshold"] for r in results]))
    oof_metrics = compute_binary_metrics(oof_subjects["label"].to_numpy(), oof_subjects["prob"].to_numpy(), mean_thresh)

    fold_df = pd.DataFrame([{"fold": r["fold"], "threshold": r["threshold"], **r["metrics"]} for r in results])
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    summary = {
        "model": "LogMelCNN",
        "parameters": results[0]["n_params"],
        "mean_threshold": mean_thresh,
        "pooled_oof_metrics": oof_metrics,
    }
    save_json(summary, output_dir / "summary.json")
    print("\n" + "=" * 78)
    print("LOG-MEL 2D CNN CROSS-VALIDATION COMPLETE")
    print("=" * 78)
    print(f"Parameters       : {results[0]['n_params']:,d}")
    print(f"Mean threshold   : {mean_thresh:.3f}")
    print(f"Pooled OOF AUROC : {oof_metrics['roc_auc']:.3f}")
    print(f"Pooled OOF AUPRC : {oof_metrics['pr_auc']:.3f}")
    print(f"Pooled OOF F1    : {oof_metrics['f1']:.3f}")
    print(f"Sensitivity      : {oof_metrics['recall_sensitivity']:.3f}")
    print(f"Specificity      : {oof_metrics['specificity']:.3f}")
    print(f"Accuracy         : {oof_metrics['accuracy']:.3f}")
    print(f"Output directory : {output_dir}")


if __name__ == "__main__":
    main()

