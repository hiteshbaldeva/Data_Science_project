"""Train Hierarchical Multimodal Cross-Attention Network across 5 Folds.

Uses:
1. 4-Site Synchronous PCG (Log-Mel 2D) + ECG (1D) per patient.
2. Cross-Modality Cross-Attention (EMAT detection).
3. 4-Site Auscultation Fusion (APEX, LLSB, LUSB, RUSB).
4. Deep Multi-Task Loss with Decision-Level Ensemble.
5. Dual-Threshold Triage Evaluation (>90% Precision & Recall).
"""

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
from src.evaluation import compute_binary_metrics, select_threshold
from src.multimodal_attention import HierarchicalLGCNet, PatientMultiSiteDataset
from src.pipeline import cache_path, windows_path
from src.training import compute_class_weights
from src.utils import ensure_dir, save_json, set_global_seed

LOGGER = logging.getLogger("scripts.train_multimodal_attention")


def evaluate_subjects(
    model: nn.Module,
    dataset: PatientMultiSiteDataset,
    device: torch.device,
    n_passes: int = 5,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Evaluates subjects by averaging predictions across multiple random window passes."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Accumulate probabilities for each subject across passes
    subjs_all = []
    labels_all = []
    probs_accum = []

    with torch.no_grad():
        for p in range(n_passes):
            pass_probs = []
            for batch in loader:
                if p == 0:
                    subjs_all.extend(batch["subject"])
                    labels_all.extend(batch["label"].numpy())

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    out = model(
                        ecg_apex=batch["ecg_apex"].to(device, non_blocking=True),
                        pcg_apex=batch["pcg_apex"].to(device, non_blocking=True),
                        ecg_llsb=batch["ecg_llsb"].to(device, non_blocking=True),
                        pcg_llsb=batch["pcg_llsb"].to(device, non_blocking=True),
                        ecg_lusb=batch["ecg_lusb"].to(device, non_blocking=True),
                        pcg_lusb=batch["pcg_lusb"].to(device, non_blocking=True),
                        ecg_rusb=batch["ecg_rusb"].to(device, non_blocking=True),
                        pcg_rusb=batch["pcg_rusb"].to(device, non_blocking=True),
                    )
                p_pos = out["p_final"][:, 1].cpu().numpy()
                pass_probs.extend(p_pos)
            probs_accum.append(pass_probs)

    # Average across passes
    mean_probs = np.mean(probs_accum, axis=0)
    return pd.DataFrame({
        "subject": [str(s) for s in subjs_all],
        "label": np.array(labels_all, dtype=int),
        "prob": np.array(mean_probs, dtype=float),
    })


def train_one_fold(
    fold: int,
    subjects_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    pcg_memmap: np.ndarray,
    ecg_memmap: np.ndarray,
    output_dir: Path,
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 32,
) -> dict:
    seed = 42 + fold
    set_global_seed(seed)

    # Split subjects: fold is test, remaining are train/val
    test_subjs = subjects_df[subjects_df["fold"] == fold]["subject"].tolist()
    dev_df = subjects_df[subjects_df["fold"] != fold].reset_index(drop=True)

    # Stratified validation split (15% of dev subjects)
    rng = np.random.default_rng(seed)
    pos_dev = dev_df[dev_df["label"] == 1]["subject"].tolist()
    neg_dev = dev_df[dev_df["label"] == 0]["subject"].tolist()
    rng.shuffle(pos_dev)
    rng.shuffle(neg_dev)

    n_val_pos = max(1, int(len(pos_dev) * 0.15))
    n_val_neg = max(1, int(len(neg_dev) * 0.15))

    val_subjs = pos_dev[:n_val_pos] + neg_dev[:n_val_neg]
    train_subjs = pos_dev[n_val_pos:] + neg_dev[n_val_neg:]

    train_ds = PatientMultiSiteDataset(
        train_subjs, windows_df, metadata_df, pcg_memmap, ecg_memmap, is_train=True, seed=seed
    )
    val_ds = PatientMultiSiteDataset(
        val_subjs, windows_df, metadata_df, pcg_memmap, ecg_memmap, is_train=False, seed=seed
    )
    test_ds = PatientMultiSiteDataset(
        test_subjs, windows_df, metadata_df, pcg_memmap, ecg_memmap, is_train=False, seed=seed
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = HierarchicalLGCNet(embed_dim=128, dropout=0.2).to(device)
    train_labels = [train_ds.labels[s] for s in train_subjs]
    class_weights = compute_class_weights(np.array(train_labels), mode="sqrt").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auroc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 8
    bad_epochs = 0

    LOGGER.info(
        "Fold %d | Train subjs: %d (pos: %d) | Val subjs: %d (pos: %d) | Test subjs: %d (pos: %d)",
        fold, len(train_subjs), sum(train_labels), len(val_subjs), n_val_pos, len(test_subjs),
        int(metadata_df[metadata_df['subject'].isin(test_subjs)]['label'].iloc[0])
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(
                    ecg_apex=batch["ecg_apex"].to(device, non_blocking=True),
                    pcg_apex=batch["pcg_apex"].to(device, non_blocking=True),
                    ecg_llsb=batch["ecg_llsb"].to(device, non_blocking=True),
                    pcg_llsb=batch["pcg_llsb"].to(device, non_blocking=True),
                    ecg_lusb=batch["ecg_lusb"].to(device, non_blocking=True),
                    pcg_lusb=batch["pcg_lusb"].to(device, non_blocking=True),
                    ecg_rusb=batch["ecg_rusb"].to(device, non_blocking=True),
                    pcg_rusb=batch["pcg_rusb"].to(device, non_blocking=True),
                )
                # Deep auxiliary supervision (Equation 15)
                l_apex = criterion(out["logits_apex"], y)
                l_llsb = criterion(out["logits_llsb"], y)
                l_lusb = criterion(out["logits_lusb"], y)
                l_rusb = criterion(out["logits_rusb"], y)
                l_cs = criterion(out["logits_cs"], y)
                l_joint = criterion(out["logits_joint"], y)

                loss = l_apex + l_llsb + l_lusb + l_rusb + l_cs + l_joint

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.detach()) * len(y)
            n_train += len(y)

        train_loss /= max(n_train, 1)

        # Validation on Val set
        val_eval = evaluate_subjects(model, val_ds, device, n_passes=3, batch_size=batch_size)
        val_metrics = compute_binary_metrics(val_eval["label"].to_numpy(), val_eval["prob"].to_numpy(), 0.20)
        auroc = val_metrics["roc_auc"]
        auprc = val_metrics["pr_auc"]
        elapsed = time.time() - t0

        LOGGER.info(
            "fold %d epoch %2d/%d | train loss %.4f | val AUROC %.3f | val AUPRC %.3f | val Sens %.2f | val Spec %.2f | %.1fs",
            fold, epoch, epochs, train_loss, auroc, auprc, val_metrics["recall_sensitivity"], val_metrics["specificity"], elapsed
        )

        if auroc > best_auroc:
            best_auroc = auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOGGER.info("Early stopping at epoch %d (best epoch %d with AUROC %.3f)", epoch, best_epoch, best_auroc)
                break

    # Evaluate best model on test fold
    model.load_state_dict(best_state)
    test_eval = evaluate_subjects(model, test_ds, device, n_passes=10, batch_size=batch_size)
    test_eval["fold"] = fold

    # Standard metrics at tau = 0.20
    test_metrics_20 = compute_binary_metrics(test_eval["label"].to_numpy(), test_eval["prob"].to_numpy(), 0.20)

    fold_dir = ensure_dir(output_dir / f"fold_{fold}")
    torch.save(best_state, fold_dir / "checkpoint.pt")
    test_eval.to_csv(fold_dir / "test_subject_predictions.csv", index=False)
    save_json({
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "metrics_tau_0.20": test_metrics_20,
    }, fold_dir / "fold_summary.json")

    return {
        "fold": fold,
        "test_subjects": test_eval,
        "metrics": test_metrics_20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--run-name", default="multimodal_attention")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cache and windows
    cache = load_signal_cache(cache_path(cfg, dataset_name="main"))
    pcg_mem = np.load(cache.pcg_path, mmap_mode="r")
    ecg_mem = np.load(cache.ecg_path, mmap_mode="r")

    p = windows_path(cfg, "e2", dataset_name="main")
    windows_df = pd.read_csv(p / "windows.csv")
    meta_df = cache.metadata.copy()
    meta_df["subject"] = meta_df["subject"].astype(str)

    # Subject unique table
    subjects_df = meta_df.drop_duplicates("subject")[["subject", "fold", "label"]].reset_index(drop=True)

    output_dir = ensure_dir(Path(cfg["_project_root"]) / "results" / "multimodal_attention" / args.run_name)

    all_test_subjects = []
    fold_results = []

    for fold in [0, 1, 2, 3, 4]:
        res = train_one_fold(
            fold=fold,
            subjects_df=subjects_df,
            windows_df=windows_df,
            metadata_df=meta_df,
            pcg_memmap=pcg_mem,
            ecg_memmap=ecg_mem,
            output_dir=output_dir,
            device=device,
            epochs=25,
            batch_size=32,
        )
        fold_results.append(res)
        all_test_subjects.append(res["test_subjects"])

    # Pooled Out-Of-Fold evaluation
    oof_df = pd.concat(all_test_subjects, ignore_index=True)
    oof_df.to_csv(output_dir / "oof_subject_predictions.csv", index=False)

    y_true = oof_df["label"].to_numpy()
    y_prob = oof_df["prob"].to_numpy()

    # Multi-threshold sweep
    sweep_results = []
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        m = compute_binary_metrics(y_true, y_prob, tau)
        sweep_results.append({"threshold": tau, **m})
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)

    # Dual-Threshold Triage Evaluation (>90% target)
    tau_low = 0.15
    tau_high = 0.65

    rule_in_mask = y_prob >= tau_high
    rule_out_mask = y_prob <= tau_low
    decisive_mask = rule_in_mask | rule_out_mask

    tp_rule_in = int(((y_true == 1) & rule_in_mask).sum())
    fp_rule_in = int(((y_true == 0) & rule_in_mask).sum())
    rule_in_precision = float(tp_rule_in / max(tp_rule_in + fp_rule_in, 1))

    tn_rule_out = int(((y_true == 0) & rule_out_mask).sum())
    fn_rule_out = int(((y_true == 1) & rule_out_mask).sum())
    rule_out_npv = float(tn_rule_out / max(tn_rule_out + fn_rule_out, 1))
    rule_out_recall_loss = float(fn_rule_out / max((y_true == 1).sum(), 1))

    decisive_y_true = y_true[decisive_mask]
    decisive_y_pred = (y_prob[decisive_mask] >= 0.5).astype(int)
    decisive_metrics = compute_binary_metrics(decisive_y_true, y_prob[decisive_mask], 0.50)

    triage_summary = {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "total_subjects": len(y_true),
        "decisive_subjects": int(decisive_mask.sum()),
        "indeterminate_subjects": int((~decisive_mask).sum()),
        "indeterminate_percentage": float((~decisive_mask).sum() / len(y_true) * 100),
        "rule_in_precision": rule_in_precision,
        "rule_in_tp": tp_rule_in,
        "rule_in_fp": fp_rule_in,
        "rule_out_npv": rule_out_npv,
        "decisive_accuracy": decisive_metrics["accuracy"],
        "decisive_precision": decisive_metrics["precision"],
        "decisive_recall": decisive_metrics["recall_sensitivity"],
        "decisive_specificity": decisive_metrics["specificity"],
        "decisive_f1": decisive_metrics["f1"],
    }
    save_json(triage_summary, output_dir / "triage_summary.json")

    # Pooled metrics at tau = 0.20
    m_20 = compute_binary_metrics(y_true, y_prob, 0.20)
    save_json({
        "pooled_metrics_tau_0.20": m_20,
        "triage_summary": triage_summary,
    }, output_dir / "final_summary.json")

    print("\n" + "=" * 78)
    print("HIERARCHICAL MULTIMODAL CROSS-ATTENTION 5-FOLD COMPLETE")
    print("=" * 78)
    print(f"Pooled AUROC           : {m_20['roc_auc']:.3f}")
    print(f"Pooled AUPRC           : {m_20['pr_auc']:.3f}")
    print(f"At tau = 0.20:")
    print(f"  Recall (Sensitivity) : {m_20['recall_sensitivity']:.3f}")
    print(f"  Specificity          : {m_20['specificity']:.3f}")
    print(f"  Precision            : {m_20['precision']:.3f}")
    print(f"  Accuracy             : {m_20['accuracy']:.3f}")
    print("-" * 78)
    print("Dual-Threshold Triage (Rule-In >= 0.65, Rule-Out <= 0.15):")
    print(f"  Decisive Subjects    : {triage_summary['decisive_subjects']} / {triage_summary['total_subjects']} ({100 - triage_summary['indeterminate_percentage']:.1f}%)")
    print(f"  Decisive Precision   : {decisive_metrics['precision'] * 100:.1f}%")
    print(f"  Decisive Recall      : {decisive_metrics['recall_sensitivity'] * 100:.1f}%")
    print(f"  Decisive Specificity : {decisive_metrics['specificity'] * 100:.1f}%")
    print(f"  Rule-In Precision    : {rule_in_precision * 100:.1f}%")
    print(f"  Rule-Out NPV         : {rule_out_npv * 100:.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
