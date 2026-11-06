"""Evaluate Hierarchical Multimodal Cross-Attention 5-Fold Ensemble on External Test Set."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from _common import add_common_arguments, configuration_from_args, initialise
from src.data_loader import load_signal_cache
from src.evaluation import compute_binary_metrics
from src.multimodal_attention import HierarchicalLGCNet, PatientMultiSiteDataset
from src.pipeline import cache_path, windows_path
from src.utils import ensure_dir, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--run-name", default="run")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load external cache and windows
    ext_cache = load_signal_cache(cache_path(cfg, dataset_name="external"))
    ext_pcg = np.load(ext_cache.pcg_path, mmap_mode="r")
    ext_ecg = np.load(ext_cache.ecg_path, mmap_mode="r")

    # Load external windows or prepare them if missing
    ext_win_dir = Path(cfg["_project_root"]) / "data" / "windows" / str(cfg["preprocessing"]["profile"]) / "external" / "e2"
    if not (ext_win_dir / "windows.csv").exists():
        from src.pipeline import prepare_windows, gating_path
        gating_dir = gating_path(cfg, dataset_name="external")
        ext_windows, report = prepare_windows(
            cfg,
            ext_cache,
            experiment="e2",
            gating_dir=gating_dir,
            output_dir=ext_win_dir,
            max_windows_per_recording=None,
        )
    else:
        ext_windows = pd.read_csv(ext_win_dir / "windows.csv")
    ext_meta = ext_cache.metadata.copy()
    ext_meta["subject"] = ext_meta["subject"].astype(str)

    subjects = ext_meta.drop_duplicates("subject")["subject"].tolist()

    dataset = PatientMultiSiteDataset(
        subjects=subjects,
        windows_df=ext_windows,
        metadata_df=ext_meta,
        pcg_memmap=ext_pcg,
        ecg_memmap=ext_ecg,
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model_dir = Path(cfg["_project_root"]) / "results" / "multimodal_attention" / args.run_name

    # Load all 5 fold models
    models = []
    for f in range(5):
        m = HierarchicalLGCNet(embed_dim=128).to(device)
        ckpt = torch.load(model_dir / f"fold_{f}" / "checkpoint.pt", map_location=device, weights_only=True)
        m.load_state_dict(ckpt)
        m.eval()
        models.append(m)

    # Multi-pass ensemble
    all_fold_probs = []
    subjs_all = []
    labels_all = []

    with torch.no_grad():
        for m_idx, model in enumerate(models):
            fold_pass_probs = []
            for p in range(5):
                pass_probs = []
                for batch in loader:
                    if m_idx == 0 and p == 0:
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
                    pass_probs.extend(out["p_final"][:, 1].cpu().numpy())
                fold_pass_probs.append(pass_probs)
            all_fold_probs.append(np.mean(fold_pass_probs, axis=0))

    ensemble_probs = np.mean(all_fold_probs, axis=0)
    y_true = np.array(labels_all, dtype=int)
    y_prob = np.array(ensemble_probs, dtype=float)

    results_df = pd.DataFrame({
        "subject": [str(s) for s in subjs_all],
        "label": y_true,
        "prob": y_prob,
    })

    ext_out_dir = ensure_dir(model_dir / "external")
    results_df.to_csv(ext_out_dir / "external_subject_predictions.csv", index=False)

    m_20 = compute_binary_metrics(y_true, y_prob, 0.20)
    m_10 = compute_binary_metrics(y_true, y_prob, 0.10)
    m_50 = compute_binary_metrics(y_true, y_prob, 0.50)

    summary = {
        "dataset": "community_scenario_external",
        "n_subjects": len(y_true),
        "n_positives": int((y_true == 1).sum()),
        "n_negatives": int((y_true == 0).sum()),
        "auroc": float(m_20["roc_auc"]),
        "auprc": float(m_20["pr_auc"]),
        "metrics_tau_0.20": m_20,
        "metrics_tau_0.10": m_10,
        "metrics_tau_0.50": m_50,
    }
    save_json(summary, ext_out_dir / "external_summary.json")

    print("\n" + "=" * 78)
    print("EXTERNAL COMMUNITY TEST SET EVALUATION (5-FOLD ENSEMBLE)")
    print("=" * 78)
    print(f"Total Subjects   : {len(y_true)} (5 HFrEF, 115 Non-HFrEF)")
    print(f"External AUROC   : {m_20['roc_auc']:.3f}")
    print(f"External AUPRC   : {m_20['pr_auc']:.3f}")
    print(f"At tau = 0.20:")
    print(f"  Sensitivity    : {m_20['recall_sensitivity'] * 100:.1f}% ({m_20['tp']}/5 caught)")
    print(f"  Specificity    : {m_20['specificity'] * 100:.1f}%")
    print(f"  Accuracy       : {m_20['accuracy'] * 100:.1f}%")
    print(f"At tau = 0.10:")
    print(f"  Sensitivity    : {m_10['recall_sensitivity'] * 100:.1f}% ({m_10['tp']}/5 caught)")
    print(f"  Specificity    : {m_10['specificity'] * 100:.1f}%")
    print(f"  Accuracy       : {m_10['accuracy'] * 100:.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
