"""Dataset discovery, label parsing and structural validation.

The Zenodo archive does not always extract into the exact folder layout shown
on the website.  In particular, the downloaded folders can be nested one level
deeper.  This module therefore discovers files by content (WAV names and label
CSVs) rather than assuming a fixed path.

Expected WAV naming convention (from the dataset description)::

    [ID]_[APEX/LLSB/LUSB/RUSB]_[ECG/PCG].wav

Labels are read from ``*label*.csv``.  The official ``fold_i_label.csv`` files
contain one row per subject/site with the LVEF as a fraction (for example
``0.35``); the community test CSV contains one row per subject.  Both layouts
are accepted.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import save_json

LOGGER = logging.getLogger(__name__)

SITE_PATTERN = r"(?P<site>APEX|LLSB|LUSB|RUSB)"
MODALITY_PATTERN = r"(?P<modality>ECG|PCG)"
_WAV_RE = re.compile(
    rf"^(?P<subject>.+)_{SITE_PATTERN}_{MODALITY_PATTERN}$", re.IGNORECASE
)
_FOLD_RE = re.compile(r"fold[_-]?(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RecordingFile:
    path: Path
    subject: str
    site: str
    modality: str
    fold: int | None = None


@dataclass
class RecordingPair:
    subject: str
    site: str
    fold: int | None
    ecg_path: Path
    pcg_path: Path
    label: int | None = None
    lvef_percent: float | None = None


@dataclass
class SubjectInfo:
    subject: str
    label: int | None
    lvef_percent: float | None
    fold: int | None
    sites: set[str] = field(default_factory=set)


@dataclass
class DatasetIndex:
    root: Path
    external: bool
    pairs: list[RecordingPair]
    subjects: dict[str, SubjectInfo]
    unpaired: list[dict[str, Any]]
    unparsed: list[Path]
    label_sources: list[Path]
    warnings: list[str]

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    def label_counts(self) -> dict[str, int]:
        labels = [s.label for s in self.subjects.values() if s.label is not None]
        return {
            "positive": int(sum(labels)),
            "negative": int(len(labels) - sum(labels)),
            "labelled": int(len(labels)),
        }

    def fold_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for subject in self.subjects.values():
            key = str(subject.fold) if subject.fold is not None else "unknown"
            bucket = out.setdefault(key, {"subjects": 0, "positive": 0, "negative": 0})
            bucket["subjects"] += 1
            if subject.label == 1:
                bucket["positive"] += 1
            elif subject.label == 0:
                bucket["negative"] += 1
        return out

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for pair in self.pairs:
            rows.append(
                {
                    "subject": pair.subject,
                    "site": pair.site,
                    "fold": pair.fold,
                    "label": pair.label,
                    "lvef_percent": pair.lvef_percent,
                    "ecg_path": str(pair.ecg_path),
                    "pcg_path": str(pair.pcg_path),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "external": self.external,
            "n_pairs": self.n_pairs,
            "n_subjects": self.n_subjects,
            "labels": self.label_counts(),
            "folds": self.fold_counts(),
            "n_unpaired": len(self.unpaired),
            "n_unparsed": len(self.unparsed),
            "label_sources": [str(p) for p in self.label_sources],
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# File-name and CSV parsing
# ---------------------------------------------------------------------------
def parse_recording_filename(path: str | Path) -> RecordingFile | None:
    """Parse ``ID_SITE_MODALITY.wav`` and return ``None`` for other files."""
    path = Path(path)
    match = _WAV_RE.match(path.stem)
    if not match:
        return None
    return RecordingFile(
        path=path,
        subject=match.group("subject"),
        site=match.group("site").upper(),
        modality=match.group("modality").upper(),
    )


def _extract_subject(raw: str) -> str:
    """Extract a subject ID from a recording name or a bare subject ID."""
    raw = str(raw).strip()
    match = _WAV_RE.match(Path(raw).stem)
    if match:
        return match.group("subject")
    # The official fold CSVs use ``subject_SITE`` without the modality.
    site_match = re.match(rf"^(?P<subject>.+)_{SITE_PATTERN}$", raw, re.IGNORECASE)
    if site_match:
        return site_match.group("subject")
    return raw


def _looks_numeric(value: str) -> bool:
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _normalise_lvef(value: float) -> float:
    """Return LVEF as a percentage.

    The official CSVs use fractions (0-1).  Some derived tables use percent
    (0-100); both are accepted.  Values outside [0, 100] are rejected.
    """
    value = float(value)
    if 0.0 <= value <= 1.0:
        value *= 100.0
    if not np.isfinite(value) or value < 0.0 or value > 100.0:
        raise ValueError(f"LVEF out of physiological range: {value}")
    return value


def _read_csv_rows(path: Path) -> list[list[str]]:
    """Read CSV rows, sniffing the delimiter and handling a UTF-8 BOM."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        return [
            [cell.strip() for cell in row]
            for row in csv.reader(fh, delimiter=delimiter)
            if row and any(cell.strip() for cell in row)
        ]


def parse_label_csv(path: str | Path) -> list[dict[str, Any]]:
    """Parse a label CSV into ``subject`` / ``lvef_percent`` / ``fold`` rows.

    The parser is intentionally tolerant because the official files have two
    slightly different layouts (recording-level rows in the folds and
    subject-level rows in the community test set).
    """
    path = Path(path)
    rows = _read_csv_rows(path)
    if not rows:
        return []

    header = [h.lower() for h in rows[0]]
    has_header = any(
        any(token in h for token in ("lvef", "ejection", "label", "subject", "record", "id"))
        for h in header
    )
    # A numeric second field in the first row suggests there is no header.
    if has_header and len(rows) > 1 and _looks_numeric(rows[0][1] if len(rows[0]) > 1 else ""):
        # Ambiguous: if the "header" row itself contains a valid LVEF number,
        # treat it as data as well.
        header_is_data = True
    else:
        header_is_data = False

    id_col = 0
    lvef_col = 1
    fold_col: int | None = None
    if has_header:
        for idx, name in enumerate(header):
            if any(token in name for token in ("lvef", "ejection", "ef")):
                lvef_col = idx
            if name in {"id", "subject", "subject_id", "patient", "patient_id", "record", "recording", "name", "file"}:
                id_col = idx
            if "fold" in name:
                fold_col = idx

    data_rows = rows[1:] if has_header and not header_is_data else rows
    parsed: list[dict[str, Any]] = []
    for row in data_rows:
        if len(row) <= max(id_col, lvef_col):
            continue
        raw_id = row[id_col]
        # If the configured LVEF column is not numeric, search the row.
        candidates = [lvef_col] + [i for i in range(len(row)) if i != id_col]
        lvef_value: float | None = None
        for idx in candidates:
            if idx < len(row) and _looks_numeric(row[idx]):
                try:
                    lvef_value = _normalise_lvef(float(row[idx]))
                    break
                except ValueError:
                    continue
        if lvef_value is None:
            LOGGER.warning("Skipping label row without a valid LVEF in %s: %s", path, row)
            continue
        fold: int | None = None
        if fold_col is not None and fold_col < len(row):
            fold_match = re.search(r"\d+", row[fold_col])
            if fold_match:
                fold = int(fold_match.group())
        if fold is None:
            fold_match = _FOLD_RE.search(path.stem)
            if fold_match:
                fold = int(fold_match.group(1))
        parsed.append(
            {
                "subject": _extract_subject(raw_id),
                "lvef_percent": lvef_value,
                "fold": fold,
                "source": str(path),
            }
        )
    return parsed


def _fold_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = _FOLD_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _is_external_path(root: Path, explicit: bool) -> bool:
    if explicit:
        return True
    name = root.name.lower()
    return "community" in name or name == "test_set"


def discover_dataset(
    root: str | Path,
    lvef_threshold_percent: float = 40.0,
    external: bool = False,
) -> DatasetIndex:
    """Discover all ECG/PCG pairs, labels and folds below *root*.

    The function is read-only and does not load waveform samples.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    external = _is_external_path(root, explicit=external)
    warnings: list[str] = []

    # 1. Find and parse all WAV files.
    files: list[RecordingFile] = []
    unparsed: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        parsed = parse_recording_filename(path)
        if parsed is None:
            unparsed.append(path)
            continue
        parsed.fold = _fold_from_path(path)
        files.append(parsed)

    # 2. Parse every CSV whose name looks like a label file.
    label_paths = [
        p
        for p in sorted(root.rglob("*.csv"))
        if "label" in p.name.lower() or "lvef" in p.name.lower()
    ]
    if not label_paths:
        # Fall back to any CSV; the parser will skip rows without a valid LVEF.
        label_paths = sorted(root.rglob("*.csv"))
    label_rows: list[dict[str, Any]] = []
    for path in label_paths:
        try:
            label_rows.extend(parse_label_csv(path))
        except Exception as exc:  # keep discovery robust
            warnings.append(f"Could not parse {path}: {exc}")

    # 3. Merge labels per subject.  All rows for a subject must agree.
    lvef_by_subject: dict[str, list[float]] = {}
    fold_by_subject: dict[str, set[int]] = {}
    for row in label_rows:
        subject = row["subject"]
        lvef_by_subject.setdefault(subject, []).append(float(row["lvef_percent"]))
        if row.get("fold") is not None:
            fold_by_subject.setdefault(subject, set()).add(int(row["fold"]))

    subject_info: dict[str, SubjectInfo] = {}
    for subject, values in lvef_by_subject.items():
        if max(values) - min(values) > 1e-6:
            warnings.append(
                f"Inconsistent LVEF values for subject {subject}: {sorted(set(values))}; "
                "using the median."
            )
        lvef = float(np.median(values))
        folds = fold_by_subject.get(subject, set())
        if len(folds) > 1:
            raise ValueError(
                f"Subject {subject} appears in multiple folds: {sorted(folds)}. "
                "Subject-level folds must be disjoint."
            )
        fold = next(iter(folds)) if folds else None
        subject_info[subject] = SubjectInfo(
            subject=subject,
            label=int(lvef <= float(lvef_threshold_percent)),
            lvef_percent=lvef,
            fold=fold,
        )

    # 4. Pair ECG and PCG files by (subject, site).
    by_key: dict[tuple[str, str], dict[str, RecordingFile]] = {}
    for rec in files:
        by_key.setdefault((rec.subject, rec.site), {})[rec.modality] = rec

    pairs: list[RecordingPair] = []
    unpaired: list[dict[str, Any]] = []
    for (subject, site), modalities in sorted(by_key.items()):
        if "ECG" not in modalities or "PCG" not in modalities:
            unpaired.append(
                {
                    "subject": subject,
                    "site": site,
                    "missing": [m for m in ("ECG", "PCG") if m not in modalities],
                    "files": [str(r.path) for r in modalities.values()],
                }
            )
            continue
        rec = modalities["ECG"]
        fold = rec.fold
        if fold is None:
            fold = subject_info.get(subject, SubjectInfo(subject, None, None, None)).fold
        if fold is None and not external:
            # The official layout provides a fold directory or CSV name.  If
            # both are absent we leave it as None; the splitter can fall back
            # to StratifiedGroupKFold later.
            warnings.append(f"No fold information for {subject} / {site}.")
        info = subject_info.get(subject)
        if info is None:
            info = SubjectInfo(subject=subject, label=None, lvef_percent=None, fold=fold)
            subject_info[subject] = info
            warnings.append(f"No label found for subject {subject}.")
        if info.fold is None and fold is not None:
            info.fold = fold
        info.sites.add(site)
        pairs.append(
            RecordingPair(
                subject=subject,
                site=site,
                fold=fold,
                ecg_path=rec.path,
                pcg_path=modalities["PCG"].path,
                label=info.label,
                lvef_percent=info.lvef_percent,
            )
        )

    # 5. Check that each subject has the expected number of sites.
    for subject, info in subject_info.items():
        if len(info.sites) not in (0, 4):
            warnings.append(
                f"Subject {subject} has {len(info.sites)} paired site(s): "
                f"{sorted(info.sites)}"
            )

    index = DatasetIndex(
        root=root,
        external=external,
        pairs=pairs,
        subjects=subject_info,
        unpaired=unpaired,
        unparsed=unparsed,
        label_sources=label_paths,
        warnings=warnings,
    )
    LOGGER.info(
        "Discovered %d pairs / %d subjects under %s",
        index.n_pairs,
        index.n_subjects,
        root,
    )
    return index


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _read_wav_info(path: Path) -> dict[str, Any]:
    """Read WAV header information without loading the samples."""
    import soundfile as sf

    info = sf.info(str(path))
    return {
        "fs": int(info.samplerate),
        "frames": int(info.frames),
        "duration": float(info.duration),
        "channels": int(info.channels),
        "subtype": str(info.subtype),
    }


def validate_dataset(
    index: DatasetIndex,
    expected_ecg_fs: int = 500,
    expected_pcg_fs: int = 4000,
    min_duration_seconds: float = 10.0,
    max_duration_seconds: float = 90.0,
    duration_tolerance_seconds: float = 0.25,
    max_recordings: int | None = None,
    check_all_sites: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check WAV headers, pairing, labels, folds and durations.

    Returns a per-pair DataFrame and a summary dictionary.  No samples are
    loaded; this function is fast enough to run on the whole dataset.
    """
    rows: list[dict[str, Any]] = []
    pairs = index.pairs[:max_recordings] if max_recordings else index.pairs
    for pair in pairs:
        row: dict[str, Any] = {
            "subject": pair.subject,
            "site": pair.site,
            "fold": pair.fold,
            "label": pair.label,
            "lvef_percent": pair.lvef_percent,
            "ecg_path": str(pair.ecg_path),
            "pcg_path": str(pair.pcg_path),
            "status": "ok",
            "warnings": "",
        }
        problems: list[str] = []
        try:
            ecg = _read_wav_info(pair.ecg_path)
            pcg = _read_wav_info(pair.pcg_path)
            row.update(
                {
                    "ecg_fs": ecg["fs"],
                    "pcg_fs": pcg["fs"],
                    "ecg_duration": ecg["duration"],
                    "pcg_duration": pcg["duration"],
                    "ecg_channels": ecg["channels"],
                    "pcg_channels": pcg["channels"],
                    "ecg_subtype": ecg["subtype"],
                    "pcg_subtype": pcg["subtype"],
                }
            )
            if ecg["fs"] != expected_ecg_fs:
                problems.append(f"ECG fs={ecg['fs']} (expected {expected_ecg_fs})")
            if pcg["fs"] != expected_pcg_fs:
                problems.append(f"PCG fs={pcg['fs']} (expected {expected_pcg_fs})")
            if not (min_duration_seconds <= ecg["duration"] <= max_duration_seconds):
                problems.append(f"ECG duration={ecg['duration']:.2f}s")
            if not (min_duration_seconds <= pcg["duration"] <= max_duration_seconds):
                problems.append(f"PCG duration={pcg['duration']:.2f}s")
            diff = abs(ecg["duration"] - pcg["duration"])
            row["duration_diff"] = diff
            if diff > duration_tolerance_seconds:
                problems.append(f"duration mismatch {diff:.3f}s")
            if ecg["channels"] != 1 or pcg["channels"] != 1:
                problems.append("expected mono WAV files")
        except Exception as exc:
            problems.append(f"read error: {exc}")
            row["status"] = "error"
        if pair.label is None:
            problems.append("missing label")
        if pair.fold is None and not index.external:
            problems.append("missing fold")
        if problems:
            row["warnings"] = "; ".join(problems)
            if row["status"] != "error":
                row["status"] = "warning"
        rows.append(row)

    df = pd.DataFrame(rows)
    subject_sites = {
        s: sorted(info.sites) for s, info in index.subjects.items() if info.sites
    }
    bad_sites = {s: sites for s, sites in subject_sites.items() if check_all_sites and len(sites) != 4}
    labels = index.label_counts()
    summary = {
        "root": str(index.root),
        "external": index.external,
        "n_subjects": index.n_subjects,
        "n_pairs": index.n_pairs,
        "n_checked": len(df),
        "labels": labels,
        "folds": index.fold_counts(),
        "subjects_not_4_sites": bad_sites,
        "n_unpaired": len(index.unpaired),
        "n_unparsed": len(index.unparsed),
        "n_errors": int((df["status"] == "error").sum()) if len(df) else 0,
        "n_warnings": int((df["status"] == "warning").sum()) if len(df) else 0,
        "warnings": index.warnings,
    }
    return df, summary


# ---------------------------------------------------------------------------
# Synchronisation diagnostic
# ---------------------------------------------------------------------------
def check_synchronization(
    ecg: np.ndarray,
    pcg: np.ndarray,
    fs_ecg: int,
    fs_pcg: int,
    max_lag_seconds: float = 0.2,
    step_seconds: float = 0.005,
) -> dict[str, Any]:
    """Estimate whether PCG energy follows the ECG R peaks.

    The dataset declares ECG and PCG within a recording to be synchronous, so
    the expected lag is zero.  This function does not *apply* a delay; it
    reports the lag that would maximise the average PCG energy after R peaks.
    A large non-zero lag or a low contrast is a warning sign, not a command to
    shift the signal.
    """
    from .ecg_processing import detect_r_peaks_simple
    from .pcg_processing import pcg_envelope

    quality = {}
    peaks = detect_r_peaks_simple(ecg, fs_ecg, return_diagnostics=True)
    r_samples = np.asarray(peaks["peaks"], dtype=int)
    r_times = r_samples.astype(float) / float(fs_ecg)
    if len(r_times) < 3:
        return {
            "ok": False,
            "reason": "fewer than 3 R peaks",
            "n_rpeaks": int(len(r_times)),
            "best_lag_seconds": None,
            "contrast": None,
        }
    env = pcg_envelope(pcg, fs_pcg)
    lags = np.arange(-max_lag_seconds, max_lag_seconds + 1e-12, step_seconds)
    scores = []
    for lag in lags:
        on_values = []
        off_values = []
        for r in r_times:
            on0 = int(round((r + lag + 0.02) * fs_pcg))
            on1 = int(round((r + lag + 0.18) * fs_pcg))
            off0 = int(round((r + lag - 0.25) * fs_pcg))
            off1 = int(round((r + lag - 0.08) * fs_pcg))
            if on0 < 0 or on1 >= len(env) or off0 < 0 or off1 >= len(env):
                continue
            on_values.append(float(np.mean(env[on0:on1])))
            off_values.append(float(np.mean(env[off0:off1])))
        if on_values and off_values:
            scores.append(float(np.mean(on_values) / (np.mean(off_values) + 1e-12)))
        else:
            scores.append(np.nan)
    scores_arr = np.asarray(scores, dtype=float)
    if np.all(np.isnan(scores_arr)):
        return {
            "ok": False,
            "reason": "not enough complete beats for lag scan",
            "n_rpeaks": int(len(r_times)),
            "best_lag_seconds": None,
            "contrast": None,
        }
    best_idx = int(np.nanargmax(scores_arr))
    best_lag = float(lags[best_idx])
    contrast = float(scores_arr[best_idx])
    return {
        "ok": True,
        "n_rpeaks": int(len(r_times)),
        "best_lag_seconds": best_lag,
        "contrast": contrast,
        "zero_lag_contrast": float(scores_arr[np.argmin(np.abs(lags))]),
        "lag_curve": {"lags": lags.tolist(), "contrast": scores_arr.tolist()},
        "note": "Diagnostic only; delay is not applied unless gating.delay.apply=true.",
    }


def save_index(index: DatasetIndex, output_dir: str | Path) -> None:
    """Save the discovery summary and a pair table for inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(index.summary(), output_dir / "dataset_index_summary.json")
    index.to_dataframe().to_csv(output_dir / "dataset_pairs.csv", index=False)
    pd.DataFrame(index.unpaired).to_csv(output_dir / "dataset_unpaired.csv", index=False)
    pd.DataFrame({"path": [str(p) for p in index.unparsed]}).to_csv(
        output_dir / "dataset_unparsed.csv", index=False
    )


if __name__ == "__main__":  # pragma: no cover - manual convenience
    import argparse

    parser = argparse.ArgumentParser(description="Quick dataset discovery")
    parser.add_argument("root")
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    idx = discover_dataset(args.root, external=args.external)
    print(idx.summary())
