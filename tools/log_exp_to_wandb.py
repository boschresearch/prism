#!/usr/bin/env python3
# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""
Script to log formatted metrics to wandb from experiment folder structure.
Enhanced with epoch tracking to avoid duplicate syncing.

Now also reports, at the end, which meta folders were not processed successfully and why.
"""

import os
import json
import re
import wandb
from pathlib import Path
from typing import List, Tuple, Dict, Any

import re
import hashlib


def shorten_run_name(
    name: str,
    max_len: int = 128,
    banned_tokens=("exp", "faris", "andreas", "juergen", "a100", "g2", "b4", "w4"),
) -> str:
    """
    Normalize a run name to <= max_len while keeping it meaningful:
      - remove banned tokens (underscore-separated, case-insensitive)
      - detect date/time in several common formats and append as _DDMM_HHMMSS (if found)
      - preserve informative tokens; compress gracefully before truncation
      - if still too long, add stable hash suffix
    """

    raw = name

    # ---------- 1) tokenize and remove banned underscore-tokens ----------
    toks = [t for t in name.split("_") if t]  # keep hyphens inside tokens
    banned = {b.lower() for b in banned_tokens}
    toks = [t for t in toks if t.lower() not in banned]
    base = "_".join(toks)

    # ---------- 2) extract date/time anywhere (last occurrence wins) ----------
    # Dates: dd-mm-yyyy / yyyy-mm-dd / dd-mm-yy / dd_mm_yyyy / yyyy_mm_dd
    date_patterns = [
        r"\b(?P<d>\d{2})[-_](?P<m>\d{2})[-_](?P<y>\d{4})\b",  # dd-mm-yyyy
        r"\b(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<d>\d{2})\b",  # yyyy-mm-dd
        r"\b(?P<d>\d{2})[-_](?P<m>\d{2})[-_](?P<y>\d{2})\b",  # dd-mm-yy
    ]
    time_patterns = [
        r"\b(?P<h>\d{2})[-_](?P<mi>\d{2})[-_](?P<s>\d{2})\b",  # HH-MM-SS / HH_MM_SS
        r"\b(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})\b",  # HHMMSS
    ]

    def last_match(pats, text):
        m = None
        for pat in pats:
            for x in re.finditer(pat, text):
                m = x
        return m

    m_date = last_match(date_patterns, base)
    m_time = last_match(time_patterns, base)

    ddmm = None
    hhmmss = None
    if m_date:
        gd = m_date.groupdict()
        y = gd["y"]
        d = int(gd["d"])
        m = int(gd["m"])
        # compact: ddmm ; we drop 4-digit year, keep yy if provided
        if len(y) == 4:
            ddmm = f"{d:02d}{m:02d}"
        else:
            ddmm = f"{d:02d}{m:02d}{int(y):02d}"
    if m_time:
        h = int(m_time.group("h"))
        mi = int(m_time.group("mi"))
        s = int(m_time.group("s"))
        hhmmss = f"{h:02d}{mi:02d}{s:02d}"

    # remove matched date/time substrings from base (to avoid duplication)
    text = base
    if m_date:
        text = text[: m_date.start()] + text[m_date.end() :]
    if m_time:
        # remove only one occurrence; re-search in current text
        text = re.sub(time_patterns[0], "", text, count=1) or re.sub(time_patterns[1], "", text, count=1)
    text = re.sub(r"_+", "_", text).strip("_")

    # ---------- 3) rebuild meaningful tokens ----------
    tokens = [t for t in text.split("_") if t]

    # Heuristics for importance: prefer earlier tokens and non-numeric tokens
    def token_score(t: str) -> int:
        score = 0
        if re.search(r"[A-Za-z]", t):
            score += 3
        if "-" in t:
            score += 1  # hyphenated often informative
        if re.search(r"\d", t):
            score += 0  # neutral
        return score

    # ---------- 4) gentle abbreviation pass (no length change yet) ----------
    def abbrev(tok: str, limit: int = 28) -> str:
        if len(tok) <= limit:
            return tok
        # For hyphenated tokens, abbreviate each chunk a bit
        parts = tok.split("-")
        if len(parts) > 1:
            parts = [p if len(p) <= 16 else (p[:10] + "…" + p[-5:]) for p in parts]
            out = "-".join(parts)
            if len(out) <= limit:
                return out
        # fallback: head…tail
        return tok[:14] + "…" + tok[-13:]

    tokens = [abbrev(t) for t in tokens]

    # ---------- 5) assemble with date/time suffix if present ----------
    suffix = []
    if ddmm:
        suffix.append(ddmm)
    if hhmmss:
        suffix.append(hhmmss)
    out = "_".join(tokens + (["_".join(suffix)] if suffix else []))
    out = re.sub(r"_+", "_", out).strip("_")

    # ---------- 6) if still too long, compress progressively ----------
    if len(out) > max_len:
        # 6a) more aggressive per-token trim
        hard_limit = 20
        trimmed = []
        for t in tokens:
            if len(t) > hard_limit:
                trimmed.append(t[:10] + "…" + t[-9:])
            else:
                trimmed.append(t)
        out = "_".join(trimmed + (["_".join(suffix)] if suffix else []))

    if len(out) > max_len:
        # 6b) drop least-informative tokens from the END first (later tokens are often less critical)
        ranked = sorted(
            [(i, t, token_score(t)) for i, t in enumerate(tokens)],
            key=lambda x: (x[2], x[0]),  # by score asc, then position asc
        )
        to_keep = set(range(len(tokens)))
        for i, t, _ in ranked:
            if len(out) <= max_len:
                break
            if i in to_keep:
                to_keep.remove(i)
                kept_tokens = [tokens[j] for j in sorted(to_keep)]
                out = "_".join(kept_tokens + (["_".join(suffix)] if suffix else []))

    if len(out) > max_len:
        # 6c) final safety net: stable hash suffix
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
        # leave room for _hXXXXXXXX
        room = max_len - (2 + len(h))  # "_h" + 8
        out = (out[:room]).rstrip("_") + f"_h{h}"

    return out


def extract_run_info(meta_folder_path):
    path = Path(meta_folder_path)
    run_name = path.name
    print(f"Extracted run name: {run_name}")

    if len(run_name) > 128:
        run_name = shorten_run_name(run_name)
        print(f"Shortened run name to: {run_name}")
        assert len(run_name) <= 128, "Run name is still too long after shortening!"
        # write shorten run name to txt file
        with open(path / "wandb_run_name.txt", "w") as f:
            f.write(run_name + "\n")

    path_parts = path.parts
    eval_index = path_parts.index("eval")
    group_parts = path_parts[eval_index + 1 : -1]
    group_name = "/".join(group_parts)
    print(f"Extracted group name: {group_name}")

    tags = group_parts
    return run_name, group_name, tags


def extract_epoch_from_folder(folder_name, verbose=True):
    pattern = r"_e(\d+)(?:_|$)"
    match = re.search(pattern, folder_name)
    if match:
        return int(match.group(1))
    else:
        if verbose:
            print(f"  Warning: Could not extract epoch from folder: {folder_name}")
        return None


def load_formatted_metrics(json_path):
    try:
        with open(json_path, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"  Error loading metrics from {json_path}: {e}")
        return None


def load_synced_epochs(sync_file_path):
    if sync_file_path.exists():
        try:
            with open(sync_file_path, "r") as f:
                data = json.load(f)
            synced_epochs = set(data.get("synced_epochs", []))
            print(f"Loaded {len(synced_epochs)} already synced epochs: {sorted(synced_epochs)}")
            return synced_epochs
        except Exception as e:
            print(f"Error loading sync file {sync_file_path}: {e}")
            print("Creating new sync tracking file...")
            return set()
    else:
        print("No sync tracking file found. Will create new one.")
        return set()


def save_synced_epochs(sync_file_path, synced_epochs):
    try:
        data = {"synced_epochs": sorted(list(synced_epochs)), "last_updated": wandb.util.generate_id()}
        with open(sync_file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated sync tracking file with {len(synced_epochs)} epochs")
    except Exception as e:
        print(f"Error saving sync file {sync_file_path}: {e}")


def process_experiment_folder(meta_folder_path: str) -> Dict[str, Any]:
    """
    Process the experiment folder and log metrics to wandb.

    Returns a dict summary:
      {
        'meta_folder': <str>,
        'status': 'success' | 'partial' | 'failed',
        'errors': [<str>],
        'epochs_logged': <int>,
        'epochs_to_sync': <int>,
      }
    """
    summary = {
        "meta_folder": meta_folder_path,
        "status": "success",
        "errors": [],
        "epochs_logged": 0,
        "epochs_to_sync": 0,
    }

    print(f"Processing experiment folder: {meta_folder_path}")

    # Extract run name and group
    try:
        run_name, group_name, tags = extract_run_info(meta_folder_path)
    except Exception as e:
        msg = f"extract_run_info failed: {e}"
        print("  " + msg)
        summary["status"] = "failed"
        summary["errors"].append(msg)
        return summary

    # Setup sync tracking file path
    meta_path = Path(meta_folder_path)
    sync_file_path = meta_path / "wandb_synced_epochs.json"

    # Load already synced epochs
    synced_epochs = load_synced_epochs(sync_file_path)

    # Get all subfolders and extract epochs
    subfolders = [f for f in meta_path.iterdir() if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders to process")

    # Determine which epochs need to be synced
    epochs_to_sync: List[Tuple[int, Path]] = []
    for subfolder in subfolders:
        metrics_file = subfolder / "formatted_metrics.json"
        if not metrics_file.exists():
            print(f"  Skipping {subfolder.name}: No formatted_metrics.json found")
            continue

        epoch = extract_epoch_from_folder(subfolder.name)
        if epoch is None:
            print(f"  Skipping {subfolder.name}: Could not extract epoch")
            continue

        if epoch in synced_epochs:
            continue

        epochs_to_sync.append((epoch, subfolder))

    summary["epochs_to_sync"] = len(epochs_to_sync)

    if not epochs_to_sync:
        print("No new epochs to sync. All epochs are already synced!")
        # Nothing to do, but it's still a success
        return summary

    # Initialize wandb
    print(f"Initializing wandb with run_name='{run_name}', group='{group_name}'")
    try:
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            id=run_name,
            group=group_name,
            tags=tags,
            resume="allow",
        )
    except Exception as e:
        msg = f"wandb.init failed: {e}"
        print("  " + msg)
        summary["status"] = "failed"
        summary["errors"].append(msg)
        return summary

    metrics_logged = 0
    newly_synced_epochs = set()

    # Process epochs in sorted order
    for epoch, subfolder in sorted(epochs_to_sync, key=lambda x: x[0]):
        print(f"\nProcessing epoch {epoch} from subfolder: {subfolder.name}")

        metrics_file = subfolder / "formatted_metrics.json"
        metrics = load_formatted_metrics(metrics_file)
        if metrics is None:
            summary["errors"].append(f"{subfolder.name}: failed to load {metrics_file.name}")
            continue

        metrics["epoch"] = epoch

        try:
            wandb.log(metrics)
            metrics_logged += 1
            newly_synced_epochs.add(epoch)
            print(f"  Successfully logged metrics for epoch {epoch}")
        except Exception as e:
            err = f"{subfolder.name}: wandb.log failed for epoch {epoch}: {e}"
            print("  " + err)
            summary["errors"].append(err)

    print(f"\nCompleted processing. Logged metrics for {metrics_logged} new epochs.")
    print("Finishing wandb run...")
    try:
        wandb.finish()
    except Exception as e:
        # finishing issues shouldn't mark entire folder failed, but record them
        fin_msg = f"wandb.finish warning: {e}"
        print("  " + fin_msg)
        summary["errors"].append(fin_msg)

    # Update sync tracking file
    if newly_synced_epochs:
        updated_synced_epochs = synced_epochs | newly_synced_epochs
        save_synced_epochs(sync_file_path, updated_synced_epochs)
        print(f"Added {len(newly_synced_epochs)} new epochs to sync tracking: {sorted(newly_synced_epochs)}")
    else:
        print("No new epochs were successfully synced.")

    summary["epochs_logged"] = metrics_logged

    # Decide overall status
    if summary["errors"]:
        if metrics_logged > 0:
            summary["status"] = "partial"
        else:
            # had something to sync but logged none
            summary["status"] = "failed"
    else:
        summary["status"] = "success"

    return summary


def find_meta_folders(root_path: str) -> List[Path]:
    root_path = Path(root_path)
    meta_folders = []
    for folder in root_path.rglob("*"):
        if folder.is_dir() and is_meta_folder(folder):
            meta_folders.append(folder)
    return meta_folders


def is_meta_folder(folder_path: Path) -> bool:
    if not folder_path.is_dir():
        return False
    folder_name = folder_path.name
    subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
    for subfolder in subfolders:
        epoch = extract_epoch_from_folder(subfolder.name, verbose=False)
        if epoch is not None and subfolder.name.startswith(folder_name):
            return True
    return False


def main():
    base_path = "/home/hao1rng/core/VAD/exps/eval/vad-fixed"

    results: List[Dict[str, Any]] = []

    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        print("\n================ SUMMARY ================\nNo folders processed.")
        return

    meta_folders = find_meta_folders(base_path)

    if not meta_folders:
        print(f"No meta folders found in: {base_path}")
        print("\n================ SUMMARY ================\nNo folders processed.")
        return

    print(f"Found {len(meta_folders)} meta folders to process:")
    for folder in sorted(meta_folders):
        print(f"  - {folder.name}")

    processed_count = 0
    for i, meta_folder in enumerate(sorted(meta_folders), start=1):
        print(f"\n{'='*80}")
        print(f"PROCESSING META FOLDER {i}/{len(meta_folders)}: {meta_folder.name}")
        print(f"{'='*80}")

        try:
            summary = process_experiment_folder(str(meta_folder))
            results.append(summary)
            if summary["status"] in ("success", "partial"):
                processed_count += 1
            print(
                f"Status: {summary['status']}. "
                f"Epochs to sync: {summary['epochs_to_sync']}, logged: {summary['epochs_logged']}."
            )
            if summary["errors"]:
                print("  Issues:")
                for err in summary["errors"]:
                    print(f"    - {err}")
        except Exception as e:
            print(f"Error processing {meta_folder.name}: {e}")
            print("Continuing with next folder...")
            results.append(
                {
                    "meta_folder": str(meta_folder),
                    "status": "failed",
                    "errors": [f"Unhandled exception: {e}"],
                    "epochs_logged": 0,
                    "epochs_to_sync": 0,
                }
            )

    # Final report
    print(f"\n{'='*80}")
    print(f"SUMMARY: Completed {processed_count}/{len(meta_folders)} meta folders (success or partial).")
    print(f"{'='*80}")

    failed_or_partial = [r for r in results if r["status"] in ("failed", "partial")]
    if not failed_or_partial:
        print("All meta folders processed successfully. 🎉")
        return

    print("\nFolders not processed successfully (failed or partial) and causes:")
    for r in failed_or_partial:
        print(f"- {Path(r['meta_folder']).name} [{r['status']}]:")
        if r["errors"]:
            for err in r["errors"]:
                print(f"    • {err}")
        else:
            print("    • No specific errors recorded, but status not 'success'.")


if __name__ == "__main__":
    WANDB_PROJECT = "VLP-eval"
    main()
