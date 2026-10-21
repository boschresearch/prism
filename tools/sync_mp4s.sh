#!/usr/bin/env bash

# sync_mp4s.sh - script to sync .mp4 files from a remote server to local machine.
# Mirror only .mp4 files from a remote root to a local root, preserving subfolders.
# Usage:
#   ./sync_mp4s.sh user@host:/remote/root /local/root
#   ./sync_mp4s.sh --dry-run user@host:/remote/root /local/root

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [--dry-run] user@host:/remote/root /local/root" >&2
  exit 1
fi

REMOTE_ROOT="$1"   # e.g. hao1rng@cluster:/home/hao1rng/core/VAD/exps
LOCAL_ROOT="$2"    # e.g. ~/Downloads/vad_videos

# Sanity checks
if [[ "$REMOTE_ROOT" != *:* ]]; then
  echo "Error: REMOTE_ROOT must be in the form user@host:/path" >&2
  exit 2
fi

mkdir -p "$LOCAL_ROOT"

# Build rsync options
RSYNC_OPTS=(
  -avz                        # archive (perm/times), verbose, compress
  --progress                  # show progress
  --prune-empty-dirs          # don't create empty folders
  --include '*/'              # include all directories (so structure is mirrored)
  --include '*.mp4'           # include only mp4 files
  --exclude '*'               # exclude everything else
  --ignore-existing           # don't overwrite files that already exist locally
  --partial                   # keep partial downloads to resume later
)

# If you prefer to verify changed files, swap --ignore-existing for --update
# (rsync defaults to mtime+size checks). For heavy verification, add --checksum (slower).

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_OPTS+=(--dry-run)
  echo "[DRY RUN] Listing actions without copying…"
fi

# Execute
# Trailing slash on REMOTE_ROOT means “copy contents of this dir into LOCAL_ROOT”
rsync "${RSYNC_OPTS[@]}" "$REMOTE_ROOT"/ "$LOCAL_ROOT"/

echo "Done."

