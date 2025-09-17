#!/usr/bin/env python3
"""
Collect NoXi transcripts and engagement annotations into two folders.

It searches recursively under a ROOT directory for files named:
  - *.audio.transcript.annotation.csv
  - *.engagement.annotation.csv

For each match, it copies the file into:
  OUT/transcripts/  (for transcripts)
  OUT/annotations/  (for engagement)

Destination filenames are prefixed with the session folder name (e.g., "001")
and keep the original participant prefix (e.g., "expert", "novice"), e.g.:
  001_expert.audio.transcript.annotation.csv
  001_novice.engagement.annotation.csv

Usage:
  python collect_noxi_files.py --root /path/to/NoXi_dataset --out ./noxi_exports

If --out is omitted, it writes to the current directory.
"""
import argparse
import shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str,
                    help="Root directory that contains NoXi_* folders (search is recursive).")
    ap.add_argument("--out", default=".", type=str,
                    help="Output base directory (will create transcripts/ and annotations/ inside).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be copied without writing files.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_base = Path(args.out).expanduser().resolve()
    out_transcripts = out_base / "transcripts"
    out_annotations = out_base / "annotations"
    out_transcripts.mkdir(parents=True, exist_ok=True)
    out_annotations.mkdir(parents=True, exist_ok=True)

    # Globs to find the files of interest
    transcript_glob = "**/*.audio.transcript.annotation.csv"
    engagement_glob = "**/*.engagement.annotation.csv"

    def dest_name(src: Path) -> str:
        """
        Build destination filename using parent folder (session id)
        and keep the participant prefix (expert/novice) that is already in the basename.
        Example: src.name = 'expert.audio.transcript.annotation.csv', parent.name = '001'
                 -> '001_expert.audio.transcript.annotation.csv'
        """
        session = src.parent.name
        return f"{session}_{src.name}"

    copied = {"transcripts": 0, "annotations": 0}
    missed = {"transcripts": 0, "annotations": 0}

    # Process transcripts
    for src in root.rglob(transcript_glob):
        if not src.is_file():
            continue
        dst = out_transcripts / dest_name(src)
        if args.dry_run:
            print(f"[DRY-RUN] TRANSCRIPT: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        copied["transcripts"] += 1

    # Process engagement annotations
    for src in root.rglob(engagement_glob):
        if not src.is_file():
            continue
        dst = out_annotations / dest_name(src)
        if args.dry_run:
            print(f"[DRY-RUN] ANNOTATION: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        copied["annotations"] += 1

    print("Done.")
    print(f"Transcripts copied: {copied['transcripts']}")
    print(f"Engagement annotations copied: {copied['annotations']}")
    if copied['annotations'] == 0:
        print("Note: No engagement annotations found (expected if you only pointed at NoXi_test/Additional_Test).")

if __name__ == "__main__":
    main()
