"""Phase 0: print the structure of every downloaded file, so loaders can be written
against the REAL formats rather than guessed ones. Send the output to Claude."""
import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to a legacy code page and crash on Arabic text.
sys.stdout.reconfigure(encoding="utf-8")

RAW = Path("data/raw")
EXTS = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow"}


def preview(path):
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                first = json.loads(f.readline())
            n = sum(1 for _ in path.open(encoding="utf-8")) + 0
            return f"jsonl, {n} lines\n    keys: {list(first)}\n    first: {str(first)[:400]}"
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return f"json dict, keys: {list(data)[:15]}\n    sample: {str(data)[:400]}"
            return (f"json list, {len(data)} items\n    keys: "
                    f"{list(data[0]) if data and isinstance(data[0], dict) else type(data[0])}"
                    f"\n    first: {str(data[0])[:400]}")
        import pandas as pd
        if path.suffix in {".csv", ".tsv"}:
            df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", nrows=3)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            return "arrow file (HF dataset shard), see dataset_info.json in the same folder"
        return f"table, columns: {list(df.columns)}\n    first row: {df.iloc[0].to_dict()}"
    except Exception as e:
        return f"could not parse: {e}"


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="inspect_report.txt",
                help="report file (written as UTF-8; do NOT use PowerShell '>' redirection)")
args = ap.parse_args()
report = open(args.out, "w", encoding="utf-8")
_print = print


def print(*a, **k):  # noqa: A001 - echo to console AND the UTF-8 report file
    _print(*a, **k)
    _print(*a, **k, file=report)


for ds_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
    print(f"\n==================== {ds_dir.name} ====================")
    files = [f for f in ds_dir.rglob("*") if f.suffix in EXTS and ".git" not in f.parts]
    if not files:
        print("  (no data files found yet)")
    for f in files[:12]:
        size_mb = f.stat().st_size / 1e6
        print(f"\n  {f.relative_to(RAW)}  ({size_mb:.1f} MB)\n    {preview(f)}")
    if len(files) > 12:
        print(f"\n  ... and {len(files) - 12} more files")

report.close()
_print(f"\nReport written to {args.out}")
