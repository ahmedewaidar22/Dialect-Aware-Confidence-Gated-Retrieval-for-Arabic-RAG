"""Download the saved models and the retrieval corpus from the Hugging Face Hub.

    python scripts/download_models.py --repo <hf-user>/ar-route

Restores models/dialect_head, models/final_router and data/corpus.
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "dialect_head": ROOT / "models" / "dialect_head",
    "final_router": ROOT / "models" / "final_router",
    "corpus": ROOT / "data" / "corpus",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. user/ar-route")
    ap.add_argument("--only", nargs="*", choices=list(TARGETS), help="download only these parts")
    a = ap.parse_args()

    parts = a.only or list(TARGETS)
    snap = Path(snapshot_download(a.repo, allow_patterns=[f"{p}/*" for p in parts]))
    for p in parts:
        src, dst = snap / p, TARGETS[p]
        if not src.exists():
            print(f"missing in repo: {p}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"{p} -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
