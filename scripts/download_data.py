"""Phase 0: download every public dataset into data/raw/.

Usage:
    python scripts/download_data.py --ardqa_repo <git URL of the ArDQA repo>
MAFQA must be downloaded manually (see message printed at the end).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path("data/raw")


def clone(url, dest):
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    if shutil.which("git") is None:
        raise SystemExit("git not found. Install Git for Windows (https://git-scm.com) "
                         "and reopen the terminal.")
    print(f"[git] cloning {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ardqa_repo", default=None,
                    help="git URL of the ArDQA repo (find it on github.com/Maha-J-Althobaiti)")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    # 1) ArabicaQA: single-hop questions + Wikipedia retrieval corpus
    clone("https://github.com/DataScienceUIBK/ArabicaQA", RAW / "ArabicaQA")
    print("    NOTE: if the repo links the corpus on Google Drive/HF instead of storing it,\n"
          "    follow its README and put the files inside data/raw/ArabicaQA/")

    # 2) ArabicRAGB: dialect + complexity labels (Hugging Face)
    out = RAW / "ArabicRAGB"
    if not out.exists():
        from datasets import load_dataset
        print("[hf] downloading HeshamHaroon/ArabicRAGB")
        load_dataset("HeshamHaroon/ArabicRAGB").save_to_disk(str(out))
    else:
        print(f"[skip] {out} already exists")

    # 3) ArDQA: parallel dialect QA
    if args.ardqa_repo:
        clone(args.ardqa_repo, RAW / "ArDQA")
    else:
        print("[todo] ArDQA: open github.com/Maha-J-Althobaiti, find the ArDQA repo,\n"
              "       then re-run with --ardqa_repo <url>")

    # 4) MAFQA: manual download
    (RAW / "MAFQA").mkdir(exist_ok=True)
    print("[todo] MAFQA (CC-BY 4.0): open https://www.mdpi.com/2306-5729/11/3/64,\n"
          "       use the link in the 'Data Availability Statement', and put the files\n"
          "       in data/raw/MAFQA/")

    # 5) Your validated pilot set
    (RAW / "pilot").mkdir(exist_ok=True)
    print("[todo] copy pilot_dataset_full200.json into data/raw/pilot/")
    print("\nThen run: python scripts\\inspect_data.py")


if __name__ == "__main__":
    main()
