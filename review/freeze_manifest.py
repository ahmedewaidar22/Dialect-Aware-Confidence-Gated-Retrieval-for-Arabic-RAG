"""Freeze the final pipeline BEFORE a confirmatory test, and verify later that nothing changed.

  python scripts/freeze_manifest.py --out freeze_manifest.json \
      --files models/dialect_head runs_v2/v2_config.json arroute/pipeline/strategies.py arroute/egy/rewrite.py \
              arroute/pipeline/final.py data/corpus \
      --param k=2 --param lo=0.2 --param hi=0.8 --param n_cands=20 --param generator=gpt-4.1-mini --param temperature=0
  git add freeze_manifest.json && git commit   # commit BEFORE touching the new test questions

  python scripts/freeze_manifest.py --verify freeze_manifest.json    # run again after the test; must print FROZEN OK

Directories are hashed file by file (sorted). The manifest also records package versions, hardware and the git commit
when available, so the run can be reproduced and a reviewer can see the pipeline was fixed first.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(p):
    p = Path(p)
    if p.is_file():
        return {str(p): sha(p)}
    if p.is_dir():
        return {str(q): sha(q) for q in sorted(p.rglob("*")) if q.is_file() and "__pycache__" not in q.parts}
    raise SystemExit(f"not found: {p}")


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scikit-learn", "torch", "transformers", "sentence-transformers", "openai", "rank-bm25"):
        try:
            from importlib import metadata
            out[m] = metadata.version(m)
        except Exception:                                      # noqa: BLE001
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:                                          # noqa: BLE001
        pass
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                          # noqa: BLE001
        return None


def build(files, params):
    hashes = {}
    for f in files:
        hashes.update(hash_path(f))
    return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
            "params": dict(params), "versions": versions(), "files": hashes}


def verify(manifest_path):
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    bad = []
    for path, h in m["files"].items():
        if not Path(path).is_file():
            bad.append((path, "missing"))
        elif sha(path) != h:
            bad.append((path, "CHANGED"))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="freeze_manifest.json")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--param", action="append", default=[], help="key=value, repeatable")
    ap.add_argument("--verify", default=None, help="manifest to check instead of writing one")
    a = ap.parse_args(argv)
    if a.verify:
        bad = verify(a.verify)
        if bad:
            for p, why in bad:
                print(f"  {why}: {p}")
            print("NOT FROZEN: the pipeline changed after the freeze")
            return 1
        print("FROZEN OK: every hashed file is identical to the manifest")
        return 0
    params = dict(x.split("=", 1) for x in a.param)
    m = build(a.files, params)
    Path(a.out).write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}: {len(m['files'])} files hashed, {len(params)} parameters, git {m['git_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
