"""Tiny, torch-free helpers for environment checks."""
import re


def version_tuple(v):
    """'2.5.1+cu121' -> (2, 5, 1);  '2.13.0' -> (2, 13, 0)"""
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+")[0])[:3])


def torch_at_least(v, minimum=(2, 6)):
    return version_tuple(v)[:len(minimum)] >= tuple(minimum)


TORCH_HELP = (
    "\nThis model only ships '.bin' weights. Recent transformers refuse to load those with PyTorch < 2.6 "
    "(security fix CVE-2025-32434).\nFix (run with the venv active and no Python process running):\n"
    "    python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126\n"
    "Then run the script again; finished seeds are skipped.\n")
