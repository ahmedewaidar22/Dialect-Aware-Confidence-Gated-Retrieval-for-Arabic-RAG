"""Text helpers shared by the baselines and (later) the router."""
import re

_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9&'.\-]*")
_DIGITS = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
_PUNCT = re.compile(r"[،؛؟?!.,:;\"'«»“”‘’()\[\]{}\-–—/\\]")


def neutralize(text):
    """Remove Latin text, digits and punctuation, leaving only Arabic words. Used to measure how much
    of a classifier's accuracy comes from surface cues (commas, titles in Latin, numbers) rather than words."""
    t = _DIGITS.sub(" ", _LATIN.sub(" ", text))
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", t)).strip()


def surface_cues(text):
    """The cheap formatting cues that differed between the two translated classes."""
    return {
        "latin word": bool(re.search(r"[A-Za-z]{2,}", text)),
        "digits": bool(_DIGITS.search(text)),
        "arabic comma": "،" in text,
        "quote marks": bool(re.search(r"[\"'«»“”]", text)),
        "no final ؟": not text.strip().endswith("؟"),
    }
