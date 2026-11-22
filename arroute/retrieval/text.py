"""Arabic normalisation + tokenisation for lexical retrieval.

FIX vs the pilot: Arabic punctuation (؟ ، ؛) lives in the Arabic Unicode block, so a regex over
[\u0600-\u06FF] glued it to the last word ("العالي؟") and that word could never match. Tokens are
now built from Arabic LETTERS only.
"""
import re

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
_TOKEN = re.compile(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]+|[a-z0-9\u0660-\u0669]+")


def normalize_ar(text):
    text = _DIACRITICS.sub("", text).replace("\u0640", "")          # diacritics, tatweel
    text = re.sub("[إأآٱا]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip()


def tokenize_ar(text):
    return _TOKEN.findall(normalize_ar(text).lower())
