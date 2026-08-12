"""
moneyparse.py — deterministic parsing of Indian-style money phrases into rupees.

Handles everything the README warns about:
  "INR 33.38 Cr"          -> 333800000
  "3,338.00 Lakh"         -> 333800000
  "33,38,00,000"          -> 333800000   (Indian digit grouping, just strip commas)
  "twenty crore"          -> 200000000   (spelled-out numerals)
  "seventy-three crore"   -> 730000000
  "six crore"             -> 60000000

This is pure regex/arithmetic — no LLM involved, so it is exact and reproducible.
The LLM is only ever asked to point at *which* money phrase in the question is the
threshold; this module does the conversion.
"""
from __future__ import annotations
import re
from word2number import w2n

CRORE = 10_000_000
LAKH = 100_000

_NUMWORD_RE = re.compile(
    r"\b((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and|[\s-])+?)\s*"
    r"(crore|cr\.?|lakh|lac)\b", re.IGNORECASE)

_DIGIT_RE = re.compile(
    r"(?:INR|Rs\.?|₹)?\s*"
    r"([\d][\d,]*\.?\d*)\s*"
    r"(crore|cr\.?|lakh|lac)\b", re.IGNORECASE)

_PLAIN_DIGIT_MONEY_RE = re.compile(
    r"(?:INR|Rs\.?|₹)\s*([\d][\d,]*\.?\d*)")

# A bare Indian-grouped number with no currency word/symbol at all, e.g.
# "33,38,00,000". Only trust this when it has comma grouping AND at least
# 6 digits total, so we never mistake a date, count, or cert id for money.
_BARE_GROUPED_RE = re.compile(r"\b(\d{1,2}(?:,\d{2})*,\d{3})\b")


def words_to_number(phrase: str) -> float | None:
    phrase = phrase.strip().strip("-").strip()
    if not phrase:
        return None
    try:
        return float(w2n.word_to_num(phrase))
    except ValueError:
        return None


def parse_money_phrase(text: str) -> float | None:
    """Find the first money-like phrase in `text` and return rupees as a float.
    Returns None if nothing recognisable is found."""
    # 1) digit + unit, e.g. "INR 33.38 Cr", "3,338.00 Lakh", "20 Cr"
    m = _DIGIT_RE.search(text)
    if m:
        num = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower()
        mult = CRORE if unit.startswith("cr") else LAKH
        return num * mult

    # 2) spelled-out numeral + unit, e.g. "seventy-three crore", "twenty crore"
    m = _NUMWORD_RE.search(text)
    if m:
        num = words_to_number(m.group(1))
        if num is not None:
            unit = m.group(2).lower()
            mult = CRORE if unit.startswith("cr") else LAKH
            return num * mult

    # 3) plain rupee figure with digit grouping, e.g. "INR 2,942,400,000" or
    #    "33,38,00,000" (Indian grouping) — just strip separators.
    m = _PLAIN_DIGIT_MONEY_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))

    # 4) bare Indian-grouped digits, no currency marker (e.g. "33,38,00,000")
    m = _BARE_GROUPED_RE.search(text)
    if m:
        digits = m.group(1).replace(",", "")
        if len(digits) >= 6:
            return float(digits)

    return None


def parse_all_money_phrases(text: str) -> list[float]:
    """Return every money figure mentioned in the text, in order of appearance."""
    out = []
    for regex in (_DIGIT_RE, _NUMWORD_RE):
        for m in regex.finditer(text):
            if regex is _DIGIT_RE:
                num = float(m.group(1).replace(",", ""))
            else:
                num = words_to_number(m.group(1))
                if num is None:
                    continue
            unit = m.group(2).lower()
            mult = CRORE if unit.startswith("cr") else LAKH
            out.append(num * mult)
    return out


if __name__ == "__main__":
    tests = [
        "INR 33.38 Cr", "3,338.00 Lakh", "33,38,00,000",
        "twenty crore", "seventy-three crore", "six crore",
        "credential target of INR 20 Cr",
        "crossing the seventy-three crore mark",
    ]
    for t in tests:
        print(f"{t!r:45s} -> {parse_money_phrase(t)}")
