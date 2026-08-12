"""
heuristic_parser.py — a zero-dependency, zero-API-cost shape classifier.

This exists for two reasons:
1. So the pipeline is fully testable/demoable right now, without a Groq/
   OpenRouter key or network access.
2. As a genuine runtime fallback: if the LLM call errors out or the account
   is rate-limited mid-run, the pipeline degrades to this instead of losing
   the row. It is less robust to novel phrasing than the LLM, but on the
   sample set it is exact (see run.py --selftest).

It does not try to extract entity mentions itself (that's noisy without an
LLM) — instead it just classifies the shape from keyword cues, and lets
run.py fall back to resolver.scan_for_* over the raw question text to find
the person/client/project/category, and moneyparse over the raw text to find
thresholds. That scan-the-whole-question approach is exactly what the LLM
slot-extraction is meant to make more precise for messier phrasing.
"""
from __future__ import annotations
import re
from llm_parser import ParsedQuestion


def classify_shape(q: str) -> str:
    low = q.lower()

    if re.search(r"no\s+(?:client\s+)?reference letter|lack(?:s)?\s+a?\s*(?:client\s+)?reference letter|"
                 r"without\s+a?\s*reference letter", low):
        return "absence"

    if re.search(r"\bdays?\b|interval|number of days|how long", low) and \
       re.search(r"issu|certif|pmp", low):
        return "date_span"

    if re.search(r"distinct|different (?:work )?categor|how many.*categor", low):
        return "distinct_count"

    if re.search(r"\bexcluding\b|excludes?\b|apart from|other than", low):
        return "exclusion_aggregate"

    if re.search(r"additional (?:work|amount)|how much more|to reach|reach (?:our|the) (?:credential )?target|"
                 r"still need|shortfall|gap to", low):
        return "gap_to_threshold"

    if re.search(r"largest.*second largest|second largest.*largest|exceed the second|"
                 r"top two|difference between the largest", low):
        return "rank_value"

    if re.search(r"out of one hundred|percent|percentage|\bshare\b.*(?:reference|verif)|"
                 r"(?:reference|verif).*\bshare\b", low):
        return "referenced_share"

    if re.search(r"crossing|hitting|above|exceeding|over the|more than|the .* mark\b|"
                 r"the .* line\b", low) and re.search(r"crore|lakh|inr|rs\.?\s*\d|₹", low):
        return "threshold_aggregate"

    if re.search(r"\baverage\b|\bmean\b", low):
        return "avg_work_size"

    if re.search(r"after (?:that date|her|his|the)|completed after|wrapped up after|after.*issu", low):
        return "temporal_chain"

    if re.search(r"combined value|total value|aggregate value|value of every|value of all", low):
        return "hop_aggregate"

    return "simple_lookup"


def parse(question: str) -> ParsedQuestion:
    return ParsedQuestion(shape=classify_shape(question))
