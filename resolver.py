"""
resolver.py — grounds free-text mentions in a question to exact KG entities.

This is the anti-hallucination layer. The LLM is good at noticing *that* a
question refers to "the commissioning client" or "Rajesh Rao's Six Sigma cert",
but it must never be trusted to spell out the exact client string or invent a
value — every entity it points at is re-resolved here against the closed list
of names that actually exist in the knowledge graph, using exact match first
and fuzzy match only as a fallback. If nothing clears the similarity floor we
return None rather than guess, and the caller can flag the question instead of
silently answering wrong.
"""
from __future__ import annotations
import re
from datetime import date
from dateutil import parser as dateparser
from rapidfuzz import fuzz, process

from kg import KnowledgeGraph

FUZZY_FLOOR = 82  # rapidfuzz token_sort_ratio; below this we refuse to guess

_CERT_ID_RE = re.compile(r"\bPMI-\d{4,}\b", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?,?\s+\d{4})\b",
    re.IGNORECASE)


class Resolver:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    # ---- exact-first, fuzzy-fallback lookups against closed lists ----
    def resolve_person(self, mention: str) -> str | None:
        return self._resolve(mention, self.kg.all_persons)

    def resolve_client(self, mention: str) -> str | None:
        return self._resolve(mention, self.kg.all_clients)

    def resolve_project(self, mention: str) -> str | None:
        return self._resolve(mention, self.kg.all_project_names)

    def resolve_category(self, mention: str) -> str | None:
        return self._resolve(mention, self.kg.all_categories, floor=70)

    def _resolve(self, mention: str, candidates: list[str], floor: int = FUZZY_FLOOR) -> str | None:
        if not mention:
            return None
        mention = mention.strip()
        if mention in candidates:
            return mention
        # substring containment (handles "Cable Stayed Bridge — Jharkhand Pkg-115"
        # appearing verbatim inside a longer question sentence)
        for c in candidates:
            if c.lower() in mention.lower() or mention.lower() in c.lower():
                return c
        match = process.extractOne(mention, candidates, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= floor:
            return match[0]
        return None

    # find any known person/client/project/category name that appears
    # *somewhere* inside a full question string (for when the LLM slot is
    # missing/wrong, as a safety net)
    def scan_for_person(self, text: str) -> str | None:
        return self._scan(text, self.kg.all_persons)

    def scan_for_client(self, text: str) -> str | None:
        return self._scan(text, self.kg.all_clients)

    def scan_for_project(self, text: str) -> str | None:
        return self._scan(text, self.kg.all_project_names)

    def scan_for_category(self, text: str, exclude: bool = False) -> str | None:
        return self._scan(text, self.kg.all_categories, floor=70)

    def _scan(self, text: str, candidates: list[str], floor: int = FUZZY_FLOOR) -> str | None:
        low = text.lower()
        hits = [c for c in candidates if c.lower() in low]
        if hits:
            # prefer the longest match (most specific) — guards against a short
            # candidate (e.g. category "Irrigation") accidentally matching
            # inside an unrelated longer name (e.g. client "Irrigation &
            # Waterways Dept..."). Longest verbatim hit wins.
            return max(hits, key=len)
        # fuzzy fallback: the mention may be paraphrased ("Package 51" vs
        # "Pkg-51", "project in West Bengal" vs the em-dash project name).
        # partial_ratio tolerates the candidate being a substring-ish match
        # inside a much longer question sentence.
        match = process.extractOne(text, candidates, scorer=fuzz.token_set_ratio)
        if match and match[1] >= floor:
            return match[0]
        return None

    @staticmethod
    def extract_exclusion_phrase(text: str) -> str | None:
        """Pull just the span naming what's excluded, e.g. 'excluding buildings,'
        -> 'buildings'. Scoping the category lookup to this span (instead of the
        whole question) stops a short category name from accidentally matching
        inside an unrelated client/department name elsewhere in the sentence."""
        m = re.search(
            r"\bexclud\w*\s+(.+?)(?:[,.;]| what | so | before | to | for |$)",
            text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b(?:except|other than|apart from)\s+(.+?)(?:[,.;]| what | so |$)",
            text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    # ---- unambiguous regex extraction (no fuzziness needed) ----
    @staticmethod
    def extract_cert_id(text: str) -> str | None:
        m = _CERT_ID_RE.search(text)
        return m.group(0).upper() if m else None

    @staticmethod
    def extract_dates(text: str) -> list[date]:
        out = []
        for m in _DATE_RE.finditer(text):
            try:
                out.append(dateparser.parse(m.group(0)).date())
            except Exception:
                pass
        return out
