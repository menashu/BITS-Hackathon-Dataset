"""
llm_parser.py — the ONLY place an LLM is called.

Its job is narrow and deliberately limited: read a question and point at which
spans of text refer to which slot (person, client, project, category, cert id,
threshold phrase, date phrase) and which of the known question "shapes" it is.
It never computes anything and never invents a canonical name — those spans
get re-grounded against the closed KG entity lists in resolver.py, and all
arithmetic happens in solve.py. If the LLM hallucinates a client name that
doesn't exist, resolver.py will simply fail to match it and the pipeline will
flag the question rather than answer wrong.

Works with any OpenAI-compatible chat completions endpoint — OpenRouter and
Groq both qualify. Configure with environment variables (or a .env file,
loaded automatically by run.py):

    LLM_PROVIDER   = "openrouter" | "groq"     (default: openrouter)
    LLM_API_KEY    = your API key
    LLM_MODEL      = e.g. "openai/gpt-oss-120b:free" (openrouter free tier,
                     supports native structured/JSON output) or
                     "llama-3.3-70b-versatile" (groq)

Calls are BATCHED — several questions go into one request instead of one
request per question — because free-tier limits (on both Groq and OpenRouter)
bite on total daily tokens *and* request count, and the system prompt + few-shot
examples below are the dominant cost per call. Batching amortizes that fixed
cost across many questions instead of paying it 371 times.

Example:
    LLM_PROVIDER=openrouter
    LLM_API_KEY=sk-or-v1-...
    LLM_MODEL=openai/gpt-oss-120b:free
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

SHAPES = [
    "absence", "date_span", "distinct_count", "hop_aggregate", "temporal_chain",
    "avg_work_size", "exclusion_aggregate", "gap_to_threshold", "rank_value",
    "referenced_share", "threshold_aggregate", "simple_lookup",
]

SLOT_KEYS = [
    "shape", "person_mention", "client_mention", "project_mention", "cert_id",
    "cert_type", "category_exclude_mention", "threshold_phrase", "comparison",
    "date_phrase",
]

SYSTEM_PROMPT = f"""You are a slot-extraction and classification engine for a bid-desk \
question answering system over a construction company's project archive. \
You NEVER compute a numeric answer. For EACH question given to you, you ONLY:
  1. Classify it into exactly one of these shapes: {", ".join(SHAPES)}
  2. Extract the exact text spans (verbatim substrings of the question, or your \
best paraphrase of the entity name if not verbatim) for each slot that applies.

Shape definitions:
- absence: count of a client's completed works that have NO reference letter on file.
- date_span: days between a person's credential issue date and a named project's completion date.
- distinct_count: number of distinct work categories a person has led to completion.
- hop_aggregate: total value of EVERY project for a client (all leads, not just the \
named person) — the person/project mentioned is only used to identify WHICH client.
- temporal_chain: total value of a person's OWN led projects that completed AFTER \
their credential's issue date.
- avg_work_size: average project value across a client's full portfolio — the \
person/project mentioned is only used to identify WHICH client.
- exclusion_aggregate: total value of a client's projects, excluding one named category.
- gap_to_threshold: money still needed for a client's current total to reach a stated target.
- rank_value: difference between a client's largest and second-largest project value.
- referenced_share: percent of a client's projects that have a reference letter on file.
- threshold_aggregate: total value of a client's projects strictly above (or below) a stated cutoff.
- simple_lookup: anything that doesn't fit the above — a direct fact lookup.

You will be given a JSON array of questions, each with an "idx". Return STRICT JSON only
(no prose, no markdown fences) with this exact shape:
{{
  "parses": [
    {{
      "idx": <same idx as input>,
      "shape": "<one of the shapes above>",
      "person_mention": "<string or null>",
      "client_mention": "<string or null>",
      "project_mention": "<string or null>",
      "cert_id": "<string or null, e.g. PMI-200029>",
      "cert_type": "<string or null, e.g. PMP>",
      "category_exclude_mention": "<string or null>",
      "threshold_phrase": "<verbatim span containing the money threshold, or null>",
      "comparison": "<'above' or 'below' or null>",
      "date_phrase": "<verbatim span containing an explicit date, or null>"
    }},
    ...
  ]
}}
Return exactly one parse object per input question, same order, matching idx values.

Notes:
- A question can mention a person only to identify a client or project — that does \
not make the aggregation person-filtered unless the shape is temporal_chain or distinct_count.
- If unsure between two shapes, prefer the one whose description matches the \
question's final ask (the number it wants), not the narrative framing.
"""

FEWSHOT_QUESTIONS = [
    "Cross-checking against the Public Health Engineering Dept, Gujarat, how many "
    "works have no client reference letter on file?",
    "Starting from Rahul Menon\u2019s PMP certification (PMI-200029) for the Ring Road "
    "\u2014 Maharashtra Pkg-125, what is the combined value of every completed "
    "assignment he has delivered for the Public Works Department, Govt of Maharashtra?",
    "Jal Nigam, Jharkhand, what\u2019s the combined value of their works crossing the "
    "seventy-three crore mark so I can lock the bid before the deadline?",
]
FEWSHOT_ANSWERS = [
    {"shape": "absence", "person_mention": None,
     "client_mention": "Public Health Engineering Dept, Gujarat",
     "project_mention": None, "cert_id": None, "cert_type": None,
     "category_exclude_mention": None, "threshold_phrase": None,
     "comparison": None, "date_phrase": None},
    {"shape": "hop_aggregate", "person_mention": "Rahul Menon",
     "client_mention": "Public Works Department, Govt of Maharashtra",
     "project_mention": "Ring Road \u2014 Maharashtra Pkg-125",
     "cert_id": "PMI-200029", "cert_type": "PMP",
     "category_exclude_mention": None, "threshold_phrase": None,
     "comparison": None, "date_phrase": None},
    {"shape": "threshold_aggregate", "person_mention": None,
     "client_mention": "Jal Nigam, Jharkhand", "project_mention": None,
     "cert_id": None, "cert_type": None, "category_exclude_mention": None,
     "threshold_phrase": "seventy-three crore", "comparison": "above",
     "date_phrase": None},
]


@dataclass
class ParsedQuestion:
    shape: str
    person_mention: Optional[str] = None
    client_mention: Optional[str] = None
    project_mention: Optional[str] = None
    cert_id: Optional[str] = None
    cert_type: Optional[str] = None
    category_exclude_mention: Optional[str] = None
    threshold_phrase: Optional[str] = None
    comparison: Optional[str] = None
    date_phrase: Optional[str] = None


PROVIDER_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        # gpt-oss-120b:free supports native structured/JSON output and is
        # currently one of the stronger free-tier OpenRouter models for this
        # kind of extraction task. Free-tier availability rotates — check
        # https://openrouter.ai/models?max_price=0 if this stops working and
        # swap LLM_MODEL, no code change needed.
        "default_model": "openai/gpt-oss-120b:free",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
}


def _client(provider: str, api_key: str) -> tuple[OpenAI, str]:
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"unknown LLM_PROVIDER {provider!r} (use 'openrouter' or 'groq')")
    cfg = PROVIDER_DEFAULTS[provider]
    extra_headers = {}
    if provider == "openrouter":
        # optional but recommended by OpenRouter for routing/analytics; safe to omit
        extra_headers = {
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_NAME", "bid-intel"),
        }
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key,
                     default_headers=extra_headers or None)
    return client, cfg["default_model"]


class LLMParser:
    def __init__(self, provider: str = None, api_key: str = None, model: str = None,
                 batch_size: int = 10):
        provider = provider or os.environ.get("LLM_PROVIDER", "openrouter")
        api_key = api_key or os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("Set LLM_API_KEY (an OpenRouter or Groq key), e.g. via a .env file.")
        self.client, default_model = _client(provider, api_key)
        self.model = model or os.environ.get("LLM_MODEL", default_model)
        self.batch_size = int(os.environ.get("LLM_BATCH_SIZE", batch_size))

    def _few_shot_messages(self):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        demo_in = [{"idx": i, "question": q} for i, q in enumerate(FEWSHOT_QUESTIONS)]
        demo_out = {"parses": [
            {"idx": i, **a} for i, a in enumerate(FEWSHOT_ANSWERS)
        ]}
        msgs.append({"role": "user", "content": json.dumps({"questions": demo_in})})
        msgs.append({"role": "assistant", "content": json.dumps(demo_out)})
        return msgs

    def parse(self, question: str, retries: int = 3) -> ParsedQuestion:
        """Single-question convenience wrapper around parse_batch."""
        return self.parse_batch([question], retries=retries)[0]

    def parse_batch(self, questions: list[str], retries: int = 3) -> list[ParsedQuestion]:
        """Parse many questions in as few LLM calls as possible. Splits into
        chunks of self.batch_size, and on failure bisects a chunk rather than
        giving up on the whole thing, so one bad question doesn't sink an
        entire batch's worth of tokens."""
        results: list[Optional[ParsedQuestion]] = [None] * len(questions)
        self._errors: list[str] = []
        self._parse_range(questions, list(range(len(questions))), results, retries)
        n_failed = sum(1 for r in results if r is None)
        if n_failed:
            uniq_errs = sorted(set(self._errors))[:5]
            print(f"[llm_parser] {n_failed}/{len(questions)} questions fell back to "
                  f"simple_lookup after retries. Sample error(s):")
            for e in uniq_errs:
                print(f"    - {e}")
        # anything still None (persistent failure) gets a safe fallback
        for i, r in enumerate(results):
            if r is None:
                results[i] = ParsedQuestion(shape="simple_lookup")
        return results

    def _parse_range(self, all_questions, idxs, results, retries):
        if not idxs:
            return
        for start in range(0, len(idxs), self.batch_size):
            chunk = idxs[start:start + self.batch_size]
            try:
                parsed = self._call_batch([all_questions[i] for i in chunk], retries=retries)
                for local_i, global_i in enumerate(chunk):
                    results[global_i] = parsed[local_i]
            except Exception as e:  # noqa: BLE001
                if len(chunk) == 1:
                    results[chunk[0]] = None  # give up, fallback applied by caller
                    self._errors.append(str(e)[:200])
                else:
                    mid = len(chunk) // 2
                    self._parse_range(all_questions, chunk[:mid], results, retries)
                    self._parse_range(all_questions, chunk[mid:], results, retries)

    def _call_batch(self, questions: list[str], retries: int) -> list[ParsedQuestion]:
        payload = {"questions": [{"idx": i, "question": q} for i, q in enumerate(questions)]}
        msgs = self._few_shot_messages()
        msgs.append({"role": "user", "content": json.dumps(payload)})

        last_err = None
        for attempt in range(retries):
            # Some OpenRouter free models reject response_format={"type":
            # "json_object"} outright (400 error) rather than just ignoring
            # it. Use it on the first attempt; if that specific request
            # fails, drop it on subsequent attempts and rely on the system
            # prompt's "strict JSON only" instruction + _extract_json's
            # markdown-fence stripping instead.
            use_json_mode = (attempt == 0)
            try:
                kwargs = dict(model=self.model, messages=msgs, temperature=0)
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                if not raw or not raw.strip():
                    raise RuntimeError("model returned an empty response")
                try:
                    data = _extract_json(raw)
                except json.JSONDecodeError as je:
                    snippet = raw.strip()[:150].replace("\n", " ")
                    raise RuntimeError(f"non-JSON response ({je}): {snippet!r}") from je
                parse_list = data.get("parses", [])
                if not isinstance(parse_list, list):
                    raise RuntimeError(f"response JSON had no 'parses' list (keys: {list(data.keys())})")

                # Normalize idx robustly — some models (especially smaller
                # free ones) return "idx" as a string ("0") instead of an
                # int, or drop it. A naive int-keyed lookup then silently
                # matches nothing and every question falls back to
                # simple_lookup, which tanks accuracy without ever raising
                # an error. Coerce what we can; if idx is unusable, fall
                # back to positional order.
                parses = {}
                for i, p in enumerate(parse_list):
                    idx = p.get("idx", i)
                    try:
                        idx = int(idx)
                    except (TypeError, ValueError):
                        idx = i
                    parses[idx] = p

                # If we got a response but couldn't match ANY of it back to
                # a question, that's a malformed batch, not a "no slots
                # found" situation — treat it as a failure so the retry/
                # bisection logic in parse_batch kicks in instead of quietly
                # answering every question with simple_lookup.
                if questions and not any(k in parses for k in range(len(questions))):
                    raise RuntimeError(
                        f"batch response idx values didn't match any question "
                        f"(got {[p.get('idx') for p in parse_list][:5]}...)")

                out = []
                for i in range(len(questions)):
                    p = parses.get(i, {})
                    shape = p.get("shape") if p.get("shape") in SHAPES else "simple_lookup"
                    out.append(ParsedQuestion(**{
                        **{k: p.get(k) for k in SLOT_KEYS if k != "shape"},
                        "shape": shape,
                    }))
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM batch parse failed after {retries} attempts: {last_err}")


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)
