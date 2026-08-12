# Bid Intelligence — reasoning layer over your knowledge graph

This is the **question-answering layer** on top of the knowledge graph you already
built (`knowledge_graph.json`: 155 projects + 39 credentials, fully extracted from
the 687-document corpus). It deliberately keeps the LLM **out of every numeric
step** — the LLM only reads a question and points at which entities/shape it's
about; a plain Python layer re-grounds those pointers against your KG and does
all arithmetic. That's what makes the answers reproducible and auditable, which
is what the scoring formula (`1 - |error|/gold`) rewards.

```
question text
     │
     ▼
llm_parser.py / heuristic_parser.py   →  ParsedQuestion(shape, person_mention, client_mention,
     │                                                    project_mention, cert_id, threshold_phrase, ...)
     ▼
resolver.py   →  re-grounds every mention against the closed KG entity lists
     │            (exact match → substring → fuzzy, never invented)
     ▼
solve.py      →  deterministic shape handler, pure arithmetic over kg.py objects
     │
     ▼
plain number, formatted per answer_type (money/count/percent/days)
```

## Files

| file | role |
|---|---|
| `kg.py` | loads `knowledge_graph.json`, builds indices (`by_client`, `by_lead`, `cred_by_person`, ...) |
| `moneyparse.py` | converts money phrases ("INR 33.38 Cr", "twenty crore", "33,38,00,000") to rupees — pure regex/arithmetic, no LLM |
| `resolver.py` | fuzzy-grounds a mention against the KG's closed entity lists; also pulls dates, cert IDs, exclusion phrases with regex |
| `solve.py` | one function per question "shape" (`absence`, `hop_aggregate`, `avg_work_size`, ...); pure arithmetic, no LLM |
| `llm_parser.py` | the **only** file that calls an LLM (OpenRouter or Groq, batched calls). Classifies shape + extracts slot mentions. Never computes a number. |
| `heuristic_parser.py` | zero-API-cost keyword classifier — used for the self-test below, and as an automatic runtime fallback if the LLM call fails |
| `run.py` | CLI: self-test against `sample_questions.json`, or answer a full `questions.json` and write `submission.csv` |

## Validate the deterministic engine (no API key needed)

```bash
pip install -r requirements.txt --break-system-packages
python run.py --selftest
```

This scores the pipeline against all 21 worked examples in `sample_questions.json`
using the exact scoring formula from the README. On this bundle it currently gets
**1.0000** — i.e. every sample question is answered exactly — using only the
offline heuristic classifier, which confirms `kg.py` / `resolver.py` / `solve.py`
correctly reproduce the reasoning chains the samples document (hop-aggregate really
does total the *whole client*, not just the named person; exclusion really does
read the category field; etc).

**Don't over-read that 1.0 as "solved."** The heuristic classifier is keyword
rules tuned to the 21 samples' phrasing. The README is explicit that the hidden
371 are "not templated, and no two are phrased alike" — that's exactly the gap
the LLM step is for. Wire up a real key and re-run with `--llm` before trusting it
on `questions.json`.

## Setting your API key

### Easiest: a `.env` file (works the same on Windows/Mac/Linux)

1. Copy `.env.example` to a new file named `.env` in the same folder as `run.py`.
2. Open it in Notepad and fill in your key:
   ```
   LLM_PROVIDER=openrouter
   LLM_API_KEY=sk-or-v1-your_actual_key
   LLM_MODEL=openai/gpt-oss-120b:free
   ```
3. Save it. `run.py` loads this file automatically on every run — nothing else
   to set up, and no risk of the key vanishing when you close the terminal.

`.env` is just a private config file sitting next to your code; it isn't sent
anywhere and doesn't need admin rights. Don't commit it if this folder ever goes
into git (add `.env` to `.gitignore`).

### Why OpenRouter instead of Groq by default now

Groq's free tier caps at 100K *tokens/day total* — and this pipeline was
originally sending the full system prompt + few-shot examples on every single
question, so it burned through that budget partway through one run. Two
things changed to fix this:

1. **Default provider is now OpenRouter**, model `openai/gpt-oss-120b:free` —
   a capable free-tier model with native structured/JSON output support.
   OpenRouter's free lineup rotates; if this model stops working, swap
   `LLM_MODEL` in `.env` for anything ending in `:free` from
   [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0) —
   no code change needed.
2. **Calls are now batched** (see below) — the fixed prompt cost is paid once
   per ~10 questions instead of once per question, which cuts total token
   usage roughly 10x regardless of which provider you use.

If you'd rather stick with Groq, uncomment the Groq block in `.env.example`.

### Alternative: setting environment variables directly

If you'd rather not use a `.env` file, `export` (Mac/Linux) has Windows equivalents:

**PowerShell**:
```powershell
$env:LLM_PROVIDER = "openrouter"
$env:LLM_API_KEY  = "sk-or-v1-your_actual_key"
$env:LLM_MODEL    = "openai/gpt-oss-120b:free"
python run.py --selftest --llm
```
These only last for the current PowerShell window — set them again if you open a
new one (or just use the `.env` file instead, which is why it's the recommended
route above).

**Command Prompt (cmd.exe)**:
```cmd
set LLM_PROVIDER=openrouter
set LLM_API_KEY=sk-or-v1-your_actual_key
set LLM_MODEL=openai/gpt-oss-120b:free
python run.py --selftest --llm
```

Either way, once the key is set (via `.env` or your shell), run:

```bash
python run.py --selftest --llm
```

### Rate limits and batching

Free tiers on both Groq and OpenRouter cap you on tokens/day *and* request
count, and the fixed cost per call here — the system prompt plus three
few-shot examples — is the dominant chunk of that budget if you send one
request per question. `llm_parser.py` now batches multiple questions into a
single call (`LLM_BATCH_SIZE`, default 10) so that fixed cost is paid once per
10 questions instead of 371 times. If a batch's JSON comes back malformed
(small free models occasionally do this), the pipeline automatically bisects
that batch and retries in half rather than losing the whole chunk.

Tune it in `.env` (`LLM_BATCH_SIZE=15`) or per-run (`--batch-size 15`):
- Higher (15–20): fewer calls, more efficient, but a bad response corrupts a
  bigger chunk before bisection kicks in.
- Lower (5–8): more resilient to a flaky model, more total calls.

If you still hit a rate limit mid-run, nothing is lost — `run.py` catches a
total parser failure and falls back to the heuristic parser for the rest of
the run, so you'll still get a full CSV; check the log for which rows used
the fallback and consider re-running just those once your quota resets.

Compare the `--llm` self-test score/log against the heuristic one. Any question
where the LLM's shape/slot call disagrees with the heuristic's and scores worse is
worth reading by hand — that's your signal for which shape definitions or few-shot
examples in `llm_parser.py` need sharpening before the full run.

## Answer the real question set

Once `--selftest --llm` looks solid:

```bash
python run.py --questions questions.json --out submission.csv --llm
python evaluate.py --submission submission.csv --questions sample_questions.json   # sanity check
```

`run.py` prints every `UNRESOLVED` question_id with the reason (couldn't ground a
client, couldn't find a credential, etc.) before you submit — treat that list as a
todo list, not noise. A row is never left blank (an unresolved question still gets
a `0` fallback so it doesn't outright fail the CSV format), but you want that list
as short as possible; each one is scoring 0 until you either fix the resolver/solver
for that pattern or hand-answer it from the source PDF.

## Extending to shapes beyond the 21 samples

`solve.py`'s docstring lists the 11 shapes the samples exhibit plus a
`simple_lookup` fallback for anything else. The hidden 371 almost certainly reuse
these shapes (README: "the hidden scoring set is larger and harder, but is the
same kinds of question") but at more hops and with more paraphrase variety. If you
hit a genuinely new shape while reading `run.py`'s `UNRESOLVED` log:

1. Add the shape name to `SHAPES` in `llm_parser.py` and describe it in the system
   prompt + one few-shot example.
2. Add a matching keyword rule to `heuristic_parser.classify_shape` (keeps the
   offline fallback useful).
3. Add a pure-arithmetic handler function to `solve.py` and a branch in
   `run.py:resolve_and_solve`.

Keep the same discipline: the new handler takes only already-grounded KG objects
and returns a number — no LLM call inside it.

## Why not just ask the LLM for the number directly?

Two reasons this design holds up better under the scoring formula:

1. **Determinism.** The KG's numbers (`project.value`, `.completion_date`,
   `.reference_letter`) are already exact facts pulled out of the PDFs. Routing
   them through an LLM's arithmetic re-introduces the risk of transcription and
   rounding errors that the extraction step already eliminated.
2. **Auditability of failure.** When `resolver.py` can't ground a mention, you get
   an explicit `Unresolvable` with a reason — a todo list you can work through — 
   instead of a confident-sounding wrong number. The README calls this out
   directly: *"A system that hallucinates connections will confidently say zero"* —
   the flip side of that risk is a system that confidently invents a value when it
   should say "I don't know."
