"""
run.py — end-to-end pipeline.

    question text  --[llm_parser or heuristic_parser]-->  ParsedQuestion (shape + slot mentions)
                   --[resolver]-->  grounded KG entities (person/client/project/category/dates/threshold)
                   --[solve]-->     plain number
                   --[format_answer]--> value formatted per answer_type

Two modes:
  --selftest      score against sample_questions.json (ground truth included) — no LLM needed by
                   default (uses heuristic parser), or pass --llm to score the real LLM parser.
  --questions F --out submission.csv
                   answer every question in F (same schema as questions.json) and write a
                   submission CSV in the exact format evaluate.py expects.

Usage:
    python run.py --selftest
    python run.py --selftest --llm                       # requires LLM_API_KEY
    python run.py --questions questions.json --out submission.csv
    python run.py --questions questions.json --out submission.csv --llm
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current directory, if present
except ImportError:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current folder, if present
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

from kg import KnowledgeGraph
from resolver import Resolver
import moneyparse
import solve
from solve import Unresolvable
from llm_parser import ParsedQuestion


def resolve_and_solve(kg: KnowledgeGraph, resolver: Resolver, question: str,
                       answer_type: str, parsed: ParsedQuestion):
    shape = parsed.shape

    # ---- ground every mention against the closed KG entity lists ----
    person = (resolver.resolve_person(parsed.person_mention) if parsed.person_mention else None) \
        or resolver.scan_for_person(question)

    client = (resolver.resolve_client(parsed.client_mention) if parsed.client_mention else None) \
        or resolver.scan_for_client(question)

    proj_name = resolver.resolve_project(parsed.project_mention) if parsed.project_mention else None
    project = kg.by_name.get(proj_name) if proj_name else None

    # If we know the person, narrow the fuzzy project search to just *their*
    # projects (a handful of names) rather than all 155 — this must run
    # BEFORE any global scan, since a global fuzzy match over noisy question
    # prose can lock onto a same-named project belonging to a different
    # person/package number ("WTP Augmentation" exists under several
    # packages/leads, e.g. Pkg-30 vs Pkg-51).
    if not project and person:
        from rapidfuzz import fuzz as _fuzz
        cands = kg.projects_for_lead(person)
        if cands:
            scored = [(p, _fuzz.token_set_ratio(question, p.name)) for p in cands]
            best, best_score = max(scored, key=lambda t: t[1])
            # require a clear margin over the runner-up, not just an absolute
            # floor, since project names share a lot of common tokens
            # ("West Bengal", "Pkg-", category words) with each other.
            runner_up = max((s for p, s in scored if p is not best), default=0)
            if best_score >= 60 and best_score - runner_up >= 15:
                project = best

    # last resort: fuzzy match against the full 155-project list
    if not project:
        proj_name = resolver.scan_for_project(question)
        project = kg.by_name.get(proj_name) if proj_name else None

    # derive client from project if still missing
    if not client and project:
        client = project.client

    # derive client via person's own projects matching the project mention, if still missing
    if not client and person and parsed.project_mention:
        hint = parsed.project_mention.lower()
        cand = [p for p in kg.projects_for_lead(person)
                if hint in p.name.lower() or p.name.lower() in hint]
        if cand:
            client = cand[0].client
            project = project or cand[0]

    # ---- credential issue date ----
    issued_date = None
    if person:
        cred = kg.credential_for_person(person, cert_type=parsed.cert_type, cert_id=parsed.cert_id)
        if cred:
            issued_date = cred.issued
    if not issued_date:
        phrase = parsed.date_phrase or question
        dates = resolver.extract_dates(phrase)
        if dates:
            issued_date = dates[0]

    # ---- money threshold ----
    threshold = None
    if parsed.threshold_phrase:
        threshold = moneyparse.parse_money_phrase(parsed.threshold_phrase)
    if threshold is None:
        threshold = moneyparse.parse_money_phrase(question)

    # ---- category to exclude ----
    category_exclude = None
    if parsed.category_exclude_mention:
        category_exclude = resolver.resolve_category(parsed.category_exclude_mention)
    if not category_exclude and shape == "exclusion_aggregate":
        phrase = resolver.extract_exclusion_phrase(question)
        category_exclude = resolver.resolve_category(phrase) if phrase else None
        if not category_exclude:
            category_exclude = resolver.scan_for_category(question)

    comparison = parsed.comparison or "above"

    if shape == "absence":
        return solve.absence(kg, client)
    if shape == "date_span":
        return solve.date_span(kg, issued_date, project)
    if shape == "distinct_count":
        return solve.distinct_count(kg, person)
    if shape == "hop_aggregate":
        return solve.hop_aggregate(kg, client)
    if shape == "temporal_chain":
        return solve.temporal_chain(kg, person, issued_date)
    if shape == "avg_work_size":
        return solve.avg_work_size(kg, client)
    if shape == "exclusion_aggregate":
        return solve.exclusion_aggregate(kg, client, category_exclude)
    if shape == "gap_to_threshold":
        return solve.gap_to_threshold(kg, client, threshold)
    if shape == "rank_value":
        return solve.rank_value(kg, client)
    if shape == "referenced_share":
        return solve.referenced_share(kg, client)
    if shape == "threshold_aggregate":
        return solve.threshold_aggregate(kg, client, threshold, comparison)
    return solve.simple_lookup(kg, project=project, client=client, answer_type=answer_type)


def format_answer(value, answer_type: str):
    if value is None:
        return None
    if answer_type == "percent":
        return round(float(value), 2)
    if answer_type in ("count", "days"):
        return int(round(float(value)))
    if answer_type == "money":
        return int(round(float(value)))
    return value


def get_parse_all_fn(use_llm: bool):
    """Returns a function questions:list[str] -> list[ParsedQuestion], done in
    as few LLM calls as possible when use_llm=True."""
    if use_llm:
        from llm_parser import LLMParser
        llm = LLMParser()
        return lambda qs: llm.parse_batch(qs)
    else:
        import heuristic_parser
        return lambda qs: [heuristic_parser.parse(q) for q in qs]


def answer_all(kg, resolver, questions, use_llm: bool, verbose: bool = False):
    parse_all = get_parse_all_fn(use_llm)
    qtexts = [item["question"] for item in questions]
    try:
        parsed_list = parse_all(qtexts)
    except Exception as e:  # noqa: BLE001
        # total parser failure (e.g. no network, bad key) — fall back to
        # heuristics for the whole run rather than losing every row
        print(f"[warn] LLM parsing failed entirely ({e}); falling back to heuristic parser.")
        import heuristic_parser
        parsed_list = [heuristic_parser.parse(q) for q in qtexts]

    results = []
    for item, qtext, parsed in zip(questions, qtexts, parsed_list):
        qid = item.get("qid") or item.get("question_id")
        atype = item.get("answer_type", "money")
        try:
            raw = resolve_and_solve(kg, resolver, qtext, atype, parsed)
            ans = format_answer(raw, atype)
            status = "ok" if ans is not None else "unresolved"
        except Unresolvable as e:
            ans, status = None, f"unresolved: {e}"
        except Exception as e:  # noqa: BLE001
            ans, status = None, f"error: {e}"
            if verbose:
                traceback.print_exc()
        results.append({"qid": qid, "question": qtext, "answer_type": atype,
                         "answer": ans, "status": status,
                         "shape": getattr(parsed, "shape", None)})
    return results


def score(results, gold_by_qid):
    scored, total = 0.0, 0
    breakdown = []
    for r in results:
        gold = gold_by_qid.get(r["qid"])
        if gold is None:
            continue
        total += 1
        if r["answer"] is None:
            s = 0.0
        else:
            s = max(0.0, 1 - abs(r["answer"] - gold) / gold) if gold != 0 else (1.0 if r["answer"] == 0 else 0.0)
        breakdown.append((r["qid"], r["shape"], r["answer"], gold, round(s, 4), r["status"]))
        scored += s
    return (scored / total if total else 0.0), breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="./knowledge_graph.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--samples", default="./sample_questions.json")
    ap.add_argument("--questions")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--llm", action="store_true", help="use the real LLM parser instead of heuristics")
    ap.add_argument("--batch-size", type=int, default=None,
                     help="questions per LLM call (default: LLM_BATCH_SIZE env var, or 10)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.batch_size:
        os.environ["LLM_BATCH_SIZE"] = str(args.batch_size)

    kg = KnowledgeGraph(args.kg)
    resolver = Resolver(kg)

    if args.selftest:
        data = json.load(open(args.samples))
        questions = data["questions"]
        gold_by_qid = {q["qid"]: q["answer"] for q in questions}
        results = answer_all(kg, resolver, questions, use_llm=args.llm, verbose=args.verbose)
        avg, breakdown = score(results, gold_by_qid)
        print(f"{'qid':12s} {'shape':20s} {'answer':>14s} {'gold':>14s} {'score':>7s}  status")
        for qid, shape, ans, gold, s, status in breakdown:
            print(f"{qid:12s} {str(shape):20s} {str(ans):>14s} {str(gold):>14s} {s:7.3f}  {status}")
        print(f"\nSelf-test mean score: {avg:.4f} over {len(breakdown)} questions "
              f"(parser={'LLM' if args.llm else 'heuristic'})")
        return

    if args.questions:
        data = json.load(open(args.questions))
        questions = data["questions"] if isinstance(data, dict) and "questions" in data else data
        results = answer_all(kg, resolver, questions, use_llm=args.llm, verbose=args.verbose)
        n_unresolved = sum(1 for r in results if r["answer"] is None)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question_id", "answer"])
            for r in results:
                # never leave a row blank — a 0 guess scores 0 like an unanswered
                # row does, but a *plausible* fallback can still score partial
                # credit, so fall back to a coarse default instead of leaving it empty.
                ans = r["answer"] if r["answer"] is not None else 0
                w.writerow([r["qid"], ans])
        print(f"Wrote {len(results)} rows to {args.out}. {n_unresolved} were unresolved "
              f"(check the log below before submitting).")
        for r in results:
            if r["answer"] is None:
                print(f"  UNRESOLVED {r['qid']} [{r['shape']}] {r['status']}: {r['question'][:90]}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
