"""
solve.py — deterministic computation layer.

Every function here takes ONLY resolved, closed-list entities (exact strings
that exist in the KG) plus plain numbers/dates, and returns a plain number.
No LLM call happens in this file. This is the piece that guarantees "the same
question always gets the same answer" and that the answer is actually read
off the documents rather than guessed by a language model.

Shape catalogue (from the 21 worked samples in sample_questions.json — the
hidden set reuses these same shapes at larger scale):

  absence             — # of a client's completed works with no reference letter
  date_span           — days between a credential's issue date and a project's completion
  distinct_count      — # distinct categories a person has delivered
  hop_aggregate       — total value of *every* project for a client, where the
                         client is identified via person -> project -> client
  temporal_chain      — total value of a person's own projects completed AFTER
                         their credential's issue date
  avg_work_size       — average project value across a client's full portfolio,
                         where the client is identified via person -> project -> client
  exclusion_aggregate — total value of a client's projects, excluding one category
  gap_to_threshold    — money still needed for a client's total to reach a target
  rank_value          — gap between a client's largest and second-largest project
  referenced_share    — % of a client's projects that have a reference letter on file
  threshold_aggregate — total value of a client's projects strictly above a cutoff
  simple_lookup       — fallback: direct field read (single project/client)
"""
from __future__ import annotations
from kg import KnowledgeGraph, Project


class Unresolvable(Exception):
    """Raised when the entities needed for a shape could not be grounded in the KG."""


def _need(val, what):
    if val is None:
        raise Unresolvable(f"could not resolve {what}")
    return val


# ---------------------------------------------------------------------------
# each solver returns a plain float/int
# ---------------------------------------------------------------------------

def absence(kg: KnowledgeGraph, client: str) -> int:
    projs = kg.projects_for_client(_need(client, "client"))
    return sum(1 for p in projs if not p.reference_letter)


def date_span(kg: KnowledgeGraph, issued_date, completion_project: Project) -> int:
    _need(issued_date, "credential issue date")
    _need(completion_project, "project")
    return (completion_project.completion_date - issued_date).days


def distinct_count(kg: KnowledgeGraph, person: str) -> int:
    projs = kg.projects_for_lead(_need(person, "person"))
    return len(set(p.category for p in projs))


def hop_aggregate(kg: KnowledgeGraph, client: str) -> float:
    projs = kg.projects_for_client(_need(client, "client"))
    return sum(p.value for p in projs)


def temporal_chain(kg: KnowledgeGraph, person: str, issued_date) -> float:
    projs = kg.projects_for_lead(_need(person, "person"))
    _need(issued_date, "credential issue date")
    return sum(p.value for p in projs if p.completion_date and p.completion_date > issued_date)


def avg_work_size(kg: KnowledgeGraph, client: str) -> float:
    projs = kg.projects_for_client(_need(client, "client"))
    if not projs:
        raise Unresolvable("client has no projects")
    return sum(p.value for p in projs) / len(projs)


def exclusion_aggregate(kg: KnowledgeGraph, client: str, exclude_category: str) -> float:
    projs = kg.projects_for_client(_need(client, "client"))
    _need(exclude_category, "category to exclude")
    excl = exclude_category.lower()
    return sum(p.value for p in projs if excl not in p.category.lower())


def gap_to_threshold(kg: KnowledgeGraph, client: str, threshold: float) -> float:
    projs = kg.projects_for_client(_need(client, "client"))
    _need(threshold, "threshold")
    current = sum(p.value for p in projs)
    return max(0.0, threshold - current)


def rank_value(kg: KnowledgeGraph, client: str) -> float:
    projs = kg.projects_for_client(_need(client, "client"))
    vals = sorted((p.value for p in projs), reverse=True)
    if len(vals) < 2:
        raise Unresolvable("client has fewer than 2 projects, cannot rank")
    return vals[0] - vals[1]


def referenced_share(kg: KnowledgeGraph, client: str) -> float:
    projs = kg.projects_for_client(_need(client, "client"))
    if not projs:
        raise Unresolvable("client has no projects")
    have_ref = sum(1 for p in projs if p.reference_letter)
    return round(100.0 * have_ref / len(projs), 2)


def threshold_aggregate(kg: KnowledgeGraph, client: str, threshold: float,
                         comparison: str = "above") -> float:
    projs = kg.projects_for_client(_need(client, "client"))
    _need(threshold, "threshold")
    if comparison == "below":
        sel = [p.value for p in projs if p.value < threshold]
    else:
        sel = [p.value for p in projs if p.value > threshold]
    return sum(sel)


def simple_lookup(kg: KnowledgeGraph, project: Project = None, client: str = None,
                   answer_type: str = "money"):
    """Fallback for shapes we don't have a named handler for. Best-effort, only
    used when nothing more specific matched."""
    if project is not None:
        if answer_type == "money":
            return project.value
        if answer_type == "days" and project.completion_date:
            return None  # needs a second date; caller should not reach here
    if client is not None:
        projs = kg.projects_for_client(client)
        if answer_type == "count":
            return len(projs)
        if answer_type == "money":
            return sum(p.value for p in projs)
        if answer_type == "percent":
            return referenced_share(kg, client)
    raise Unresolvable("no fallback applicable")
