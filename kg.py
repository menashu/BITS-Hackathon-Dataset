"""
kg.py — loads knowledge_graph.json and builds fast, queryable indices.

The knowledge graph already carries the *extracted facts* out of the 687-document
corpus (project name, client, category, value, completion date, lead engineer,
reference-letter presence, client certificate ref). We do NOT re-parse PDFs here —
that extraction step already happened when the graph was built. This module just
turns the flat lists into something a deterministic solver can query in O(1)/O(n).

If you regenerate knowledge_graph.json with more fields (e.g. from workbooks,
ledgers, tender dossiers) later, extend Project / index_by_* accordingly — the
solver layer (solve.py) only touches these accessors, never the raw JSON.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from dateutil import parser as dateparser


@dataclass
class Project:
    id: str
    name: str
    client: str
    category: str
    value: float
    completion_date: Optional[date]
    lead: Optional[str]
    client_certificate_ref: Optional[str]
    documents: list
    quality: Optional[str]
    reference_letter: bool
    raw: dict = field(repr=False, default=None)


@dataclass
class Credential:
    person: str
    type: str
    id: str
    issued: Optional[date]
    document: str


class KnowledgeGraph:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.projects: list[Project] = []
        for p in data.get("projects", []):
            self.projects.append(Project(
                id=p["id"],
                name=p["name"],
                client=p["client"],
                category=p["category"],
                value=float(p["value"]),
                completion_date=_safe_date(p.get("completion_date")),
                lead=p.get("lead"),
                client_certificate_ref=p.get("client_certificate_ref"),
                documents=p.get("documents", []),
                quality=p.get("quality"),
                reference_letter=bool(p.get("reference_letter")),
                raw=p,
            ))

        self.credentials: list[Credential] = []
        for c in data.get("credentials", []):
            self.credentials.append(Credential(
                person=c["person"],
                type=c["type"],
                id=c["id"],
                issued=_safe_date(c.get("issued")),
                document=c.get("document"),
            ))

        # ---- indices ----
        self.by_id = {p.id: p for p in self.projects}
        self.by_name = {p.name: p for p in self.projects}
        self.by_client: dict[str, list[Project]] = {}
        self.by_lead: dict[str, list[Project]] = {}
        for p in self.projects:
            self.by_client.setdefault(p.client, []).append(p)
            if p.lead:
                self.by_lead.setdefault(p.lead, []).append(p)

        self.cred_by_person: dict[str, list[Credential]] = {}
        self.cred_by_id: dict[str, Credential] = {}
        for c in self.credentials:
            self.cred_by_person.setdefault(c.person, []).append(c)
            self.cred_by_id[c.id] = c

        self.all_persons = sorted(set(
            [p.lead for p in self.projects if p.lead] +
            [c.person for c in self.credentials]
        ))
        self.all_clients = sorted(set(p.client for p in self.projects))
        self.all_categories = sorted(set(p.category for p in self.projects))
        self.all_project_names = sorted(set(p.name for p in self.projects))
        self.all_cert_ids = sorted(self.cred_by_id.keys())

    # ---- convenience queries ----
    def projects_for_client(self, client: str) -> list[Project]:
        return list(self.by_client.get(client, []))

    def projects_for_lead(self, person: str) -> list[Project]:
        return list(self.by_lead.get(person, []))

    def credential_for_person(self, person: str, cert_type: str = None,
                               cert_id: str = None) -> Optional[Credential]:
        cands = self.cred_by_person.get(person, [])
        if cert_id:
            for c in cands:
                if c.id == cert_id:
                    return c
        if cert_type:
            cands = [c for c in cands if c.type == cert_type]
        return cands[0] if cands else None


def _safe_date(s):
    if not s:
        return None
    try:
        return dateparser.parse(s).date()
    except Exception:
        return None
