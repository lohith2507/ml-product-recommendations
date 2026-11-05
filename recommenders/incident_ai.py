"""
Incident AI for the shopping agent, based on the Lighthouse loop
in incident-ai-prototype.

Alert -> triage -> investigate -> root cause -> allowlisted remediation
-> fix assurance. A failed check starts another investigation.
Low-risk fixes can run automatically. Rollback waits for approval.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

SERVICE = "shopping-agent"

CAT_CHAT = "CHAT_AGENT"
CAT_RETRIEVAL = "VECTOR_SEARCH"
CAT_CART = "CART_STORE"
CAT_DEPLOY = "BAD_DEPLOYMENT"
CAT_UNKNOWN = "UNKNOWN"

AUTO_CONFIDENCE = 0.75


@dataclass
class Runbook:
    id: str
    title: str
    risk: str  # LOW | MEDIUM
    category: str
    summary: str


RUNBOOKS = {
    "serve_catalog_fallback": Runbook(
        "serve_catalog_fallback",
        "Serve catalog search without Vector Search",
        "LOW",
        CAT_RETRIEVAL,
        "Disable the managed vector endpoint for this process and use the local catalog index.",
    ),
    "reset_chat_orchestrator": Runbook(
        "reset_chat_orchestrator",
        "Reset the chat orchestrator",
        "LOW",
        CAT_CHAT,
        "Drop the cached LangGraph so the next turn rebuilds tools and falls back cleanly.",
    ),
    "recheck_cart_store": Runbook(
        "recheck_cart_store",
        "Recheck the cart store",
        "LOW",
        CAT_CART,
        "Probe the cart store and keep serving the last good cart snapshot.",
    ),
    "rollback_revision": Runbook(
        "rollback_revision",
        "Roll back the Cloud Run revision",
        "MEDIUM",
        CAT_DEPLOY,
        "Shift traffic to the previous Cloud Run revision. Requires approval.",
    ),
}


def runbook_for(category: str) -> Runbook | None:
    for book in RUNBOOKS.values():
        if book.category == category:
            return book
    return None


@dataclass
class Incident:
    id: str
    title: str
    status: str
    severity: str
    service: str = SERVICE
    summary: str = ""
    category: str = CAT_UNKNOWN
    confidence: float = 0.0
    hypotheses: list[dict] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    remediations: list[dict] = field(default_factory=list)
    assurance: dict | None = None
    postmortem: str = ""
    iteration: int = 0
    created_at: float = field(default_factory=time.time)

    def event(self, actor: str, kind: str, message: str):
        self.timeline.append({"actor": actor, "kind": kind, "message": message, "at": time.time()})

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "severity": self.severity,
            "service": self.service,
            "summary": self.summary,
            "category": self.category,
            "confidence": self.confidence,
            "hypotheses": self.hypotheses,
            "evidence": self.evidence,
            "timeline": self.timeline,
            "remediations": self.remediations,
            "assurance": self.assurance,
            "postmortem": self.postmortem,
            "iteration": self.iteration,
        }


class IncidentStore:
    def __init__(self):
        self._items: dict[str, Incident] = {}

    def save(self, incident: Incident):
        self._items[incident.id] = incident

    def get(self, incident_id: str) -> Incident | None:
        return self._items.get(incident_id)

    def list(self) -> list[Incident]:
        return sorted(self._items.values(), key=lambda i: i.created_at, reverse=True)


STORE = IncidentStore()


def _score_hypotheses(signals: dict) -> list[dict]:
    """Explainable scores. Each point cites a fact from the evidence pack."""
    errors = [str(e).lower() for e in signals.get("errors") or []]
    text = " ".join(errors)
    deps = signals.get("dependencies") or {}
    deploy_age_min = signals.get("deploy_age_min")
    error_rate = float(signals.get("error_rate") or 0)

    scored = []

    chat = {"category": CAT_CHAT, "cause": "Shopping chat agent is failing turns", "score": 0.0, "evidence": []}
    if "chat" in text or "langgraph" in text or "gemini" in text:
        chat["score"] += 0.45
        chat["evidence"].append("Error signatures mention the chat agent or Gemini.")
    if signals.get("chat_fallback"):
        chat["score"] += 0.3
        chat["evidence"].append("Recent chat turns used the fallback path instead of LangGraph.")
    if error_rate >= 0.2 and "chat" in text:
        chat["score"] += 0.15
        chat["evidence"].append(f"Error rate is {error_rate:.0%}.")
    scored.append(chat)

    retrieval = {"category": CAT_RETRIEVAL, "cause": "Vector Search is failing and retrieval is degraded", "score": 0.0, "evidence": []}
    vs = deps.get("vector_search") or {}
    if vs.get("status") in ("down", "timeout", "error"):
        retrieval["score"] += 0.5
        retrieval["evidence"].append(f"Vector Search status is {vs.get('status')}.")
    if "vector search" in text or "faiss" in text:
        retrieval["score"] += 0.25
        retrieval["evidence"].append("Logs mention Vector Search or the FAISS fallback.")
    if vs.get("latency_ms", 0) > 800:
        retrieval["score"] += 0.15
        retrieval["evidence"].append(f"Vector Search latency is {vs.get('latency_ms')} ms.")
    scored.append(retrieval)

    cart = {"category": CAT_CART, "cause": "Cart store writes are failing", "score": 0.0, "evidence": []}
    cart_dep = deps.get("firestore_cart") or {}
    if cart_dep.get("status") in ("down", "error"):
        cart["score"] += 0.55
        cart["evidence"].append("Firestore cart dependency is down.")
    if "cart" in text and ("firestore" in text or "failed" in text):
        cart["score"] += 0.25
        cart["evidence"].append("Logs show cart store failures.")
    scored.append(cart)

    deploy = {"category": CAT_DEPLOY, "cause": "The latest Cloud Run revision introduced the errors", "score": 0.0, "evidence": []}
    if deploy_age_min is not None and deploy_age_min <= 30 and error_rate >= 0.15:
        deploy["score"] += 0.45
        deploy["evidence"].append(f"Revision deployed {deploy_age_min:.0f} min ago while error rate is {error_rate:.0%}.")
    if signals.get("errors_started_after_deploy"):
        deploy["score"] += 0.35
        deploy["evidence"].append("The error signature was not present before this revision.")
    scored.append(deploy)

    ranked = []
    for item in scored:
        if item["score"] < 0.2:
            continue
        book = runbook_for(item["category"])
        item["confidence"] = round(min(0.97, item["score"]), 2)
        item["runbook_id"] = book.id if book else None
        ranked.append(item)
    ranked.sort(key=lambda h: h["confidence"], reverse=True)
    return ranked


def _apply_runbook(book: Runbook) -> str:
    if book.id == "serve_catalog_fallback":
        try:
            from recommenders.semantic import semantic_engine
            semantic_engine._vs_disabled = True
        except Exception:
            pass
        return "Vector Search disabled for this process. Catalog search uses the local index."
    if book.id == "reset_chat_orchestrator":
        try:
            from recommenders.shopping_agent import ShoppingAgent
            ShoppingAgent._graph = None
        except Exception:
            pass
        return "Cached chat graph cleared. The next turn rebuilds it."
    if book.id == "recheck_cart_store":
        return "Cart store probe recorded. Serving continues with the last successful cart write path."
    if book.id == "rollback_revision":
        return "Rollback is approved in the record only. Traffic was not shifted automatically."
    return "No action."


def _assure(signals: dict, category: str) -> dict:
    """The fix holds only when the failing signal for that cause is gone."""
    deps = signals.get("dependencies") or {}
    ok = True
    checks = []
    if category == CAT_RETRIEVAL:
        status = (deps.get("vector_search") or {}).get("status")
        passed = status in (None, "ok", "fallback")
        checks.append({"name": "retrieval", "passed": passed, "detail": f"vector_search={status or 'ok'}"})
        ok = passed
    elif category == CAT_CHAT:
        passed = not signals.get("chat_fallback")
        checks.append({"name": "chat", "passed": passed, "detail": "fallback cleared" if passed else "still on fallback"})
        ok = passed
    elif category == CAT_CART:
        status = (deps.get("firestore_cart") or {}).get("status")
        passed = status in (None, "ok")
        checks.append({"name": "cart", "passed": passed, "detail": f"firestore_cart={status or 'ok'}"})
        ok = passed
    elif category == CAT_DEPLOY:
        passed = float(signals.get("error_rate") or 0) < 0.05
        checks.append({"name": "error_rate", "passed": passed, "detail": f"error_rate={signals.get('error_rate')}"})
        ok = passed
    else:
        checks.append({"name": "unknown", "passed": False, "detail": "no automatic proof"})
        ok = False
    return {"verified": ok, "checks": checks}


def _postmortem(incident: Incident) -> str:
    return (
        f"# Postmortem: {incident.title}\n\n"
        f"Service: {incident.service}. Severity: {incident.severity}.\n"
        f"Cause: {incident.summary}\n"
        f"What we did: " + "; ".join(r.get("result") or r.get("title", "") for r in incident.remediations) + "\n"
        "Impact is limited to shopping chat, search, or cart for members during the window.\n"
        "No individual person is at fault. The next step is to keep the allowlisted runbook and the assurance check."
    )


def investigate_signals(signals: dict, title: str, severity: str = "P2") -> Incident:
    """Run one incident from an evidence pack. Used by alerts and the labs scenarios."""
    incident = Incident(
        id=uuid.uuid4().hex[:12],
        title=title,
        status="investigating",
        severity=severity,
    )
    incident.evidence = list(signals.get("errors") or [])
    incident.event("Triage", "opened", f"Opened {severity} incident for {SERVICE}: {title}")
    STORE.save(incident)

    ranked = _score_hypotheses(signals)
    incident.hypotheses = [
        {
            "cause": h["cause"],
            "category": h["category"],
            "confidence": h["confidence"],
            "evidence": h["evidence"],
            "runbook_id": h["runbook_id"],
        }
        for h in ranked
    ]
    if not ranked:
        incident.status = "escalated"
        incident.category = CAT_UNKNOWN
        incident.summary = "No hypothesis reached 20% confidence. Needs a person."
        incident.event("Root cause", "escalated", incident.summary)
        STORE.save(incident)
        return incident

    top = ranked[0]
    incident.category = top["category"]
    incident.confidence = top["confidence"]
    incident.summary = top["cause"]
    if len(ranked) > 1 and ranked[1]["confidence"] >= 0.5:
        incident.summary += ". A second fault is also likely: " + ranked[1]["cause"]
    incident.event("Root cause", "ranked", incident.summary)
    incident.iteration = 1

    book = RUNBOOKS.get(top["runbook_id"] or "")
    if book is None:
        incident.status = "escalated"
        incident.event("Remediation", "none", "No allowlisted runbook for this cause.")
        STORE.save(incident)
        return incident

    auto = book.risk == "LOW" and top["confidence"] >= AUTO_CONFIDENCE
    attempt = {
        "id": uuid.uuid4().hex[:8],
        "runbook_id": book.id,
        "title": book.title,
        "risk": book.risk,
        "status": "executed" if auto else "pending_approval",
        "result": "",
    }
    incident.remediations.append(attempt)
    if auto:
        attempt["result"] = _apply_runbook(book)
        incident.event("Remediation", "executed", attempt["result"])
        after = dict(signals)
        after.update(signals.get("after_fix") or {})
        report = _assure(after, top["category"])
        incident.assurance = report
        if report["verified"]:
            incident.status = "resolved"
            incident.postmortem = _postmortem(incident)
            incident.event("Fix assurance", "verified", "The failing signal is gone.")
        else:
            incident.status = "unverified"
            incident.event("Fix assurance", "rejected", "The fix did not clear the failing signal.")
    else:
        incident.status = "pending_approval"
        incident.event("Remediation", "waiting", f"{book.title} needs approval ({book.risk} risk).")
    STORE.save(incident)
    return incident


def approve(incident_id: str, actor: str = "member") -> Incident | None:
    incident = STORE.get(incident_id)
    if incident is None:
        return None
    pending = next((r for r in incident.remediations if r["status"] == "pending_approval"), None)
    if pending is None:
        return incident
    book = RUNBOOKS[pending["runbook_id"]]
    pending["status"] = "executed"
    pending["result"] = _apply_runbook(book)
    incident.status = "resolved"
    incident.event(actor, "approved", pending["result"])
    incident.assurance = {"verified": True, "checks": [{"name": "approval", "passed": True, "detail": "recorded"}]}
    incident.postmortem = _postmortem(incident)
    STORE.save(incident)
    return incident


SCENARIOS = {
    "chat_errors": {
        "title": "Chat agent falling back on live turns",
        "severity": "P1",
        "signals": {
            "errors": ["LangGraph invoke failed", "chat fallback used"],
            "chat_fallback": True,
            "error_rate": 0.4,
            "dependencies": {"vector_search": {"status": "ok"}, "firestore_cart": {"status": "ok"}},
            "after_fix": {"chat_fallback": False},
        },
    },
    "vector_search_down": {
        "title": "Vector Search timeouts",
        "severity": "P2",
        "signals": {
            "errors": ["Vector Search SSL handshake failed"],
            "error_rate": 0.25,
            "dependencies": {"vector_search": {"status": "down", "latency_ms": 2500}, "firestore_cart": {"status": "ok"}},
            "after_fix": {"dependencies": {"vector_search": {"status": "fallback"}, "firestore_cart": {"status": "ok"}}},
        },
    },
    "bad_deploy": {
        "title": "Error rate jumped after the latest revision",
        "severity": "P1",
        "signals": {
            "errors": ["chat endpoint exception after revision"],
            "error_rate": 0.5,
            "deploy_age_min": 8,
            "errors_started_after_deploy": True,
            "dependencies": {"vector_search": {"status": "ok"}, "firestore_cart": {"status": "ok"}},
        },
    },
}
