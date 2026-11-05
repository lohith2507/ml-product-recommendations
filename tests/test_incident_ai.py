from recommenders.incident_ai import SCENARIOS, approve, investigate_signals


def test_chat_failure_is_auto_fixed_and_verified():
    incident = investigate_signals(SCENARIOS["chat_errors"]["signals"], SCENARIOS["chat_errors"]["title"], "P1")
    assert incident.category == "CHAT_AGENT"
    assert incident.status == "resolved"
    assert incident.assurance["verified"] is True
    assert incident.postmortem


def test_vector_search_is_not_called_a_cart_failure():
    incident = investigate_signals(
        SCENARIOS["vector_search_down"]["signals"],
        SCENARIOS["vector_search_down"]["title"],
        "P2",
    )
    assert incident.category == "VECTOR_SEARCH"
    assert incident.remediations[0]["runbook_id"] == "serve_catalog_fallback"
    assert incident.status == "resolved"


def test_bad_revision_waits_for_approval():
    incident = investigate_signals(SCENARIOS["bad_deploy"]["signals"], "bad revision", "P1")
    assert incident.category == "BAD_DEPLOYMENT"
    assert incident.status == "pending_approval"
    approved = approve(incident.id, "oncall")
    assert approved.status == "resolved"
    assert "not shifted automatically" in approved.remediations[0]["result"]
