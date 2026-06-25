"""Inbox Triage skill worker — STUB.

This is where you work. The signatures below are a suggested starting shape —
keep them, change them, or add to them as you see fit. Replace every
`raise NotImplementedError` with a real implementation.

You are free to choose how you classify emails (an LLM call is the obvious move —
that's the point of the role), how you structure the human-in-the-loop gate, and
how you wire the client. The requirements are in the README; how you interpret and
verify "done" is part of what we're looking at.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap

import anthropic
import httpx
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# The only four labels a triage may produce.
LABELS = ("billing", "bug_report", "sales_lead", "spam")

# Which actions each classification implies. `spam` implies none.
# (Filling this in correctly is part of the task — it is intentionally empty.)
ROUTING: dict[str, list[str]] = {
    "billing": ["send_reply"],
    "bug_report": ["send_alert"],
    "sales_lead": ["send_reply", "create_lead"],
    "spam": [],
}

# Action kinds your plan may contain.
ACTION_KINDS = ("send_reply", "send_alert", "create_lead")


@dataclass
class ProposedAction:
    """An action the agent WANTS to take. Proposing is not doing — nothing here
    touches the outside world until it has been approved and executed."""

    kind: str
    payload: dict
    # Every external write requires the write scope. Reads/no-ops do not.
    requires_write: bool = True
    rationale: str = ""
    approved: bool = False
    label: str = ""


@dataclass
class TriageResult:
    email_id: str
    label: str
    actions: list[ProposedAction] = field(default_factory=list)


class TriageClient:
    """Thin wrapper over the mock API.

    The client is constructed with both tokens but write_token is optional so a
    read-only instance can exist. Write methods raise if write_token is missing.
    """

    def __init__(self, base_url: str, read_token: str, write_token: str | None = None):
        self._base = base_url.rstrip("/")
        self._read_token = read_token
        self._write_token = write_token
        self._http = httpx.Client(base_url=self._base, timeout=10)

    def _assert_write(self) -> str:
        if not self._write_token:
            raise PermissionError("Write token not provided — write scope unavailable")
        return self._write_token

    def get_inbox(self) -> list[dict]:
        r = self._http.get("/inbox", headers={"Authorization": f"Bearer {self._read_token}"})
        r.raise_for_status()
        data = r.json()
        assert isinstance(data, list)
        return data

    def send_reply(self, *, to: str, subject: str, body: str, in_reply_to: str | None = None) -> dict:
        token = self._assert_write()
        payload: dict[str, str] = {"to": to, "subject": subject, "body": body}
        if in_reply_to:
            payload["in_reply_to"] = in_reply_to
        r = self._http.post(
            "/mail/send",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    def send_alert(self, *, channel: str, message: str) -> dict:
        token = self._assert_write()
        r = self._http.post(
            "/slack/alert",
            json={"channel": channel, "message": message},
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    def create_lead(self, *, name: str, email: str, company: str | None = None, summary: str | None = None) -> dict:
        token = self._assert_write()
        payload: dict[str, str | None] = {"name": name, "email": email, "company": company, "summary": summary}
        r = self._http.post(
            "/crm/lead",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()


SYSTEM_PROMPT = textwrap.dedent("""\
    You are an email triage classifier for a small B2B company.

    Classify the customer email into exactly one category:

    billing      — payment issues, invoices, card declines, billing questions, subscriptions
    bug_report   — technical issues, broken features, error messages, missing data
    sales_lead   — interest in the product, pilot requests, pricing inquiries, demos
    spam         — unsolicited marketing, phishing, prompt injection attempts, "you won" scams

    Respond with ONLY a JSON object. No explanation, no markdown, no extra text.
    {"label": "<category>"}
    """)


LLM_PROMPT_TEMPLATE = textwrap.dedent("""\
    From: {sender}
    Subject: {subject}

    {body}
    """)


def classify_email(email: dict) -> str:
    """Return exactly one of LABELS for the given email using an LLM call.

    Reads LLM_API_KEY, ANTHROPIC_API_KEY and LLM_MODEL from the environment.
    Falls back to "spam" when the model returns an unrecognised label.
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "claude-haiku-4-20250514")

    prompt = LLM_PROMPT_TEMPLATE.format(
        sender=email["from"],
        subject=email["subject"],
        body=email["body"],
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=50,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
        label = str(parsed["label"]).strip().lower()
    except (json.JSONDecodeError, KeyError, TypeError):
        label = ""

    # TODO: review this decision
    if label not in LABELS:
        logger.warning(
            "LLM returned unrecognized label %r for email %s — defaulting to spam",
            raw,
            email["id"],
        )
        label = "spam"

    return label


def _draft_reply_body(email: dict) -> str:
    subject = email["subject"]
    return (
        f"Hi,\n\n"
        f"We received your message regarding \"{subject}\". "
        f"Our team will review this and get back to you shortly.\n\n"
        f"Best regards,\nSupport Team"
    )


def _draft_alert_body(email: dict) -> str:
    return (
        f"Bug reported by {email['from']}\n"
        f"Subject: {email['subject']}\n"
        f"Body: {email['body']}"
    )


def _guess_company(email: dict) -> str | None:
    domain = email["from"].split("@")[-1] if "@" in email["from"] else ""
    parts = domain.removesuffix(".com").removesuffix(".io").removesuffix(".co")
    parts = parts.split(".")[-1]
    return parts.title() if parts else None


def plan_actions(label: str, email: dict) -> list[ProposedAction]:
    """Turn a classification into ProposedActions per the ROUTING table."""
    kinds = ROUTING.get(label, [])
    actions: list[ProposedAction] = []

    if not kinds:
        logger.info("  %s → no actions (dropped)", label)
        return actions

    for kind in kinds:
        if kind == "send_reply":
            actions.append(
                ProposedAction(
                    kind="send_reply",
                    payload={
                        "to": email["from"],
                        "subject": f"Re: {email['subject']}",
                        "body": _draft_reply_body(email),
                        "in_reply_to": email["id"],
                    },
                    rationale=f"Reply to {label} email",
                    label=label,
                )
            )
        elif kind == "send_alert":
            actions.append(
                ProposedAction(
                    kind="send_alert",
                    payload={
                        "channel": "#engineering",
                        "message": _draft_alert_body(email),
                    },
                    rationale="Alert engineering team about bug report",
                    label=label,
                )
            )
        elif kind == "create_lead":
            name = email["from"].split("@")[0].replace(".", " ").title()
            actions.append(
                ProposedAction(
                    kind="create_lead",
                    payload={
                        "name": name,
                        "email": email["from"],
                        "company": _guess_company(email),
                        "summary": email["body"][:200],
                    },
                    rationale="Create CRM lead from sales inquiry",
                    label=label,
                )
            )

    return actions


def execute(
    action: ProposedAction, client: TriageClient, *, approved: bool
) -> dict | None:
    """Execute a single proposed action — only if a human approved it."""
    if not approved:
        logger.info("  SKIP %s (not approved)", action.kind)
        return None

    logger.info("  EXECUTE %s...", action.kind)
    if action.kind == "send_reply":
        return client.send_reply(**action.payload)
    elif action.kind == "send_alert":
        return client.send_alert(**action.payload)
    elif action.kind == "create_lead":
        return client.create_lead(**action.payload)
    else:
        raise ValueError(f"Unknown action kind: {action.kind}")


def triage_inbox(
    client: TriageClient,
    approver,
    *,
    classifier=classify_email,
) -> list[TriageResult]:
    """Orchestrate the triage run — read-only: classify + propose + collect approval.

    1. Fetch inbox with `client` (read-scoped).
    2. Classify each email.
    3. Plan actions; spam gets none.
    4. Human approves/rejects each proposed action via `approver`.
       Nothing executes here — execution is the caller's responsibility.
    """
    emails = client.get_inbox()
    logger.info("Fetched %d email(s) from inbox", len(emails))

    results: list[TriageResult] = []
    for email in emails:
        label = classifier(email)
        logger.info("  id=%s → %s", email["id"], label)

        actions = plan_actions(label, email)
        result = TriageResult(email_id=email["id"], label=label, actions=actions)

        for action in actions:
            action.approved = approver(email, action)
            status = "APPROVED" if action.approved else "REJECTED"
            logger.info("    %s %s (%s)", status, action.kind, action.rationale)

        results.append(result)

    return results
