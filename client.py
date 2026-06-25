"""Entry point: load env, fetch inbox, classify via LLM, collect human approval,
execute approved actions with a write-scoped client.

Two-pass architecture (least privilege):
  Pass 1 — read-only client: fetch + classify + plan + human approval.
            The WRITE_TOKEN is never loaded into this process.
  Pass 2 — write-capable client, created only after human approval.
            Executes only the approved actions.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap

from dotenv import load_dotenv

from src.triage_skill import TriageClient, TriageResult, ProposedAction, execute, triage_inbox

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("client")

# Suppress noisy httpx logs unless debugging.
logging.getLogger("httpx").setLevel(logging.WARNING)


def build_read_client() -> TriageClient:
    base = os.getenv("API_BASE_URL", "http://127.0.0.1:8099")
    read_token = os.getenv("READ_TOKEN", "read-token-dev")
    return TriageClient(base_url=base, read_token=read_token, write_token=None)


def build_write_client() -> TriageClient:
    base = os.getenv("API_BASE_URL", "http://127.0.0.1:8099")
    read_token = os.getenv("READ_TOKEN", "read-token-dev")
    write_token = os.getenv("WRITE_TOKEN", "write-token-dev")
    return TriageClient(base_url=base, read_token=read_token, write_token=write_token)


def cli_approver(email: dict, action: ProposedAction) -> bool:
    """Interactive CLI approver — asks y/n for each proposed action."""
    os.system("clear")
    print()
    print(f"  ── Email ──")
    print(f"  From:    {email['from']}")
    print(f"  Subject: {email['subject']}")
    print(f"  Body:    {email['body']}")
    print("\n\n")
    print(f"  ── Proposed Action ──")
    print(f"  Classification: {action.label}")
    print(f"  Kind:           {action.kind}")
    print(f"  Rationale:      {action.rationale}")
    for k, v in action.payload.items():
        print(f"  {k}: {v}")
    print("\n\n")

    while True:
        ans = input("  Approve? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False


def print_summary(results: list[TriageResult]) -> None:
    approved_count = sum(1 for r in results for a in r.actions if a.approved)
    total = sum(len(r.actions) for r in results)
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in results:
        flags = " ".join(
            "✅" if a.approved else "❌" for a in r.actions
        ) or "—"
        print(f"  {r.email_id:6s}  {r.label:12s}  {flags}")
    print(f"\n  Approved: {approved_count}/{total} action(s)")
    print()


def main() -> None:
    # ── Pass 1: classify & collect approvals (NO write token in memory) ─
    logger.info("Pass 1 — fetching inbox and classifying (read-only scope)")

    read_client = build_read_client()
    results = triage_inbox(read_client, cli_approver)

    print_summary(results)

    # ── Gate — only proceed if human explicitly confirms ───────────────
    approved_actions = [
        (r, a) for r in results for a in r.actions if a.approved
    ]
    if not approved_actions:
        logger.info("No actions approved. Exiting.")
        return

    ans = input(f"Execute {len(approved_actions)} approved action(s)? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        logger.info("Aborted by user.")
        return

    # ── Pass 2: execute approved actions (WRITE token enters memory) ───
    logger.info("Pass 2 — loading write scope and executing approved actions")

    write_client = build_write_client()
    for _result, action in approved_actions:
        try:
            execute(action, write_client, approved=True)
        except PermissionError:
            logger.error(
                "Write client missing write token — cannot execute %s for %s",
                action.kind,
                _result.email_id,
            )

    logger.info("Done. Run `make audit` to inspect side effects.")


if __name__ == "__main__":
    main()
