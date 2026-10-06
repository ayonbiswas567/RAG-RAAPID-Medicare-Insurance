"""Mock member-eligibility service.

Mirrors policy RA-110 section 3 in the corpus: eligibility is checked by member id and date
of service, the response carries an ACTIVE/INACTIVE status, and every lookup is stamped so
it can be logged for audit.  Member ``M-TEST-1004`` is retroactively terminated on
2025-03-31, which makes the 2025-04-02 telehealth note in the corpus ineligible for a second,
independent reason (the first being that the visit was audio-only).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

# member_id -> plan, coverage windows, optional termination metadata
MEMBERS: dict[str, dict[str, Any]] = {
    "M-TEST-1001": {"plan": "Example Health Plan Medicare Advantage HMO", "coverage": [("2024-01-01", "2025-12-31")]},
    "M-TEST-1002": {"plan": "Example Health Plan Medicare Advantage PPO", "coverage": [("2023-01-01", "2025-12-31")]},
    "M-TEST-1003": {"plan": "Example Health Plan Medicare Advantage HMO", "coverage": [("2025-01-01", "2025-12-31")]},
    "M-TEST-1004": {
        "plan": "Example Health Plan Medicare Advantage HMO",
        "coverage": [("2024-06-01", "2025-03-31")],
        "termination_type": "retroactive",
        "termination_processed": "2025-04-15",
    },
}


def _parse(value: str) -> date:
    return date.fromisoformat(value.strip())


def check_eligibility(member_id: str, date_of_service: str) -> dict[str, Any]:
    """Return coverage status for ``member_id`` on ``date_of_service`` (ISO date, YYYY-MM-DD)."""
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        dos = _parse(date_of_service)
    except (ValueError, AttributeError):
        return {"error": f"date_of_service must be an ISO date (YYYY-MM-DD), got {date_of_service!r}"}

    key = (member_id or "").strip().upper()
    member = MEMBERS.get(key)
    base = {"member_id": key, "date_of_service": dos.isoformat(), "checked_at": checked_at}
    if member is None:
        return {**base, "status": "NOT_FOUND", "plan": None, "coverage_start": None, "coverage_end": None,
                "termination_type": None, "note": "No member with this id in the eligibility system."}

    for start, end in member["coverage"]:
        if _parse(start) <= dos <= _parse(end):
            return {**base, "status": "ACTIVE", "plan": member["plan"], "coverage_start": start, "coverage_end": end,
                    "termination_type": None, "note": "Coverage active on the date of service."}

    last_start, last_end = member["coverage"][-1]
    termination_type = member.get("termination_type")
    note = "Coverage not active on the date of service."
    if termination_type == "retroactive":
        note += (f" Coverage was terminated retroactively effective {last_end}"
                 f" (processed {member.get('termination_processed')}).")
    return {**base, "status": "INACTIVE", "plan": member["plan"], "coverage_start": last_start, "coverage_end": last_end,
            "termination_type": termination_type, "note": note}
