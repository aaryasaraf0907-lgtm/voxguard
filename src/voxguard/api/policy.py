"""Risk policy for transfers asked for during (or after) a call: graded verification, not a
binary accusation.

A call's risk becomes a tier:

- ``unknown``: a live call with fewer than ``MIN_ALERT_WINDOWS`` scored windows (not enough audio;
  the live monitor does not alert on one window either);
- ``low``: the score is below the threshold;
- ``caution``: at or above the threshold but below ``CAUTION_FACTOR`` x it (the UI's caution band);
- ``high``: at or above ``CAUTION_FACTOR`` x the threshold.

A caller already on the blocklist moves one tier up (``unknown`` counts as ``low`` for that).
The tier and the transfer's value (``high_value``: at or above ``VOXGUARD_HIGH_VALUE_INR``) pick
an action from the policy table, replaceable with ``VOXGUARD_TRANSACTION_POLICY`` (JSON,
``{"tier": ["normal-value action", "high-value action"], ...}``):

- ``allow``: no extra step;
- ``step_up``: any one of a call-back on a number already on file, an OTP, or a second approver;
- ``hold``: both a second approver and a call-back.

Steps are recorded as attestations by the person who did them; this prototype sends no OTP and
places no call.
"""

from __future__ import annotations

import json
from typing import Literal

from voxguard.api.risk import CAUTION_FACTOR, MIN_ALERT_WINDOWS

Tier = Literal["unknown", "low", "caution", "high"]
Action = Literal["allow", "step_up", "hold"]
Method = Literal["callback", "otp", "second_approver"]

TIERS: tuple[Tier, ...] = ("unknown", "low", "caution", "high")
ACTIONS: tuple[Action, ...] = ("allow", "step_up", "hold")
METHODS: tuple[Method, ...] = ("callback", "otp", "second_approver")
DEFAULT_HIGH_VALUE_INR = 50_000.0
DEFAULT_POLICY: dict[str, tuple[str, str]] = {  # tier -> (normal value, high value)
    "unknown": ("step_up", "step_up"),
    "low": ("allow", "allow"),
    "caution": ("step_up", "hold"),
    "high": ("hold", "hold"),
}
STEP_UP_ANY: tuple[Method, ...] = ("callback", "otp", "second_approver")  # one of these
HOLD_ALL: tuple[Method, ...] = ("second_approver", "callback")  # all of these


def parse_policy(raw: str) -> dict[str, tuple[str, str]]:
    """``VOXGUARD_TRANSACTION_POLICY``: empty for the default; else JSON covering every tier.
    Raises ValueError with a message that says what to fix."""
    if not raw.strip():
        return dict(DEFAULT_POLICY)
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError("VOXGUARD_TRANSACTION_POLICY must be JSON") from None
    if not isinstance(data, dict) or set(data) != set(TIERS):
        raise ValueError(f"VOXGUARD_TRANSACTION_POLICY must give every tier: {list(TIERS)}")
    policy = {}
    for tier, pair in data.items():
        if not (isinstance(pair, list) and len(pair) == 2 and all(a in ACTIONS for a in pair)):
            raise ValueError(
                f"VOXGUARD_TRANSACTION_POLICY[{tier!r}] must be [normal, high-value] actions "
                f"from {list(ACTIONS)}"
            )
        policy[tier] = (pair[0], pair[1])
    return policy


def tier_for(score: float, threshold: float, windows: int | None) -> Tier:
    """``windows`` is the number of scored windows of a live call, None for a finished one."""
    if windows is not None and windows < MIN_ALERT_WINDOWS:
        return "unknown"
    if score < threshold:
        return "low"
    return "caution" if score < CAUTION_FACTOR * threshold else "high"


def escalate(tier: Tier, flagged_before: bool) -> Tier:
    if not flagged_before:
        return tier
    return {"unknown": "caution", "low": "caution", "caution": "high", "high": "high"}[tier]  # type: ignore[return-value]


def decide(
    tier: Tier, amount_inr: float, high_value_inr: float, policy: dict[str, tuple[str, str]]
) -> tuple[Action, bool]:
    """(action, high_value)."""
    high_value = amount_inr >= high_value_inr
    return policy[tier][1 if high_value else 0], high_value  # type: ignore[return-value]


def required_steps(action: Action) -> list[Method]:
    return {"allow": [], "step_up": list(STEP_UP_ANY), "hold": list(HOLD_ALL)}[action]


def released(action: Action, completed: set[str]) -> bool:
    """Whether the completed verification methods release a transfer with this action."""
    if action == "allow":
        return True
    if action == "step_up":
        return bool(completed & set(STEP_UP_ANY))
    return set(HOLD_ALL) <= completed


def initial_status(action: Action) -> str:
    return {"allow": "allowed", "step_up": "pending", "hold": "held"}[action]
