"""Surgical Clause Router for Multi-Request, Change-of-Mind, and Escalation Rules.

Implements the cleaned cue set verified in the Step 5 audit (`results/noise_flags_audit.md`):
  - Rule 2 (Change-of-mind: last request wins): matches explicit reversal phrases
    (`actually never mind,`, `actually forget it,`, `no wait, forget that,`,
    `scratch that,`, `ignore what i just said,`, including 1-char generator typos),
    never bare `instead of` or `actually help`.
  - Rule 5 + Priority Override (`, but first` / `, but before that`): second clause wins.
  - Rule 1 & Rule 5 Additive Multi-Request (`, also `, ` and also `, ` plus `, ` and by the way `):
    never bare ` also ` (which matches distractor chitchat `my neighbour also uses this app`
    and `my wifi is also acting up`). First clause wins unless first clause is `talk_to_agent`
    or `out_of_scope`.
  - Never splits on contrastive `but` / `par` / `lekin`.
"""
from __future__ import annotations

import re
import numpy as np

# Regex patterns (tolerating single-character generator typos before the comma)
RE_CHANGE_OF_MIND = re.compile(
    r"(?:(?:act\w{3,6}|ac\w{4,6})\s+)?(?:never\s*m\w{2,4}|n\w{3,5}\s*m\w{2,4}|forget\s+it|f\w{4,6}\s+it)\s*,\s*"
    r"|(?:no\s+wait\s*,\s*|n\w\s+w\w{2,4}\s*,\s*)?(?:forget\s+that|f\w{4,6}\s+that|f\w{4,6}\s+th\w{1,3})\s*,\s*"
    r"|(?:scratch\s+that|sc\w{4,6}\s+that|s\w{5,7}\s+th\w{1,3})\s*,\s*"
    r"|(?:sorry|s\w{3,5})\s*,\s*(?:wrong|w\w{3,5})\s+(?:thing|th\w{2,4})\s*,\s*"
    r"|ignore\s+what\s+i\s+(?:just|j\w{2,4})\s+(?:said|s\w{2,4})\s*,\s*"
    r"|(?:on\s+second\s+thought|rehne\s+do|chhod\s+do|nvm)\s*,\s*",
    re.IGNORECASE,
)

RE_BUT_FIRST = re.compile(
    r",\s*but\s+(?:first|f\w{3,4}|before\s+that|b\w{4,6}\s+th\w{1,3})\s+",
    re.IGNORECASE,
)

RE_ADDITIVE = re.compile(
    r",\s*also\s+|\s+also\s*,\s*|\s+and\s+also\s+|\s+plus\s+|\s+and\s+by\s+the\s+way\s+",
    re.IGNORECASE,
)

ESCALATION_HINTS = (
    "agent", "human", "person", "supervisor", "manager", "representative",
    "executive", "customer care", "customer support", "senior team", "bot",
    "speak to", "talk to", "connect me", "connect karo", "call me back",
)


def parse_clauses(norm_text: str) -> tuple[str, list[str]]:
    """Return (route_type, clauses_list).

    If route_type == 'single', returns ('single', [norm_text]).
    If route_type in ('change_of_mind', 'but_first', 'additive'), returns
    (route_type, [clause_1, clause_2, ...]) with >= 2 clauses so all clauses in a
    flagged message are batched into a single encoder/vectorizer call.
    """
    t = norm_text.strip()
    if not t:
        return "single", [t]

    # 1. Change of mind (Rule 2: last request wins)
    com_matches = list(RE_CHANGE_OF_MIND.finditer(t))
    if com_matches:
        last_m = com_matches[-1]
        c1 = t[: last_m.start()].strip(" ,.!?;:-")
        c2 = t[last_m.end() :].strip(" ,.!?;:-")
        if len(c1.split()) >= 2 and len(c2.split()) >= 2:
            # Check if c2 itself has an additive split (e.g. "..., also ...")
            add_m = RE_ADDITIVE.search(c2)
            if add_m:
                c2a = c2[: add_m.start()].strip(" ,.!?;:-")
                c2b = c2[add_m.end() :].strip(" ,.!?;:-")
                if len(c2a.split()) >= 2 and len(c2b.split()) >= 2:
                    return "change_of_mind_additive", [c1, c2a, c2b]
            return "change_of_mind", [c1, c2]

    # 2. ", but first" / ", but before that" (Rule 5 override: second clause wins)
    bf_m = RE_BUT_FIRST.search(t)
    if bf_m:
        c1 = t[: bf_m.start()].strip(" ,.!?;:-")
        c2 = t[bf_m.end() :].strip(" ,.!?;:-")
        if len(c1.split()) >= 2 and len(c2.split()) >= 2:
            return "but_first", [c1, c2]

    # 3. Additive multi-request (", also ", " and also ", " plus ")
    add_matches = list(RE_ADDITIVE.finditer(t))
    if add_matches:
        parts = []
        last_idx = 0
        for m in add_matches:
            seg = t[last_idx : m.start()].strip(" ,.!?;:-")
            if len(seg.split()) >= 2:
                parts.append(seg)
            last_idx = m.end()
        tail = t[last_idx:].strip(" ,.!?;:-")
        if len(tail.split()) >= 2:
            parts.append(tail)
        if len(parts) >= 2:
            return "additive", parts

    return "single", [t]


def has_escalation_hint(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in ESCALATION_HINTS)


def resolve_clause_probs(
    route_type: str,
    clauses: list[str],
    clause_probs: np.ndarray,
    full_prob: np.ndarray,
    talk_to_agent_id: int,
    out_of_scope_id: int,
) -> np.ndarray:
    """Given per-clause probability vectors (k, 12) and full_prob (12,), return final (12,) prob vector."""
    if route_type == "single" or len(clauses) < 2:
        return full_prob

    if route_type in ("change_of_mind", "but_first"):
        # Second clause is the active request (Rule 2 / Rule 5 override)
        p2 = clause_probs[-1]
        if p2.argmax() != out_of_scope_id and p2.max() >= 0.35:
            return p2
        return full_prob

    if route_type == "change_of_mind_additive":
        # After change of mind, there are two additive clauses: c2a and c2b
        p2a, p2b = clause_probs[1], clause_probs[2]
        c2a_text = clauses[1]
        if p2a.argmax() == talk_to_agent_id or has_escalation_hint(c2a_text) or p2a.argmax() == out_of_scope_id:
            return p2b if p2b.max() >= 0.35 else full_prob
        return p2a if p2a.max() >= 0.35 else full_prob

    if route_type == "additive":
        # Rule 1: first non-escalation, non-OOS request wins (Rule 5: talk_to_agent yields to other request)
        for idx, c_text in enumerate(clauses):
            p_c = clause_probs[idx]
            top_c = int(p_c.argmax())
            if top_c == talk_to_agent_id or has_escalation_hint(c_text):
                continue
            if top_c == out_of_scope_id:
                continue
            if p_c.max() >= 0.35:
                return p_c
        # Fallback if all clauses were talk_to_agent or low confidence
        return full_prob

    return full_prob
