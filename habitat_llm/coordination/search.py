"""Deterministic candidate sampling; no simulator or model dependencies."""

import hashlib
from habitat_llm.coordination.core import layout_fingerprint


def select_search_candidates(candidates, limit, seed):
    if limit < 1:
        raise ValueError("search.limit must be positive")
    ranked = sorted(
        candidates,
        key=lambda ep: hashlib.sha256(
            (str(seed) + ":" + str(ep["episode_id"])).encode()
        ).hexdigest(),
    )
    selected, seen = [], set()
    for ep in ranked:
        key = layout_fingerprint(ep)
        if key not in seen:
            selected.append(ep)
            seen.add(key)
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError("No compatible allocation tasks in dataset")
    return selected
