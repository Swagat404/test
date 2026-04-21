"""Score parser. Tolerant of prose responses; uses regex fallback."""
from __future__ import annotations

import json
import re
from typing import Optional

_INT_RE = re.compile(r"\b(\d{1,3})\b")


def parse_score(text: str) -> Optional[int]:
    """Extract an integer score in [0, 100] from a model response.

    Strategy:
      1. Try to parse the entire response as an integer (scalar prompts).
      2. Try to parse as JSON and pull "score_0_100" (rubric prompts).
      3. Regex fallback: first 1-3 digit integer in [0, 100].

    Returns None if no score can be extracted.
    """
    if text is None:
        return None

    s = text.strip()

    try:
        v = int(s)
        if 0 <= v <= 100:
            return v
    except ValueError:
        pass

    candidate = s
    if "{" in candidate and "}" in candidate:
        start = candidate.index("{")
        end = candidate.rindex("}") + 1
        snippet = candidate[start:end]
        try:
            obj = json.loads(snippet)
            v = obj.get("score_0_100")
            if isinstance(v, (int, float)) and 0 <= v <= 100:
                return int(v)
        except (json.JSONDecodeError, AttributeError):
            pass

    for match in _INT_RE.finditer(s):
        v = int(match.group(1))
        if 0 <= v <= 100:
            return v

    return None
