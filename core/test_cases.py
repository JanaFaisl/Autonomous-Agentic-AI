"""Turn acceptance criteria into structured test cases and runnable test stubs."""
import json
import re
from typing import Any, Dict, List, Optional

_GWT = re.compile(r"given\s+(?P<given>.+?)[,;]?\s+when\s+(?P<when>.+?)[,;]?\s+then\s+(?P<then>.+)", re.I | re.S)


def generate_test_cases(requirements: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One case per acceptance criterion: id, feature, Given/When/Then (or the raw text as 'then')."""
    if not isinstance(requirements, dict) or "error" in requirements:
        return []
    cases: List[Dict[str, Any]] = []
    for f in requirements.get("features") or []:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id") or f.get("name"))
        for n, text in enumerate(f.get("acceptance_criteria") or [], 1):
            m = _GWT.search(str(text))
            cases.append({
                "id": f"TC-{fid}-{n}",
                "feature_id": fid,
                "feature": f.get("name"),
                "priority": f.get("priority"),
                "given": m.group("given").strip() if m else "",
                "when": m.group("when").strip() if m else "",
                "then": m.group("then").strip().rstrip(".") if m else str(text).strip(),
                "criterion": str(text),
            })
    return cases


def render_test_stubs(cases: List[Dict[str, Any]]) -> str:
    """A node:test file with one todo per case, so `node --test` lists what is still unwritten."""
    lines = ['// Acceptance tests generated from requirements. Replace each todo with a real test.',
             'const test = require("node:test");', ""]
    for c in cases:
        title = json.dumps(f"{c['id']} {c['feature']}: {c['then']}")
        lines.append(f"test.todo({title});")
        detail = " / ".join(x for x in (f"Given {c['given']}" if c["given"] else "",
                                         f"When {c['when']}" if c["when"] else "") if x)
        if detail:
            lines.append("// " + detail.replace("\n", " "))
    return "\n".join(lines) + "\n"
