"""ML-10 grounding check: every number in the fused answer must trace to a
measured ToolResult metric — catches hallucinated figures."""
import re

# standalone numbers only: not glued to a letter/digit (skips the "2" in "m2", "T1")
_NUM = re.compile(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?![A-Za-z0-9])")


def extract_numbers(text):
    return [float(x) for x in _NUM.findall((text or "").replace(",", ""))]


def _metrics(r):
    m = r.metrics if hasattr(r, "metrics") else r.get("metrics", {})
    return [float(v) for v in m.values()]


def grounding_report(answer, tool_results, tol=0.01):
    """answer: str; tool_results: list of ToolResult (or dicts with 'metrics').
    A number is grounded if it matches any metric within `tol` relative error."""
    vals = [v for r in tool_results for v in _metrics(r)]
    grounded, ungrounded = [], []
    for n in extract_numbers(answer):
        if any(abs(n - v) <= tol * max(1.0, abs(v)) for v in vals):
            grounded.append(n)
        else:
            ungrounded.append(n)
    total = len(grounded) + len(ungrounded)
    return {
        "grounded": grounded,
        "ungrounded": ungrounded,
        "ratio": len(grounded) / total if total else 1.0,
        "all_grounded": not ungrounded,
    }
