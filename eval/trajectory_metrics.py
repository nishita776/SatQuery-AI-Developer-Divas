"""ML-10 trajectory metrics: compare a planned tool sequence to a gold plan.
Tool-Exact-Match, Tool-In-Order (LCS/len(gold)), Parameter Accuracy."""
import functools


def _tools(seq):
    return [s.tool if hasattr(s, "tool") else s["tool"] for s in seq]


def _params(s):
    return s.params if hasattr(s, "params") else s.get("params", {})


def tool_exact_match(pred, gold):
    return _tools(pred) == _tools(gold)


def tool_in_order(pred, gold):
    p, g = _tools(pred), _tools(gold)

    @functools.lru_cache(None)
    def lcs(i, j):
        if i == len(p) or j == len(g):
            return 0
        if p[i] == g[j]:
            return 1 + lcs(i + 1, j + 1)
        return max(lcs(i + 1, j), lcs(i, j + 1))

    return lcs(0, 0) / len(g) if g else 1.0


def parameter_accuracy(pred, gold):
    total, correct = 0, 0
    for ps, gs in zip(pred, gold):
        gp, pp = _params(gs), _params(ps)
        for k, v in gp.items():
            total += 1
            if str(pp.get(k)) == str(v):
                correct += 1
    return correct / total if total else 1.0


def trajectory_report(pred, gold):
    return {
        "tool_exact_match": tool_exact_match(pred, gold),
        "tool_in_order": tool_in_order(pred, gold),
        "parameter_accuracy": parameter_accuracy(pred, gold),
    }
