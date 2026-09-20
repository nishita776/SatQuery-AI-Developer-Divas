"""Number formatting for answers (master F5.5): thousands separators; m² up to 1 km², then km²."""

from __future__ import annotations


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


def fmt_area(m2: float) -> str:
    if m2 >= 1_000_000:
        return f"{m2 / 1_000_000:,.2f} km²"
    return f"{int(round(m2)):,} m²"
