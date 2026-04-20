"""
tools.py – Tool implementations for Study Buddy Physics.

Rules
-----
- Every tool must NEVER raise an exception.
- Every tool must return a plain string.
- Tools are self-contained and import-safe.
"""

import math
import datetime
import re


# ── CALCULATOR ────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression and return the result as a string.

    Supports standard arithmetic, exponentiation (**), and all math module
    functions (sin, cos, sqrt, log, pi, e, …).

    Parameters
    ----------
    expression : str
        A Python-compatible mathematical expression, e.g. "0.5 * 2 * 5**2".

    Returns
    -------
    str
        Human-readable result string, or an error message.
    """
    try:
        # Build a safe namespace with math functions only
        safe_globals: dict = {"__builtins__": {}}
        safe_locals: dict = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_")
        }
        safe_locals.update(
            {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            }
        )

        # Sanitise: reject obvious injection attempts
        banned = ["import", "exec", "eval", "open", "os", "sys", "__"]
        for token in banned:
            if token in expression:
                return (
                    f"Calculator refused to evaluate '{expression}' "
                    "because it contains disallowed keywords."
                )

        result = eval(str(expression), safe_globals, safe_locals)  # noqa: S307

        # Format nicely
        if isinstance(result, float):
            formatted = f"{result:.6g}"  # strip trailing zeros
        else:
            formatted = str(result)

        return (
            f"Calculator result for expression '{expression}': {formatted}"
        )

    except ZeroDivisionError:
        return f"Calculator error: division by zero in expression '{expression}'."
    except SyntaxError as exc:
        return f"Calculator error: invalid syntax in '{expression}'. Detail: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Calculator error for '{expression}': {exc}"


# ── DATETIME ─────────────────────────────────────────────────────────────────

def get_datetime() -> str:
    """
    Return the current local date, time, and additional calendar info.

    Returns
    -------
    str
        Formatted date/time string, or an error message.
    """
    try:
        now = datetime.datetime.now()
        day_of_year = now.timetuple().tm_yday
        week_number = now.isocalendar()[1]
        return (
            f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}. "
            f"Day of year: {day_of_year}. "
            f"ISO week number: {week_number}."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Datetime tool error: {exc}"


# ── TOOL REGISTRY ─────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict = {
    "calculator": calculator,
    "datetime": get_datetime,
}


# ── SELF-TEST ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(calculator("0.5 * 2 * 5**2"))       # 25.0
    print(calculator("math.sqrt(144)"))         # 12.0
    print(calculator("6.674e-11 * 5.97e24 / (6.371e6)**2"))  # ~9.8
    print(get_datetime())
