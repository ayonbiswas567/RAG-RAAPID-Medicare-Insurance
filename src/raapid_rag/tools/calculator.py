"""Safe arithmetic and date arithmetic.

``calculate`` walks the expression's abstract syntax tree and evaluates only a whitelist of
node types: numbers, + - * / // % **, unary +/-, parentheses, and calls to ``abs``, ``round``,
``min``, ``max``.  Anything else (names, attribute access, subscripts, strings, imports) is
rejected before evaluation.  ``eval`` is never used.  Exponents are capped so a model cannot
request ``2 ** 100000`` and stall the worker.
"""

from __future__ import annotations

import ast
import operator
from datetime import date
from typing import Any

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"abs": abs, "round": round, "min": min, "max": max}
_MAX_EXPONENT = 1000
_MAX_EXPRESSION_LENGTH = 200


class _Unsafe(ValueError):
    pass


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise _Unsafe(f"exponent too large (limit {_MAX_EXPONENT})")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*(_eval(a) for a in node.args))
    raise _Unsafe(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate a numeric expression such as ``84 / (1.65 ** 2)`` or ``round(7.4 * 12, 1)``."""
    expr = (expression or "").strip()
    if not expr:
        return {"error": "expression must be non-empty"}
    if len(expr) > _MAX_EXPRESSION_LENGTH:
        return {"error": f"expression too long (limit {_MAX_EXPRESSION_LENGTH} characters)"}
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
    except SyntaxError as exc:
        return {"error": f"could not parse expression: {exc.msg}"}
    except _Unsafe as exc:
        return {"error": f"expression rejected: {exc}"}
    except ZeroDivisionError:
        return {"error": "division by zero"}
    except (OverflowError, ValueError, TypeError) as exc:
        return {"error": f"could not evaluate expression: {exc}"}
    return {"expression": expr, "result": result}


def days_between(start_date: str, end_date: str) -> dict[str, Any]:
    """Signed number of days from ``start_date`` to ``end_date`` (ISO dates, YYYY-MM-DD)."""
    try:
        start = date.fromisoformat((start_date or "").strip())
        end = date.fromisoformat((end_date or "").strip())
    except ValueError as exc:
        return {"error": f"dates must be ISO format (YYYY-MM-DD): {exc}"}
    return {"start_date": start.isoformat(), "end_date": end.isoformat(), "days": (end - start).days}
