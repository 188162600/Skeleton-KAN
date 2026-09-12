"""Shared declared-input semantics for extraction and structural grouping."""

from __future__ import annotations

import re
import keyword
from typing import Iterable, Mapping, Any

import sympy as sp


CONSTANT_POLICY = "every symbol absent from variables is constant"
EQUIVALENCE_VERSION = "affine-free-declared-input-v2"


def parse_declared_expression(expression, inputs: Iterable[str | sp.Symbol] | None = None):
    """Parse input names even when they collide with pi, E, or Python keywords.

    None is a legacy convenience for standalone expressions only. Dataset and
    builder entry points must supply their declared input list, including [].
    """
    if inputs is None:
        parsed = sp.sympify(expression)
        return parsed, tuple(sorted(parsed.free_symbols, key=str))
    names = tuple(str(value) for value in inputs)
    if len(names) != len(set(names)):
        raise ValueError("declared input names must be unique")
    symbols = tuple(sp.Symbol(name) for name in names)
    if isinstance(expression, sp.Basic):
        replacements = {symbol: symbols[names.index(str(symbol))]
                        for symbol in expression.free_symbols if str(symbol) in names}
        return expression.xreplace(replacements), symbols
    source = str(expression)
    # Simultaneous substitution prevents one renamed input from matching a
    # later input name. Dummy objects avoid collisions with source identifiers.
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", source))
    function_names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", source))
    # Bare parameters such as beta/gamma/N/Q must not resolve to SymPy
    # functions or assumption objects. Standard numerical constants retain
    # their values, unless explicitly declared as inputs above.
    fixed_names = sorted(identifiers - function_names - set(names)
                         - {"pi", "E", "I", "oo", "nan", "zoo", "True", "False", "None"})
    parse_names = names + tuple(fixed_names)
    parse_symbols = symbols + tuple(sp.Symbol(name) for name in fixed_names)
    tokens = {name: f"_declared_input_placeholder_{i}" for i, name in enumerate(parse_names)}
    while any(token in source for token in tokens.values()):
        tokens = {name: "_" + token for name, token in tokens.items()}
    if parse_names:
        pattern = r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(n) for n in sorted(parse_names, key=len, reverse=True)) + r")(?![A-Za-z0-9_])"
        source = re.sub(pattern, lambda match: tokens[match.group(0)], source)
    parsed = sp.sympify(source, locals={tokens[name]: symbol for name, symbol in zip(parse_names, parse_symbols)})
    return parsed, symbols


def depends_on_inputs(expression, inputs) -> bool:
    return bool(sp.sympify(expression).free_symbols.intersection(inputs))


def declared_inputs(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Require names, never infer model inputs from every free symbol."""
    spec = record.get("operator_variable_spec", {})
    if "variables" in spec:
        return tuple(str(value) for value in spec["variables"])
    for key in ("input_variables", "declared_variables", "variables"):
        value = record.get(key)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
    raise ValueError("record has no declared input names; an integer dimension is insufficient")


def constant_audit(expression, inputs):
    parsed, symbols = parse_declared_expression(expression, inputs)
    # Record maximal constant-only subexpressions, not every numeric atom in
    # them; constants in exponents remain numerical metadata, not input nodes.
    constants = set()
    def visit(node):
        if not depends_on_inputs(node, symbols):
            constants.add(sp.sstr(node))
            return
        for child in node.args:
            visit(child)
    visit(parsed)
    return {
        "constant_policy": CONSTANT_POLICY,
        "declared_inputs": [str(symbol) for symbol in symbols],
        "noninput_symbols": sorted(str(symbol) for symbol in parsed.free_symbols - set(symbols)),
        "constant_subexpressions": sorted(constants),
    }
