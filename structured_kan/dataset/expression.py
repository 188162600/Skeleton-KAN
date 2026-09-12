"""Faithful float64 frozen-expression evaluation; no training or simplification."""
import ast
import math
from typing import Any
import numpy as np
import torch

def _unary(torch_function: Any, scalar_function: Any):
    def evaluate(value: Any):
        return torch_function(value) if isinstance(value, torch.Tensor) else scalar_function(float(value))

    return evaluate


def _heaviside(value: Any, at_zero: Any = 0.5):
    if isinstance(value, torch.Tensor):
        zero = torch.as_tensor(at_zero, dtype=value.dtype, device=value.device)
        return torch.heaviside(value, zero)
    scalar = float(value)
    return 0.0 if scalar < 0.0 else 1.0 if scalar > 0.0 else float(at_zero)


def _maximum(*values: Any):
    tensor = next((value for value in values if isinstance(value, torch.Tensor)), None)
    if tensor is None:
        return max(float(value) for value in values)
    result = torch.as_tensor(values[0], dtype=tensor.dtype, device=tensor.device)
    for value in values[1:]:
        result = torch.maximum(result, torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device))
    return result


def _rational_power(value: Any, numerator: Any, denominator: Any):
    p, q = int(numerator), int(denominator)
    if q == 0:
        raise ZeroDivisionError("rational exponent has zero denominator")
    if q < 0:
        p, q = -p, -q
    exponent = p / q
    if q % 2:
        if isinstance(value, torch.Tensor):
            magnitude = torch.abs(value) ** exponent
            return torch.sign(value) * magnitude if abs(p) % 2 else magnitude
        magnitude = abs(float(value)) ** exponent
        return math.copysign(magnitude, float(value)) if abs(p) % 2 else magnitude
    return value**exponent


TORCH_NAMESPACE = {
    "acos": _unary(torch.acos, math.acos),
    "arccos": _unary(torch.acos, math.acos),
    "arcsin": _unary(torch.asin, math.asin),
    "atan": _unary(torch.atan, math.atan),
    "arctan": _unary(torch.atan, math.atan),
    "Abs": _unary(torch.abs, abs),
    "abs": _unary(torch.abs, abs),
    "asin": _unary(torch.asin, math.asin),
    "cos": _unary(torch.cos, math.cos),
    "cosh": _unary(torch.cosh, math.cosh),
    "exp": _unary(torch.exp, math.exp),
    "floor": _unary(torch.floor, math.floor),
    "Heaviside": _heaviside,
    "log": _unary(torch.log, math.log),
    "Max": _maximum,
    "sign": _unary(torch.sign, lambda value: -1.0 if value < 0 else 1.0 if value > 0 else 0.0),
    "sin": _unary(torch.sin, math.sin),
    "sqrt": _unary(torch.sqrt, math.sqrt),
    "tan": _unary(torch.tan, math.tan),
    "tanh": _unary(torch.tanh, math.tanh),
    "pi": math.pi,
    "E": math.e,
    "_rpow": _rational_power,
}


class FloatConstants(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(float(node.value)), node)
        return node


class ExactRationalPowers(ast.NodeTransformer):
    @staticmethod
    def _integer(node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = ExactRationalPowers._integer(node.operand)
            return None if value is None else -value
        return None

    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Pow) or not isinstance(node.right, ast.BinOp) or not isinstance(node.right.op, ast.Div):
            return node
        numerator = self._integer(node.right.left)
        denominator = self._integer(node.right.right)
        if numerator is None or denominator is None:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="_rpow", ctx=ast.Load()),
                args=[node.left, ast.Constant(numerator), ast.Constant(denominator)],
                keywords=[],
            ),
            node,
        )


def evaluate_expression(row: dict[str, Any], x: np.ndarray) -> np.ndarray:
    expression = str(row.get("analysis_expression") or row["expression"])
    syntax = ExactRationalPowers().visit(ast.parse(expression, mode="eval"))
    syntax = ast.fix_missing_locations(FloatConstants().visit(syntax))
    code = compile(syntax, f"<range-audit:{row['equation']}>", "eval")
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    namespace = dict(TORCH_NAMESPACE)
    namespace.update({f"x_{index}": xt[:, index] for index in range(xt.shape[1])})
    with torch.no_grad():
        value = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 - frozen generated artifact
    if not isinstance(value, torch.Tensor):
        value = torch.full((xt.shape[0],), float(value), dtype=torch.float64)
    return value.detach().cpu().numpy().reshape(-1)


def original_text(row):
    """Preserve original expression syntax when renaming, without simplification."""
    import re
    source=row.get('fixed_parameter_expression') or row.get('original_expression')
    if not source: return None
    names=row['original_symbols']
    aliases={name:f'x_{i}' for i,name in enumerate(names)}
    pattern=r'(?<![A-Za-z0-9_])(?:'+'|'.join(re.escape(n) for n in sorted(names,key=len,reverse=True))+r')(?![A-Za-z0-9_])'
    return re.sub(pattern,lambda m:aliases[m[0]],source).replace('^','**')

