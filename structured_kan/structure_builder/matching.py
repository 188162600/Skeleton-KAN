"""Total rooted-tree M/U matching with a single shared reserve budget.

This extends the project's rooted unordered child-assignment formulation to
all tree pairs. Missing target nodes/routes cost M; surplus builder nodes/core
routes and unallocated shared reserve cost U. Operator substitution costs one
M, except that a truly unary target is colour-neutral. No target is omitted.

Without a shared budget the child assignment decomposes into the usual nested
Hungarian problem. A shared reserve couples those assignments: an integral
assignment formulation avoids greedy topology-first capacity decisions. Exact
coverage is a separate minimum-M solve, not whether the M+U winner has M=0.
This is a structural lane-capacity certificate, not a G=3 approximation theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


MATCHER_VERSION = "rooted-total-shared-capacity-mu-v2"


@dataclass(frozen=True)
class Node:
    operator: str
    raw: int
    children: tuple["Node", ...] = ()

    @property
    def arity(self):
        return self.raw + len(self.children)


@dataclass(frozen=True)
class Match:
    mismatch: int
    unused: int
    covered: bool
    minimum_mismatch: int
    reserve_used: int
    node_pairs: tuple[tuple[int, int], ...]

    @property
    def total(self):
        return self.mismatch + self.unused


def target_node(rich):
    return Node(rich.operator, len(rich.raw_ports), tuple(target_node(child.node) for child in rich.children))


def builder_node(payload, dimension, core_by_path, path="root"):
    # Read the original operator before a Boolean-arity projection can turn a
    # multi-lane product into an apparent unary sum.
    return Node(str(payload["operator"]), min(dimension, int(core_by_path.get(path, 0))), tuple(
        builder_node(child, dimension, core_by_path, f"{path}.{i}")
        for i, child in enumerate(payload.get("children", ()))
    ))


def flatten(root):
    rows = []
    def visit(node, parent, depth):
        index = len(rows)
        rows.append((node, parent, depth))
        for child in node.children:
            visit(child, index, depth + 1)
    visit(root, None, 0)
    return rows


@lru_cache(maxsize=32768)
def match(target: Node, builder: Node, reserve: int = 0) -> Match:
    if reserve < 0 or any(n.raw < 0 for n, _, _ in flatten(target) + flatten(builder)):
        raise ValueError("capacities must be nonnegative")
    targets, builders = flatten(target), flatten(builder)
    pairs = [(i, j) for i, (_, _, td) in enumerate(targets)
             for j, (_, _, bd) in enumerate(builders) if td == bd]
    positions = {pair: index for index, pair in enumerate(pairs)}
    z = len(pairs)
    constraints, lower, upper = [], [], []
    def constrain(coefficients, low=-np.inf, high=np.inf):
        constraints.append(coefficients)
        lower.append(low)
        upper.append(high)
    for i in range(len(targets)):
        constrain({k: 1 for (t, _), k in positions.items() if t == i}, high=1)
    for j in range(len(builders)):
        constrain({k: 1 for (_, b), k in positions.items() if b == j}, high=1)
    constrain({positions[0, 0]: 1}, low=1, high=1)
    m_coefs, u_coefs = np.zeros(z + 1), np.zeros(z + 1)
    deficits = {}
    for (i, j), k in positions.items():
        t, tp, _ = targets[i]
        b, bp, _ = builders[j]
        if tp is not None:
            constrain({k: 1, positions[tp, bp]: -1}, high=0)
        substitution = int(t.operator != b.operator and t.arity != 1)
        deficit = max(0, t.raw - b.raw)
        deficits[k] = -deficit
        m_coefs[k] = substitution + deficit - 1 - t.raw
        u_coefs[k] = -1 - min(t.raw, b.raw)
    # One allocation budget for the entire builder, never reserve copies at
    # every destination. Extra unused reserve is counted once in U.
    constrain({**deficits, z: 1}, high=0)
    m_coefs[z] = u_coefs[z] = -1
    m_base = sum(1 + node.raw for node, _, _ in targets)
    u_base = sum(1 + node.raw for node, _, _ in builders) + reserve
    matrix = lil_matrix((len(constraints), z + 1), dtype=float)
    for row, values in enumerate(constraints):
        for column, value in values.items():
            matrix[row, column] = value
    bounds = Bounds(np.zeros(z + 1), np.asarray([1] * z + [reserve], dtype=float))
    constraint = LinearConstraint(matrix.tocsr(), lower, upper)
    def solve(cost):
        result = milp(cost, integrality=np.ones(z + 1), bounds=bounds, constraints=constraint,
                      options={"mip_rel_gap": 0.0})
        if not result.success or result.x is None:
            raise RuntimeError(f"total matching did not certify an optimum: {result.message}")
        value = np.rint(result.x).astype(np.int64)
        if np.max(np.abs(result.x - value)) > 1e-5:
            raise RuntimeError("nonintegral alignment solution")
        return value
    # Integer scaling makes the lexicographic objectives exact. First minimize
    # M+U and then M, avoiding an arbitrary floating-point epsilon tie break.
    bound = m_base + u_base + len(targets) + 1
    solution = solve(bound * (m_coefs + u_coefs) + m_coefs)
    mismatch = int(round(m_base + m_coefs @ solution))
    unused = int(round(u_base + u_coefs @ solution))
    if mismatch < 0 or unused < 0:
        raise RuntimeError("negative mismatch or unused capacity")
    if mismatch == 0:
        minimum = 0
    else:
        exact_solution = solve(bound * m_coefs + u_coefs)
        minimum = int(round(m_base + m_coefs @ exact_solution))
    return Match(mismatch, unused, minimum == 0, minimum, int(solution[z]), tuple(
        pair for pair, k in positions.items() if solution[k]
    ))
